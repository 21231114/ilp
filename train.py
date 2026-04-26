"""
Unsupervised Training with Tolerance-Relaxed State-Aware ALM.

Trains a GNN to solve ILP problems without labeled solutions.
The loss function uses an Augmented Lagrangian Method (ALM) with
state-aware margins to ensure rounded solutions satisfy constraints.
"""

import argparse
import os
import math
import time
import random
import copy

import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.utils import unbatch

from utils import TASKS, extract_raw_ilp
from gnn import GNNPolicy
from dataset.unsupervised_dataset import UnsupervisedGraphDataset

os.environ['TORCH'] = torch.__version__
os.environ['DGLBACKEND'] = "pytorch"
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.sparse.check_sparse_tensor_invariants.disable()

# ============================================================
#  Task-specific defaults
# ============================================================
TASK_BATCH_SIZE = {'CA': 4, 'WA': 4, 'IP': 4, 'SC': 1, 'IS': 4}


# ============================================================
#  Gumbel-Softmax Sampling Utilities
# ============================================================

def gumbel_sample(logits, N, tau=1.0):
    """
    Gumbel-Softmax sampling for differentiable binary decisions.

    Args:
        logits: [n_vars] or [n_vars, 1] raw logits from the GNN.
        N: number of solutions to sample.
        tau: Gumbel-Softmax temperature.

    Returns:
        [N, n_vars] binary samples (hard via straight-through estimator).
    """
    logits = logits.reshape(-1, 1)
    logits = logits.repeat(N, 1, 1)                                  # [N, n_vars, 1]
    logits = torch.cat([torch.zeros_like(logits), logits], dim=-1)    # [N, n_vars, 2]
    return torch.nn.functional.gumbel_softmax(logits, tau=tau, hard=True)[:, :, 1]


def build_dense_A(raw_cons_indices, raw_cons_values, n_cons, n_vars, device):
    """Build dense constraint matrix A from sparse representation."""
    A = torch.zeros(n_cons, n_vars, device=device)
    row = raw_cons_indices[0].to(device)
    col = raw_cons_indices[1].to(device)
    val = raw_cons_values.to(device)
    A[row, col] = val
    return A


def map_logits_to_raw(logits_gnn, gnn_to_raw_map, n_raw_vars, device):
    """
    Map GNN-ordered logits to raw ILP variable order via scatter-average.
    """
    logits_raw = torch.zeros(n_raw_vars, device=device)
    count = torch.zeros(n_raw_vars, device=device)
    gnn_to_raw = gnn_to_raw_map.to(device)
    logits_raw.scatter_add_(0, gnn_to_raw, logits_gnn)
    count.scatter_add_(0, gnn_to_raw, torch.ones_like(logits_gnn))
    count = count.clamp(min=1)
    return logits_raw / count


@torch.no_grad()
def evaluate_by_sampling(logits_gnn, graph, n_eval_samples, device):
    """
    Evaluate a graph by sampling many solutions and finding the best feasible one.

    Returns:
        best_feasible_obj: best objective among feasible solutions (inf if none).
        best_obj: best objective among all solutions.
        mean_obj: mean objective over all solutions.
        mean_feasible_obj: mean objective among feasible solutions (inf if none).
        n_feasible: number of feasible solutions found.
    """
    n_raw_vars = graph.obj_coeffs.shape[0]
    n_cons = graph.raw_n_cons if isinstance(graph.raw_n_cons, int) else graph.raw_n_cons.item()

    logits_raw = map_logits_to_raw(logits_gnn, graph.gnn_to_raw_map, n_raw_vars, device)

    A = build_dense_A(graph.raw_cons_indices, graph.raw_cons_values, n_cons, n_raw_vars, device)
    b = graph.raw_rhs.to(device).reshape(-1, 1)
    c = graph.obj_coeffs.to(device).reshape(-1, 1)

    # Sample solutions
    xx = gumbel_sample(logits_raw, n_eval_samples, tau=1.0).float().reshape(n_eval_samples, -1)

    # Objectives for all samples
    objs = (xx @ c).squeeze(-1)  # [n_eval_samples]

    # Find feasible solutions: all constraints satisfied
    violations = torch.relu(A @ xx.T - b).sum(dim=0)  # [n_eval_samples]
    feasible_mask = (violations == 0)
    n_feasible = feasible_mask.sum().item()

    if n_feasible > 0:
        best_feasible_obj = objs[feasible_mask].min().item()
        mean_feasible_obj = objs[feasible_mask].mean().item()
    else:
        best_feasible_obj = float('inf')
        mean_feasible_obj = float('inf')

    best_obj = objs.min().item()
    mean_obj = objs.mean().item()

    return best_feasible_obj, best_obj, mean_obj, mean_feasible_obj, n_feasible


# ============================================================
#  ALM Helper Functions
# ============================================================

def compute_K(gamma, K_max=10.0):
    """
    Rounding error bound constant K(gamma).
    Has a phase transition at gamma_c ~ 19.56.
    Capped at K_max for numerical stability.
    """
    gamma_c = 19.56
    if gamma <= gamma_c:
        return 0.5
    disc = 0.25 - 4.0 / gamma
    if disc < 0:
        return 0.5
    x_star = 0.75 + 0.5 * math.sqrt(disc)
    K = (1.0 - x_star) * math.exp(gamma / 2.0 * (x_star - 0.5) ** 2)
    return K


def compute_state_uncertainty(x_hat, gamma):
    """
    State uncertainty field:  u_i = exp(-gamma * (x_hat_i - 0.5)^2)
    Returns sqrt(u_i) for direct use in margin computation.
    """
    u = torch.exp(-gamma * (x_hat - 0.5) ** 2)
    return torch.sqrt(u)


def compute_alm_loss(
    x_hat,           # [total_vars] GNN sigmoid output
    batch,           # PyG batch object
    gamma,           # current gamma
    tau_x0,          # reference point x_0 in (0,1) for per-constraint tau_j(gamma)
    lambda_global,   # scalar global Lagrangian multiplier
    rho,             # quadratic penalty parameter
    cons_norm_cache=None,  # optional per-constraint normalization factors
    entropy_weight=0.0,    # binary entropy regularization weight
    tau_min=None,    # minimum value for tau_j(gamma); clamp tau_vec to this floor
    obj_margin_weight=1.0, # weight for margin term in objective function
    cons_loss_normalize=False, # use xi.mean() instead of xi.sum() for scale-invariant constraint cost
    tau_scale=1.0,   # dynamic scaling factor for tau tolerance (reduced when model collapses)
):
    """
    Compute the full Augmented Lagrangian loss.

    tau_j(gamma) = K(gamma) * sqrt(u(x_0)) * sum_i |A_ji|  (per-constraint tolerance)

    Returns:
        loss:           total ALM loss (scalar)
        f_tilde:        margin-aware objective value (scalar)
        xi:             [total_cons] per-constraint violation vector (with tau)
        xi_no_tau:      [total_cons] violation without tau (for threshold metrics)
        max_violation:  scalar max violation
        mean_violation: scalar mean violation
        entropy_val:    scalar entropy regularization value
    """
    device = x_hat.device
    K = compute_K(gamma)

    # --- 1. State uncertainty ---
    sqrt_u = compute_state_uncertainty(x_hat, gamma)

    # --- 2. Map GNN outputs to raw ILP variable order ---
    # gnn_to_raw_map maps each GNN variable to its index in the raw ILP
    gnn_to_raw = batch.gnn_to_raw_map.to(device)
    n_raw_vars = batch.obj_coeffs.shape[0]

    # Scatter GNN outputs to raw ILP order
    x_raw = torch.zeros(n_raw_vars, device=device)
    sqrt_u_raw = torch.zeros(n_raw_vars, device=device)
    count_raw = torch.zeros(n_raw_vars, device=device)

    x_raw.scatter_add_(0, gnn_to_raw, x_hat)
    sqrt_u_raw.scatter_add_(0, gnn_to_raw, sqrt_u)
    count_raw.scatter_add_(0, gnn_to_raw, torch.ones_like(x_hat))
    # Avoid division by zero for unmapped variables
    count_raw = count_raw.clamp(min=1)
    x_raw = x_raw / count_raw
    sqrt_u_raw = sqrt_u_raw / count_raw

    # --- 3. Margin-aware objective ---
    c = batch.obj_coeffs.to(device)
    f_base = (c * x_raw).sum()
    f_margin = K * (c.abs() * sqrt_u_raw).sum()
    f_tilde = f_base + obj_margin_weight * f_margin

    # Normalize objective by sum(|c_i|) for scale balance with constraints
    c_norm = c.abs().sum().clamp(min=1.0)
    f_tilde_normalized = f_tilde / c_norm

    # --- 4. Constraint violations with margin and tolerance ---
    cons_idx = batch.raw_cons_indices.to(device)   # [2, n_edges]
    cons_val = batch.raw_cons_values.to(device)     # [n_edges]
    rhs = batch.raw_rhs.to(device)                  # [n_cons]
    n_cons = rhs.shape[0]

    cons_row = cons_idx[0]  # constraint indices
    cons_col = cons_idx[1]  # variable indices

    # A_j @ x_hat
    Ax_per_edge = cons_val * x_raw[cons_col]
    Ax = torch.zeros(n_cons, device=device)
    Ax.scatter_add_(0, cons_row, Ax_per_edge)

    # sum_i |A_ji| * sqrt(u_i)  for margin
    abs_cons_val = cons_val.abs()
    abs_A_sqrt_u_per_edge = abs_cons_val * sqrt_u_raw[cons_col]
    margin_sum = torch.zeros(n_cons, device=device)
    margin_sum.scatter_add_(0, cons_row, abs_A_sqrt_u_per_edge)

    # Raw violation without tau: ReLU(Ax - b + K*margin_sum)
    raw_no_tau = Ax - rhs + K * margin_sum
    xi_no_tau = torch.relu(raw_no_tau)

    # Raw violation without margin and tau: ReLU(Ax - b)
    xi_raw = torch.relu(Ax - rhs)

    # Per-constraint tolerance: tau_j(gamma) = K(gamma) * sqrt(u(x_0)) * sum_i |A_ji|
    sqrt_u_0 = math.exp(-gamma / 2.0 * (tau_x0 - 0.5) ** 2)
    sum_abs_A = torch.zeros(n_cons, device=device)
    sum_abs_A.scatter_add_(0, cons_row, abs_cons_val)
    tau_vec = K * sqrt_u_0 * sum_abs_A * tau_scale

    # Clamp tau_vec to tau_min floor
    if tau_min is not None:
        tau_vec = torch.clamp(tau_vec, min=tau_min)

    # Violation with tolerance: ReLU(Ax - b + K*margin_sum - tau_j)
    raw_violation = raw_no_tau - tau_vec

    # Optional: normalize by constraint scale for balanced penalties
    if cons_norm_cache is not None:
        raw_violation = raw_violation / cons_norm_cache.to(device)

    xi = torch.relu(raw_violation)

    # --- 5. Augmented Lagrangian loss ---
    if cons_loss_normalize and xi.numel() > 0:
        lagrangian_term = lambda_global * xi.mean()
        penalty_term = (rho / 2.0) * (xi ** 2).mean()
    else:
        lagrangian_term = lambda_global * xi.sum()
        penalty_term = (rho / 2.0) * (xi ** 2).sum()
    loss = f_tilde_normalized + lagrangian_term + penalty_term

    # --- 6. Binary entropy regularization ---
    # Encourages x_hat toward 0 or 1: H(x) = -[x*log(x) + (1-x)*log(1-x)]
    # We MINIMIZE negative entropy (= maximize entropy early, then gamma takes over)
    # Actually we want to push toward 0/1, so we ADD entropy as penalty
    entropy_val = torch.tensor(0.0, device=device)
    if entropy_weight > 0:
        x_clamped = x_hat.clamp(1e-6, 1 - 1e-6)
        binary_entropy = -(x_clamped * x_clamped.log() + (1 - x_clamped) * (1 - x_clamped).log())
        # binary_entropy is maximal at x=0.5, zero at x=0 or x=1
        # We want to push toward 0/1, so minimize this (it's already positive)
        entropy_val = binary_entropy.mean()
        loss = loss + entropy_weight * entropy_val

    # --- 7. Statistics ---
    max_violation = xi.max().item() if xi.numel() > 0 else 0.0
    mean_violation = xi.mean().item() if xi.numel() > 0 else 0.0

    return loss, f_tilde.item(), f_base.item(), xi, xi_no_tau, xi_raw, max_violation, mean_violation, entropy_val.item(), tau_vec


def compute_constraint_norms(batch, device):
    """
    Compute L1 norm of each constraint row for normalization.
    Returns [n_cons] tensor of norms (clamped to avoid division by zero).
    """
    cons_idx = batch.raw_cons_indices.to(device)
    cons_val = batch.raw_cons_values.to(device)
    rhs = batch.raw_rhs.to(device)
    n_cons = rhs.shape[0]

    norms = torch.zeros(n_cons, device=device)
    norms.scatter_add_(0, cons_idx[0], cons_val.abs())
    # Add |b_j| to the norm for numerical stability
    norms = norms + rhs.abs()
    return norms.clamp(min=1e-4)


@torch.no_grad()
def evaluate_discrete(x_hat, batch, device):
    """
    Round x_hat to 0/1 and evaluate on the original ILP.

    Returns:
        feasibility_rate: fraction of constraints satisfied
        discrete_obj:     c^T * round(x_hat) (raw objective, not margin-aware)
        polarization_rate: fraction of variables near 0 or 1
        mean_uncertainty:  mean of u_i
        avg_violation_per_instance: average sum of ReLU(Ax-b) per instance
        n_feasible_instances: number of instances where ALL constraints are satisfied
        n_infeasible_instances: number of instances with at least one violated constraint
    """
    # Map to raw order
    gnn_to_raw = batch.gnn_to_raw_map.to(device)
    n_raw_vars = batch.obj_coeffs.shape[0]
    x_raw = torch.zeros(n_raw_vars, device=device)
    count_raw = torch.zeros(n_raw_vars, device=device)
    x_raw.scatter_add_(0, gnn_to_raw, x_hat)
    count_raw.scatter_add_(0, gnn_to_raw, torch.ones_like(x_hat))
    count_raw = count_raw.clamp(min=1)
    x_raw = x_raw / count_raw

    # Round
    x_rounded = torch.round(x_raw)

    # Discrete objective
    c = batch.obj_coeffs.to(device)
    discrete_obj = (c * x_rounded).sum().item()

    # Check constraint satisfaction: A @ x_rounded <= b
    cons_idx = batch.raw_cons_indices.to(device)
    cons_val = batch.raw_cons_values.to(device)
    rhs = batch.raw_rhs.to(device)
    n_cons = rhs.shape[0]

    cons_row = cons_idx[0]
    cons_col = cons_idx[1]

    Ax_rounded = torch.zeros(n_cons, device=device)
    Ax_rounded.scatter_add_(0, cons_row, cons_val * x_rounded[cons_col])

    violations = Ax_rounded - rhs
    satisfied = (violations <= 1e-6).float()
    feasibility_rate = satisfied.mean().item() if n_cons > 0 else 1.0

    # Per-instance violation and feasibility
    violation_per_cons = torch.relu(violations)
    n_instances = batch.num_graphs
    raw_n_cons_tensor = batch.raw_n_cons.long().to(device) if torch.is_tensor(batch.raw_n_cons) else torch.tensor([batch.raw_n_cons], device=device).long()
    raw_cons_batch = torch.repeat_interleave(
        torch.arange(n_instances, device=device),
        raw_n_cons_tensor,
    )

    # Sum of violations per instance
    violation_per_instance = torch.zeros(n_instances, device=device)
    violation_per_instance.scatter_add_(0, raw_cons_batch, violation_per_cons)
    avg_violation_per_instance = violation_per_instance.mean().item()

    # Count violated constraints per instance
    violated_flags = (violations > 1e-6).float()
    violated_per_instance = torch.zeros(n_instances, device=device)
    violated_per_instance.scatter_add_(0, raw_cons_batch, violated_flags)
    n_feasible_instances = (violated_per_instance == 0).sum().item()
    n_infeasible_instances = n_instances - n_feasible_instances

    # Polarization rate: |x_hat - 0.5| > 0.45 (i.e., x < 0.05 or x > 0.95)
    polarized = ((x_hat < 0.05) | (x_hat > 0.95)).float()
    polarization_rate = polarized.mean().item()

    # Mean uncertainty
    # We don't have gamma here, so use a simple measure
    mean_uncertainty = (4 * x_hat * (1 - x_hat)).mean().item()  # max at 0.5, 0 at 0/1

    return (feasibility_rate, discrete_obj, polarization_rate, mean_uncertainty,
            avg_violation_per_instance, n_feasible_instances, n_infeasible_instances)


def obj_is_better(curr, best, is_minimize, tol=1e-6):
    """Check if curr objective is better than best, respecting optimization direction."""
    if is_minimize:
        return curr < best - tol
    else:
        return curr > best + tol


# ============================================================
#  EMA Model
# ============================================================

class EMAModel:
    """Exponential Moving Average of model parameters for stable inference."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply(self, model):
        """Apply EMA weights to model (returns backup for restoration)."""
        backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
        return backup

    def restore(self, model, backup):
        """Restore original weights from backup."""
        for name, param in model.named_parameters():
            if name in backup:
                param.data.copy_(backup[name])

    def state_dict(self, model):
        """Merge EMA trainable params with model buffers (e.g. BatchNorm running stats)."""
        state = model.state_dict()
        for name in self.shadow:
            state[name] = self.shadow[name]
        return state


# ============================================================
#  Training Loop
# ============================================================

def train_epoch(model, data_loader, optimizer, scheduler, ema,
                gamma, tau, lambda_global, rho, prev_violation,
                inner_steps, beta, rho_max, gamma_max, delta_gamma,
                entropy_weight, cons_normalize, grad_clip_norm,
                device, step_counter,
                freeze_lambda=False, freeze_gamma=False, freeze_rho=False,
                tau_min=None, obj_margin_weight=1.0, cons_loss_normalize=False,
                lambda_ema_alpha=1.0,
                update_mode='alm',
                lambda_lr=1.0, target_violation=0.1,
                lambda_max=1000.0, lambda_min=0.0, rho_min=0.01,
                lambda_delta_max=5.0, tau_scale=1.0,
                freeze_tau_scale=False):
    """
    One epoch of ALM training with inner/outer loop.

    Returns updated ALM state: (gamma, lambda_global, rho, prev_violation, step_counter, metrics)
    """
    model.train()

    # Accumulators for epoch-level metrics
    total_loss = 0.0
    total_f_tilde = 0.0
    total_f_base = 0.0
    total_max_viol = 0.0
    total_mean_viol = 0.0
    total_entropy = 0.0
    total_xi_sum = 0.0
    total_xi_with_tau_sum = 0.0
    total_xi_raw_sum = 0.0
    total_num_graphs = 0
    total_pred0_ratio = 0.0
    total_pred1_ratio = 0.0
    total_xi_mean_per_cons = 0.0
    total_tau_mean = 0.0
    # Discrete metrics accumulators
    total_discrete_obj = 0.0
    total_discrete_viol = 0.0
    total_feasible_inst = 0
    total_infeasible_inst = 0
    n_batches = 0

    for batch in data_loader:
        batch = batch.to(device)

        # Precompute constraint norms if normalizing
        cons_norm = compute_constraint_norms(batch, device) if cons_normalize else None

        # --- Forward pass ---
        constraint_features_batch = torch.repeat_interleave(
            torch.arange(len(batch.ntcons), device=device),
            batch.ntcons.clone().detach().long()
        )
        variable_features_batch = torch.repeat_interleave(
            torch.arange(len(batch.ntvars), device=device),
            batch.ntvars.clone().detach().long()
        )

        batch.constraint_features[torch.isinf(batch.constraint_features)] = 10

        logits = model(
            batch.constraint_features,
            batch.edge_index,
            batch.edge_attr,
            batch.variable_features,
            batch.n_constraints,
            constraint_features_batch,
            variable_features_batch,
        )
        x_hat = logits.sigmoid()

        # --- Compute ALM loss ---
        loss, f_tilde, f_base, xi, xi_no_tau, xi_raw, max_viol, mean_viol, ent_val, tau_vec = compute_alm_loss(
            x_hat, batch, gamma, tau, lambda_global, rho,
            cons_norm_cache=cons_norm,
            entropy_weight=entropy_weight,
            tau_min=tau_min,
            obj_margin_weight=obj_margin_weight,
            cons_loss_normalize=cons_loss_normalize,
            tau_scale=tau_scale,
        )

        # Normalize by number of graphs in batch
        loss = loss / max(batch.num_graphs, 1)

        # --- Backward + optimize ---
        optimizer.zero_grad()
        loss.backward()
        if grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        if ema is not None:
            ema.update(model)

        # --- Accumulate metrics ---
        total_loss += loss.item()
        total_f_tilde += f_tilde
        total_f_base += f_base
        total_max_viol += max_viol
        total_mean_viol += mean_viol
        total_entropy += ent_val
        total_xi_sum += xi_no_tau.sum().item()
        total_xi_with_tau_sum += xi.sum().item()
        total_xi_raw_sum += xi_raw.sum().item()
        total_num_graphs += batch.num_graphs

        # Predicted 0/1 ratio and discrete evaluation
        with torch.no_grad():
            x_rounded = torch.round(x_hat)
            total_pred0_ratio += (x_rounded == 0).float().mean().item()
            total_pred1_ratio += (x_rounded == 1).float().mean().item()
            # Mean xi_j per constraint (average across constraints)
            total_xi_mean_per_cons += (xi_no_tau.sum().item() / max(xi_no_tau.numel(), 1))
            # Mean tau_j
            total_tau_mean += tau_vec.mean().item()

            # Discrete evaluation
            feas, disc_obj, _, _, avg_viol_inst, n_feas, n_infeas = evaluate_discrete(x_hat, batch, device)
            total_discrete_obj += disc_obj
            total_discrete_viol += avg_viol_inst * batch.num_graphs
            total_feasible_inst += n_feas
            total_infeasible_inst += n_infeas

        n_batches += 1
        step_counter += 1

        # --- Outer loop update (ALM mode only) ---
        if update_mode == 'alm' and step_counter % inner_steps == 0:
            with torch.no_grad():
                curr_viol = xi.sum().item() / max(batch.num_graphs, 1)
                # EMA smoothing of violation for lambda update
                # Guard against prev_violation=inf on the first outer step:
                # use curr_viol directly (no smoothing) when prev is uninitialized
                if math.isinf(prev_violation):
                    smoothed_viol = curr_viol
                else:
                    smoothed_viol = lambda_ema_alpha * curr_viol + (1 - lambda_ema_alpha) * prev_violation
                # Update global lambda
                if not freeze_lambda:
                    lambda_global = max(0.0, lambda_global + rho * smoothed_viol)
                # Update rho if violations aren't decreasing fast enough
                if not freeze_rho:
                    if curr_viol > 0.8 * prev_violation and curr_viol > 1e-4:
                        rho = min(rho * beta, rho_max)
                # Gamma annealing
                if not freeze_gamma:
                    gamma = min(gamma + delta_gamma, gamma_max)
                prev_violation = curr_viol

    n_batches = max(n_batches, 1)
    total_num_graphs = max(total_num_graphs, 1)

    # --- Epoch-level adaptive update (adaptive mode only) ---
    if update_mode == 'adaptive':
        with torch.no_grad():
            avg_disc_viol = total_discrete_viol / total_num_graphs

            # Collapse detection: ALL training instances infeasible after rounding.
            all_infeasible = (total_feasible_inst == 0)

            if not freeze_lambda:
                if all_infeasible:
                    # Infeasible: INCREASE lambda to strengthen constraint enforcement.
                    # Also REDUCE tau_scale so that the margin tolerance shrinks,
                    # making xi > 0 and giving lambda/rho actual gradient signal.
                    lambda_global = min(lambda_global + lambda_delta_max, lambda_max)
                else:
                    # Normal bidirectional update with clamped step size.
                    raw_delta = lambda_lr * (avg_disc_viol - target_violation)
                    clamped_delta = max(min(raw_delta, lambda_delta_max), -lambda_delta_max)
                    lambda_global = lambda_global + clamped_delta
                    lambda_global = max(min(lambda_global, lambda_max), lambda_min)

            if not freeze_rho:
                if all_infeasible:
                    # Infeasible: increase rho (bounded) for stronger quadratic penalty.
                    rho = min(rho * beta, rho_max)
                else:
                    if avg_disc_viol > target_violation * 2 and avg_disc_viol > 1e-4:
                        rho = min(rho * beta, rho_max)
                    elif avg_disc_viol < target_violation * 0.5 and rho > rho_min:
                        rho = max(rho / beta, rho_min)

            # Dynamic tau_scale: the key mechanism to prevent death spirals.
            # When xi=0 (tau absorbs all violations), lambda*0 + rho*0 = 0,
            # so increasing lambda/rho alone provides NO gradient signal.
            # Reducing tau_scale makes tau smaller → xi becomes non-zero →
            # the penalty terms produce useful gradients.
            if not freeze_tau_scale:
                if all_infeasible:
                    tau_scale = max(tau_scale * 0.5, 0.01)
                else:
                    # Gradually restore tau_scale when feasibility improves,
                    # but only partially — don't snap back to 1.0 instantly.
                    feas_ratio = total_feasible_inst / total_num_graphs
                    if feas_ratio > 0.5:
                        tau_scale = min(tau_scale * 1.2, 1.0)

            # Gamma: linear annealing per epoch
            if not freeze_gamma:
                gamma = min(gamma + delta_gamma, gamma_max)

            prev_violation = avg_disc_viol

    metrics = {
        'loss_total': total_loss / n_batches,
        'objective_margin': total_f_tilde / n_batches,
        'max_violation': total_max_viol / n_batches,
        'mean_violation': total_mean_viol / n_batches,
        'entropy': total_entropy / n_batches,
        'xi_sum_per_sample': total_xi_sum / total_num_graphs,
        'pred0_ratio': total_pred0_ratio / n_batches,
        'pred1_ratio': total_pred1_ratio / n_batches,
        'xi_mean_per_cons': total_xi_mean_per_cons / n_batches,
        'tau_mean': total_tau_mean / n_batches,
        'gamma': gamma,
        'rho': rho,
        'lambda_global': lambda_global,
        'K_gamma': compute_K(gamma),
        'tau_scale': tau_scale,
        # Per-instance continuous metrics
        'obj_margin_per_inst': total_f_tilde / total_num_graphs,
        'obj_raw_per_inst': total_f_base / total_num_graphs,
        'xi_margin_tau_per_inst': total_xi_with_tau_sum / total_num_graphs,
        'xi_raw_per_inst': total_xi_raw_sum / total_num_graphs,
        'xi_margin_per_inst': total_xi_sum / total_num_graphs,
        # Per-instance discrete metrics
        'disc_obj_per_inst': total_discrete_obj / total_num_graphs,
        'disc_viol_per_inst': total_discrete_viol / total_num_graphs,
        'n_feasible_inst': total_feasible_inst,
        'n_infeasible_inst': total_infeasible_inst,
    }

    return gamma, lambda_global, rho, prev_violation, step_counter, tau_scale, metrics


@torch.no_grad()
def validate_epoch(model, data_loader, gamma, tau, lambda_global, rho,
                   entropy_weight, cons_normalize, device, tau_min=None,
                   n_eval_samples=0, obj_margin_weight=1.0, cons_loss_normalize=False,
                   tau_scale=1.0):
    """
    Validate: compute ALM loss + discrete rounding metrics + sampling metrics.
    """
    model.eval()

    total_loss = 0.0
    total_f_tilde = 0.0
    total_f_base = 0.0
    total_max_viol = 0.0
    total_mean_viol = 0.0
    total_feasibility = 0.0
    total_discrete_obj = 0.0
    total_polarization = 0.0
    total_uncertainty = 0.0
    total_xi_sum = 0.0
    total_xi_with_tau_sum = 0.0
    total_xi_raw_sum = 0.0
    total_num_graphs = 0
    total_pred0_ratio = 0.0
    total_pred1_ratio = 0.0
    total_xi_mean_per_cons = 0.0
    total_tau_mean = 0.0
    # Per-instance discrete metrics
    total_discrete_viol = 0.0
    total_feasible_inst = 0
    total_infeasible_inst = 0
    # Sampling metrics accumulators
    sample_best_feasible_sum = 0.0          # sum of best feasible obj over feasible instances
    sample_mean_feasible_obj_sum = 0.0      # sum of mean feasible obj over feasible instances
    sample_best_obj_sum = 0.0               # sum of best obj (all samples) over all instances
    sample_mean_obj_sum = 0.0               # sum of mean obj (all samples) over all instances
    sample_total_feasible = 0               # total number of feasible samples
    sample_n_feasible_instances = 0         # instances with at least one feasible sample
    sample_total_samples = 0               # total number of samples across all instances
    n_batches = 0

    for batch in data_loader:
        batch = batch.to(device)

        cons_norm = compute_constraint_norms(batch, device) if cons_normalize else None

        constraint_features_batch = torch.repeat_interleave(
            torch.arange(len(batch.ntcons), device=device),
            batch.ntcons.clone().detach().long()
        )
        variable_features_batch = torch.repeat_interleave(
            torch.arange(len(batch.ntvars), device=device),
            batch.ntvars.clone().detach().long()
        )

        batch.constraint_features[torch.isinf(batch.constraint_features)] = 10

        logits = model(
            batch.constraint_features,
            batch.edge_index,
            batch.edge_attr,
            batch.variable_features,
            batch.n_constraints,
            constraint_features_batch,
            variable_features_batch,
        )
        x_hat = logits.sigmoid()

        loss, f_tilde, f_base, xi, xi_no_tau, xi_raw, max_viol, mean_viol, _, tau_vec = compute_alm_loss(
            x_hat, batch, gamma, tau, lambda_global, rho,
            cons_norm_cache=cons_norm,
            entropy_weight=entropy_weight,
            tau_min=tau_min,
            obj_margin_weight=obj_margin_weight,
            cons_loss_normalize=cons_loss_normalize,
            tau_scale=tau_scale,
        )
        loss = loss / max(batch.num_graphs, 1)

        # Discrete evaluation
        feas, disc_obj, polar, uncert, avg_viol_inst, n_feas, n_infeas = evaluate_discrete(x_hat, batch, device)

        total_loss += loss.item()
        total_f_tilde += f_tilde
        total_f_base += f_base
        total_max_viol += max_viol
        total_mean_viol += mean_viol
        total_feasibility += feas
        total_discrete_obj += disc_obj
        total_polarization += polar
        total_uncertainty += uncert
        total_xi_sum += xi_no_tau.sum().item()
        total_xi_with_tau_sum += xi.sum().item()
        total_xi_raw_sum += xi_raw.sum().item()
        total_num_graphs += batch.num_graphs

        # Per-instance discrete
        total_discrete_viol += avg_viol_inst * batch.num_graphs
        total_feasible_inst += n_feas
        total_infeasible_inst += n_infeas

        # Predicted 0/1 ratio
        x_rounded = torch.round(x_hat)
        total_pred0_ratio += (x_rounded == 0).float().mean().item()
        total_pred1_ratio += (x_rounded == 1).float().mean().item()
        # Mean xi_j per constraint
        total_xi_mean_per_cons += (xi_no_tau.sum().item() / max(xi_no_tau.numel(), 1))
        # Mean tau_j
        total_tau_mean += tau_vec.mean().item()

        # Sampling evaluation (per-graph)
        if n_eval_samples > 0:
            logits_per_graph = unbatch(logits, variable_features_batch)
            graphs = batch.to_data_list()
            for i, g in enumerate(graphs):
                best_feas, best_obj, mean_obj, mean_feas_obj, n_feas_s = evaluate_by_sampling(
                    logits_per_graph[i], g, n_eval_samples, device
                )
                sample_best_obj_sum += best_obj
                sample_mean_obj_sum += mean_obj
                sample_total_feasible += n_feas_s
                sample_total_samples += n_eval_samples
                if n_feas_s > 0:
                    sample_best_feasible_sum += best_feas
                    sample_mean_feasible_obj_sum += mean_feas_obj
                    sample_n_feasible_instances += 1

        n_batches += 1

    n_batches = max(n_batches, 1)
    total_num_graphs = max(total_num_graphs, 1)
    metrics = {
        'loss_total': total_loss / n_batches,
        'objective_margin': total_f_tilde / n_batches,
        'max_violation': total_max_viol / n_batches,
        'mean_violation': total_mean_viol / n_batches,
        'feasibility_rate': total_feasibility / n_batches,
        'discrete_objective': total_discrete_obj / n_batches,
        'polarization_rate': total_polarization / n_batches,
        'mean_uncertainty': total_uncertainty / n_batches,
        'xi_sum_per_sample': total_xi_sum / total_num_graphs,
        'objective_per_sample': total_discrete_obj / total_num_graphs,
        'pred0_ratio': total_pred0_ratio / n_batches,
        'pred1_ratio': total_pred1_ratio / n_batches,
        'xi_mean_per_cons': total_xi_mean_per_cons / n_batches,
        'tau_mean': total_tau_mean / n_batches,
        # Per-instance continuous metrics
        'obj_margin_per_inst': total_f_tilde / total_num_graphs,
        'obj_raw_per_inst': total_f_base / total_num_graphs,
        'xi_margin_tau_per_inst': total_xi_with_tau_sum / total_num_graphs,
        'xi_raw_per_inst': total_xi_raw_sum / total_num_graphs,
        'xi_margin_per_inst': total_xi_sum / total_num_graphs,
        # Per-instance discrete metrics
        'disc_obj_per_inst': total_discrete_obj / total_num_graphs,
        'disc_viol_per_inst': total_discrete_viol / total_num_graphs,
        'n_feasible_inst': total_feasible_inst,
        'n_infeasible_inst': total_infeasible_inst,
        # Sampling metrics
        'sample_best_feasible_obj': sample_best_feasible_sum / sample_n_feasible_instances if sample_n_feasible_instances > 0 else float('inf'),
        'sample_mean_feasible_obj': sample_mean_feasible_obj_sum / sample_n_feasible_instances if sample_n_feasible_instances > 0 else float('inf'),
        'sample_best_obj': sample_best_obj_sum / total_num_graphs,
        'sample_mean_obj': sample_mean_obj_sum / total_num_graphs,
        'sample_feasible_rate': sample_total_feasible / max(sample_total_samples, 1),
        'sample_n_feasible_instances': sample_n_feasible_instances,
        'sample_total_feasible': sample_total_feasible,
        'sample_total_samples': sample_total_samples,
        'n_valid_instances': total_num_graphs,
    }
    return metrics


# ============================================================
#  Logging Utilities
# ============================================================

def format_metrics(train_metrics, val_metrics, epoch, elapsed, is_minimize=True):
    """Format metrics for console and file logging."""
    obj_sign = 1.0 if is_minimize else -1.0
    opt_label = "↓" if is_minimize else "↑"
    lines = [
        f"@epoch{epoch}  TIME:{elapsed:.1f}s",
        f"  [Train] Loss={train_metrics['loss_total']:.6f}  "
        f"Obj_margin={train_metrics['objective_margin']:.4f}  "
        f"MaxViol={train_metrics['max_violation']:.6f}  "
        f"MeanViol={train_metrics['mean_violation']:.6f}  "
        f"XiSum/s={train_metrics['xi_sum_per_sample']:.6f}  "
        f"Entropy={train_metrics['entropy']:.4f}",
        f"  [Pred]  Pred0={train_metrics['pred0_ratio']:.4f}  "
        f"Pred1={train_metrics['pred1_ratio']:.4f}  "
        f"Xi_mean/cons={train_metrics['xi_mean_per_cons']:.6f}  "
        f"Tau_mean={train_metrics['tau_mean']:.6f}",
        f"  [ALM]   gamma={train_metrics['gamma']:.2f}  "
        f"rho={train_metrics['rho']:.4f}  "
        f"lambda={train_metrics['lambda_global']:.4f}  "
        f"K(gamma)={train_metrics['K_gamma']:.6f}  "
        f"tau_scale={train_metrics.get('tau_scale', 1.0):.4f}",
        # Train discrete per-instance
        f"  [TDisc] AvgObj/inst={train_metrics['disc_obj_per_inst'] * obj_sign:.4f}{opt_label}  "
        f"AvgViol/inst={train_metrics['disc_viol_per_inst']:.6f}  "
        f"FeasInst={train_metrics['n_feasible_inst']}  "
        f"InfeasInst={train_metrics['n_infeasible_inst']}",
        # Train continuous per-instance
        f"  [TCont] ObjMargin/inst={train_metrics['obj_margin_per_inst'] * obj_sign:.4f}  "
        f"ObjRaw/inst={train_metrics['obj_raw_per_inst'] * obj_sign:.4f}  "
        f"Xi_m_t/inst={train_metrics['xi_margin_tau_per_inst']:.6f}  "
        f"Xi_raw/inst={train_metrics['xi_raw_per_inst']:.6f}  "
        f"Xi_m/inst={train_metrics['xi_margin_per_inst']:.6f}",
    ]
    if val_metrics:
        lines.append(
            f"  [Valid] Loss={val_metrics['loss_total']:.6f}  "
            f"MaxViol={val_metrics['max_violation']:.6f}  "
            f"MeanViol={val_metrics['mean_violation']:.6f}  "
            f"XiSum/s={val_metrics['xi_sum_per_sample']:.6f}  "
            f"AvgObj/s={val_metrics['objective_per_sample'] * obj_sign:.4f}{opt_label}"
        )
        lines.append(
            f"  [VPred] Pred0={val_metrics['pred0_ratio']:.4f}  "
            f"Pred1={val_metrics['pred1_ratio']:.4f}  "
            f"Xi_mean/cons={val_metrics['xi_mean_per_cons']:.6f}  "
            f"Tau_mean={val_metrics['tau_mean']:.6f}"
        )
        lines.append(
            f"  [Disc]  Feasibility={val_metrics['feasibility_rate']:.4f}  "
            f"Objective={val_metrics['discrete_objective'] * obj_sign:.4f}{opt_label}  "
            f"Polarization={val_metrics['polarization_rate']:.4f}  "
            f"Uncertainty={val_metrics['mean_uncertainty']:.4f}"
        )
        # Validation discrete per-instance
        lines.append(
            f"  [VDisc] AvgObj/inst={val_metrics['disc_obj_per_inst'] * obj_sign:.4f}{opt_label}  "
            f"AvgViol/inst={val_metrics['disc_viol_per_inst']:.6f}  "
            f"FeasInst={val_metrics['n_feasible_inst']}  "
            f"InfeasInst={val_metrics['n_infeasible_inst']}"
        )
        # Validation continuous per-instance
        lines.append(
            f"  [VCont] ObjMargin/inst={val_metrics['obj_margin_per_inst'] * obj_sign:.4f}  "
            f"ObjRaw/inst={val_metrics['obj_raw_per_inst'] * obj_sign:.4f}  "
            f"Xi_m_t/inst={val_metrics['xi_margin_tau_per_inst']:.6f}  "
            f"Xi_raw/inst={val_metrics['xi_raw_per_inst']:.6f}  "
            f"Xi_m/inst={val_metrics['xi_margin_per_inst']:.6f}"
        )
        # Sampling metrics
        if val_metrics.get('sample_total_samples', 0) > 0:
            best_feas = val_metrics['sample_best_feasible_obj'] * obj_sign
            mean_feas = val_metrics['sample_mean_feasible_obj'] * obj_sign
            best_feas_str = f"{best_feas:.4f}" if not math.isinf(best_feas) else "N/A"
            mean_feas_str = f"{mean_feas:.4f}" if not math.isinf(mean_feas) else "N/A"
            lines.append(
                f"  [Sample] BestFeasObj={best_feas_str}{opt_label}  "
                f"MeanFeasObj={mean_feas_str}{opt_label}  "
                f"FeasRate={val_metrics['sample_feasible_rate']:.4f}  "
                f"FeasInst={val_metrics['sample_n_feasible_instances']}/{val_metrics['n_valid_instances']}  "
                f"FeasSamples={val_metrics['sample_total_feasible']}/{val_metrics['sample_total_samples']}"
            )
    return '\n'.join(lines)


try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


def log_to_tensorboard(writer, train_metrics, val_metrics, epoch, is_minimize=True):
    """Log all metrics to TensorBoard."""
    if writer is None:
        return

    obj_sign = 1.0 if is_minimize else -1.0

    # Loss & game dynamics
    writer.add_scalar('Loss/Total', train_metrics['loss_total'], epoch)
    writer.add_scalar('Loss/Objective_Margin', train_metrics['objective_margin'], epoch)
    writer.add_scalar('Loss/Entropy', train_metrics['entropy'], epoch)
    writer.add_scalar('Violation/Train_Max', train_metrics['max_violation'], epoch)
    writer.add_scalar('Violation/Train_Mean', train_metrics['mean_violation'], epoch)

    # ALM environment
    writer.add_scalar('ALM/Gamma', train_metrics['gamma'], epoch)
    writer.add_scalar('ALM/Rho', train_metrics['rho'], epoch)
    writer.add_scalar('ALM/Lambda_Global', train_metrics['lambda_global'], epoch)
    writer.add_scalar('ALM/K_gamma', train_metrics['K_gamma'], epoch)
    writer.add_scalar('ALM/Tau_Scale', train_metrics.get('tau_scale', 1.0), epoch)

    # New metrics
    writer.add_scalar('Prediction/Pred0_Ratio', train_metrics['pred0_ratio'], epoch)
    writer.add_scalar('Prediction/Pred1_Ratio', train_metrics['pred1_ratio'], epoch)
    writer.add_scalar('Violation/Xi_Mean_PerCons', train_metrics['xi_mean_per_cons'], epoch)
    writer.add_scalar('ALM/Tau_Mean', train_metrics['tau_mean'], epoch)

    # Train per-instance metrics
    writer.add_scalar('PerInst/Train_Obj_Margin', train_metrics['obj_margin_per_inst'], epoch)
    writer.add_scalar('PerInst/Train_Obj_Raw', train_metrics['obj_raw_per_inst'], epoch)
    writer.add_scalar('PerInst/Train_Xi_Margin_Tau', train_metrics['xi_margin_tau_per_inst'], epoch)
    writer.add_scalar('PerInst/Train_Xi_Raw', train_metrics['xi_raw_per_inst'], epoch)
    writer.add_scalar('PerInst/Train_Xi_Margin', train_metrics['xi_margin_per_inst'], epoch)
    writer.add_scalar('PerInst/Train_Disc_Obj', train_metrics['disc_obj_per_inst'], epoch)
    writer.add_scalar('PerInst/Train_Disc_Obj_Orig', train_metrics['disc_obj_per_inst'] * obj_sign, epoch)
    writer.add_scalar('PerInst/Train_Disc_Viol', train_metrics['disc_viol_per_inst'], epoch)
    writer.add_scalar('PerInst/Train_Feasible', train_metrics['n_feasible_inst'], epoch)
    writer.add_scalar('PerInst/Train_Infeasible', train_metrics['n_infeasible_inst'], epoch)

    if val_metrics:
        writer.add_scalar('Loss/Valid_Total', val_metrics['loss_total'], epoch)
        writer.add_scalar('Violation/Valid_Max', val_metrics['max_violation'], epoch)
        writer.add_scalar('Violation/Valid_Mean', val_metrics['mean_violation'], epoch)
        writer.add_scalar('Violation/Valid_XiSum_PerSample', val_metrics['xi_sum_per_sample'], epoch)
        writer.add_scalar('Objective/Valid_PerSample', val_metrics['objective_per_sample'], epoch)

        # Discrete ground truth
        writer.add_scalar('Discrete/Feasibility_Rate', val_metrics['feasibility_rate'], epoch)
        writer.add_scalar('Discrete/Objective', val_metrics['discrete_objective'], epoch)
        writer.add_scalar('State/Polarization_Rate', val_metrics['polarization_rate'], epoch)
        writer.add_scalar('State/Mean_Uncertainty', val_metrics['mean_uncertainty'], epoch)

        # Validation new metrics
        writer.add_scalar('Prediction/Valid_Pred0_Ratio', val_metrics['pred0_ratio'], epoch)
        writer.add_scalar('Prediction/Valid_Pred1_Ratio', val_metrics['pred1_ratio'], epoch)
        writer.add_scalar('Violation/Valid_Xi_Mean_PerCons', val_metrics['xi_mean_per_cons'], epoch)
        writer.add_scalar('ALM/Valid_Tau_Mean', val_metrics['tau_mean'], epoch)

        # Validation per-instance metrics
        writer.add_scalar('PerInst/Valid_Obj_Margin', val_metrics['obj_margin_per_inst'], epoch)
        writer.add_scalar('PerInst/Valid_Obj_Raw', val_metrics['obj_raw_per_inst'], epoch)
        writer.add_scalar('PerInst/Valid_Xi_Margin_Tau', val_metrics['xi_margin_tau_per_inst'], epoch)
        writer.add_scalar('PerInst/Valid_Xi_Raw', val_metrics['xi_raw_per_inst'], epoch)
        writer.add_scalar('PerInst/Valid_Xi_Margin', val_metrics['xi_margin_per_inst'], epoch)
        writer.add_scalar('PerInst/Valid_Disc_Obj', val_metrics['disc_obj_per_inst'], epoch)
        writer.add_scalar('PerInst/Valid_Disc_Obj_Orig', val_metrics['disc_obj_per_inst'] * obj_sign, epoch)
        writer.add_scalar('PerInst/Valid_Disc_Viol', val_metrics['disc_viol_per_inst'], epoch)
        writer.add_scalar('PerInst/Valid_Feasible', val_metrics['n_feasible_inst'], epoch)
        writer.add_scalar('PerInst/Valid_Infeasible', val_metrics['n_infeasible_inst'], epoch)

        # Sampling metrics
        if val_metrics.get('sample_total_samples', 0) > 0:
            writer.add_scalar('Sample/Best_Feasible_Obj', val_metrics['sample_best_feasible_obj'], epoch)
            writer.add_scalar('Sample/Mean_Feasible_Obj', val_metrics['sample_mean_feasible_obj'], epoch)
            writer.add_scalar('Sample/Best_Obj', val_metrics['sample_best_obj'], epoch)
            writer.add_scalar('Sample/Mean_Obj', val_metrics['sample_mean_obj'], epoch)
            writer.add_scalar('Sample/Feasible_Rate', val_metrics['sample_feasible_rate'], epoch)
            writer.add_scalar('Sample/Feasible_Instances', val_metrics['sample_n_feasible_instances'], epoch)
            writer.add_scalar('Sample/Total_Feasible', val_metrics['sample_total_feasible'], epoch)


# ============================================================
#  Argument Parser
# ============================================================

def get_parser():
    parser = argparse.ArgumentParser(description="Unsupervised ALM Training for ILP via GNN.")

    # Problem
    parser.add_argument("--problem_type", choices=TASKS, default='SC')

    # Model architecture (unchanged)
    parser.add_argument("--gnn_type", default='gcn')
    parser.add_argument("--emb_size", type=int, default=64)
    parser.add_argument("--cons_nfeats", type=int, default=4)
    parser.add_argument("--edge_nfeats", type=int, default=1)
    parser.add_argument("--var_nfeats", type=int, default=6)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument('--Intra_Constraint_Competitive', default=False, action='store_true')

    # Training hyperparameters
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="L2 regularization (default: %(default)s)")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override task-specific batch size")

    # ALM hyperparameters
    parser.add_argument("--tau", type=float, default=0.9,
                        help="Reference point x_0 for per-constraint tolerance "
                             "tau_j(gamma) = K(gamma)*sqrt(u(x_0))*sum|A_ji| (default: %(default)s)")
    parser.add_argument("--tau_min", type=float, default=0.99,
                        help="Minimum value for tau_j(gamma); values below this are clamped "
                             "(default: %(default)s). Set to 0 to disable.")
    parser.add_argument("--gamma_init", type=float, default=1.0,
                        help="Initial state sharpness (default: %(default)s)")
    parser.add_argument("--gamma_max", type=float, default=50.0,
                        help="Maximum gamma (default: %(default)s)")
    parser.add_argument("--delta_gamma", type=float, default=0.3,
                        help="Gamma increment per outer step (default: %(default)s)")
    parser.add_argument("--rho_init", type=float, default=1.0,
                        help="Initial penalty parameter (default: %(default)s)")
    parser.add_argument("--rho_max", type=float, default=1e5,
                        help="Maximum rho (default: %(default)s)")
    parser.add_argument("--beta", type=float, default=1.5,
                        help="Rho amplification factor (default: %(default)s)")
    parser.add_argument("--inner_steps", type=int, default=240,
                        help="Inner loop steps between ALM updates (default: %(default)s)")

    # Regularization & training tricks
    parser.add_argument("--grad_clip_norm", type=float, default=1.0,
                        help="Max gradient norm for clipping (0 = no clipping)")
    parser.add_argument("--entropy_weight", type=float, default=0.00,
                        help="Binary entropy regularization weight (default: %(default)s)")
    parser.add_argument("--obj_margin_weight", type=float, default=0.3,
                        help="Weight for margin term in objective function: "
                             "f_tilde = f_base + obj_margin_weight * K * margin "
                             "(default: %(default)s)")
    parser.add_argument("--cons_loss_normalize", action='store_true', default=True,
                        help="Use xi.mean() instead of xi.sum() for lagrangian and penalty terms, "
                             "making constraint cost scale-invariant w.r.t. number of constraints "
                             "(default: %(default)s)")
    parser.add_argument("--lambda_ema_alpha", type=float, default=0.8,
                        help="EMA smoothing factor for violation used in lambda update. "
                             "1.0 = no smoothing (original behavior), "
                             "0.3 = heavy smoothing to reduce oscillation "
                             "(default: %(default)s)")
    parser.add_argument("--n_eval_samples", type=int, default=50,
                        help="Number of Gumbel-Softmax samples for validation sampling evaluation "
                             "(0 = disabled) (default: %(default)s)")
    parser.add_argument("--cons_normalize", action='store_true', default=True,
                        help="Normalize constraint violations by row norm")
    parser.add_argument("--no_cons_normalize", action='store_false', dest='cons_normalize')
    parser.add_argument("--ema_decay", type=float, default=0.999,
                        help="EMA decay (0 = no EMA)")
    parser.add_argument("--warmup_epochs", type=int, default=10,
                        help="LR warmup epochs (default: %(default)s)")
    parser.add_argument("--lr_schedule", choices=['cosine', 'step', 'none'], default='cosine',
                        help="LR schedule type (default: %(default)s)")

    # Paths
    parser.add_argument("--instance_dir",
                        default="/home/lmh/autodl-tmp/data/l2o_milp",
                        help="Directory containing .lp/.mps instance files")
    parser.add_argument("--cache_dir", default=None,
                        help="Cache directory for preprocessed data")
    parser.add_argument("--model_save_dir", default="./pretrain_models")
    parser.add_argument("--log_save_dir", default="./train_logs")
    parser.add_argument("--tensorboard_dir", default="./tb_logs",
                        help="TensorBoard log directory")

    # Resume from checkpoint
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to .pth checkpoint to resume training from. "
                             "Supports both full checkpoints (with optimizer/ALM state) "
                             "and plain model state_dicts.")

    # Device
    parser.add_argument("--device", default="cuda:0")

    # Validation frequency
    parser.add_argument("--val_every", type=int, default=1,
                        help="Validate every N epochs (default: %(default)s)")

    # Early stopping
    parser.add_argument("--es_xi_threshold", type=float, default=1.0,
                        help="Early-stop feasibility threshold: avg sum_j xi_j per sample (default: %(default)s)")
    parser.add_argument("--patience", type=int, default=100,
                        help="Early-stop patience: epochs without improvement (default: %(default)s)")

    # ALM freezing: freeze lambda (and optionally gamma/rho) when xi_sum < threshold2
    parser.add_argument("--es_xi_threshold2", type=float, default=None,
                        help="When xi_sum/sample < this, freeze lambda. "
                             "None = disabled (default: %(default)s)")
    parser.add_argument("--threshold2_on", choices=['train', 'valid'], default='valid',
                        help="Whether es_xi_threshold2 acts on train or valid metrics "
                             "(default: %(default)s)")
    parser.add_argument("--freeze_gamma_on_feasible", action='store_true', default=False,
                        help="Also freeze gamma when xi_sum < es_xi_threshold2")
    parser.add_argument("--freeze_rho_on_feasible", action='store_true', default=False,
                        help="Also freeze rho when xi_sum < es_xi_threshold2")

    # Adaptive ALM freezing based on sampling feasibility
    parser.add_argument("--freeze_on_all_sample_feasible", action='store_true', default=False,
                        help="Freeze lambda when ALL validation instances have at least one "
                             "feasible sampled solution; resume normal updates when any instance "
                             "has no feasible samples. Requires --n_eval_samples > 0. "
                             "Use --freeze_gamma_on_feasible and --freeze_rho_on_feasible "
                             "to also freeze gamma and rho.")
    parser.add_argument("--freeze_tau_scale_on_feasible", action='store_true', default=False,
                        help="Also freeze tau_scale when ALM params are frozen due to "
                             "feasibility (es_xi_threshold2 or freeze_on_all_sample_feasible). "
                             "Prevents tau tolerance from drifting while other params are locked.")

    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducibility (default: %(default)s)")

    # ---- Adaptive update mode (DiffILO-inspired) ----
    parser.add_argument("--update_mode", choices=['alm', 'adaptive'], default='alm',
                        help="Parameter update mode: 'alm' = standard ALM outer loop, "
                             "'adaptive' = bidirectional DiffILO-style update using raw "
                             "constraint violations (default: %(default)s)")
    parser.add_argument("--lambda_lr", type=float, default=1.0,
                        help="[adaptive] Step size for lambda update: "
                             "lambda += lambda_lr * (avg_raw_viol - target_violation) "
                             "(default: %(default)s)")
    parser.add_argument("--target_violation", type=float, default=0.1,
                        help="[adaptive] Target raw violation level per instance. "
                             "Lambda increases when above target, decreases when below. "
                             "Set > 0 to allow bidirectional updates (default: %(default)s)")
    parser.add_argument("--lambda_max", type=float, default=1000.0,
                        help="[adaptive] Maximum lambda value (default: %(default)s)")
    parser.add_argument("--lambda_min", type=float, default=0.0,
                        help="[adaptive] Minimum lambda value (default: %(default)s)")
    parser.add_argument("--rho_min", type=float, default=0.01,
                        help="[adaptive] Minimum rho value for rho decay "
                             "(default: %(default)s)")
    parser.add_argument("--lambda_delta_max", type=float, default=5.0,
                        help="[adaptive] Maximum per-epoch change in lambda. "
                             "Prevents instant saturation when discrete violation "
                             "is much larger than target (default: %(default)s)")

    return parser


# ============================================================
#  Main
# ============================================================

def main():
    parser = get_parser()
    args = parser.parse_args()

    device = args.device
    problem_type = args.problem_type
    batch_size = args.batch_size or TASK_BATCH_SIZE.get(problem_type, 4)

    # Fix random seed for reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    save_name = (
        f'ALM_tau{args.tau}_taumin{args.tau_min}_gamma{args.gamma_init}_rho{args.rho_init}'
        f'_inner{args.inner_steps}_ent{args.entropy_weight}'
        f'_ICC{args.Intra_Constraint_Competitive}'
        f'_esxi{args.es_xi_threshold}_esxi2{args.es_xi_threshold2}_t2on{args.threshold2_on}'
        f'_omw{args.obj_margin_weight}'
        f'_cln{args.cons_loss_normalize}'
        f'_lema{args.lambda_ema_alpha}'
    )
    if args.update_mode == 'adaptive':
        save_name += f'_adaptive_llr{args.lambda_lr}_tv{args.target_violation}'

    # Create directories
    model_save_path = os.path.join(args.model_save_dir, problem_type)
    log_save_path = os.path.join(args.log_save_dir, problem_type)
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(log_save_path, exist_ok=True)

    log_file = open(f'{log_save_path}/{save_name}_train.log', 'w')

    # TensorBoard
    tb_writer = None
    if HAS_TENSORBOARD:
        tb_dir = os.path.join(args.tensorboard_dir, problem_type, save_name)
        os.makedirs(tb_dir, exist_ok=True)
        tb_writer = SummaryWriter(tb_dir)
        print(f"TensorBoard logging to {tb_dir}")

    # ---- Data loading ----
    ins_dir = os.path.join(args.instance_dir, problem_type)
    all_instances = sorted([
        os.path.join(ins_dir, f)
        for f in os.listdir(ins_dir)
        if f.endswith(('.lp', '.mps'))
    ])

    random.shuffle(all_instances)
    split = int(0.8 * len(all_instances))
    train_files = all_instances[:split]
    valid_files = all_instances[split:]

    cache_dir = args.cache_dir
    if cache_dir is None:
        cache_dir = os.path.join(args.log_save_dir, problem_type, 'unsup_cache')

    train_data = UnsupervisedGraphDataset(train_files, cache_dir=cache_dir)
    valid_data = UnsupervisedGraphDataset(valid_files, cache_dir=cache_dir)

    train_loader = torch_geometric.loader.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=args.num_workers
    )
    valid_loader = torch_geometric.loader.DataLoader(
        valid_data, batch_size=batch_size, shuffle=False, num_workers=args.num_workers
    )

    print(f"Train instances: {len(train_files)}, Valid instances: {len(valid_files)}")
    print(f"Batch size: {batch_size}")

    # Determine objective sense (minimize or maximize)
    first_data = train_data[0]
    if hasattr(first_data, 'obj_sense_min'):
        is_minimize = bool(first_data.obj_sense_min.item())
    else:
        _raw = extract_raw_ilp(train_files[0])
        is_minimize = _raw['obj_sense_min']
    obj_sign = 1.0 if is_minimize else -1.0
    opt_dir = "MINIMIZE" if is_minimize else "MAXIMIZE"
    print(f"Objective sense: {opt_dir}")

    # ---- Model ----
    model = GNNPolicy(
        emb_size=args.emb_size,
        cons_nfeats=args.cons_nfeats,
        edge_nfeats=args.edge_nfeats,
        var_nfeats=args.var_nfeats,
        depth=args.depth,
        Intra_Constraint_Competitive=args.Intra_Constraint_Competitive,
    ).to(device)

    # Initialize output layer bias to 0 (sigmoid(0) = 0.5, neutral start)
    for m in model.vars_output_layer:
        if isinstance(m, nn.Linear) and m.out_features == 1:
            nn.init.zeros_(m.bias)
            nn.init.xavier_uniform_(m.weight, gain=0.1)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # ---- LR Schedule ----
    total_steps = args.num_epochs * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)

    if args.lr_schedule == 'cosine' and total_steps > 0:
        def lr_lambda(step):
            if step < warmup_steps:
                return max(step / max(warmup_steps, 1), 0.01)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return max(0.5 * (1 + math.cos(math.pi * progress)), 0.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif args.lr_schedule == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=total_steps // 5, gamma=0.5)
    else:
        scheduler = None

    # ---- EMA ----
    ema = EMAModel(model, decay=args.ema_decay) if args.ema_decay > 0 else None

    # ---- ALM State ----
    gamma = args.gamma_init
    rho = args.rho_init
    lambda_global = 0.0
    prev_violation = float('inf')
    step_counter = 0
    start_epoch = 0
    tau_scale = 1.0  # dynamic tau tolerance scaling (reduced on collapse, restored on recovery)

    # ---- Resume from checkpoint ----
    if args.resume_from is not None:
        assert os.path.isfile(args.resume_from), f"Checkpoint not found: {args.resume_from}"
        print(f"Loading checkpoint from {args.resume_from} ...")
        ckpt = torch.load(args.resume_from, map_location=device)

        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            # Full checkpoint with training state
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if scheduler is not None and 'scheduler_state_dict' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            if ema is not None and 'ema_shadow' in ckpt:
                ema.shadow = ckpt['ema_shadow']
            gamma = ckpt.get('gamma', gamma)
            rho = ckpt.get('rho', rho)
            lambda_global = ckpt.get('lambda_global', lambda_global)
            prev_violation = ckpt.get('prev_violation', prev_violation)
            step_counter = ckpt.get('step_counter', step_counter)
            tau_scale = ckpt.get('tau_scale', tau_scale)
            start_epoch = ckpt.get('epoch', 0) + 1
            print(f"  Resumed full checkpoint: epoch={start_epoch}, gamma={gamma:.2f}, "
                  f"rho={rho:.4f}, lambda={lambda_global:.4f}, tau_scale={tau_scale:.4f}")
        else:
            # Plain state_dict (model weights only)
            model.load_state_dict(ckpt)
            print("  Loaded model weights (no optimizer/ALM state). Training from epoch 0.")

    # ---- Training ----
    best_val_xi_sum = float('inf')
    best_val_obj = float('inf') if is_minimize else float('-inf')
    best_feasible = False  # whether best model satisfies xi threshold
    patience_counter = 0
    alm_frozen = False  # whether ALM params are frozen due to es_xi_threshold2
    sample_alm_frozen = False  # whether ALM params are frozen due to all-sample-feasible

    # Restore frozen states from checkpoint if resuming
    if args.resume_from is not None:
        try:
            alm_frozen = ckpt.get('alm_frozen', False)
            sample_alm_frozen = ckpt.get('sample_alm_frozen', False)
        except (NameError, AttributeError):
            pass
    best_allfeas_obj = float('inf') if is_minimize else float('-inf')  # best discrete obj when ALL val instances are feasible
    best_sample_allfeas_obj = float('inf') if is_minimize else float('-inf')  # best sampling obj when ALL val instances have feasible samples

    # Resolve tau_min: 0 means disabled
    tau_min = args.tau_min if args.tau_min > 0 else None

    print(f"\n{'='*70}")
    print(f"Starting Unsupervised ALM Training for {problem_type} ({opt_dir})")
    print(f"  update_mode={args.update_mode}")
    print(f"  tau={args.tau}, tau_min={tau_min}, gamma_init={args.gamma_init}, rho_init={args.rho_init}")
    print(f"  inner_steps={args.inner_steps}, beta={args.beta}")
    print(f"  entropy_weight={args.entropy_weight}, obj_margin_weight={args.obj_margin_weight}, grad_clip={args.grad_clip_norm}")
    print(f"  cons_loss_normalize={args.cons_loss_normalize}, lambda_ema_alpha={args.lambda_ema_alpha}")
    print(f"  LR={args.lr}, schedule={args.lr_schedule}, warmup={args.warmup_epochs}")
    print(f"  n_eval_samples={args.n_eval_samples}")
    if args.update_mode == 'adaptive':
        print(f"  [adaptive] lambda_lr={args.lambda_lr}, target_violation={args.target_violation}, "
              f"lambda_delta_max={args.lambda_delta_max}")
        print(f"  [adaptive] lambda_range=[{args.lambda_min}, {args.lambda_max}], rho_min={args.rho_min}")
    if args.freeze_on_all_sample_feasible:
        print(f"  freeze_on_all_sample_feasible=True "
              f"(freeze_gamma={args.freeze_gamma_on_feasible}, "
              f"freeze_rho={args.freeze_rho_on_feasible}, "
              f"freeze_tau_scale={args.freeze_tau_scale_on_feasible})")
    print(f"{'='*70}\n")

    for epoch in range(start_epoch, args.num_epochs):
        t0 = time.time()

        # Train
        any_frozen = alm_frozen or sample_alm_frozen
        freeze_lambda = any_frozen
        freeze_gamma = any_frozen and args.freeze_gamma_on_feasible
        freeze_rho = any_frozen and args.freeze_rho_on_feasible
        freeze_tau_scale = any_frozen and args.freeze_tau_scale_on_feasible
        gamma, lambda_global, rho, prev_violation, step_counter, tau_scale, train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, ema,
            gamma, args.tau, lambda_global, rho, prev_violation,
            args.inner_steps, args.beta, args.rho_max, args.gamma_max, args.delta_gamma,
            args.entropy_weight, args.cons_normalize, args.grad_clip_norm,
            device, step_counter,
            freeze_lambda=freeze_lambda, freeze_gamma=freeze_gamma, freeze_rho=freeze_rho,
            tau_min=tau_min, obj_margin_weight=args.obj_margin_weight,
            cons_loss_normalize=args.cons_loss_normalize,
            lambda_ema_alpha=args.lambda_ema_alpha,
            update_mode=args.update_mode,
            lambda_lr=args.lambda_lr, target_violation=args.target_violation,
            lambda_max=args.lambda_max, lambda_min=args.lambda_min,
            rho_min=args.rho_min,
            lambda_delta_max=args.lambda_delta_max,
            tau_scale=tau_scale,
            freeze_tau_scale=freeze_tau_scale,
        )

        # Validate periodically
        val_metrics = None
        if (epoch + 1) % args.val_every == 0 or epoch == 0:
            # Use EMA weights for validation
            if ema is not None:
                backup = ema.apply(model)

            val_metrics = validate_epoch(
                model, valid_loader,
                gamma, args.tau, lambda_global, rho,
                args.entropy_weight, args.cons_normalize, device,
                tau_min=tau_min,
                n_eval_samples=args.n_eval_samples,
                obj_margin_weight=args.obj_margin_weight,
                cons_loss_normalize=args.cons_loss_normalize,
                tau_scale=tau_scale,
            )

            if ema is not None:
                ema.restore(model, backup)

            # Save best model
            # Primary: xi_sum below threshold → feasible; secondary: lowest discrete objective per instance
            curr_xi = val_metrics['xi_sum_per_sample']
            curr_obj_orig = val_metrics['disc_obj_per_inst'] * obj_sign  # original scale
            curr_feasible = curr_xi < args.es_xi_threshold

            is_best = False
            if curr_feasible and not best_feasible:
                # First time becoming feasible — always save
                is_best = True
            elif curr_feasible and best_feasible:
                # Both feasible — save if objective improved
                if obj_is_better(curr_obj_orig, best_val_obj, is_minimize):
                    is_best = True
            elif not curr_feasible and not best_feasible:

                # Neither feasible — save if xi_sum improved
                if curr_xi < best_val_xi_sum - 1e-6:
                    is_best = True

            if is_best:
                best_feasible = curr_feasible
                best_val_xi_sum = curr_xi
                best_val_obj = curr_obj_orig
                patience_counter = 0
                save_state = ema.state_dict(model) if ema is not None else model.state_dict()
                torch.save(save_state, os.path.join(model_save_path, f'{save_name}_model_best.pth'))
            else:
                patience_counter += 1

            # Save best all-feasible model:
            # ALL validation instances feasible after discretization + best discrete objective
            n_infeas = val_metrics['n_infeasible_inst']
            if n_infeas == 0:
                if obj_is_better(curr_obj_orig, best_allfeas_obj, is_minimize):
                    best_allfeas_obj = curr_obj_orig
                    save_state = ema.state_dict(model) if ema is not None else model.state_dict()
                    af_path = os.path.join(model_save_path, f'{save_name}_model_best_allfeas.pth')
                    torch.save(save_state, af_path)
                    print(f"  [AllFeas] Saved best all-feasible model: "
                          f"disc_obj/inst={curr_obj_orig:.4f}, n_infeasible=0")

            # Save best sampling-all-feasible model:
            # ALL validation instances have at least one feasible sample + best average best feasible obj
            if val_metrics.get('sample_total_samples', 0) > 0:
                n_feas_inst_sample = val_metrics['sample_n_feasible_instances']
                n_total_inst = val_metrics['n_valid_instances']
                if n_feas_inst_sample == n_total_inst:
                    curr_sample_obj = val_metrics['sample_best_feasible_obj'] * obj_sign
                    if obj_is_better(curr_sample_obj, best_sample_allfeas_obj, is_minimize):
                        best_sample_allfeas_obj = curr_sample_obj
                        save_state = ema.state_dict(model) if ema is not None else model.state_dict()
                        sf_path = os.path.join(model_save_path, f'{save_name}_model_best_sample.pth')
                        torch.save(save_state, sf_path)
                        print(f"  [SampleFeas] Saved best sampling-all-feasible model: "
                              f"avg_best_feas_obj={curr_sample_obj:.4f}, "
                              f"feas_inst={n_feas_inst_sample}/{n_total_inst}")

        # Check ALM freeze/unfreeze condition (can act on train or valid metrics)
        if args.es_xi_threshold2 is not None:
            if args.threshold2_on == 'train':
                freeze_xi = train_metrics['xi_sum_per_sample']
                freeze_src = 'train'
            else:
                # Only check when validation was performed this epoch
                if val_metrics is not None:
                    freeze_xi = val_metrics['xi_sum_per_sample']
                    freeze_src = 'valid'
                else:
                    freeze_xi = None
                    freeze_src = None

            if freeze_xi is not None:
                if freeze_xi < args.es_xi_threshold2:
                    if not alm_frozen:
                        alm_frozen = True
                        frozen_parts = ["lambda"]
                        if args.freeze_gamma_on_feasible:
                            frozen_parts.append("gamma")
                        if args.freeze_rho_on_feasible:
                            frozen_parts.append("rho")
                        if args.freeze_tau_scale_on_feasible:
                            frozen_parts.append("tau_scale")
                        print(f"  [Freeze] {freeze_src} xi_sum/sample={freeze_xi:.6f} < threshold2={args.es_xi_threshold2} "
                              f"=> freezing {', '.join(frozen_parts)} "
                              f"(lambda={lambda_global:.4f}, gamma={gamma:.2f}, rho={rho:.4f}, tau_scale={tau_scale:.4f})")
                else:
                    if alm_frozen:
                        alm_frozen = False
                        print(f"  [Unfreeze] {freeze_src} xi_sum/sample={freeze_xi:.6f} >= threshold2={args.es_xi_threshold2} "
                              f"=> resuming all ALM updates "
                              f"(lambda={lambda_global:.4f}, gamma={gamma:.2f}, rho={rho:.4f}, tau_scale={tau_scale:.4f})")

        # Check ALM freeze/unfreeze based on sampling feasibility
        if args.freeze_on_all_sample_feasible and val_metrics is not None:
            if val_metrics.get('sample_total_samples', 0) > 0:
                n_feas_inst_sample = val_metrics['sample_n_feasible_instances']
                n_total_inst = val_metrics['n_valid_instances']
                if n_feas_inst_sample == n_total_inst:
                    if not sample_alm_frozen:
                        sample_alm_frozen = True
                        frozen_parts = ["lambda"]
                        if args.freeze_gamma_on_feasible:
                            frozen_parts.append("gamma")
                        if args.freeze_rho_on_feasible:
                            frozen_parts.append("rho")
                        if args.freeze_tau_scale_on_feasible:
                            frozen_parts.append("tau_scale")
                        print(f"  [SampleFreeze] All {n_total_inst} val instances have feasible samples "
                              f"=> freezing {', '.join(frozen_parts)} "
                              f"(lambda={lambda_global:.4f}, gamma={gamma:.2f}, rho={rho:.4f}, tau_scale={tau_scale:.4f})")
                else:
                    if sample_alm_frozen:
                        sample_alm_frozen = False
                        print(f"  [SampleUnfreeze] {n_total_inst - n_feas_inst_sample}/{n_total_inst} val instances "
                              f"have no feasible samples => resuming all ALM updates "
                              f"(lambda={lambda_global:.4f}, gamma={gamma:.2f}, rho={rho:.4f}, tau_scale={tau_scale:.4f})")

        # Save latest (full checkpoint for resumable training)
        full_ckpt = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'gamma': gamma,
            'rho': rho,
            'lambda_global': lambda_global,
            'prev_violation': prev_violation,
            'step_counter': step_counter,
            'tau_scale': tau_scale,
            'alm_frozen': alm_frozen,
            'sample_alm_frozen': sample_alm_frozen,
        }
        if scheduler is not None:
            full_ckpt['scheduler_state_dict'] = scheduler.state_dict()
        if ema is not None:
            full_ckpt['ema_shadow'] = ema.shadow
        torch.save(full_ckpt, os.path.join(model_save_path, f'{save_name}_model_last.pth'))

        elapsed = time.time() - t0
        log_str = format_metrics(train_metrics, val_metrics, epoch, elapsed, is_minimize)
        print(log_str)
        log_file.write(log_str + '\n')
        log_file.flush()

        log_to_tensorboard(tb_writer, train_metrics, val_metrics, epoch, is_minimize)
        if tb_writer is not None:
            tb_writer.add_scalar('ALM/ALM_Frozen', int(alm_frozen), epoch)
            tb_writer.add_scalar('ALM/Sample_ALM_Frozen', int(sample_alm_frozen), epoch)

        # Early stopping based on patience
        if (val_metrics is not None
                and patience_counter >= args.patience
                and epoch > args.patience):
            print(f"\nEarly stopping at epoch {epoch}: no improvement for {args.patience} epochs.")
            print(f"  Best XiSum/sample={best_val_xi_sum:.6f}  "
                  f"Best AvgObj/sample={best_val_obj:.4f}({opt_dir})  "
                  f"Feasible={best_feasible}")
            break

    log_file.close()
    if tb_writer is not None:
        tb_writer.close()
    print("Training completed successfully.")


if __name__ == '__main__':
    main()
