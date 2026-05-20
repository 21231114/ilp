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
TASK_BATCH_SIZE = {'CA': 4, 'WA': 4, 'IP': 4, 'SC': 1, 'IS': 4, '2club': 1}


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
    gamma,           # retained for scheduler/log compatibility
    tau,             # fixed uniform constraint tolerance tau_j = tau
    mu,              # scalar constraint penalty parameter
    loss_config='sum',      # reduction/normalization mode for objective + violations
    cons_norm_cache=None,  # optional per-constraint normalization factors
    entropy_weight=0.0,    # optional binary entropy regularization weight
    tau_min=None,    # deprecated/ignored for fixed-tau loss
    tau_obj=0.0,     # objective margin tolerance
):
    """
    Compute the fixed-tau scalar-penalty loss:

        c^T x + ReLU(2 * sum_i |c_i| x_i(1-x_i) - tau_obj)
        + mu * configured_reduction_j ReLU(A_j x - b_j + 2 * sum_i |A_ji| x_i(1-x_i) - tau)

    loss_config controls how the objective and constraint violations are reduced.
    """
    device = x_hat.device

    # --- 1. Map GNN outputs to raw ILP variable order ---
    # gnn_to_raw_map maps each GNN variable to its index in the raw ILP
    gnn_to_raw = batch.gnn_to_raw_map.to(device)
    n_raw_vars = batch.obj_coeffs.shape[0]

    x_raw = torch.zeros(n_raw_vars, device=device)
    count_raw = torch.zeros(n_raw_vars, device=device)

    x_raw.scatter_add_(0, gnn_to_raw, x_hat)
    count_raw.scatter_add_(0, gnn_to_raw, torch.ones_like(x_hat))
    count_raw = count_raw.clamp(min=1)
    x_raw = x_raw / count_raw
    binary_relax = x_raw * (1.0 - x_raw)

    # --- 2. Objective: c^T x + ReLU(2 * sum_i |c_i| x_i(1-x_i) - tau_obj) ---
    c = batch.obj_coeffs.to(device)
    f_base = (c * x_raw).sum()
    f_margin_raw = 2.0 * (c.abs() * binary_relax).sum()
    f_margin = torch.relu(f_margin_raw - float(tau_obj))
    f_tilde = f_base + f_margin

    # --- 3. Constraint violations with fixed uniform tau ---
    cons_idx = batch.raw_cons_indices.to(device)   # [2, n_edges]
    cons_val = batch.raw_cons_values.to(device)     # [n_edges]
    rhs = batch.raw_rhs.to(device)                  # [n_cons]
    n_cons = rhs.shape[0]

    cons_row = cons_idx[0]
    cons_col = cons_idx[1]

    Ax_per_edge = cons_val * x_raw[cons_col]
    Ax = torch.zeros(n_cons, device=device)
    Ax.scatter_add_(0, cons_row, Ax_per_edge)

    abs_A_relax_per_edge = cons_val.abs() * binary_relax[cons_col]
    margin_sum = torch.zeros(n_cons, device=device)
    margin_sum.scatter_add_(0, cons_row, abs_A_relax_per_edge)

    raw_no_tau = Ax - rhs + 2.0 * margin_sum
    xi_no_tau = torch.relu(raw_no_tau)
    xi_raw = torch.relu(Ax - rhs)

    tau_vec = torch.full_like(rhs, fill_value=float(tau))
    raw_violation = raw_no_tau - tau_vec

    # Legacy opt-in normalization. Skip it for baseline-style normalize mode to avoid double normalization.
    if cons_norm_cache is not None and loss_config != 'normalize':
        raw_violation = raw_violation / cons_norm_cache.to(device)

    xi = torch.relu(raw_violation)

    # --- 4. Scalar penalty term ---
    mu_val = float(mu)
    positive_mask = xi > 0
    num_positive = positive_mask.sum()

    if loss_config == 'normalize':
        c_norm = c.norm().clamp(min=1e-8)
        obj_term = f_tilde / c_norm
        if num_positive.item() > 0:
            row_l2 = compute_constraint_l2_row_norms(cons_idx, cons_val, n_cons, device)
            viol_term = (xi[positive_mask] / row_l2[positive_mask]).sum() / num_positive
        else:
            viol_term = torch.zeros((), device=device, dtype=f_tilde.dtype)
        loss = obj_term + mu_val * viol_term
    elif loss_config == 'sum':
        loss = f_tilde + mu_val * xi.sum()
    elif loss_config == 'nonzero_mean':
        if num_positive.item() > 0:
            loss = f_tilde + mu_val * xi[positive_mask].sum() / num_positive
        else:
            loss = f_tilde
    elif loss_config == 'mean':
        loss = f_tilde + mu_val * xi.mean()
    else:
        raise ValueError(f"Unknown loss_config: {loss_config}")

    # --- 5. Optional binary entropy regularization ---
    entropy_val = torch.tensor(0.0, device=device)
    if entropy_weight > 0:
        x_clamped = x_hat.clamp(1e-6, 1 - 1e-6)
        binary_entropy = -(x_clamped * x_clamped.log() + (1 - x_clamped) * (1 - x_clamped).log())
        entropy_val = binary_entropy.mean()
        loss = loss + entropy_weight * entropy_val

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


def compute_constraint_l2_row_norms(cons_idx, cons_val, n_cons, device):
    """Compute baseline-style L2 norm of each sparse constraint row."""
    row_sq_sum = torch.zeros(n_cons, device=device)
    row_sq_sum.scatter_add_(0, cons_idx[0], cons_val.pow(2))
    return torch.sqrt(row_sq_sum).clamp(min=1e-8)


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


@torch.no_grad()
def evaluate_discrete_single(x_hat, graph, device):
    """Round one graph's probabilities and evaluate the original ILP."""
    gnn_to_raw = graph.gnn_to_raw_map.to(device)
    n_raw_vars = graph.obj_coeffs.shape[0]
    x_raw = torch.zeros(n_raw_vars, device=device)
    count_raw = torch.zeros(n_raw_vars, device=device)
    x_raw.scatter_add_(0, gnn_to_raw, x_hat)
    count_raw.scatter_add_(0, gnn_to_raw, torch.ones_like(x_hat))
    x_raw = x_raw / count_raw.clamp(min=1)
    x_rounded = torch.round(x_raw)

    c = graph.obj_coeffs.to(device)
    discrete_obj = (c * x_rounded).sum().item()

    cons_idx = graph.raw_cons_indices.to(device)
    cons_val = graph.raw_cons_values.to(device)
    rhs = graph.raw_rhs.to(device)
    n_cons = rhs.shape[0]

    Ax_rounded = torch.zeros(n_cons, device=device)
    Ax_rounded.scatter_add_(0, cons_idx[0], cons_val * x_rounded[cons_idx[1]])

    violations = Ax_rounded - rhs
    violation_pos = torch.relu(violations)
    feasibility_rate = (violations <= 1e-6).float().mean().item() if n_cons > 0 else 1.0
    violation_sum = violation_pos.sum().item()
    is_feasible = violation_sum <= 1e-6

    polarization_rate = ((x_hat < 0.05) | (x_hat > 0.95)).float().mean().item()
    mean_uncertainty = (4 * x_hat * (1 - x_hat)).mean().item()

    return feasibility_rate, discrete_obj, polarization_rate, mean_uncertainty, violation_sum, is_feasible


def model_forward(model, batch, device):
    """Run the GNN forward pass and return logits plus variable graph ids."""
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

    return logits, variable_features_batch


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

def train_epoch(model, data_loader, optimizer, ema,
                gamma, tau, tau_obj, mu, loss_config,
                entropy_weight, cons_normalize, grad_clip_norm,
                device, tau_min=None):
    """
    One epoch of fixed-tau scalar-penalty training, averaged per graph.
    """
    model.train()

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
    total_discrete_obj = 0.0
    total_discrete_viol = 0.0
    total_feasible_inst = 0
    total_infeasible_inst = 0

    for batch in data_loader:
        batch = batch.to(device)

        logits, variable_features_batch = model_forward(model, batch, device)
        logits_per_graph = unbatch(logits, variable_features_batch)
        graphs = batch.to_data_list()

        batch_loss = torch.zeros((), device=device)
        n_graphs = len(graphs)

        for logits_i, graph in zip(logits_per_graph, graphs):
            x_hat_i = logits_i.sigmoid()
            cons_norm = compute_constraint_norms(graph, device) if cons_normalize else None

            loss_i, f_tilde, f_base, xi, xi_no_tau, xi_raw, max_viol, mean_viol, ent_val, tau_vec = compute_alm_loss(
                x_hat_i, graph, gamma, tau, mu,
                tau_obj=tau_obj,
                loss_config=loss_config,
                cons_norm_cache=cons_norm,
                entropy_weight=entropy_weight,
                tau_min=tau_min,
            )
            batch_loss = batch_loss + loss_i

            total_loss += loss_i.item()
            total_f_tilde += f_tilde
            total_f_base += f_base
            total_max_viol += max_viol
            total_mean_viol += mean_viol
            total_entropy += ent_val
            total_xi_sum += xi_no_tau.sum().item()
            total_xi_with_tau_sum += xi.sum().item()
            total_xi_raw_sum += xi_raw.sum().item()
            total_xi_mean_per_cons += xi_no_tau.sum().item() / max(xi_no_tau.numel(), 1)
            total_tau_mean += tau_vec.mean().item()

            with torch.no_grad():
                x_rounded = torch.round(x_hat_i)
                total_pred0_ratio += (x_rounded == 0).float().mean().item()
                total_pred1_ratio += (x_rounded == 1).float().mean().item()
                _, disc_obj, _, _, viol_sum, is_feasible = evaluate_discrete_single(x_hat_i, graph, device)
                total_discrete_obj += disc_obj
                total_discrete_viol += viol_sum
                total_feasible_inst += int(is_feasible)
                total_infeasible_inst += int(not is_feasible)

        loss = batch_loss / max(n_graphs, 1)

        optimizer.zero_grad()
        loss.backward()
        if grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()

        if ema is not None:
            ema.update(model)

        total_num_graphs += n_graphs

    total_num_graphs = max(total_num_graphs, 1)
    metrics = {
        'loss_total': total_loss / total_num_graphs,
        'objective_margin': total_f_tilde / total_num_graphs,
        'max_violation': total_max_viol / total_num_graphs,
        'mean_violation': total_mean_viol / total_num_graphs,
        'entropy': total_entropy / total_num_graphs,
        'xi_sum_per_sample': total_xi_sum / total_num_graphs,
        'pred0_ratio': total_pred0_ratio / total_num_graphs,
        'pred1_ratio': total_pred1_ratio / total_num_graphs,
        'xi_mean_per_cons': total_xi_mean_per_cons / total_num_graphs,
        'tau_mean': total_tau_mean / total_num_graphs,
        'gamma': gamma,
        'mu': mu,
        'K_gamma': compute_K(gamma),
        'obj_margin_per_inst': total_f_tilde / total_num_graphs,
        'obj_raw_per_inst': total_f_base / total_num_graphs,
        'xi_margin_tau_per_inst': total_xi_with_tau_sum / total_num_graphs,
        'xi_raw_per_inst': total_xi_raw_sum / total_num_graphs,
        'xi_margin_per_inst': total_xi_sum / total_num_graphs,
        'disc_obj_per_inst': total_discrete_obj / total_num_graphs,
        'disc_viol_per_inst': total_discrete_viol / total_num_graphs,
        'n_feasible_inst': total_feasible_inst,
        'n_infeasible_inst': total_infeasible_inst,
    }

    return metrics


@torch.no_grad()
def validate_epoch(model, data_loader, gamma, tau, tau_obj, mu, loss_config,
                   entropy_weight, cons_normalize, device, tau_min=None):
    """
    Validate with the current ALM loss averaged per graph plus rounding metrics.
    """
    model.eval()

    total_loss = 0.0
    total_f_tilde = 0.0
    total_f_base = 0.0
    total_max_viol = 0.0
    total_mean_viol = 0.0
    total_feasibility = 0.0
    total_discrete_obj = 0.0
    total_round_obj_feasible = 0.0
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
    total_discrete_viol = 0.0
    total_feasible_inst = 0
    total_infeasible_inst = 0

    for batch in data_loader:
        batch = batch.to(device)

        logits, variable_features_batch = model_forward(model, batch, device)
        logits_per_graph = unbatch(logits, variable_features_batch)
        graphs = batch.to_data_list()

        for logits_i, graph in zip(logits_per_graph, graphs):
            x_hat_i = logits_i.sigmoid()
            cons_norm = compute_constraint_norms(graph, device) if cons_normalize else None

            loss, f_tilde, f_base, xi, xi_no_tau, xi_raw, max_viol, mean_viol, _, tau_vec = compute_alm_loss(
                x_hat_i, graph, gamma, tau, mu,
                tau_obj=tau_obj,
                loss_config=loss_config,
                cons_norm_cache=cons_norm,
                entropy_weight=entropy_weight,
                tau_min=tau_min,
            )

            feas, disc_obj, polar, uncert, viol_sum, is_feasible = evaluate_discrete_single(x_hat_i, graph, device)

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
            total_discrete_viol += viol_sum
            total_feasible_inst += int(is_feasible)
            total_infeasible_inst += int(not is_feasible)
            if is_feasible:
                total_round_obj_feasible += disc_obj

            x_rounded = torch.round(x_hat_i)
            total_pred0_ratio += (x_rounded == 0).float().mean().item()
            total_pred1_ratio += (x_rounded == 1).float().mean().item()
            total_xi_mean_per_cons += xi_no_tau.sum().item() / max(xi_no_tau.numel(), 1)
            total_tau_mean += tau_vec.mean().item()
            total_num_graphs += 1

    total_num_graphs = max(total_num_graphs, 1)
    metrics = {
        'loss_total': total_loss / total_num_graphs,
        'objective_margin': total_f_tilde / total_num_graphs,
        'max_violation': total_max_viol / total_num_graphs,
        'mean_violation': total_mean_viol / total_num_graphs,
        'feasibility_rate': total_feasibility / total_num_graphs,
        'discrete_objective': total_discrete_obj / total_num_graphs,
        'polarization_rate': total_polarization / total_num_graphs,
        'mean_uncertainty': total_uncertainty / total_num_graphs,
        'xi_sum_per_sample': total_xi_sum / total_num_graphs,
        'objective_per_sample': total_discrete_obj / total_num_graphs,
        'pred0_ratio': total_pred0_ratio / total_num_graphs,
        'pred1_ratio': total_pred1_ratio / total_num_graphs,
        'xi_mean_per_cons': total_xi_mean_per_cons / total_num_graphs,
        'tau_mean': total_tau_mean / total_num_graphs,
        'obj_margin_per_inst': total_f_tilde / total_num_graphs,
        'obj_raw_per_inst': total_f_base / total_num_graphs,
        'xi_margin_tau_per_inst': total_xi_with_tau_sum / total_num_graphs,
        'xi_raw_per_inst': total_xi_raw_sum / total_num_graphs,
        'xi_margin_per_inst': total_xi_sum / total_num_graphs,
        'disc_obj_per_inst': total_discrete_obj / total_num_graphs,
        'disc_viol_per_inst': total_discrete_viol / total_num_graphs,
        'n_feasible_inst': total_feasible_inst,
        'n_infeasible_inst': total_infeasible_inst,
        'round_n_feasible': total_feasible_inst,
        'round_avg_obj': total_round_obj_feasible / max(total_feasible_inst, 1),
        'n_valid_instances': total_feasible_inst + total_infeasible_inst,
    }
    return metrics


# ============================================================
#  Logging Utilities
# ============================================================

def format_metrics(train_metrics, val_metrics, epoch, elapsed):
    """Format metrics for console and file logging."""
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
        f"  [Penalty] mu={train_metrics['mu']:.4f}  "
        f"gamma={train_metrics['gamma']:.2f}  "
        f"K(gamma)={train_metrics['K_gamma']:.6f}  "
        f"lr_o={train_metrics['lr_o']:.2e}  "
        f"lr_i={train_metrics['lr_i']:.2e}",
        # Train discrete per-instance
        f"  [TDisc] AvgObj/inst={train_metrics['disc_obj_per_inst']:.4f}  "
        f"AvgViol/inst={train_metrics['disc_viol_per_inst']:.6f}  "
        f"FeasInst={train_metrics['n_feasible_inst']}  "
        f"InfeasInst={train_metrics['n_infeasible_inst']}",
        # Train continuous per-instance
        f"  [TCont] ObjMargin/inst={train_metrics['obj_margin_per_inst']:.4f}  "
        f"ObjRaw/inst={train_metrics['obj_raw_per_inst']:.4f}  "
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
            f"AvgObj/s={val_metrics['objective_per_sample']:.4f}"
        )
        lines.append(
            f"  [VPred] Pred0={val_metrics['pred0_ratio']:.4f}  "
            f"Pred1={val_metrics['pred1_ratio']:.4f}  "
            f"Xi_mean/cons={val_metrics['xi_mean_per_cons']:.6f}  "
            f"Tau_mean={val_metrics['tau_mean']:.6f}"
        )
        lines.append(
            f"  [Disc]  Feasibility={val_metrics['feasibility_rate']:.4f}  "
            f"Objective={val_metrics['discrete_objective']:.4f}  "
            f"Polarization={val_metrics['polarization_rate']:.4f}  "
            f"Uncertainty={val_metrics['mean_uncertainty']:.4f}"
        )
        lines.append(
            f"  [Round] Feasible={val_metrics['round_n_feasible']}/{val_metrics['n_valid_instances']}  "
            f"AvgObj={val_metrics['round_avg_obj']:.4f}  "
            f"BestFeas={val_metrics.get('best_round_feasible', -1)}  "
            f"BestObj={val_metrics.get('best_round_obj', float('inf')):.4f}"
        )
        if val_metrics.get('best_saved'):
            lines.append("  >> Best model saved by rounded feasibility")
        # Validation discrete per-instance
        lines.append(
            f"  [VDisc] AvgObj/inst={val_metrics['disc_obj_per_inst']:.4f}  "
            f"AvgViol/inst={val_metrics['disc_viol_per_inst']:.6f}  "
            f"FeasInst={val_metrics['n_feasible_inst']}  "
            f"InfeasInst={val_metrics['n_infeasible_inst']}"
        )
        # Validation continuous per-instance
        lines.append(
            f"  [VCont] ObjMargin/inst={val_metrics['obj_margin_per_inst']:.4f}  "
            f"ObjRaw/inst={val_metrics['obj_raw_per_inst']:.4f}  "
            f"Xi_m_t/inst={val_metrics['xi_margin_tau_per_inst']:.6f}  "
            f"Xi_raw/inst={val_metrics['xi_raw_per_inst']:.6f}  "
            f"Xi_m/inst={val_metrics['xi_margin_per_inst']:.6f}"
        )
    return '\n'.join(lines)


try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


def log_to_tensorboard(writer, train_metrics, val_metrics, epoch):
    """Log all metrics to TensorBoard."""
    if writer is None:
        return

    # Loss & game dynamics
    writer.add_scalar('Loss/Total', train_metrics['loss_total'], epoch)
    writer.add_scalar('Loss/Objective_Margin', train_metrics['objective_margin'], epoch)
    writer.add_scalar('Loss/Entropy', train_metrics['entropy'], epoch)
    writer.add_scalar('Violation/Train_Max', train_metrics['max_violation'], epoch)
    writer.add_scalar('Violation/Train_Mean', train_metrics['mean_violation'], epoch)

    # Penalty environment
    writer.add_scalar('Params/mu', train_metrics['mu'], epoch)
    writer.add_scalar('Params/lr_o', train_metrics['lr_o'], epoch)
    writer.add_scalar('Params/lr_i', train_metrics['lr_i'], epoch)
    writer.add_scalar('ALM/Gamma', train_metrics['gamma'], epoch)
    writer.add_scalar('ALM/K_gamma', train_metrics['K_gamma'], epoch)

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
        writer.add_scalar('Round/feasible_count', val_metrics['round_n_feasible'], epoch)
        writer.add_scalar('Round/avg_obj', val_metrics['round_avg_obj'], epoch)
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
        writer.add_scalar('PerInst/Valid_Disc_Viol', val_metrics['disc_viol_per_inst'], epoch)
        writer.add_scalar('PerInst/Valid_Feasible', val_metrics['n_feasible_inst'], epoch)
        writer.add_scalar('PerInst/Valid_Infeasible', val_metrics['n_infeasible_inst'], epoch)


# ============================================================
#  Argument Parser
# ============================================================

def get_parser():
    parser = argparse.ArgumentParser(description="Unsupervised fixed-tau scalar-penalty training for ILP via GNN.")

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
    parser.add_argument("--lr", type=float, default=None,
                        help="Deprecated compatibility alias: if set, overrides both lr_output and lr_inner")
    parser.add_argument("--lr_output", type=float, default=5e-4,
                        help="Learning rate for output layers (default: %(default)s)")
    parser.add_argument("--lr_inner", type=float, default=5e-4,
                        help="Learning rate for GNN body (default: %(default)s)")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="L2 regularization (default: %(default)s)")
    parser.add_argument("--num_epochs", type=int, default=12)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override task-specific batch size")

    # Fixed-tau scalar penalty hyperparameters
    parser.add_argument("--tau", type=float, default=0.9,
                        help="Fixed uniform tolerance tau_j=tau applied to every constraint "
                             "(default: %(default)s)")
    parser.add_argument("--tau_obj", type=float, default=0.0,
                        help="Objective margin tolerance in ReLU(2*sum_i |c_i|x_i(1-x_i) - tau_obj) "
                             "(default: %(default)s)")
    parser.add_argument("--tau_min", type=float, default=0.0,
                        help="Deprecated/ignored by the fixed-tau loss; kept for CLI compatibility "
                             "(default: %(default)s).")
    parser.add_argument("--loss_config", choices=['normalize', 'mean', 'sum', 'nonzero_mean'], default='normalize',
                        help="Loss reduction mode; formulas remain the current fixed-tau ALM loss "
                             "(default: %(default)s)")
    parser.add_argument("--mu_init", type=float, default=0.3,
                        help="Initial scalar constraint penalty mu (default: %(default)s)")
    parser.add_argument("--mu_step_size", type=float, default=0.01,
                        help="Step size for epoch-level mu update (default: %(default)s)")
    parser.add_argument("--mu_value", type=float, default=1.0,
                        help="Target average constraint violation for mu update (default: %(default)s)")
    parser.add_argument("--mu_max", type=float, default=5.0,
                        help="Maximum scalar constraint penalty mu (default: %(default)s)")
    parser.add_argument("--mu_min", type=float, default=0.01,
                        help="Minimum scalar constraint penalty mu (default: %(default)s)")
    parser.add_argument("--gamma_init", type=float, default=1.0,
                        help="Initial state sharpness retained for compatibility (default: %(default)s)")
    parser.add_argument("--gamma_max", type=float, default=50.0,
                        help="Deprecated/ignored by scalar-mu training; kept for CLI compatibility")
    parser.add_argument("--delta_gamma", type=float, default=0.3,
                        help="Deprecated/ignored by scalar-mu training; kept for CLI compatibility")
    parser.add_argument("--rho_init", type=float, default=1.0,
                        help="Deprecated/ignored by scalar-mu training; kept for CLI compatibility")
    parser.add_argument("--rho_max", type=float, default=1e5,
                        help="Deprecated/ignored by scalar-mu training; kept for CLI compatibility")
    parser.add_argument("--beta", type=float, default=1.5,
                        help="Deprecated/ignored by scalar-mu training; kept for CLI compatibility")
    parser.add_argument("--inner_steps", type=int, default=20,
                        help="Deprecated/ignored by scalar-mu training; kept for CLI compatibility")

    # Regularization & training tricks
    parser.add_argument("--grad_clip_norm", type=float, default=1.0,
                        help="Max gradient norm for clipping (0 = no clipping)")
    parser.add_argument("--entropy_weight", type=float, default=0.0,
                        help="Optional binary entropy regularization weight; 0 keeps the requested loss exact "
                             "(default: %(default)s)")
    parser.add_argument("--cons_normalize", action='store_true', default=False,
                        help="Explicitly normalize constraint violations by row norm; ignored when loss_config=normalize")
    parser.add_argument("--no_cons_normalize", action='store_false', dest='cons_normalize')
    parser.add_argument("--ema_decay", type=float, default=0.0,
                        help="EMA decay; default 0 disables EMA to match dilbaseline")
    parser.add_argument("--warmup_epochs", type=int, default=0,
                        help="Deprecated/ignored by dilbaseline-style scheduler; kept for CLI compatibility")
    parser.add_argument("--lr_schedule", choices=['cos', 'cosrestart', 'exp', 'none'], default='exp',
                        help="LR schedule type (default: %(default)s)")
    parser.add_argument("--cos_T", type=int, default=200,
                        help="T_max/T_0 for cosine schedulers (default: %(default)s)")
    parser.add_argument("--cos_min", type=float, default=0.0,
                        help="Minimum LR for cosine schedulers (default: %(default)s)")
    parser.add_argument("--lr_anneal_factor", type=float, default=0.88,
                        help="Multiplicative factor for ExponentialLR (default: %(default)s)")

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
                        help="Deprecated for best-model selection; kept for CLI compatibility")
    parser.add_argument("--patience", type=int, default=200,
                        help="Early-stop patience: epochs without rounded-feasibility improvement (default: %(default)s)")

    # Deprecated ALM freezing arguments kept for CLI compatibility
    parser.add_argument("--es_xi_threshold2", type=float, default=None,
                        help="Deprecated/ignored by scalar-mu training; kept for CLI compatibility")
    parser.add_argument("--threshold2_on", choices=['train', 'valid'], default='valid',
                        help="Deprecated/ignored by scalar-mu training; kept for CLI compatibility")
    parser.add_argument("--freeze_gamma_on_feasible", action='store_true', default=False,
                        help="Deprecated/ignored by scalar-mu training; kept for CLI compatibility")
    parser.add_argument("--freeze_rho_on_feasible", action='store_true', default=False,
                        help="Deprecated/ignored by scalar-mu training; kept for CLI compatibility")

    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducibility (default: %(default)s)")

    return parser


# ============================================================
#  Main
# ============================================================

def main():
    parser = get_parser()
    args = parser.parse_args()

    device = args.device
    problem_type = args.problem_type
    if args.lr is not None:
        args.lr_output = args.lr
        args.lr_inner = args.lr
    batch_size = args.batch_size or TASK_BATCH_SIZE.get(problem_type, 1)

    # Fix random seed for reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    save_name = (
        f'ScalarMu_{args.loss_config}_lr{args.lr_output}_{args.lr_inner}_tau{args.tau}_tauobj{args.tau_obj}'
        f'_mu{args.mu_init}_mustep{args.mu_step_size}_mutarget{args.mu_value}'
        f'_murange{args.mu_min}-{args.mu_max}_sched{args.lr_schedule}'
        f'_ent{args.entropy_weight}_ICC{args.Intra_Constraint_Competitive}'
    )

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
    # Try instance_dir/problem_type first; if it doesn't exist, use instance_dir directly
    ins_dir = os.path.join(args.instance_dir, problem_type)
    if not os.path.isdir(ins_dir):
        ins_dir = args.instance_dir
    all_instances = sorted([
        os.path.join(ins_dir, f)
        for f in os.listdir(ins_dir)
        if f.endswith(('.lp', '.mps'))
    ])

    random.shuffle(all_instances)
    split = int(0.8 * len(all_instances))
    # Single-instance case: use the same instance for both train and valid
    if split == 0 or len(all_instances) == 1:
        train_files = all_instances
        valid_files = all_instances
    else:
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

    first_data = train_data[0]
    if hasattr(first_data, 'obj_sense_min'):
        is_minimize = bool(first_data.obj_sense_min.item())
    else:
        is_minimize = extract_raw_ilp(train_files[0])['obj_sense_min']
    opt_dir = "MINIMIZE" if is_minimize else "MAXIMIZE"
    print(f"Objective sense: {opt_dir} (coefficients are sign-adjusted for the current loss)")

    # ---- Model ----
    model = GNNPolicy(
        emb_size=args.emb_size,
        cons_nfeats=args.cons_nfeats,
        edge_nfeats=args.edge_nfeats,
        var_nfeats=args.var_nfeats,
        depth=args.depth,
        Intra_Constraint_Competitive=args.Intra_Constraint_Competitive,
    ).to(device)

    # ---- Optimizer with separate learning rates ----
    output_param_ids = set()
    for layer in (model.vars_output_layer, model.cons_output_layer):
        for p in layer.parameters():
            output_param_ids.add(id(p))
    other_params = [p for p in model.parameters() if id(p) not in output_param_ids]

    optimizer = torch.optim.Adam([
        {'params': model.vars_output_layer.parameters(), 'lr': args.lr_output},
        {'params': model.cons_output_layer.parameters(), 'lr': args.lr_output},
        {'params': other_params, 'lr': args.lr_inner},
    ], weight_decay=args.weight_decay)

    # ---- LR Schedule ----
    if args.lr_schedule == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.cos_T, eta_min=args.cos_min
        )
    elif args.lr_schedule == 'cosrestart':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=args.cos_T, T_mult=2, eta_min=args.cos_min
        )
    elif args.lr_schedule == 'exp':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=args.lr_anneal_factor
        )
    else:
        scheduler = None

    # ---- EMA ----
    ema = EMAModel(model, decay=args.ema_decay) if args.ema_decay > 0 else None

    # ---- Scalar penalty state ----
    gamma = args.gamma_init
    mu = args.mu_init
    start_epoch = 0

    # ---- Resume from checkpoint ----
    if args.resume_from is not None:
        assert os.path.isfile(args.resume_from), f"Checkpoint not found: {args.resume_from}"
        print(f"Loading checkpoint from {args.resume_from} ...")
        ckpt = torch.load(args.resume_from, map_location=device)

        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            # Full checkpoint with training state
            model.load_state_dict(ckpt['model_state_dict'])
            if 'optimizer_state_dict' in ckpt:
                try:
                    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                except Exception as exc:
                    print(f"  Warning: optimizer state is incompatible with grouped Adam; using a fresh optimizer ({exc})")
            if scheduler is not None and 'scheduler_state_dict' in ckpt:
                try:
                    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                except Exception as exc:
                    print(f"  Warning: scheduler state is incompatible; using a fresh scheduler ({exc})")
            if ema is not None and 'ema_shadow' in ckpt:
                ema.shadow = ckpt['ema_shadow']
            gamma = ckpt.get('gamma', gamma)
            mu = ckpt.get('mu', ckpt.get('lambda_global', mu))
            ckpt_loss_config = ckpt.get('loss_config')
            if ckpt_loss_config is not None and ckpt_loss_config != args.loss_config:
                print(f"  Warning: checkpoint loss_config={ckpt_loss_config}, current loss_config={args.loss_config}")
            start_epoch = ckpt.get('epoch', 0) + 1
            print(f"  Resumed full checkpoint: epoch={start_epoch}, gamma={gamma:.2f}, mu={mu:.4f}")
        else:
            # Plain state_dict (model weights only)
            model.load_state_dict(ckpt)
            print("  Loaded model weights (no optimizer/ALM state). Training from epoch 0.")

    # ---- Training ----
    best_round_feasible = -1
    best_round_obj = float('inf')
    patience_counter = 0
    best_allfeas_obj = float('inf')

    # Resolve tau_min for backward-compatible call signatures; fixed-tau loss ignores it.
    tau_min = args.tau_min if args.tau_min > 0 else None

    print(f"\n{'='*70}")
    print(f"Starting Fixed-Tau ALM Training with dilbaseline-style GD for {problem_type} ({opt_dir})")
    print(f"  tau={args.tau} (uniform fixed constraint), tau_obj={args.tau_obj}, tau_min={tau_min} ignored")
    print(f"  loss_config={args.loss_config}")
    if args.loss_config == 'normalize' and args.cons_normalize:
        print("  Note: --cons_normalize is ignored when loss_config=normalize to avoid double normalization.")
    print(f"  mu_init={args.mu_init}, mu_step_size={args.mu_step_size}, mu_value={args.mu_value}")
    print(f"  mu_range=[{args.mu_min}, {args.mu_max}]")
    print(f"  entropy_weight={args.entropy_weight}, grad_clip={args.grad_clip_norm}, ema_decay={args.ema_decay}")
    print(f"  lr_output={args.lr_output}, lr_inner={args.lr_inner}, weight_decay={args.weight_decay}")
    print(f"  lr_schedule={args.lr_schedule}, cos_T={args.cos_T}, cos_min={args.cos_min}, lr_anneal_factor={args.lr_anneal_factor}")
    print(f"  batch_size={batch_size}, num_epochs={args.num_epochs}")
    print(f"{'='*70}\n")

    for epoch in range(start_epoch, args.num_epochs):
        t0 = time.time()

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, ema,
            gamma, args.tau, args.tau_obj, mu, args.loss_config,
            args.entropy_weight, args.cons_normalize, args.grad_clip_norm,
            device,
            tau_min=tau_min,
        )

        if scheduler is not None:
            scheduler.step()
            lr_list = scheduler.get_last_lr()
        else:
            lr_list = [args.lr_output, args.lr_output, args.lr_inner]
        train_metrics['lr_o'] = lr_list[0]
        train_metrics['lr_i'] = lr_list[-1]

        avg_cons = train_metrics['xi_margin_tau_per_inst']
        mu = mu + args.mu_step_size * (avg_cons - args.mu_value)
        mu = max(min(mu, args.mu_max), args.mu_min)
        train_metrics['mu'] = mu

        # Validate periodically
        val_metrics = None
        if (epoch + 1) % args.val_every == 0 or epoch == 0:
            # Use EMA weights for validation
            if ema is not None:
                backup = ema.apply(model)

            val_metrics = validate_epoch(
                model, valid_loader,
                gamma, args.tau, args.tau_obj, mu, args.loss_config,
                args.entropy_weight, args.cons_normalize, device,
                tau_min=tau_min,
            )

            if ema is not None:
                ema.restore(model, backup)

            curr_round_feas = val_metrics['round_n_feasible']
            curr_round_obj = val_metrics['round_avg_obj']
            curr_valid = val_metrics['n_valid_instances']

            is_best = False
            if curr_round_feas > best_round_feasible:
                is_best = True
            elif curr_round_feas == best_round_feasible and curr_round_feas > 0:
                if curr_round_obj < best_round_obj - 1e-6:
                    is_best = True

            if is_best:
                best_round_feasible = curr_round_feas
                best_round_obj = curr_round_obj
                patience_counter = 0
                save_state = ema.state_dict(model) if ema is not None else model.state_dict()
                torch.save(save_state, os.path.join(model_save_path, f'{save_name}_model_best.pth'))
            else:
                patience_counter += 1

            val_metrics['best_saved'] = is_best
            val_metrics['best_round_feasible'] = best_round_feasible
            val_metrics['best_round_obj'] = best_round_obj
            val_metrics['n_valid_instances'] = curr_valid

            # Extra current-project artifact: all validation instances feasible after discretization.
            n_infeas = val_metrics['n_infeasible_inst']
            curr_all_obj = val_metrics['disc_obj_per_inst']
            if n_infeas == 0:
                if curr_all_obj < best_allfeas_obj - 1e-6:
                    best_allfeas_obj = curr_all_obj
                    save_state = ema.state_dict(model) if ema is not None else model.state_dict()
                    af_path = os.path.join(model_save_path, f'{save_name}_model_best_allfeas.pth')
                    torch.save(save_state, af_path)
                    print(f"  [AllFeas] Saved best all-feasible model: "
                          f"disc_obj/inst={curr_all_obj:.4f}, n_infeasible=0")

        # Save latest (full checkpoint for resumable training)
        full_ckpt = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'gamma': gamma,
            'mu': mu,
            'tau': args.tau,
            'tau_obj': args.tau_obj,
            'loss_config': args.loss_config,
            'lr_output': args.lr_output,
            'lr_inner': args.lr_inner,
            'lr_schedule': args.lr_schedule,
        }
        if scheduler is not None:
            full_ckpt['scheduler_state_dict'] = scheduler.state_dict()
        if ema is not None:
            full_ckpt['ema_shadow'] = ema.shadow
        torch.save(full_ckpt, os.path.join(model_save_path, f'{save_name}_model_last.pth'))

        elapsed = time.time() - t0
        log_str = format_metrics(train_metrics, val_metrics, epoch, elapsed)
        print(log_str)
        log_file.write(log_str + '\n')
        log_file.flush()

        log_to_tensorboard(tb_writer, train_metrics, val_metrics, epoch)

        # Early stopping based on patience
        if (val_metrics is not None
                and patience_counter >= args.patience
                and epoch > args.patience):
            print(f"\nEarly stopping at epoch {epoch}: no improvement for {args.patience} epochs.")
            print(f"  Best round_feasible={best_round_feasible}  Best round_avg_obj={best_round_obj:.4f}")
            break

    log_file.close()
    if tb_writer is not None:
        tb_writer.close()
    print("Training completed successfully.")


if __name__ == '__main__':
    main()
