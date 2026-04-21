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

from utils import TASKS
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
    return min(K, K_max)


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
    tau,             # tolerance
    lambda_global,   # scalar global Lagrangian multiplier
    rho,             # quadratic penalty parameter
    cons_norm_cache=None,  # optional per-constraint normalization factors
    entropy_weight=0.0,    # binary entropy regularization weight
):
    """
    Compute the full Augmented Lagrangian loss.

    Returns:
        loss:           total ALM loss (scalar)
        f_tilde:        margin-aware objective value (scalar)
        xi:             [total_cons] per-constraint violation vector
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
    f_tilde = f_base + f_margin

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
    abs_A_sqrt_u_per_edge = cons_val.abs() * sqrt_u_raw[cons_col]
    margin_sum = torch.zeros(n_cons, device=device)
    margin_sum.scatter_add_(0, cons_row, abs_A_sqrt_u_per_edge)

    # Violation: ReLU(Ax - b + K*margin_sum - tau)
    raw_violation = Ax - rhs + K * margin_sum - tau

    # Optional: normalize by constraint scale for balanced penalties
    if cons_norm_cache is not None:
        raw_violation = raw_violation / cons_norm_cache.to(device)

    xi = torch.relu(raw_violation)

    # --- 5. Augmented Lagrangian loss ---
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

    return loss, f_tilde.item(), xi, max_violation, mean_violation, entropy_val.item()


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

    # Polarization rate: |x_hat - 0.5| > 0.45 (i.e., x < 0.05 or x > 0.95)
    polarized = ((x_hat < 0.05) | (x_hat > 0.95)).float()
    polarization_rate = polarized.mean().item()

    # Mean uncertainty
    # We don't have gamma here, so use a simple measure
    mean_uncertainty = (4 * x_hat * (1 - x_hat)).mean().item()  # max at 0.5, 0 at 0/1

    return feasibility_rate, discrete_obj, polarization_rate, mean_uncertainty


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


# ============================================================
#  Training Loop
# ============================================================

def train_epoch(model, data_loader, optimizer, scheduler, ema,
                gamma, tau, lambda_global, rho, prev_violation,
                inner_steps, beta, rho_max, gamma_max, delta_gamma,
                entropy_weight, cons_normalize, grad_clip_norm,
                device, step_counter):
    """
    One epoch of ALM training with inner/outer loop.

    Returns updated ALM state: (gamma, lambda_global, rho, prev_violation, step_counter, metrics)
    """
    model.train()

    # Accumulators for epoch-level metrics
    total_loss = 0.0
    total_f_tilde = 0.0
    total_max_viol = 0.0
    total_mean_viol = 0.0
    total_entropy = 0.0
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
        loss, f_tilde, xi, max_viol, mean_viol, ent_val = compute_alm_loss(
            x_hat, batch, gamma, tau, lambda_global, rho,
            cons_norm_cache=cons_norm,
            entropy_weight=entropy_weight,
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
        total_max_viol += max_viol
        total_mean_viol += mean_viol
        total_entropy += ent_val
        n_batches += 1
        step_counter += 1

        # --- Outer loop update ---
        if step_counter % inner_steps == 0:
            with torch.no_grad():
                curr_viol = xi.sum().item() / max(batch.num_graphs, 1)
                # Update global lambda
                lambda_global = max(0.0, lambda_global + rho * curr_viol)
                # Update rho if violations aren't decreasing fast enough
                if curr_viol > 0.8 * prev_violation and curr_viol > 1e-4:
                    rho = min(rho * beta, rho_max)
                # Gamma annealing
                gamma = min(gamma + delta_gamma, gamma_max)
                prev_violation = curr_viol

    n_batches = max(n_batches, 1)
    metrics = {
        'loss_total': total_loss / n_batches,
        'objective_margin': total_f_tilde / n_batches,
        'max_violation': total_max_viol / n_batches,
        'mean_violation': total_mean_viol / n_batches,
        'entropy': total_entropy / n_batches,
        'gamma': gamma,
        'rho': rho,
        'lambda_global': lambda_global,
        'K_gamma': compute_K(gamma),
    }

    return gamma, lambda_global, rho, prev_violation, step_counter, metrics


@torch.no_grad()
def validate_epoch(model, data_loader, gamma, tau, lambda_global, rho,
                   entropy_weight, cons_normalize, device):
    """
    Validate: compute ALM loss + discrete rounding metrics.
    """
    model.eval()

    total_loss = 0.0
    total_f_tilde = 0.0
    total_max_viol = 0.0
    total_mean_viol = 0.0
    total_feasibility = 0.0
    total_discrete_obj = 0.0
    total_polarization = 0.0
    total_uncertainty = 0.0
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

        loss, f_tilde, xi, max_viol, mean_viol, _ = compute_alm_loss(
            x_hat, batch, gamma, tau, lambda_global, rho,
            cons_norm_cache=cons_norm,
            entropy_weight=entropy_weight,
        )
        loss = loss / max(batch.num_graphs, 1)

        # Discrete evaluation
        feas, disc_obj, polar, uncert = evaluate_discrete(x_hat, batch, device)

        total_loss += loss.item()
        total_f_tilde += f_tilde
        total_max_viol += max_viol
        total_mean_viol += mean_viol
        total_feasibility += feas
        total_discrete_obj += disc_obj
        total_polarization += polar
        total_uncertainty += uncert
        n_batches += 1

    n_batches = max(n_batches, 1)
    metrics = {
        'loss_total': total_loss / n_batches,
        'objective_margin': total_f_tilde / n_batches,
        'max_violation': total_max_viol / n_batches,
        'mean_violation': total_mean_viol / n_batches,
        'feasibility_rate': total_feasibility / n_batches,
        'discrete_objective': total_discrete_obj / n_batches,
        'polarization_rate': total_polarization / n_batches,
        'mean_uncertainty': total_uncertainty / n_batches,
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
        f"Entropy={train_metrics['entropy']:.4f}",
        f"  [ALM]   gamma={train_metrics['gamma']:.2f}  "
        f"rho={train_metrics['rho']:.4f}  "
        f"lambda={train_metrics['lambda_global']:.4f}  "
        f"K(gamma)={train_metrics['K_gamma']:.6f}",
    ]
    if val_metrics:
        lines.append(
            f"  [Valid] Loss={val_metrics['loss_total']:.6f}  "
            f"MaxViol={val_metrics['max_violation']:.6f}  "
            f"MeanViol={val_metrics['mean_violation']:.6f}"
        )
        lines.append(
            f"  [Disc]  Feasibility={val_metrics['feasibility_rate']:.4f}  "
            f"Objective={val_metrics['discrete_objective']:.4f}  "
            f"Polarization={val_metrics['polarization_rate']:.4f}  "
            f"Uncertainty={val_metrics['mean_uncertainty']:.4f}"
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

    # ALM environment
    writer.add_scalar('ALM/Gamma', train_metrics['gamma'], epoch)
    writer.add_scalar('ALM/Rho', train_metrics['rho'], epoch)
    writer.add_scalar('ALM/Lambda_Global', train_metrics['lambda_global'], epoch)
    writer.add_scalar('ALM/K_gamma', train_metrics['K_gamma'], epoch)

    if val_metrics:
        writer.add_scalar('Loss/Valid_Total', val_metrics['loss_total'], epoch)
        writer.add_scalar('Violation/Valid_Max', val_metrics['max_violation'], epoch)
        writer.add_scalar('Violation/Valid_Mean', val_metrics['mean_violation'], epoch)

        # Discrete ground truth
        writer.add_scalar('Discrete/Feasibility_Rate', val_metrics['feasibility_rate'], epoch)
        writer.add_scalar('Discrete/Objective', val_metrics['discrete_objective'], epoch)
        writer.add_scalar('State/Polarization_Rate', val_metrics['polarization_rate'], epoch)
        writer.add_scalar('State/Mean_Uncertainty', val_metrics['mean_uncertainty'], epoch)


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
    parser.add_argument("--num_epochs", type=int, default=5000)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override task-specific batch size")

    # ALM hyperparameters
    parser.add_argument("--tau", type=float, default=0.9,
                        help="Constraint tolerance (default: %(default)s)")
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
    parser.add_argument("--inner_steps", type=int, default=20,
                        help="Inner loop steps between ALM updates (default: %(default)s)")

    # Regularization & training tricks
    parser.add_argument("--grad_clip_norm", type=float, default=1.0,
                        help="Max gradient norm for clipping (0 = no clipping)")
    parser.add_argument("--entropy_weight", type=float, default=0.01,
                        help="Binary entropy regularization weight (default: %(default)s)")
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
    parser.add_argument("--val_every", type=int, default=5,
                        help="Validate every N epochs (default: %(default)s)")

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

    save_name = (
        f'ALM_tau{args.tau}_gamma{args.gamma_init}_rho{args.rho_init}'
        f'_inner{args.inner_steps}_ent{args.entropy_weight}'
        f'_ICC{args.Intra_Constraint_Competitive}'
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
            start_epoch = ckpt.get('epoch', 0) + 1
            print(f"  Resumed full checkpoint: epoch={start_epoch}, gamma={gamma:.2f}, "
                  f"rho={rho:.4f}, lambda={lambda_global:.4f}")
        else:
            # Plain state_dict (model weights only)
            model.load_state_dict(ckpt)
            print("  Loaded model weights (no optimizer/ALM state). Training from epoch 0.")

    # ---- Training ----
    best_val_feasibility = 0.0
    best_val_loss = float('inf')

    print(f"\n{'='*70}")
    print(f"Starting Unsupervised ALM Training for {problem_type}")
    print(f"  tau={args.tau}, gamma_init={args.gamma_init}, rho_init={args.rho_init}")
    print(f"  inner_steps={args.inner_steps}, beta={args.beta}")
    print(f"  entropy_weight={args.entropy_weight}, grad_clip={args.grad_clip_norm}")
    print(f"  LR={args.lr}, schedule={args.lr_schedule}, warmup={args.warmup_epochs}")
    print(f"{'='*70}\n")

    for epoch in range(start_epoch, args.num_epochs):
        t0 = time.time()

        # Train
        gamma, lambda_global, rho, prev_violation, step_counter, train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, ema,
            gamma, args.tau, lambda_global, rho, prev_violation,
            args.inner_steps, args.beta, args.rho_max, args.gamma_max, args.delta_gamma,
            args.entropy_weight, args.cons_normalize, args.grad_clip_norm,
            device, step_counter,
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
            )

            if ema is not None:
                ema.restore(model, backup)

            # Save best model
            # Primary: best feasibility; secondary: best loss among feasible
            is_best = False
            if val_metrics['feasibility_rate'] > best_val_feasibility + 1e-4:
                best_val_feasibility = val_metrics['feasibility_rate']
                is_best = True
            elif (abs(val_metrics['feasibility_rate'] - best_val_feasibility) < 1e-4
                  and val_metrics['loss_total'] < best_val_loss):
                best_val_loss = val_metrics['loss_total']
                is_best = True

            if is_best:
                save_state = ema.shadow if ema is not None else model.state_dict()
                if isinstance(save_state, dict) and all(isinstance(v, torch.Tensor) for v in save_state.values()):
                    torch.save(save_state, os.path.join(model_save_path, f'{save_name}_model_best.pth'))
                else:
                    torch.save(model.state_dict(), os.path.join(model_save_path, f'{save_name}_model_best.pth'))

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

        # Early stopping: if feasibility is 100% and violations are near zero
        if (val_metrics is not None
                and val_metrics['feasibility_rate'] >= 0.9999
                and val_metrics['max_violation'] < 1e-4
                and epoch > 100):
            print(f"\nEarly stopping at epoch {epoch}: 100% feasibility achieved!")
            break

    log_file.close()
    if tb_writer is not None:
        tb_writer.close()
    print("Training completed successfully.")


if __name__ == '__main__':
    main()
