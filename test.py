import pickle
import argparse
import random
import os
import logging
import queue
import threading
import time
import numpy as np
import torch

from utils import get_a_new2, extract_raw_ilp, TASKS
from solver.solver_utils import SOLVER_CLASSES
from gnn import GNNPolicy


def setup_environment(seed: int = 0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

def configure_logging(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    file_handler = logging.FileHandler(os.path.join(log_dir, "test.log"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def load_pretrained_model(args: argparse.Namespace, model_path: str, device: torch.device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = GNNPolicy(
        emb_size=args.emb_size,
        cons_nfeats=args.cons_nfeats,
        edge_nfeats=args.edge_nfeats,
        var_nfeats=args.var_nfeats,
        depth=args.depth,
        Intra_Constraint_Competitive=args.Intra_Constraint_Competitive
    ).to(device)
    
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=False)
    model.eval()
    
    return model


def process_single_instance(args, ins_path, policy, device):
    A, v_map, v_nodes, c_nodes, b_vars=get_a_new2(ins_path)

    constraint_features = c_nodes.cpu()
    mask = torch.isnan(constraint_features)
    constraint_features[mask] = 1
    variable_features = v_nodes
    edge_indices = A._indices()
    edge_features = A._values().unsqueeze(1)
    edge_features=torch.ones(edge_features.shape)

    constraint_features_batch = torch.tensor([0]*len(constraint_features)).to(device)
    variable_features_batch = torch.tensor([0]*len(variable_features)).to(device)

    BD = policy(
        constraint_features.to(device),
        edge_indices.to(device),
        edge_features.to(device),
        variable_features.to(device),
        torch.tensor([constraint_features.shape[0]]).to(device),
        constraint_features_batch.to(device),
        variable_features_batch.to(device)
    )
    BD = BD.sigmoid().cpu().squeeze()

    # 对齐GNN输出和求解器之间的变量名
    all_varname=[]
    for name in v_map:
        all_varname.append(name)
    binary_name=[all_varname[i] for i in b_vars]
    
    # get a list of (index, VariableName, Prob, -1, type)
    scores=[]
    for i in range(len(v_map)):
        type="C"
        if all_varname[i] in binary_name:
            type='BINARY'
        scores.append([i, all_varname[i], BD[i].item(), -1, type])

    scores.sort(key=lambda x:x[2],reverse=True)

    scores=[x for x in scores if x[4]=='BINARY']
    return scores

def fix_pas(scores, task, args):
    # default hyperparameters {"IP": (60, 35, 55), "WA": (20, 200, 100), "CA": (400, 0, 40), "SC" : (1000, 0, 200)} 
    k0 = int(args.k0)
    k1 = int(args.k1)
    delta = int(args.delta)

    # fixing variable picked by confidence scores
    scores.sort(key=lambda x: x[2], reverse=True)
    for i in range(min(len(scores), k1)):
        scores[i][3] = 1

    scores.sort(key=lambda x: x[2], reverse=False)
    for i in range(min(len(scores), k0)):
        scores[i][3] = 0

    return scores, delta


def solve_mps(mps_file, log_dir, save_name, ins_name, scores, task, args):
    log_file = log_dir
    solver = SOLVER_CLASSES[args.solver]()
    solver.hide_output_to_console()

    solver.load_model(mps_file)
    solver.set_aggressive()

    scores, delta = fix_pas(scores, task, args)
    # trust region method implemented by adding constraints
    instance_variables = solver.get_vars()
    instance_variables.sort(key=lambda v: solver.varname(v))

    # Create a map from varname string to solver's variable object
    variables_map = {}
    for v in instance_variables:  
        variables_map[solver.varname(v)] = v

    alphas = []

    for i in range(len(scores)):
        tar_var = variables_map[scores[i][1]]
        x_star = scores[i][3]  # 1, 0, or -1 (don't fix)
        if x_star < 0:
            continue

        tmp_var = solver.create_real_var(name=f'alpha_{tar_var}')
        alphas.append(tmp_var)
        solver.add_constraint(tmp_var >= tar_var - x_star, name=f'alpha_up_{i}')
        solver.add_constraint(tmp_var >= x_star - tar_var, name=f'alpha_down_{i}')

    if len(alphas) > 0:
        all_tmp = 0
        for tmp in alphas:
            all_tmp += tmp
        solver.add_constraint(all_tmp <= delta, name="sum_alpha")

    results = solver.solve(means=args.solver, log_file=log_file, time_limit=args.max_time, threads=args.threads)
    sol_save_path = os.path.join(os.path.dirname(log_dir), save_name + ins_name.split('.')[0] + '_node_info.pkl')
    with open(sol_save_path, 'wb') as f:
        pickle.dump(results, f)



def compute_rounded_violations(instance_path, var_names, rounded_values):
    """
    加载实例（pyscipopt），代入四舍五入后的解，计算目标值和约束违反量。

    Returns
    -------
    obj_val   : float  — 四舍五入解的目标函数值
    max_viol  : float  — 最大约束违反量
    mean_viol : float  — 平均约束违反量
    n_violated: int    — 违反约束数
    n_total   : int    — 约束总数
    """
    import pyscipopt as scp

    m = scp.Model()
    m.hideOutput(True)
    m.readProblem(instance_path)

    mvars = m.getVars()
    mvars.sort(key=lambda v: v.name)
    name2idx = {v.name: i for i, v in enumerate(mvars)}

    # 构建解向量（按 pyscipopt 变量顺序）
    x = np.zeros(len(mvars))
    for name, val in zip(var_names, rounded_values):
        gname = name[2:] if (name.startswith('t_') and name not in name2idx) else name
        if gname in name2idx:
            x[name2idx[gname]] = float(val)

    # 目标值
    obj = m.getObjective()
    obj_val = 0.0
    for e in obj:
        vnm = e.vartuple[0].name
        if vnm in name2idx:
            obj_val += obj[e] * x[name2idx[vnm]]

    # 约束违反量
    cons = m.getConss()
    cons = [c for c in cons if len(m.getValsLinear(c)) > 0]

    viol_list = []
    for c in cons:
        coeff = m.getValsLinear(c)
        rhs_val = m.getRhs(c)
        lhs_val = m.getLhs(c)

        # Compute A_j @ x
        ax = sum(coeff[k] * x[name2idx[k]] for k in coeff if k in name2idx)

        if rhs_val == lhs_val:
            # Equality
            viol_list.append(abs(ax - rhs_val))
        elif rhs_val >= 1e+20:
            # >= constraint
            viol_list.append(max(0.0, lhs_val - ax))
        else:
            # <= constraint
            viol_list.append(max(0.0, ax - rhs_val))

    viol = np.array(viol_list) if len(viol_list) > 0 else np.array([0.0])

    m.freeProb()

    max_v  = float(viol.max())
    mean_v = float(viol.mean())
    total_v = float(viol.sum())
    n_viol = int((viol > 1e-6).sum())
    return obj_val, max_v, mean_v, total_v, n_viol, len(viol_list)


def load_solution(sol_path):
    """加载 solution 文件，返回 var_names 列表和最优解向量（第一个 sol，取绝对值）。"""
    with open(sol_path, 'rb') as f:
        sol_data = pickle.load(f)
    var_names = sol_data['var_names']
    # objs 最小化时第一个最优；取绝对值消除 -0.0
    best_sol = np.abs(np.round(np.array(sol_data['sols'][0]), 0))
    return var_names, best_sol


def compute_ce_and_acc(BD, v_map, b_vars, sol_var_names, best_sol, topk=500):
    """
    计算 GNN 预测概率与最优解之间的交叉熵损失、四舍五入正确率和 top-K 置信度正确率。

    Parameters
    ----------
    BD          : torch.Tensor (n_vars,)  — sigmoid 后的预测概率，按 v_map 顺序
    v_map       : dict  — {varname: index}，v_map 中键的迭代顺序即 BD 对应顺序
    b_vars      : Tensor[int]  — 二进制变量在 v_map 中的位置索引
    sol_var_names : list[str]  — solution 文件中的变量名顺序
    best_sol    : np.ndarray   — 按 sol_var_names 排列的最优解（0/1）
    topk        : int          — 取置信度最高的 K 个变量计算正确率

    Returns
    -------
    ce       : float  — 交叉熵损失（仅二进制变量）
    acc      : float  — 四舍五入后正确率（仅二进制变量）
    topk_acc : float  — 置信度最高 topk 个变量的四舍五入正确率
    mse      : float  — 均方误差损失（仅二进制变量）
    n        : int    — 参与计算的二进制变量数量
    """
    sol_name2val = {name: val for name, val in zip(sol_var_names, best_sol)}
    all_varname = list(v_map)

    preds, targets = [], []
    for idx in b_vars.tolist():
        vname = all_varname[idx]
        if vname not in sol_name2val:
            continue
        preds.append(BD[idx].item())
        targets.append(sol_name2val[vname])

    if len(preds) == 0:
        return float('nan'), float('nan'), float('nan'), float('nan'), 0

    preds = np.array(preds, dtype=np.float64)
    targets = np.array(targets, dtype=np.float64)

    # 交叉熵: -[t*log(p) + (1-t)*log(1-p)]，clip 防止 log(0)
    eps = 1e-7
    preds_clip = np.clip(preds, eps, 1 - eps)
    ce = -np.mean(targets * np.log(preds_clip) + (1 - targets) * np.log(1 - preds_clip))

    rounded = np.round(preds)
    acc = np.mean(rounded == targets)

    # top-K 置信度正确率：按 |pred - 0.5| 降序，取前 K 个
    confidence = np.abs(preds - 0.5)
    k = min(topk, len(preds))
    topk_indices = np.argpartition(confidence, -k)[-k:]
    topk_acc = float(np.mean(rounded[topk_indices] == targets[topk_indices]))

    # MSE 损失
    mse = float(np.mean((preds - targets) ** 2))

    return float(ce), float(acc), topk_acc, mse, len(preds)


def run_inference_only(args, policy):
    """推理 + 启发式固定 → 导出加了信赖域约束的新 MILP 实例文件。

    对每个实例：
      1. GNN 推理得到 scores
      2. fix_pas 决定固定哪些变量
      3. 加载原始 MILP，添加信赖域约束（alpha 辅助变量 + sum ≤ delta）
      4. 导出为独立的 .mps 文件，后续可直接用任意求解器求解
      5. 与 solution 目录的最优解计算交叉熵损失和四舍五入正确率
    """
    device = args.device
    test_instances = sorted(
        os.listdir(os.path.join(args.instance_dir, args.test_problem_type))
    )[:args.test_num]

    print(f"\n{'='*60}")
    print(f"Inference-Only Mode ({args.test_problem_type},"
          f" Intra_Constraint_Competitive={args.Intra_Constraint_Competitive})")
    print(f"{'='*60}")

    output_dir = os.path.join(args.log_dir, args.test_problem_type, 'modified_instances')
    os.makedirs(output_dir, exist_ok=True)

    # solution 目录：dataset/<problem_type>/solution/
    dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset')
    solution_dir = os.path.join(dataset_dir, args.test_problem_type, 'solution')

    sum_feat, sum_infer, sum_total = 0.0, 0.0, 0.0
    sum_conf_vars = 0
    count = 0
    conf_threshold = args.margin

    all_ce = []
    all_acc = []
    all_topk_acc = []
    all_mse = []

    all_feas = []
    all_obj_round = []
    all_max_viol = []
    all_total_viol = []
    all_n_violated = []

    for idx, ins_name in enumerate(test_instances):
        ins_path = os.path.join(args.instance_dir, args.test_problem_type, ins_name)
        if not os.path.exists(ins_path):
            print(f"[SKIP] {ins_name}: file not found")
            continue

        # ---- 1. 特征提取 ----
        t0 = time.time()
        A, v_map, v_nodes, c_nodes, b_vars = get_a_new2(ins_path)
        constraint_features = c_nodes.cpu()
        constraint_features[torch.isnan(constraint_features)] = 1
        variable_features = v_nodes
        edge_indices = A._indices()
        edge_features = torch.ones(A._values().unsqueeze(1).shape)
        constraint_features_batch = torch.tensor([0]*len(constraint_features)).to(device)
        variable_features_batch = torch.tensor([0]*len(variable_features)).to(device)
        t1 = time.time()
        feature_time = t1 - t0

        # ---- 2. GNN 推理 ----
        BD = policy(
            constraint_features.to(device),
            edge_indices.to(device),
            edge_features.to(device),
            variable_features.to(device),
            torch.tensor([constraint_features.shape[0]]).to(device),
            constraint_features_batch,
            variable_features_batch
        )
        BD = BD.sigmoid().cpu().squeeze()
        t2 = time.time()
        infer_time = t2 - t1

        # ---- 3. 对齐变量名，构造 scores ----
        all_varname = list(v_map)
        binary_name = [all_varname[i] for i in b_vars]
        scores = []
        for i in range(len(v_map)):
            vtype = 'BINARY' if all_varname[i] in binary_name else 'C'
            scores.append([i, all_varname[i], BD[i].item(), -1, vtype])
        scores.sort(key=lambda x: x[2], reverse=True)
        scores = [x for x in scores if x[4] == 'BINARY']

        # ---- 3b. 与最优解计算交叉熵和正确率 ----
        base_name = ins_name.split('.')[0]
        sol_path = os.path.join(solution_dir, base_name + '.sol')
        ce_val, acc_val, topk_acc_val = float('nan'), float('nan'), float('nan')
        mse_val = float('nan')
        if os.path.exists(sol_path):
            try:
                sol_var_names, best_sol = load_solution(sol_path)
                ce_val, acc_val, topk_acc_val, mse_val, n_vars = compute_ce_and_acc(
                    BD, v_map, b_vars, sol_var_names, best_sol, topk=args.topk
                )
                if not np.isnan(ce_val):
                    all_ce.append(ce_val)
                    all_acc.append(acc_val)
                    all_topk_acc.append(topk_acc_val)
                    all_mse.append(mse_val)
            except Exception as e:
                print(f"  [WARN] CE/Acc computation failed for {ins_name}: {e}")
        else:
            print(f"  [WARN] Solution file not found: {sol_path}")

        # ---- 3c. 四舍五入后评估可行性和约束违反量 ----
        BD_rounded = torch.round(BD).detach().numpy()
        obj_val_r, max_v, mean_v, total_v, n_viol, n_total = compute_rounded_violations(
            ins_path, all_varname, BD_rounded
        )
        feas = 1.0 if n_viol == 0 else 0.0
        all_feas.append(feas)
        all_obj_round.append(obj_val_r)
        all_max_viol.append(max_v)
        all_total_viol.append(total_v)
        all_n_violated.append(n_viol)

        # ---- 4. 启发式固定 + 加信赖域约束 + 导出新实例 ----
        #        用 pyscipopt 做模型 I/O（开源，无 license 限制）
        scores, delta = fix_pas(scores, args.test_problem_type, args)

        import pyscipopt as scip_io
        model = scip_io.Model()
        model.hideOutput(True)
        model.readProblem(ins_path)

        instance_variables = model.getVars()
        instance_variables.sort(key=lambda v: v.name)
        variables_map = {v.name: v for v in instance_variables}

        alphas = []
        for i in range(len(scores)):
            var_name = scores[i][1]
            x_star = scores[i][3]  # 1, 0, or -1 (don't fix)
            if x_star < 0:
                continue
            tar_var = variables_map[var_name]
            tmp_var = model.addVar(name=f'alpha_{var_name}', vtype='C', lb=0)
            alphas.append(tmp_var)
            model.addCons(tmp_var >= tar_var - x_star, name=f'alpha_up_{i}')
            model.addCons(tmp_var >= x_star - tar_var, name=f'alpha_down_{i}')

        if len(alphas) > 0:
            model.addCons(scip_io.quicksum(alphas) <= delta, name="sum_alpha")

        # 导出修改后的 MILP 实例
        out_path = os.path.join(output_dir, base_name + '_modified.mps')
        model.writeProblem(out_path)
        model.freeProb()

        total_time = time.time() - t0
        sum_feat += feature_time
        sum_infer += infer_time
        sum_total += total_time
        count += 1

        n_fixed = sum(1 for s in scores if s[3] >= 0)
        n_conf = sum(1 for s in scores if abs(s[2] - 0.5) * 2 >= conf_threshold)
        sum_conf_vars += n_conf

        ce_str       = f"{ce_val:.4f}"       if not np.isnan(ce_val)       else "N/A"
        acc_str      = f"{acc_val:.4f}"      if not np.isnan(acc_val)      else "N/A"
        topk_acc_str = f"{topk_acc_val:.4f}" if not np.isnan(topk_acc_val) else "N/A"
        mse_str      = f"{mse_val:.4f}"      if not np.isnan(mse_val)      else "N/A"
        print(f"[{idx+1}/{len(test_instances)}] {ins_name}  "
              f"binary={len(scores)}  fixed={n_fixed}  conf={n_conf}  delta={delta}  "
              f"CE={ce_str}  MSE={mse_str}  Acc={acc_str}  Top{args.topk}-Acc={topk_acc_str}  "
              f"Obj={obj_val_r:.4f}  MaxViol={max_v:.6f}  "
              f"Violated={n_viol}/{n_total}  Feasible={'YES' if feas else 'NO'}  "
              f"feat={feature_time:.4f}s  infer={infer_time:.4f}s  "
              f"total={total_time:.4f}s  -> {os.path.basename(out_path)}")

    if count > 0:
        print(f"\n{'='*60}")
        print(f"Summary ({count} instances):")
        print(f"  Confidence filter  : threshold={conf_threshold}")
        print(f"  Avg vars meeting confidence per instance: {sum_conf_vars/count:.1f}  "
              f"(total={sum_conf_vars})")
        print(f"  Feature extraction : total={sum_feat:.4f}s  avg={sum_feat/count:.4f}s")
        print(f"  GNN inference      : total={sum_infer:.4f}s  avg={sum_infer/count:.4f}s")
        print(f"  Overall            : total={sum_total:.4f}s  avg={sum_total/count:.4f}s")
        print(f"  Output directory   : {output_dir}")

        if all_ce:
            ce_arr       = np.array(all_ce)
            acc_arr      = np.array(all_acc)
            topk_acc_arr = np.array(all_topk_acc)
            mse_arr      = np.array(all_mse)
            print(f"\n  Cross-Entropy Loss  ({len(ce_arr)} instances with solution):")
            print(f"    mean={ce_arr.mean():.4f}  std={ce_arr.std():.4f}  "
                  f"min={ce_arr.min():.4f}  max={ce_arr.max():.4f}  "
                  f"median={np.median(ce_arr):.4f}")
            print(f"  MSE Loss            ({len(mse_arr)} instances with solution):")
            print(f"    mean={mse_arr.mean():.4f}  std={mse_arr.std():.4f}  "
                  f"min={mse_arr.min():.4f}  max={mse_arr.max():.4f}  "
                  f"median={np.median(mse_arr):.4f}")
            print(f"  Rounded Accuracy    ({len(acc_arr)} instances with solution):")
            print(f"    mean={acc_arr.mean():.4f}  std={acc_arr.std():.4f}  "
                  f"min={acc_arr.min():.4f}  max={acc_arr.max():.4f}  "
                  f"median={np.median(acc_arr):.4f}")
            print(f"  Top-{args.topk} Conf Accuracy ({len(topk_acc_arr)} instances with solution):")
            print(f"    mean={topk_acc_arr.mean():.4f}  std={topk_acc_arr.std():.4f}  "
                  f"min={topk_acc_arr.min():.4f}  max={topk_acc_arr.max():.4f}  "
                  f"median={np.median(topk_acc_arr):.4f}")
        else:
            print("\n  No solution files found; CE/Acc statistics not available.")

        if all_feas:
            feas_arr = np.array(all_feas)
            obj_arr = np.array(all_obj_round)
            viol_arr = np.array(all_max_viol)
            total_viol_arr = np.array(all_total_viol)

            feasible_mask = feas_arr > 0.5
            infeasible_mask = ~feasible_mask
            n_feas = int(feasible_mask.sum())
            n_infeas = int(infeasible_mask.sum())

            print(f"\n  Rounded Feasibility ({len(all_feas)} instances):")
            print(f"    Feasible instances:   {n_feas}")
            print(f"    Infeasible instances: {n_infeas}")

            if n_feas > 0:
                print(f"    Feasible   — Avg Objective: {obj_arr[feasible_mask].mean():.4f}")
            else:
                print(f"    Feasible   — (none)")

            if n_infeas > 0:
                print(f"    Infeasible — Avg Total Violation: {total_viol_arr[infeasible_mask].mean():.4f}")
                print(f"    Infeasible — Avg Objective:       {obj_arr[infeasible_mask].mean():.4f}")
            else:
                print(f"    Infeasible — (none)")

            print(f"    Max Violation:     mean={viol_arr.mean():.6f}  max={viol_arr.max():.6f}")
            print(f"    Total time: {sum_total:.2f}s  Avg: {sum_total/count:.3f}s")

        print(f"{'='*60}")


def run_unsupervised_eval(args, policy):
    """
    Unsupervised ALM model evaluation:
    1. GNN inference -> x_hat
    2. Round to 0/1
    3. Evaluate discrete objective and constraint satisfaction
    """
    device = args.device
    test_instances = sorted(
        os.listdir(os.path.join(args.instance_dir, args.test_problem_type))
    )[:args.test_num]

    # Filter to .lp/.mps files
    test_instances = [f for f in test_instances if f.endswith(('.lp', '.mps'))]

    print(f"\n{'='*60}")
    print(f"Unsupervised Rounding Evaluation ({args.test_problem_type})")
    print(f"{'='*60}")

    all_feas = []
    all_obj = []
    all_max_viol = []
    all_total_viol = []
    all_n_violated = []
    sum_time = 0.0

    for idx, ins_name in enumerate(test_instances):
        ins_path = os.path.join(args.instance_dir, args.test_problem_type, ins_name)
        if not os.path.exists(ins_path):
            print(f"[SKIP] {ins_name}")
            continue

        t0 = time.time()

        # Feature extraction + inference
        A, v_map, v_nodes, c_nodes, b_vars = get_a_new2(ins_path)
        constraint_features = c_nodes.cpu()
        constraint_features[torch.isnan(constraint_features)] = 1
        variable_features = v_nodes
        edge_indices = A._indices()
        edge_features = torch.ones(A._values().unsqueeze(1).shape)

        constraint_features_batch = torch.zeros(len(constraint_features), dtype=torch.long, device=device)
        variable_features_batch = torch.zeros(len(variable_features), dtype=torch.long, device=device)

        BD = policy(
            constraint_features.to(device),
            edge_indices.to(device),
            edge_features.to(device),
            variable_features.to(device),
            torch.tensor([constraint_features.shape[0]]).to(device),
            constraint_features_batch,
            variable_features_batch,
        )
        BD = BD.sigmoid().cpu().squeeze()

        # Map GNN output to variable names
        all_varname = list(v_map)

        # Round
        BD_rounded = torch.round(BD).detach().numpy()

        # Evaluate constraint violations (pyscipopt)
        obj_val, max_v, mean_v, total_v, n_viol, n_total = compute_rounded_violations(
            ins_path, all_varname, BD_rounded
        )

        elapsed = time.time() - t0
        sum_time += elapsed

        feas = 1.0 if n_viol == 0 else 0.0
        all_feas.append(feas)
        all_obj.append(obj_val)
        all_max_viol.append(max_v)
        all_total_viol.append(total_v)
        all_n_violated.append(n_viol)

        # Polarization
        polar = ((BD < 0.05) | (BD > 0.95)).float().mean().item()

        print(f"[{idx+1}/{len(test_instances)}] {ins_name}  "
              f"Obj={obj_val:.4f}  MaxViol={max_v:.6f}  "
              f"Violated={n_viol}/{n_total}  Feasible={'YES' if feas else 'NO'}  "
              f"Polar={polar:.4f}  Time={elapsed:.3f}s")

    if all_feas:
        feas_arr = np.array(all_feas)
        obj_arr = np.array(all_obj)
        viol_arr = np.array(all_max_viol)
        total_viol_arr = np.array(all_total_viol)

        feasible_mask = feas_arr > 0.5
        infeasible_mask = ~feasible_mask
        n_feas = int(feasible_mask.sum())
        n_infeas = int(infeasible_mask.sum())

        print(f"\n{'='*60}")
        print(f"Summary ({len(all_feas)} instances):")
        print(f"  Feasible instances:   {n_feas}")
        print(f"  Infeasible instances: {n_infeas}")

        if n_feas > 0:
            print(f"  Feasible   — Avg Objective: {obj_arr[feasible_mask].mean():.4f}")
        else:
            print(f"  Feasible   — (none)")

        if n_infeas > 0:
            print(f"  Infeasible — Avg Total Violation: {total_viol_arr[infeasible_mask].mean():.4f}")
            print(f"  Infeasible — Avg Objective:       {obj_arr[infeasible_mask].mean():.4f}")
        else:
            print(f"  Infeasible — (none)")

        print(f"  Max Violation:     mean={viol_arr.mean():.6f}  max={viol_arr.max():.6f}")
        print(f"  Total time: {sum_time:.2f}s  Avg: {sum_time/len(all_feas):.3f}s")
        print(f"{'='*60}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="multitest for CoCo.")
    
    exp_group = parser.add_argument_group("Experiment Settings")
    exp_group.add_argument("--test_problem_type", type=str, choices=TASKS, default='SC',help="Problem type to train on (e.g., CA, WA, IP)")
    
    exp_group.add_argument("--test_num", type=int, default=100,
                         help="Number of test instances to process")
    
    model_group = parser.add_argument_group("Model Settings")

    model_group.add_argument("--model_dir", default="./pretrain_models",
                           help="Directory containing pretrained models")
    
    gcn_group = parser.add_argument_group("GCN Settings")
    gcn_group.add_argument("--emb_size", type=int, default=64,
                       help="Embedding size for GNN (default: %(default)s)")
    gcn_group.add_argument("--cons_nfeats", type=int, default=4, 
                       help="Number of features for constraint nodes (default: %(default)s)")
    gcn_group.add_argument("--edge_nfeats", type=int, default=1, 
                       help="Number of features for edge (default: %(default)s)")
    gcn_group.add_argument("--var_nfeats", type=int, default=6, 
                       help="Number of features for variable nodes (default: %(default)s)")
    parser.add_argument("--depth", type=int, default=2)
    
    solver_group = parser.add_argument_group("Solver Settings")
    solver_group.add_argument("--solver", choices=SOLVER_CLASSES.keys(), default="gurobi",
                            help="MILP solver implementation (default: %(default)s)")
    solver_group.add_argument("--max_time", type=int, default=1000,
                            help="Maximum solving time in seconds")
    solver_group.add_argument("--threads", type=int, default=1,
                            help="Number of threads for solving")
 
    sys_group = parser.add_argument_group("System Settings")
    sys_group.add_argument("--instance_dir",
                         default="/home/lmh/autodl-tmp/data/l2o_milp_test",
                         help="Path to test instances directory")
    sys_group.add_argument("--scores_dir", default="./scores",
                         help="Path to store scores directory")
    sys_group.add_argument("--device", default="cuda:0",
                         help="Computation device (default: %(default)s)")
    sys_group.add_argument("--num_workers", type=int, default=1,
                         help="Number of parallel workers (default: %(default)s)")
    sys_group.add_argument("--log_dir", default="./test_logs/",
                         help="Path to test instances directory")
    
    parser.add_argument('--Intra_Constraint_Competitive', default=False, action='store_true')
    parser.add_argument('--inference_only', default=False, action='store_true',
                        help='仅推理+四舍五入，计算目标值和约束违反量，不调用求解器')
    parser.add_argument('--unsupervised_eval', default=False, action='store_true',
                        help='Evaluate unsupervised ALM model: round and check feasibility')

    parser.add_argument("--margin",type=float,default=0.9)
    parser.add_argument("--alpha",type=float,default=0.01)
    parser.add_argument("--tao",type=float,default=0.1)

    parser.add_argument("--k0", type=int, default=1000)
    parser.add_argument("--k1", type=int, default=0)
    parser.add_argument("--delta", type=int, default=200)
    parser.add_argument("--topk", type=int, default=500,
                        help="Top-K most confident binary variables for accuracy evaluation (default: 500)")

    return parser.parse_args()


def _solver_worker(task_queue, stop_flag, args, log_dir, save_name):
    """
    线程worker：从 task_queue 取 (ins_name, ins_path, scores)，调用求解器。
    每个 worker 独立运行，无需共享求解器对象。
    """
    logger = configure_logging(log_dir=log_dir)

    while not stop_flag.is_set():
        try:
            item = task_queue.get(timeout=1)
        except queue.Empty:
            continue
        if item is None:  # poison pill
            break

        ins_name, ins_path, scores = item
        log_path = os.path.join(log_dir, f"{save_name}_{ins_name.split('.')[0]}.log")
        logger.info(f"[worker {threading.current_thread().name}] "
                    f"Start solving {ins_name} with {args.solver}")
        try:
            solve_mps(ins_path, log_path, save_name, ins_name,
                      scores, args.test_problem_type, args)
        except Exception as e:
            logger.error(f"[worker] Error solving {ins_name}: {e}")


def main():
    setup_environment()
    args = parse_arguments()

    save_name = (
        f'Intra_Constraint_Competitive_{args.Intra_Constraint_Competitive}'
        f'_margin_{args.margin}_alpha_{args.alpha}_tao_{args.tao}'
        f'_k0_{args.k0}_k1_{args.k1}_delta_{args.delta}'
    )

    model_path = args.model_dir
    policy = load_pretrained_model(args, model_path, args.device)

    log_dir = os.path.join(args.log_dir, args.test_problem_type, save_name)
    os.makedirs(log_dir, exist_ok=True)

    scores_dir = os.path.join(
        args.scores_dir, args.test_problem_type,
        f"Intra_Constraint_Competitive_{args.Intra_Constraint_Competitive}"
        f"_margin_{args.margin}_alpha_{args.alpha}_tao_{args.tao}"
    )
    os.makedirs(scores_dir, exist_ok=True)

    if args.inference_only:
        run_inference_only(args, policy)
        return

    if args.unsupervised_eval:
        run_unsupervised_eval(args, policy)
        return

    test_instances = sorted(
        os.listdir(os.path.join(args.instance_dir, args.test_problem_type))
    )[:args.test_num]

    num_workers = args.num_workers

    conf_threshold = args.margin
    sum_conf_vars = 0
    total_instances = 0

    if num_workers <= 1:
        # ---- 单线程：顺序推理 + 顺序求解（原始行为）----
        for ins_name in test_instances:
            ins_path = os.path.join(args.instance_dir, args.test_problem_type, ins_name)
            score_path = os.path.join(scores_dir, f"scores_{ins_name.split('.')[0]}.pkl")

            if os.path.exists(score_path):
                with open(score_path, 'rb') as f:
                    scores = pickle.load(f)
            else:
                scores = process_single_instance(args, ins_path, policy, args.device)
                with open(score_path, 'wb') as f:
                    pickle.dump(scores, f)

            n_conf = sum(1 for s in scores if abs(s[2] - 0.5) * 2 >= conf_threshold)
            sum_conf_vars += n_conf
            total_instances += 1

            log_path = os.path.join(log_dir, f"{save_name}_{ins_name.split('.')[0]}.log")
            solve_mps(ins_path, log_path, save_name, ins_name,
                      scores, args.test_problem_type, args)
    else:
        # ---- 多线程：主线程顺序推理，solver workers 并行求解 ----
        # task_queue 容量限制防止推理跑太快堆积内存
        task_queue = queue.Queue(maxsize=2 * num_workers)
        stop_flag = threading.Event()

        workers = []
        for _ in range(num_workers):
            t = threading.Thread(
                target=_solver_worker,
                args=(task_queue, stop_flag, args, log_dir, save_name),
                daemon=True,
            )
            t.start()
            workers.append(t)

        # 主线程：顺序推理，dispatch 给 solver workers
        for ins_name in test_instances:
            ins_path = os.path.join(args.instance_dir, args.test_problem_type, ins_name)
            score_path = os.path.join(scores_dir, f"scores_{ins_name.split('.')[0]}.pkl")

            if os.path.exists(score_path):
                with open(score_path, 'rb') as f:
                    scores = pickle.load(f)
                print(f"[infer] {ins_name}: loaded cached scores", flush=True)
            else:
                scores = process_single_instance(args, ins_path, policy, args.device)
                with open(score_path, 'wb') as f:
                    pickle.dump(scores, f)
                print(f"[infer] {ins_name}: inference done", flush=True)

            n_conf = sum(1 for s in scores if abs(s[2] - 0.5) * 2 >= conf_threshold)
            sum_conf_vars += n_conf
            total_instances += 1

            # 阻塞直到 queue 有空位（背压控制）
            task_queue.put((ins_name, ins_path, scores))

        # 发送 poison pill 让所有 worker 退出
        for _ in workers:
            task_queue.put(None)

        for t in workers:
            t.join()

    if total_instances > 0:
        print(f"\nConfidence stats (threshold={conf_threshold}):")
        print(f"  Avg vars meeting confidence per instance: {sum_conf_vars/total_instances:.1f}  "
              f"(total={sum_conf_vars}, instances={total_instances})")

    print("Testing completed successfully.")


if __name__ == "__main__":
    main()