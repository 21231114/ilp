# 训练、调参与日志指标说明

本文档基于当前代码整理，目标是让刚接触项目的人能快速理解：

- 无监督 ALM 训练的损失函数由哪些项组成；
- 训练、验证、测试相关超参数分别控制什么；
- 模型保存、恢复、测试加载机制；
- 训练日志、验证日志、测试日志里每个字段的含义；
- 调参时优先看哪些指标、常见问题该从哪些参数入手。

数据集中只有SC,CA和IS是整数规划问题，但是目前只跑了SC和CA  
目前的运行参数设置：  
SC:python train.py --problem_type SC --instance_dir /home/lmh/autodl-tmp/data/l2o_milp --num_epochs 5000 --device cuda:0 --tau 1.0 --tau_min 0.9999 --gamma_init 22 --delta_gamma 0.1 --gamma_max 900.0 --rho_init 1.0 --beta 1.1 --rho_max 10000.0 --inner_steps 240 --entropy_weight 0.0 --lr 1e-4 --es_xi_threshold 1.1 --es_xi_threshold2 1.1 --freeze_gamma_on_feasible --freeze_rho_on_feasible --patience 5000 --threshold2_on valid
CA:python train.py --problem_type CA --instance_dir /home/lmh/autodl-tmp/data/l2o_milp --num_epochs 5000 --device cuda:0 --tau 1.0 --tau_min 0.9999 --gamma_init 5(也可能是-4,这个忘怎么设置了) --delta_gamma 0.05 --gamma_max 900.0 --rho_init 1.0 --beta 1.1 --rho_max 10000.0 --inner_steps 240 --entropy_weight 0.0 --lr 1e-4 --es_xi_threshold 1.1 --es_xi_threshold2 1.1 --freeze_gamma_on_feasible --freeze_rho_on_feasible --patience 5000 --threshold2_on valid  

核心代码位置：

| 模块 | 文件 | 主要内容 |
|---|---|---|
| 训练入口 | `train.py` | 参数解析、数据划分、训练/验证循环、日志、保存模型 |
| 损失函数 | `train.py:66` | `compute_alm_loss()`，ALM + margin + 熵正则 |
| 离散评估 | `train.py:213` | `evaluate_discrete()`，round 后检查约束和目标 |
| 模型结构 | `gnn.py` | `GNNPolicy` 和二部图消息传递层 |
| 特征/原始 ILP 提取 | `utils.py` | 图特征、原始目标/约束提取与约束方向统一 |
| 训练数据集 | `dataset/unsupervised_dataset.py` | `.lp/.mps` 读取、缓存、PyG Data 封装 |
| 测试入口 | `test.py` | 推理、变量固定、信赖域约束、solver 调用、测试指标 |


---

## 1. 训练整体流程

训练入口是：

```bash
python train.py --problem_type SC
```

`train.py` 做的事情：

1. 读取命令行参数。
2. 从 `--instance_dir` 找训练实例：优先使用 `instance_dir/problem_type`，不存在则直接使用 `instance_dir`。
3. 收集 `.lp` / `.mps` 文件，随机打乱后按 8:2 划分 train/valid；如果只有 1 个实例，则 train 和 valid 共用同一个实例。
4. 用 `UnsupervisedGraphDataset` 提取：
   - GNN 二部图特征；
   - 原始 ILP 的 `c, A, b`，供 ALM loss 计算。
5. 构建 `GNNPolicy`，输出每个变量的 logit，再经过 `sigmoid()` 得到 `x_hat in (0,1)`。
6. 训练时用 ALM loss 反向传播；每隔 `inner_steps` 个 batch 更新 ALM 状态参数 `lambda_global/rho/gamma`。
7. 每个 epoch 保存 `_model_last.pth`；验证集指标变好时保存 `_model_best.pth`；验证集所有实例离散可行且目标更好时保存 `_model_best_allfeas.pth`。

---

## 2. 数据、目标和约束约定（这块可以跳过，对实验而言意义不大）

### 2.1 原始 ILP 数据

`dataset/unsupervised_dataset.py` 会同时保存图特征和原始 ILP 数据：

| 字段 | 含义 |
|---|---|
| `obj_coeffs` | 原始目标系数 `c`，如果原问题是 maximize，会乘 `-1` 转成最小化方向 |
| `raw_cons_indices` | 稀疏约束矩阵 `A` 的 `(row, col)` 索引 |
| `raw_cons_values` | 稀疏约束矩阵 `A` 的非零值 |
| `raw_rhs` | 约束右端 `b` |
| `raw_n_cons` | 转换后的约束条数 |
| `gnn_to_raw_map` | GNN 输出变量顺序到原始 ILP 变量顺序的映射 |
| `b_vars_mask` / `b_vars` | 二进制变量标记/索引 |
| `varNames` | 变量名列表，用于测试时和 solver 变量对齐 |

### 2.2 约束方向统一

`utils.extract_raw_ilp()` 会把约束统一成 `A x <= b`：

- `<=` 约束：保持不变；
- `>=` 约束：乘 `-1` 变成 `-A x <= -lhs`；
- 等式约束：拆成两条：`A x <= b` 和 `-A x <= -b`。

因此训练 loss、训练验证离散评估里的违反量都按 `ReLU(Ax - b)` 来理解。

### 2.3 图特征维度

默认：

- 变量节点特征维度 `var_nfeats=6`；
- 约束节点特征维度 `cons_nfeats=4`；
- 边特征维度 `edge_nfeats=1`。

变量节点 6 维特征来自 `utils.get_a_new2()`：

| 维度 | 含义 |
|---|---|
| 0 | 目标系数 `c_i`，先 clamp 到 `[0, 20000]` 再归一化 |
| 1 | 该变量在约束中的系数累加，代码中为 `sum(A_ji / n_cons)` |
| 2 | 变量出现的约束数，即 degree |
| 3 | 该变量在所有约束系数中的最大值 |
| 4 | 该变量在所有约束系数中的最小值 |
| 5 | 是否为二进制变量，二进制为 1，否则为 0 |

约束节点 4 维特征：

| 维度 | 含义 |
|---|---|
| 0 | 该约束所有系数的平均值 |
| 1 | 该约束包含的变量数，即非零系数数 |
| 2 | 右端项 `rhs` |
| 3 | 约束类型：`0` 表示 `<=`，`1` 表示 `>=`，`2` 表示等式 |

注意：图特征会做 min-max 归一化并 clamp 到 `[1e-5, 1]`。训练 loss 用的是 `extract_raw_ilp()` 提取出的原始 `c/A/b`，不是归一化后的图特征。

---

## 3. 模型结构和相关超参数（如果要调整模型大小，应该调这块的参数，这里使用的不是普通的GNN,使用的是有监督baseline（coco-milp）这个工作提出的GNN）

模型定义在 `gnn.py`。

### 3.1 `GNNPolicy`

`GNNPolicy` 接收约束节点、变量节点、边特征，做二部图消息传递，最后对变量节点输出一个 logit：

```python
logits = model(...)
x_hat = logits.sigmoid()
```

`x_hat_i` 可以理解为变量 `i` 取 1 的概率/软解。

### 3.2 架构参数

| 参数 | 默认值 | 作用 | 调参说明 |
|---|---:|---|---|
| `--emb_size` | `64` | 节点嵌入维度 | 当前 `BipartiteGraphConvolution` 内部写死 `emb_size=64`，所以不建议直接改；要改需同步修改 `gnn.py` 内部层维度 |
| `--cons_nfeats` | `4` | 约束节点输入特征维度 | 需和 `utils.get_a_new2()` 输出一致 |
| `--edge_nfeats` | `1` | 边特征维度 | 当前边特征实际是全 1 |
| `--var_nfeats` | `6` | 变量节点输入特征维度 | 需和 `utils.get_a_new2()` 输出一致 |
| `--depth` | `2` | 二部图消息传递轮数 | 更大可能表达力更强，但显存/时间更高，也更难训 |
| `--Intra_Constraint_Competitive` | `False` | 是否启用约束内竞争归一化层 `ConstraintNormalization` | 打开后变量特征会减去同约束邻域聚合均值，适合尝试改善变量之间竞争关系 |
| `--gnn_type` | `gcn` | 参数保留 | 当前代码没有根据该参数切换模型，实际无效 |

输出层初始化：最后一层 bias 初始化为 0，因此初始 `sigmoid(0)=0.5`，代表中性软解。

---

## 4. 损失函数组成

损失函数在 `train.py:66` 的 `compute_alm_loss()`。

### 4.1 总公式

训练优化的 loss 可以概括为：

```text
loss = normalized_f_tilde
     + lambda_global * sum_j xi_j
     + (rho / 2) * sum_j xi_j^2
     + entropy_weight * H(x_hat)
```

在 `train_epoch()` 中，最终还会除以 batch 内实例数：

```text
loss_for_backward = loss / batch.num_graphs
```

其中：

| 项 | 代码变量 | 作用 |
|---|---|---|
| margin-aware 目标项 | `f_tilde_normalized` | 让模型优化目标函数，同时考虑四舍五入误差 margin |
| ALM 线性惩罚 | `lambda_global * xi.sum()` | 拉格朗日乘子惩罚违反约束 |
| ALM 二次惩罚 | `(rho / 2) * (xi ** 2).sum()` | 对较大的违反量施加更强惩罚 |
| 二值熵正则 | `entropy_weight * entropy` | 惩罚接近 0.5 的输出，鼓励 `x_hat` 靠近 0/1 |

### 4.2 状态不确定性 `sqrt_u`

```text
u_i = exp(-gamma * (x_hat_i - 0.5)^2)
sqrt_u_i = sqrt(u_i)
```

含义：

- `x_hat_i` 越接近 0.5，变量越不确定，`sqrt_u_i` 越大；
- `x_hat_i` 越接近 0 或 1，变量越确定，`sqrt_u_i` 越小；
- `gamma` 越大，不确定性对偏离 0.5 的变量衰减越快。

### 4.3 `K(gamma)`

`compute_K(gamma)` 定义四舍五入误差界常数：

- 当 `gamma <= 19.56` 时，`K(gamma)=0.5`；
- 当 `gamma > 19.56` 时，按代码中的闭式公式计算。

注意：`compute_K(gamma, K_max=10.0)` 虽然有 `K_max` 参数和注释，但当前实现没有实际 `min(K, K_max)` 截断。

### 4.4 margin-aware 目标 `f_tilde`

```text
f_base   = sum_i c_i * x_i
f_margin = K(gamma) * sum_i |c_i| * sqrt_u_i
f_tilde  = f_base + f_margin
f_tilde_normalized = f_tilde / max(sum_i |c_i|, 1)
```

含义：

- `f_base` 是连续软解的原始目标值；
- `f_margin` 是目标项的保守 margin，变量越不确定，该项越大；
- 训练 loss 中使用的是归一化后的 `f_tilde_normalized`；
- 日志中的 `Obj_margin` / `ObjMargin/inst` 记录的是未归一化的 `f_tilde`。

### 4.5 约束 margin、容忍度和违反量

先计算：

```text
Ax_j = sum_i A_ji * x_i
margin_j = K(gamma) * sum_i |A_ji| * sqrt_u_i
raw_no_tau_j = Ax_j - b_j + margin_j
xi_no_tau_j = ReLU(raw_no_tau_j)
xi_raw_j = ReLU(Ax_j - b_j)
```

其中：

| 变量 | 含义 | 日志相关 |
|---|---|---|
| `xi_raw` | 不带 margin、不带 tau 的原始连续违反量 `ReLU(Ax-b)` | `Xi_raw/inst` |
| `xi_no_tau` | 带 margin、不减 tau 的违反量 | `XiSum/s`、`Xi_m/inst`、`Xi_mean/cons` |
| `xi` | 带 margin、减 tau、可选归一化后的最终 loss 违反量 | `MaxViol`、`MeanViol`、`Xi_m_t/inst` |

然后计算每条约束的容忍度：

```text
sqrt_u_0 = exp(-gamma / 2 * (tau - 0.5)^2)
tau_j = K(gamma) * sqrt_u_0 * sum_i |A_ji|
```

如果设置了 `tau_min > 0`，则：

```text
tau_j = max(tau_j, tau_min)
```

最终进入 ALM loss 的违反量：

```text
raw_violation_j = raw_no_tau_j - tau_j
if cons_normalize:
    raw_violation_j /= constraint_norm_j
xi_j = ReLU(raw_violation_j)
```

`constraint_norm_j` 来自 `compute_constraint_norms()`：

```text
constraint_norm_j = max(sum_i |A_ji| + |b_j|, 1e-4)
```

### 4.6 二值熵正则（这个不用管，默认这项的系数为0就行，claude写代码时给我瞎加的）

```text
H(x_hat) = mean_i( -x_i log x_i - (1-x_i) log(1-x_i) )
loss += entropy_weight * H(x_hat)
```

`H(x)` 在 `x=0.5` 最大，在 `x=0/1` 附近最小。因为训练是最小化 loss，所以加上这项会惩罚不确定输出，推动变量极化到 0 或 1。

---

## 5. ALM 动态参数更新机制

ALM 状态包括：

| 状态 | 初始值参数 | 含义 |
|---|---|---|
| `gamma` | `--gamma_init` | 状态不确定性锐化参数，影响 margin 和 tau |
| `rho` | `--rho_init` | 二次惩罚权重 |
| `lambda_global` | 固定从 `0.0` 开始 | 全局拉格朗日乘子 |
| `prev_violation` | `inf` | 上一次外循环的违反量，用于判断是否增大 `rho` |
| `step_counter` | `0` | batch 级步数计数器 |

每训练 `inner_steps` 个 batch 后触发一次外循环更新：

```text
curr_viol = sum_j xi_j / batch.num_graphs

lambda_global = max(0, lambda_global + rho * curr_viol)

if curr_viol > 0.8 * prev_violation and curr_viol > 1e-4:
    rho = min(rho * beta, rho_max)

gamma = min(gamma + delta_gamma, gamma_max)

prev_violation = curr_viol
```

含义：

- 如果违反量下降不够快，`rho` 会乘 `beta` 放大；
- `gamma` 随外循环逐步增大，使不确定性/margin 机制逐渐变化；
- `lambda_global` 是全局标量，不是每条约束单独一个 lambda。

### 5.1 ALM freeze 机制

如果设置：

```bash
--es_xi_threshold2 <value>
```

则当 `xi_sum_per_sample < es_xi_threshold2` 时，下一轮开始冻结 `lambda` 更新。判定来源由：

```bash
--threshold2_on train|valid
```

决定。

可选：

| 参数 | 作用 |
|---|---|
| `--freeze_gamma_on_feasible` | 达到 threshold2 后同时冻结 `gamma` |
| `--freeze_rho_on_feasible` | 达到 threshold2 后同时冻结 `rho` |

如果后续 `xi_sum_per_sample` 又高于阈值，会自动 unfreeze。

---

## 6. 训练参数完整说明

参数定义在 `train.py:get_parser()`。

### 6.1 问题与模型结构参数

| 参数 | 默认值 | 含义 | 建议 |
|---|---:|---|---|
| `--problem_type` | `SC` | 问题类型，可选 `IP/WA/CA/SC/2club` | 必须和数据目录、测试任务保持一致 |
| `--gnn_type` | `gcn` | 保留参数 | 当前未实际使用 |
| `--emb_size` | `64` | 嵌入维度 | 当前不建议改，见第 3.2 节 |
| `--cons_nfeats` | `4` | 约束节点特征维度 | 通常不改 |
| `--edge_nfeats` | `1` | 边特征维度 | 通常不改 |
| `--var_nfeats` | `6` | 变量节点特征维度 | 通常不改 |
| `--depth` | `2` | 消息传递深度 | 可尝试 2/3，太深可能更慢或更难训 |
| `--Intra_Constraint_Competitive` | `False` | 是否启用约束内竞争归一化 | 可作为模型结构 ablation |

### 6.2 优化器与训练规模

| 参数 | 默认值 | 含义 | 调参方向 |
|---|---:|---|---|
| `--lr` | `1e-4` | AdamW 学习率 | loss 抖动/发散可降到 `5e-5`；收敛慢可试 `2e-4` |
| `--weight_decay` | `1e-5` | AdamW L2 正则 | 通常不优先调 |
| `--num_epochs` | `5000` | 最大 epoch 数 | 早停会提前结束 |
| `--num_workers` | `0` | DataLoader worker 数 | 数据读取慢时可增大，但 PySCIPOpt/缓存场景需实测稳定性 |
| `--batch_size` | `None` | 覆盖任务默认 batch size | 默认按任务：`CA/WA/IP=4`，`SC/2club=1` |
| `--grad_clip_norm` | `1.0` | 梯度裁剪上限；`0` 表示关闭 | loss 不稳时保留或降低；训练很慢可尝试关闭对比 |
| `--warmup_epochs` | `10` | 学习率 warmup epoch 数 | 大数据/不稳定时可增大 |
| `--lr_schedule` | `cosine` | 学习率调度：`cosine/step/none` | 默认 cosine；debug 可用 none |
| `--ema_decay` | `0.999` | EMA 权重衰减；`0` 关闭 EMA | 默认建议保留；验证/保存 best 使用 EMA 权重 |
| `--seed` | `0` | 随机种子 | 复现实验时固定 |

学习率调度细节：

- `cosine`：前 `warmup_epochs * len(train_loader)` step 线性 warmup，最低 1% lr；之后余弦下降，最低 1% lr。
- `step`：每 `total_steps // 5` step 学习率乘 0.5。
- `none`：不使用 scheduler。

### 6.3 ALM 与 loss 参数

| 参数 | 默认值 | 含义 | 调参方向 |
|---|---:|---|---|
| `--tau` | `0.9` | 计算 `tau_j(gamma)` 的参考点 `x_0` | 不是直接容忍度；越靠近 0.5，`sqrt_u_0` 越大，tau 越大 |
| `--tau_min` | `0.99` | `tau_j` 下界；设为 `0` 表示禁用 floor | 太大可能让 loss 对违反约束更宽松；太小会更严格但更难训 |
| `--gamma_init` | `1.0` | 初始 `gamma` | 影响不确定性和 margin 初始状态 |
| `--gamma_max` | `50.0` | `gamma` 上限 | 输出长期不极化时可关注 gamma 是否过低/过慢 |
| `--delta_gamma` | `0.3` | 每次外循环 `gamma` 增量 | 增大后 gamma 更快达到上限 |
| `--rho_init` | `1.0` | 初始二次惩罚系数 | 约束违反量长期高时可增大 |
| `--rho_max` | `1e5` | `rho` 上限 | 防止惩罚无限增长 |
| `--beta` | `1.5` | `rho` 放大倍率 | 违反量降不下来时可增大；过大可能训练不稳 |
| `--inner_steps` | `20` | 每多少 batch 更新一次 ALM 状态 | 减小会更频繁更新 lambda/rho/gamma；增大则更平滑 |
| `--entropy_weight` | `0.01` | 二值熵正则权重 | 输出长期接近 0.5 可增大；过大可能牺牲目标/可行性 |
| `--cons_normalize` | `True` | 按约束尺度归一化违反量 | 默认开启 |
| `--no_cons_normalize` | - | 关闭约束归一化 | 只有在确认归一化影响目标时再尝试 |

注意：`--cons_normalize` 默认已经是 True；如果要关闭，需要显式传 `--no_cons_normalize`。

### 6.4 路径、验证和早停参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--instance_dir` | `/home/lmh/autodl-tmp/data/l2o_milp` | 训练实例目录 |
| `--cache_dir` | `None` | 预处理缓存目录；默认用 `log_save_dir/problem_type/unsup_cache` |
| `--model_save_dir` | `./pretrain_models` | 模型保存根目录 |
| `--log_save_dir` | `./train_logs` | 训练日志根目录 |
| `--tensorboard_dir` | `./tb_logs` | TensorBoard 日志根目录 |
| `--resume_from` | `None` | 从 checkpoint 恢复训练 |
| `--device` | `cuda:0` | 计算设备 |
| `--val_every` | `1` | 每 N 个 epoch 验证一次 |
| `--es_xi_threshold` | `1.0` | 判定“达到可行阈值”的 `xi_sum_per_sample` 阈值 |
| `--patience` | `50` | 验证指标无提升的早停耐心 |
| `--es_xi_threshold2` | `None` | 触发 ALM freeze 的第二阈值 |
| `--threshold2_on` | `valid` | 第二阈值看 train 还是 valid 指标 |
| `--freeze_gamma_on_feasible` | `False` | threshold2 达成后是否冻结 gamma |
| `--freeze_rho_on_feasible` | `False` | threshold2 达成后是否冻结 rho |

---

## 7. 模型保存、加载和早停机制

### 7.1 保存文件名

`save_name` 由关键 ALM/结构参数拼接：

```text
ALM_tau{tau}_taumin{tau_min}_gamma{gamma_init}_rho{rho_init}
_inner{inner_steps}_ent{entropy_weight}
_ICC{Intra_Constraint_Competitive}
_esxi{es_xi_threshold}_esxi2{es_xi_threshold2}_t2on{threshold2_on}
```

保存目录：

```text
{model_save_dir}/{problem_type}/
```

训练日志目录：

```text
{log_save_dir}/{problem_type}/{save_name}_train.log
```

TensorBoard 目录：

```text
{tensorboard_dir}/{problem_type}/{save_name}/
```

### 7.2 `_model_best.pth`

验证时判断 best，核心指标：

```text
curr_xi = val_metrics['xi_sum_per_sample']
curr_obj = val_metrics['disc_obj_per_inst']
curr_feasible = curr_xi < es_xi_threshold
```

优先级：

1. 如果当前第一次达到 `curr_feasible=True`，保存 best；
2. 如果当前和历史 best 都 feasible，则比较 `curr_obj`，目标更低则保存；
3. 如果当前和历史 best 都不 feasible，则比较 `curr_xi`，违反量更低则保存。

保存内容：

- 如果 `ema_decay > 0`，保存 EMA 权重合并后的 `state_dict`；
- 否则保存当前模型 `state_dict`。

这个文件适合用于测试推理。

### 7.3 `_model_best_allfeas.pth`

当验证集离散四舍五入后所有实例都可行：

```text
val_metrics['n_infeasible_inst'] == 0
```

并且 `disc_obj_per_inst` 优于历史记录时，保存：

```text
{save_name}_model_best_allfeas.pth
```

这个文件也适合用于测试推理，通常比普通 best 更强调离散可行性。

### 7.4 `_model_last.pth`

每个 epoch 都会覆盖保存：

```text
{save_name}_model_last.pth
```

内容是完整训练 checkpoint：

| 字段 | 含义 |
|---|---|
| `model_state_dict` | 模型当前权重 |
| `optimizer_state_dict` | AdamW 状态 |
| `scheduler_state_dict` | scheduler 状态，如果有 |
| `ema_shadow` | EMA shadow 参数，如果启用 EMA |
| `epoch` | 当前 epoch |
| `gamma` / `rho` / `lambda_global` | ALM 状态 |
| `prev_violation` / `step_counter` | ALM 外循环状态 |

这个文件适合 `--resume_from` 恢复训练，不适合直接传给 `test.py` 做推理。

### 7.5 恢复训练

```bash
python train.py --resume_from path/to/xxx_model_last.pth
```

如果 checkpoint 内含 `model_state_dict`，会恢复：

- 模型；
- optimizer；
- scheduler；
- EMA；
- ALM 状态；
- 起始 epoch。

如果传入的是普通 `state_dict`，只加载模型权重，optimizer/ALM 不恢复，从 epoch 0 开始。

### 7.6 测试加载注意事项

`test.py` 里的 `--model_dir` 名字容易误导：当前代码实际把它当成“模型文件路径”使用，而不是目录。

正确示例：

```bash
python test.py \
  --test_problem_type SC \
  --model_dir ./pretrain_models/SC/xxx_model_best.pth \
  --unsupervised_eval
```

不要把 `_model_last.pth` 直接传给 `test.py`；它是完整训练 checkpoint，不是纯模型 `state_dict`。

---

## 8. 训练日志字段说明

训练日志由 `train.py:645` 的 `format_metrics()` 生成，同时打印到控制台和写入：

```text
{log_save_dir}/{problem_type}/{save_name}_train.log
```

每个 epoch 的格式大致如下：

```text
@epoch0  TIME:12.3s
  [Train] Loss=... Obj_margin=... MaxViol=... MeanViol=... XiSum/s=... Entropy=...
  [Pred]  Pred0=... Pred1=... Xi_mean/cons=... Tau_mean=...
  [ALM]   gamma=... rho=... lambda=... K(gamma)=...
  [TDisc] AvgObj/inst=... AvgViol/inst=... FeasInst=... InfeasInst=...
  [TCont] ObjMargin/inst=... ObjRaw/inst=... Xi_m_t/inst=... Xi_raw/inst=... Xi_m/inst=...
  [Valid] ...
  [VPred] ...
  [Disc] ...
  [VDisc] ...
  [VCont] ...
```

### 8.1 epoch 头

| 字段 | 含义 |
|---|---|
| `@epochN` | 当前 epoch 编号，从 0 开始 |
| `TIME` | 当前 epoch 训练 + 验证 + 保存 + 日志耗时 |

### 8.2 `[Train]` 训练 loss 与连续违反量

| 字段 | 来源 | 含义 | 趋势解读 |
|---|---|---|---|
| `Loss` | `loss_total` | batch 平均总 loss，已除以 `batch.num_graphs` | 总体应下降；但 ALM 的 `rho/lambda/gamma` 会变化，可能不单调 |
| `Obj_margin` | `objective_margin` | batch 平均 `f_tilde`，未按 `sum|c|` 归一化 | 越低越好，但要先满足可行性 |
| `MaxViol` | `max_violation` | 最终 `xi` 的 batch 平均最大值，带 margin、减 tau、可归一化 | 越低越好 |
| `MeanViol` | `mean_violation` | 最终 `xi` 的 batch 平均均值 | 越低越好 |
| `XiSum/s` | `xi_sum_per_sample` | 每实例 `sum(xi_no_tau)`，带 margin、不减 tau、不按 constraint norm 归一化 | best/early-stop 主要看这个 |
| `Entropy` | `entropy` | 二值熵均值 | 越低表示输出越接近 0/1 |

### 8.3 `[Pred]` 预测分布

| 字段 | 含义 | 趋势解读 |
|---|---|---|
| `Pred0` | `round(x_hat)==0` 的变量比例 | 和 `Pred1` 一起看是否塌缩到全 0/全 1 |
| `Pred1` | `round(x_hat)==1` 的变量比例 | 任务不同合理比例不同 |
| `Xi_mean/cons` | 每条约束平均 `xi_no_tau` | 越低越好 |
| `Tau_mean` | 当前 batch 的 `tau_j` 均值 | 用来判断 tau 容忍度规模 |

### 8.4 `[ALM]` 动态参数

| 字段 | 含义 |
|---|---|
| `gamma` | 当前状态锐化参数 |
| `rho` | 当前二次惩罚系数 |
| `lambda` | 当前全局拉格朗日乘子 |
| `K(gamma)` | 当前 rounding margin 常数 |

如果 `rho/lambda` 快速变得很大，通常说明约束违反量没有按 ALM 期望下降，训练可能开始被可行性惩罚主导。

### 8.5 `[TDisc]` 训练集离散评估

这些指标来自 `evaluate_discrete()`，先对 `x_hat` 做 `round()`，再在原始 ILP 约束上检查。

| 字段 | 含义 | 趋势解读 |
|---|---|---|
| `AvgObj/inst` | 每实例平均离散目标 `c^T round(x)` | 在可行前只作参考；可行后越低越好 |
| `AvgViol/inst` | 每实例平均离散原始违反量之和 | 越低越好，0 表示所有实例离散约束都满足 |
| `FeasInst` | 离散后所有约束都满足的训练实例数 | 越高越好 |
| `InfeasInst` | 离散后至少一条约束违反的训练实例数 | 越低越好 |

### 8.6 `[TCont]` 训练集连续 per-instance 指标

| 字段 | 含义 |
|---|---|
| `ObjMargin/inst` | 每实例平均 `f_tilde = f_base + f_margin` |
| `ObjRaw/inst` | 每实例平均连续目标 `c^T x_hat` |
| `Xi_m_t/inst` | 每实例平均 `sum(xi)`，带 margin、减 tau、可归一化，是真正进入 ALM 惩罚的违反量 |
| `Xi_raw/inst` | 每实例平均 `sum(ReLU(Ax-b))`，不带 margin、不减 tau |
| `Xi_m/inst` | 每实例平均 `sum(xi_no_tau)`，带 margin、不减 tau |

`Xi_m_t/inst` 低但 `Xi_raw/inst` 或离散 `AvgViol/inst` 高时，说明 tau/margin/归一化后的 loss 可行性和真实约束可行性存在差距，需要关注 `tau_min/tau/gamma/rho`。

### 8.7 `[Valid]` 验证主要指标

| 字段 | 含义 |
|---|---|
| `Loss` | 验证 ALM 总 loss |
| `MaxViol` | 验证最终 `xi` 最大违反量 |
| `MeanViol` | 验证最终 `xi` 平均违反量 |
| `XiSum/s` | 验证每实例 `sum(xi_no_tau)`，best 判定主指标 |
| `AvgObj/s` | 验证每实例离散目标，best 判定次指标 |

### 8.8 `[VPred]` 验证预测分布

同 `[Pred]`，但在验证集上计算。

| 字段 | 含义 |
|---|---|
| `Pred0` | 验证集 round 后为 0 的变量比例 |
| `Pred1` | 验证集 round 后为 1 的变量比例 |
| `Xi_mean/cons` | 验证集每约束平均 `xi_no_tau` |
| `Tau_mean` | 验证集 `tau_j` 均值 |

### 8.9 `[Disc]` 验证离散/极化指标

| 字段 | 含义 | 趋势解读 |
|---|---|---|
| `Feasibility` | 约束级可行率：满足约束条数 / 总约束条数 | 越接近 1 越好；注意不是实例级可行率 |
| `Objective` | batch 平均离散目标 `c^T round(x)` | 可行后越低越好 |
| `Polarization` | `x_hat < 0.05` 或 `x_hat > 0.95` 的变量比例 | 越高表示输出越接近 0/1 |
| `Uncertainty` | `mean(4*x_hat*(1-x_hat))` | 0 表示非常确定，1 表示接近 0.5；越低越极化 |

### 8.10 `[VDisc]` 验证集离散 per-instance 指标

| 字段 | 含义 |
|---|---|
| `AvgObj/inst` | 验证每实例平均离散目标 |
| `AvgViol/inst` | 验证每实例平均离散原始违反量之和 |
| `FeasInst` | 验证离散可行实例数 |
| `InfeasInst` | 验证离散不可行实例数 |

`_model_best_allfeas.pth` 只有在 `InfeasInst=0` 且目标更好时保存。

### 8.11 `[VCont]` 验证集连续 per-instance 指标

同 `[TCont]`，但在验证集上计算。

---

## 9. TensorBoard 指标

如果安装了 TensorBoard，训练会写入：

```text
{tensorboard_dir}/{problem_type}/{save_name}/
```

常用曲线：

| TensorBoard tag | 含义 |
|---|---|
| `Loss/Total` | 训练总 loss |
| `Loss/Valid_Total` | 验证总 loss |
| `Violation/Train_Max` / `Violation/Valid_Max` | 训练/验证最大最终违反量 |
| `Violation/Train_Mean` / `Violation/Valid_Mean` | 训练/验证平均最终违反量 |
| `Violation/Valid_XiSum_PerSample` | 验证 `XiSum/s`，best 判定主指标 |
| `Objective/Valid_PerSample` | 验证每实例离散目标 |
| `ALM/Gamma` / `ALM/Rho` / `ALM/Lambda_Global` | ALM 状态变化 |
| `Prediction/Pred0_Ratio` / `Prediction/Pred1_Ratio` | 训练预测 0/1 比例 |
| `Prediction/Valid_Pred0_Ratio` / `Prediction/Valid_Pred1_Ratio` | 验证预测 0/1 比例 |
| `Discrete/Feasibility_Rate` | 验证约束级可行率 |
| `State/Polarization_Rate` | 验证极化率 |
| `State/Mean_Uncertainty` | 验证不确定性 |
| `PerInst/Valid_Disc_Obj` | 验证每实例离散目标 |
| `PerInst/Valid_Disc_Viol` | 验证每实例离散违反量 |
| `PerInst/Valid_Feasible` / `PerInst/Valid_Infeasible` | 验证离散可行/不可行实例数 |

---

## 10. 测试脚本参数与模式

测试入口是 `test.py`。

### 10.1 常用命令

只评估模型 round 后的可行性和目标：

```bash
python test.py \
  --test_problem_type SC \
  --model_dir ./pretrain_models/SC/xxx_model_best.pth \
  --unsupervised_eval
```

推理、round 评估，并导出加了信赖域约束的新 `.mps`：

```bash
python test.py \
  --test_problem_type SC \
  --model_dir ./pretrain_models/SC/xxx_model_best.pth \
  --inference_only
```

默认模式：GNN 推理后固定变量/加信赖域约束，再调用 solver：

```bash
python test.py \
  --test_problem_type SC \
  --model_dir ./pretrain_models/SC/xxx_model_best.pth \
  --solver gurobi \
  --max_time 1000 \
  --threads 1
```

### 10.2 测试参数说明

| 参数 | 默认值 | 含义 | 注意 |
|---|---:|---|---|
| `--test_problem_type` | `SC` | 测试问题类型 | 需和模型训练任务匹配 |
| `--test_num` | `100` | 取前多少个测试实例 | 按文件名排序后截断 |
| `--model_dir` | `./pretrain_models` | 当前实际作为模型文件路径使用 | 建议传 `xxx_model_best.pth` 或 `xxx_model_best_allfeas.pth` |
| `--emb_size` | `64` | 模型嵌入维度 | 必须和训练时结构一致 |
| `--cons_nfeats` | `4` | 约束特征维度 | 必须和训练一致 |
| `--edge_nfeats` | `1` | 边特征维度 | 必须和训练一致 |
| `--var_nfeats` | `6` | 变量特征维度 | 必须和训练一致 |
| `--depth` | `2` | GNN 深度 | 必须和训练一致 |
| `--Intra_Constraint_Competitive` | `False` | 是否启用 ICC 结构 | 必须和训练一致，否则权重可能对不上 |
| `--solver` | `gurobi` | 求解器，`gurobi` 或 `scip` | 默认模式才用 |
| `--max_time` | `1000` | solver 时间上限，秒 | 默认模式才用 |
| `--threads` | `1` | solver 线程数 | 默认模式才用 |
| `--instance_dir` | `/home/lmh/autodl-tmp/data/l2o_milp_test` | 测试实例根目录 | 实际读取 `instance_dir/test_problem_type` |
| `--scores_dir` | `./scores` | 缓存 GNN scores 的目录 | 默认 solver 模式使用 |
| `--device` | `cuda:0` | 推理设备 | 需和环境匹配 |
| `--num_workers` | `1` | solver worker 数 | `>1` 时主线程推理、多个 worker 并行求解 |
| `--log_dir` | `./test_logs/` | 测试日志根目录 | solver 日志写到其下 |
| `--inference_only` | `False` | 只推理、评估 round、导出 modified mps，不调用 solver | 会打印 CE/Acc/可行性等 |
| `--unsupervised_eval` | `False` | 只推理、round、检查可行性，不导出、不调用 solver | 最适合快速评估 ALM 模型 |
| `--margin` | `0.9` | 置信度统计阈值：`abs(pred-0.5)*2 >= margin` | 当前主要用于统计 `conf`，不影响 `fix_pas()` 固定逻辑 |
| `--alpha` | `0.01` | 保留/命名参数 | 当前只进入 `save_name`，核心逻辑未使用 |
| `--tao` | `0.1` | 保留/命名参数 | 当前只进入 `save_name`，核心逻辑未使用 |
| `--k0` | `1000` | 按预测概率从低到高固定为 0 的二进制变量数 | 默认 solver/inference_only 模式使用 |
| `--k1` | `0` | 按预测概率从高到低固定为 1 的二进制变量数 | 默认 solver/inference_only 模式使用 |
| `--delta` | `200` | 信赖域半径，约束 `sum alpha <= delta` | 固定变量越多，delta 越关键 |
| `--topk` | `500` | 计算 top-K 置信变量准确率的 K | 需要有 solution 文件 |

### 10.3 `fix_pas()` 固定变量逻辑

`fix_pas(scores, task, args)` 当前只使用：

```text
k0, k1, delta
```

逻辑：

1. 按预测概率从高到低，取前 `k1` 个二进制变量，目标固定值设为 1；
2. 按预测概率从低到高，取前 `k0` 个二进制变量，目标固定值设为 0；
3. 对被固定变量添加辅助变量 `alpha_i >= |x_i - x*_i|`；
4. 添加信赖域约束：`sum_i alpha_i <= delta`。

因此：

- `k0/k1` 控制固定变量数量；
- `delta` 控制允许偏离固定建议的总半径；
- `margin` 当前不参与固定变量筛选，只用于统计多少变量达到置信阈值。

---

## 11. 测试日志与指标说明

### 11.1 `--unsupervised_eval` 输出

每个实例输出：

```text
[i/N] instance.mps  Obj=...  MaxViol=...  Violated=a/b  Feasible=YES/NO  Polar=...  Time=...s
```

| 字段 | 含义 |
|---|---|
| `Obj` | 原始目标函数在 round 解上的值；这里没有像训练那样把 maximize 乘 `-1` |
| `MaxViol` | round 解的最大约束违反量 |
| `Violated=a/b` | 违反约束数 / 总约束数 |
| `Feasible` | 是否没有任何违反约束 |
| `Polar` | 极化率：`pred<0.05` 或 `pred>0.95` 的比例 |
| `Time` | 单实例特征提取 + 推理 + 评估耗时 |

汇总输出：

| 字段 | 含义 |
|---|---|
| `Feasible instances` | 可行实例数 |
| `Infeasible instances` | 不可行实例数 |
| `Feasible — Avg Objective` | 可行实例平均目标 |
| `Infeasible — Avg Total Violation` | 不可行实例平均违反量总和 |
| `Infeasible — Avg Objective` | 不可行实例平均目标，仅参考 |
| `Max Violation mean/max` | 最大约束违反量的均值/最大值 |
| `Total time / Avg` | 总耗时/平均单实例耗时 |

### 11.2 `--inference_only` 输出

每个实例输出：

```text
[i/N] instance.mps  binary=...  fixed=...  conf=...  delta=...
  CE=...  MSE=...  Acc=...  Top500-Acc=...
  Obj=...  MaxViol=...  Violated=a/b  Feasible=YES/NO
  feat=...s  infer=...s  total=...s  -> xxx_modified.mps
```

| 字段 | 含义 |
|---|---|
| `binary` | 实例中的二进制变量数量 |
| `fixed` | 被 `fix_pas()` 建议固定的变量数量，即 `k0+k1` 截断后的数量 |
| `conf` | 满足 `abs(pred-0.5)*2 >= margin` 的变量数，只是统计 |
| `delta` | 信赖域半径 |
| `CE` | 和 solution 文件最优解对比的二分类交叉熵；没有 solution 时为 `N/A` |
| `MSE` | 和最优解对比的均方误差 |
| `Acc` | round 后和最优解逐变量比较的准确率 |
| `TopK-Acc` | 置信度最高的 K 个二进制变量的 round 准确率 |
| `Obj` | round 解在原始目标上的值 |
| `MaxViol` | round 解最大约束违反量 |
| `Violated=a/b` | 违反约束数 / 总约束数 |
| `Feasible` | round 解是否可行 |
| `feat` | 特征提取耗时 |
| `infer` | GNN 推理耗时 |
| `total` | 单实例总耗时 |
| `-> xxx_modified.mps` | 导出的加信赖域约束后的实例文件 |

`CE/MSE/Acc/TopK-Acc` 依赖：

```text
dataset/{test_problem_type}/solution/{instance_name}.sol
```

如果 solution 文件不存在，就不会统计这些监督指标。

### 11.3 默认 solver 模式输出与日志

默认模式不加 `--inference_only` 和 `--unsupervised_eval`，流程是：

1. 对每个实例推理得到二进制变量 scores；
2. scores 缓存到：

   ```text
   {scores_dir}/{test_problem_type}/Intra_Constraint_Competitive_.../scores_xxx.pkl
   ```

3. 根据 `k0/k1/delta` 添加固定变量与信赖域约束；
4. 调用 Gurobi 或 SCIP 求解；
5. solver 日志写到：

   ```text
   {log_dir}/{test_problem_type}/{save_name}/{save_name}_{instance}.log
   ```

同时 `test.log` 记录 worker 启动/错误等信息。

### 11.4 solver 日志汇总

用 `summarize_test_logs.py` 汇总默认 solver 模式产生的 `.log`：

```bash
python summarize_test_logs.py ./test_logs/SC/your_save_name --show-all --csv result.csv
```

汇总指标：

| 字段 | 含义 |
|---|---|
| `Status counts` | 不同求解状态数量，如 `OPTIMAL/TIME_LIMIT/INFEASIBLE/UNKNOWN` |
| `Objective` | solver 最终 primal objective |
| `BestBound` | solver best bound / dual bound |
| `Gap (%)` | MIP gap 百分比 |
| `Runtime (s)` | 求解耗时 |
| `Inferred objective sense` | 根据 objective 和 bound 推断是 minimize 还是 maximize |
| `Best/Worst instances by objective` | 按 objective 排序的最好/最差实例 |

对 `Objective`、`Gap (%)`、`Runtime (s)` 会输出：

```text
mean, std, min, q25, median, q75, max
```

如果传 `--csv`，会保存逐实例结果，字段包括：

```text
instance, instance_id, objective, best_bound, gap_pct, runtime_s, status, solver, source_file
```

---

## 12. 新人调参建议

### 12.1 先跑一个最小闭环

建议先用少量实例或较短 epoch 确认流程：

```bash
python train.py \
  --problem_type SC \
  --num_epochs 20 \
  --val_every 1 \
  --model_save_dir ./pretrain_models \
  --log_save_dir ./train_logs
```

确认有以下产物：

- `train_logs/SC/*_train.log`；
- `pretrain_models/SC/*_model_last.pth`；
- 如果验证有改善，会有 `*_model_best.pth`；
- 如果所有验证实例离散可行且目标更好，会有 `*_model_best_allfeas.pth`。

然后快速测试：

```bash
python test.py \
  --test_problem_type SC \
  --model_dir ./pretrain_models/SC/xxx_model_best.pth \
  --unsupervised_eval
```

### 12.2 训练时优先看哪些指标

优先级建议：

1. **验证离散可行性**：`[VDisc] InfeasInst`、`[Disc] Feasibility`、`AvgViol/inst`。
2. **best 判定主指标**：`[Valid] XiSum/s`。
3. **输出是否极化**：`[Disc] Polarization`、`Uncertainty`、`Entropy`、`Pred0/Pred1`。
4. **可行后的目标质量**：`[VDisc] AvgObj/inst`、`[Valid] AvgObj/s`。
5. **ALM 是否过激**：`rho/lambda/gamma` 是否爆涨。

### 12.3 常见现象与调参方向

| 现象 | 优先检查 | 可能调参方向 |
|---|---|---|
| `XiSum/s` 长期很高 | `[ALM] rho/lambda`、`MaxViol`、`MeanViol` | 增大 `rho_init` 或 `beta`；减小 `inner_steps` 让 ALM 更频繁更新；适当延长训练 |
| `MaxViol/MeanViol` 很低，但 `VDisc InfeasInst` 高 | `Xi_raw/inst`、`Xi_m_t/inst`、`AvgViol/inst` | loss 里的 tau/归一化可能过宽松；尝试降低 `tau_min` 或关闭/对比 `cons_normalize`；同时提高极化 |
| 输出长期接近 0.5 | `Entropy`、`Polarization`、`Uncertainty` | 增大 `entropy_weight`；增大 `delta_gamma` 或 `gamma_max`；训练更久 |
| `Pred0` 或 `Pred1` 接近 1，预测塌缩 | `Obj`、`Viol`、数据类别 | 降低学习率；降低过强的 `entropy_weight`；检查该任务真实解是否本来稀疏 |
| 训练 loss 抖动或发散 | `rho/lambda` 是否迅速变大 | 降低 `lr`；降低 `beta` 或 `rho_init`；保留 `grad_clip_norm` |
| 可行后目标变差 | `AvgObj/inst`、`rho/lambda` | 尝试 `es_xi_threshold2` 冻结 lambda，必要时同时冻结 rho/gamma；降低过强约束惩罚 |
| 验证比训练差很多 | train/valid 划分、实例数量、seed | 固定 `seed`；增加数据；检查是否单实例重复验证导致误判 |
| 测试加载后效果异常 | 模型路径、结构参数 | 确认传的是 `*_model_best.pth` 或 `*_model_best_allfeas.pth`；`depth/ICC/emb_size` 和训练一致 |

### 12.4 推荐的调参顺序

1. **先固定模型结构**：`emb_size=64, depth=2`，不要一开始改结构。
2. **先调可行性**：看 `XiSum/s`、`VDisc InfeasInst`、`AvgViol/inst`，优先调 `rho_init/beta/inner_steps/tau_min`。
3. **再调极化**：看 `Entropy/Polarization/Uncertainty`，调 `entropy_weight/gamma` 相关参数。
4. **最后调目标质量**：在验证可行后比较 `AvgObj/inst`，考虑 freeze ALM 或降低过强惩罚。
5. **测试阶段再调 `k0/k1/delta`**：这些不影响训练，只影响 solver 前的变量固定与信赖域。

### 12.5 测试阶段 `k0/k1/delta` 怎么理解

| 参数 | 增大后的效果 | 风险 |
|---|---|---|
| `k0` | 更多低概率变量被建议为 0 | 如果模型错判，会限制 solver 搜索 |
| `k1` | 更多高概率变量被建议为 1 | 对稀疏解任务容易引入错误固定 |
| `delta` | solver 可以更大幅度偏离固定建议 | 太大则 GNN 引导变弱；太小则可能把可行/最优解排除 |

一般做法：

1. 先用 `--unsupervised_eval` 看模型 round 解质量；
2. 如果预测较准，再增加 `k0/k1`；
3. 若 solver 经常 infeasible 或 gap 差，适当增大 `delta` 或减少固定数量；
4. 用 `summarize_test_logs.py` 对比不同配置的 objective/gap/runtime。

---

## 13. 当前代码里的几个容易踩坑点

1. **`test.py --model_dir` 实际是模型文件路径**，不是目录。
2. **测试推理不要传 `_model_last.pth`**；它是完整 checkpoint，适合 `train.py --resume_from`，不适合 `test.py`。
3. **`--emb_size` 当前不完全可自由调**；`gnn.py` 的 `BipartiteGraphConvolution` 内部写死了 `emb_size=64`。
4. **`--gnn_type` 当前未使用**；改它不会切换模型。
5. **`--alpha`、`--tao` 当前只进入测试 `save_name`**，核心求解逻辑没有用到。
6. **`--margin` 当前主要用于统计置信变量数**，不参与 `fix_pas()` 的变量固定筛选。
7. **训练目标方向和测试目标方向不完全一样**：训练中 maximize 会被转成 minimize；`test.py` 的 `compute_rounded_violations()` 直接按原始目标计算。
8. **训练日志会用 `w` 模式打开**；同一个 `save_name` 重新训练会覆盖旧日志。
9. **`tau_min` 会给每条约束 tau 设置下界**；这会直接影响进入 ALM loss 的 `xi`，但不影响 `XiSum/s` 的 `xi_no_tau` 统计。
10. **`compute_K()` 的 `K_max` 参数当前没有实际截断效果**。

---

## 14. 快速判读一段训练日志

看到一段日志时，可以按这个顺序读：

1. 看 `[Valid] XiSum/s`：best 主指标是否在下降。
2. 看 `[VDisc] InfeasInst`：round 后实例级可行性是否改善。
3. 看 `[Disc] Feasibility`：约束级可行率是否接近 1。
4. 看 `[Disc] Polarization` 和 `Uncertainty`：输出是否已经接近 0/1。
5. 看 `[ALM] rho/lambda/gamma`：ALM 惩罚是否过大或增长过快。
6. 在可行性稳定后，再比较 `[VDisc] AvgObj/inst`。

一句话总结：

```text
先让 InfeasInst / AvgViol 下降，再让 Polarization 上升，最后比较 AvgObj。
```
