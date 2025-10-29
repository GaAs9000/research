BC-GNN 数据生成（V1.1：边界 P/Q 预测）
=====================================

本模块仅负责将 OPFData 原始 JSON 转换为用于“边界 P/Q 预测”的训练数据。
不包含训练/评估/测试与示例/测试脚本，避免引入额外依赖与困惑。

功能概览
--------
- 读取 OPFData JSON（以 JSON 字段为准，所有功率统一为 p.u.）。
- 增强 DCPF 获取母线相角 θ（考虑变压器 tap/shift；参考母线 θ=0）。
- 单一路径图分区（构造式生长）：带权 Dijkstra 种子 + 多源优先队列生长 + 连通性修复 + 规模均衡。
- 生成 V1.1 数据样本（PyG Data）：节点/边特征 + 边界对象 + 端口级与母线级边界 P/Q 标签。

数据规范（PyG Data）
-------------------
- 节点特征 `x [N,20]`：
  - 基础4列：`[P_load, Q_load, theta_dcpf, is_coupling_bus]`（均为 p.u.；`theta_dcpf` 适度裁剪）
  - 发电机6列：`[P_gen, Q_gen, P_gen_min, P_gen_max, Q_gen_min, Q_gen_max]`（按母线聚合，支持区域守恒损失）
  - 预计算特征10列：`[B_decayed, S_decayed, feat_B_0, feat_B_1, feat_B_2, feat_B_3, feat_S_0, feat_S_1, feat_S_2, feat_S_3]`（Q先验已移除）
- 边 `edge_index [2,E]`：物理边双向展开
- 边特征 `edge_attr [E,8]`：
  - `[R, X_eff, S_max, is_tie, edge_type, shift, Pdc, Pdc_ratio]`
  - AC 线路：`edge_type=0, shift=0, X_eff=|x|`
  - 变压器：`edge_type=1, X_eff=|x|/|tap|, shift` 来自 JSON（若 |shift|>π 兜底按度→弧度）
  - `Pdc` 由 DCPF 计算（反向边取相反数），`Pdc_ratio=|Pdc|/S_max`（裁剪至 [0,2]）
- 分区/边界对象：`partition [N]`，`tie_corridors [C,2] (u<v)`，`tie_lines [L,3] (u,v,cid)`，`tie_buses [B]`，
  `tie_edge_corridor [E]`（每条有向边关联的走廊 id，非联络线为 -1；用于 S_max 走廊分组聚合）。
- 标签：
  - `y_corridor_pfqt [C,4]`：每个跨区走廊 (u<v) 的端口级 `(pf_u,pt_v,qf_u,qt_v)`，按并联回线聚合；若回线记录方向为 `(v→u)`，则交换使用 `(pt,qt)` 与 `(pf,qf)`；单位 p.u.
  - `y_bus_pq [B,2]`：由端口聚合得到的母线本端净注入：对走廊 `(u→v)`，在母线侧记 `u:(-Pu,-Qu)`、`v:(+Pv,+Qv)`；单位 p.u.
- 先验：
  - `pq_prior [B,2]`：默认仅提供 P 先验（第二列 Q_prior=0）。P 先验由 DCPF 的 `Pdc` 沿联络线方向聚合到母线本端净注入口径得到，单位 p.u.；
    训练时采用"残差式 + dropout"使用。
- 预计算特征：
  - `ring_decayed [B,2]`：环特征（仅在边界母线非零），支持邻域无功能力代理
  - `feat_B [B,4]`：导纳富集特征（K=3环）
  - `feat_S [B,4]`：容量富集特征（K=3环）
  - ~~`Q_ref_bus [B,1]`：Q弱先验（已移除）~~

分区算法（唯一实现）
--------------------
- 相角来源：使用增强 DCPF 计算（完整考虑变压器 tap/shift，参考母线 θ=0）。不读取或依赖 OPFData 的 AC 解。
- 种子：带权 Dijkstra 最远优先（边长 1/(ε+A)，A 为基于流量/容量与电气亲和复合的权重）。
- 生长：多源优先队列沿拓扑扩张，严格连通与规模上界；未分配节点按相邻偏好回填。
- 修复/均衡：连通性修复（规模内优先）+ 非割点边界节点迁移做规模均衡。

DCPF 说明
---------
- Bθ = P + s：
  - AC 线：`Bij = -1/x`，对角为相邻边之和。
  - 变压器：等效电抗 `x_eff = |x|/|tap|`；相移 `shift` 以等效注入 s 加入右端（以 JSON 为主，若 |shift|>π 视为度再转弧度）。
- 支路 DC 有功：
  - AC：`Pdc_ij = (θ_i-θ_j)/x`
  - 变压器：`Pdc_ij = (θ_i-θ_j-shift)/(x/|tap|)`

使用方法
--------
- 配置：
  - 在 `gnn_data/src/main.py` 的 `DATASETS` 中设置 `raw_dir/processed_dir/num_samples` 等。
  - 分区参数 `PARTITION_CONFIG`：默认 `k_values=[3,4,5,6]`，`size_tolerance="0.9,1.1"`。
  - 切分/分块 `PROCESS_CONFIG`：默认 `train/val/test = 0.90/0.05/0.05`，`chunk_size=512`。
- 执行：
  - `python gnn_data/src/main.py process --datasets ieee500 --overwrite`
  - 输出：`processed_dir/{train,val,test}/chunk_*.pt`

约定与校验
----------
- 单位：所有功率（含标签）均为 p.u.（除以 `baseMVA`，来源于 JSON）。
- 方向：走廊方向固定为 `(u<v)`；若物理回线方向为 `(v→u)`，聚合时使用 `pt/qt`。
- 一致性：
  - `is_tie ⇔ partition[u]!=partition[v] ⇔ (u,v)∈tie_corridors`
  - `端口→母线映射` 与 `y_bus_pq` 一致（数值误差 ≤ 1e-6）
  - `S_max ≥ 0`，`0 ≤ Pdc_ratio ≤ 2`

依赖
----
- Python 3.8+
- PyTorch、PyTorch Geometric（仅用于数据结构与保存）
- NumPy、tqdm

备注
----
- 本模块不包含训练/评估代码与示例/测试脚本。
- 分区算法仅提供构造式生长一条路径，避免多实现引起歧义。
- 本数据与训练侧（gnn）严格对齐：以端口为主监督（`y_end_pq`），母线由端口聚合得到（`y_bus_pq`）；`pq_prior` 作为 P 先验（Q_prior=0）。容量守护以端口差分与 `Smax_corr` 评估。
