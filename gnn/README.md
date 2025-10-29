# GNN 训练与 PQ 路线（V1.1 对齐）

## 概览
- 仅提供“直接预测边界 P/Q（PQ 路线）”，不再保留非PQ路线。
- 完全对齐 V1.1 数据：字段、方向、单位（p.u.）与口径一致。

## 数据接口（V1.1）
- 节点 `x [N,20]`：基础4列 + 发电机6列 + 预计算特征10列，支持区域守恒损失 (Q先验已移除)。
- 边 `edge_attr [E,10]`：`[R, X_eff, S_max, is_tie, edge_type(0/1), shift, Pdc, Pdc_ratio, b_fr, b_to]`。
- 边界对象：`tie_buses [B]`、`tie_corridors [C,2] (u<v)`、`tie_edge_corridor [E]`（非联络边为 -1）。
- 标签：`y_bus_pq [B,2]`（耦合母线本端净注入），`y_corridor_pfqt [C,4]`（走廊端口聚合，[pf_u, pt_v, qf_u, qt_v]）。

## 训练配置（gnn/config.py）
- 仅 PQ 路线：`use_pq_route=True`（默认且唯一）
- 物理守护：四合一损失（端口监督 + 母线一致性 + 容量守护 + 区域守恒）。
- 相关参数：`pq_capacity_alpha`，`pq_lambda_S_init`。（注意：`pq_lambda_pair_init`已废弃，因为走廊有损耗）
- 环形汇总：`ring_k/ring_decay/ring_use_decayed`（默认 K=3，decay=0.5）。
- 先验残差：默认仅 P 先验，Q 先验关闭（`pq_use_p_prior=True, pq_use_q_prior=False, pq_prior_dropout=0.2`）。数据必须提供 `pq_prior[B,2]`，第二列可全 0。
- 大 batch 默认：`batch_size=8, accum_steps=2`，损失按“逐图均值→跨图均值”公平聚合（见下）。
- 加速默认：`use_compile=True, compile_mode='reduce-overhead'`、`pin_memory=True, persistent_workers=True, prefetch_factor=4`、AMP=bf16。

## 快速开始
1) 在 `gnn/config.py` 设置 `data_dir` 指向处理好的 V1.1 数据（含 `train/val/test`）。
2) 训练：`python gnn/train.py`
   - 若遇第三方算子编译兼容问题：`use_compile=False` 一键回退（其余不变）。
3) 评估：输出 MAE(P_bd)/MAE(Q_bd)。

## 物理守护细节（PQ 路线）
- 成对一致（端口小头）：已废弃 - 走廊有损耗，Pu+Pv≠0。现在只对投影端口 (Pu,Pv,Qu,Qv) 做容量守护。
- 容量安全域：按 `tie_edge_corridor` 将 `S_max` 聚合为走廊上界；罚 `max(0,(S-α Smax)/(α Smax))`，并自动除以 2 抵消双向边重复统计。
- 区域守恒：利用发电机特征计算内部注入，确保边界净注入+内部注入≈0。
- 公平聚合：
  - collate 会在 Batch 上挂 `bus_ptr/corr_ptr`（每图前缀和）。
  - 训练端在 `bpq_training_step` 中先按 ptr 做“逐图均值”，再跨图取均值；未提供 ptr 时退回全局均值（向后兼容）。

## N‑1 场景与混合训练
- 按应急改图（删边/屏蔽）后再前向；需保证 `tie_*` 与 `tie_edge_corridor` 与当前图一致。
- 可将多套已处理数据（基准/N-1）合并到同一 processed 目录（提供 merge_datasets.py），或在自定义 DataLoader 中传入多目录列表混合加载。

## 备注
- 已提供 GPU 向量化的环形汇总与“走廊端强一致性投影”工具；后续可按需接入到 PQ 头。
- 训练加速建议：AMP=bf16、`non_blocking=True` + `pin_memory=True`、`prefetch_factor` 适度增大；在 PyTorch ≥ 2.1 上启用 `torch.compile`（已默认开启）。
