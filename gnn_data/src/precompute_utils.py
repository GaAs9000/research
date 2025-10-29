"""
预计算工具函数
用于在数据处理阶段预计算走廊容量、环形汇总等，减少训练时的计算开销
"""

import numpy as np
import torch
import os
from typing import Dict, List, Tuple, Optional, Any
import hashlib
from collections import defaultdict, deque


def precompute_corridor_capacity(
    edge_attr: np.ndarray,
    tie_edge_corridor: np.ndarray,
    num_corridors: int
) -> np.ndarray:
    """
    预计算走廊容量 Smax_corr[C]
    
    Args:
        edge_attr: [E, 8] 边属性，其中 [:, 2] 是 S_max
        tie_edge_corridor: [E] 每条边对应的走廊ID（-1表示非联络边）
        num_corridors: 走廊总数
        
    Returns:
        Smax_corr: [C] 每个走廊的容量上限
    """
    Smax_edge = edge_attr[:, 2].astype(np.float32)
    corridor_ids = tie_edge_corridor.astype(np.int64)
    
    # 初始化走廊容量
    Smax_corr = np.zeros(num_corridors, dtype=np.float32)
    
    # 按走廊ID聚合边容量
    valid_mask = corridor_ids >= 0
    valid_ids = corridor_ids[valid_mask]
    valid_smax = Smax_edge[valid_mask]
    
    for i, smax in zip(valid_ids, valid_smax):
        Smax_corr[i] += smax
    
    # 除以2，抵消有向边重复计算
    Smax_corr = 0.5 * Smax_corr
    
    return Smax_corr


def precompute_ring_sums(
    edge_index: np.ndarray,
    node_features: np.ndarray,
    tie_buses: np.ndarray,
    K: int = 3,
    decay: float = 0.5,
    mode: str = 'decayed'
) -> Dict[str, np.ndarray]:
    """
    预计算环形汇总
    
    Args:
        edge_index: [2, E] 边索引
        node_features: [N, 6] 节点特征，其中 [:, 0:2] 是 (P_load, Q_load)
        tie_buses: [B] 边界母线ID列表
        K: 环形汇总的跳数
        decay: 衰减因子
        mode: 'decayed' 或 'rings'
        
    Returns:
        Dict包含预计算结果和元数据
    """
    N = node_features.shape[0]
    B = len(tie_buses)
    
    # 构建邻接表（无向图）
    adj_list = defaultdict(list)
    for i in range(edge_index.shape[1]):
        u, v = edge_index[0, i], edge_index[1, i]
        adj_list[u].append(v)
        adj_list[v].append(u)
    
    # 提取P_load和Q_load
    P_load = node_features[:, 0].astype(np.float32)
    Q_load = node_features[:, 1].astype(np.float32)
    
    if mode == 'decayed':
        # Decayed模式：计算加权衰减和
        ring_decayed = np.zeros((B, 2), dtype=np.float32)
        
        for i, bus_id in enumerate(tie_buses):
            # BFS遍历K跳
            visited = set()
            current_ring = {bus_id}
            decayed_P = P_load[bus_id]  # 0跳：自己
            decayed_Q = Q_load[bus_id]
            
            for k in range(1, K + 1):
                next_ring = set()
                for node in current_ring:
                    if node in visited:
                        continue
                    visited.add(node)
                    for neighbor in adj_list[node]:
                        if neighbor not in visited:
                            next_ring.add(neighbor)
                
                # 计算当前环的贡献
                ring_P = sum(P_load[node] for node in next_ring)
                ring_Q = sum(Q_load[node] for node in next_ring)
                
                # 加权累加
                weight = decay ** k
                decayed_P += weight * ring_P
                decayed_Q += weight * ring_Q
                
                current_ring = next_ring
                if not current_ring:
                    break
            
            ring_decayed[i, 0] = decayed_P
            ring_decayed[i, 1] = decayed_Q
        
        return {
            'ring_decayed': ring_decayed,
            'ring_meta': {'K': K, 'decay': decay, 'mode': 'decayed'}
        }
    
    elif mode == 'rings':
        # Rings模式：保留每环的值
        ringP = np.zeros((B, K + 1), dtype=np.float32)
        ringQ = np.zeros((B, K + 1), dtype=np.float32)
        
        for i, bus_id in enumerate(tie_buses):
            # BFS遍历K跳
            visited = set()
            current_ring = {bus_id}
            
            # 0跳：自己
            ringP[i, 0] = P_load[bus_id]
            ringQ[i, 0] = Q_load[bus_id]
            
            for k in range(1, K + 1):
                next_ring = set()
                for node in current_ring:
                    if node in visited:
                        continue
                    visited.add(node)
                    for neighbor in adj_list[node]:
                        if neighbor not in visited:
                            next_ring.add(neighbor)
                
                # 记录当前环的值
                ring_P = sum(P_load[node] for node in next_ring)
                ring_Q = sum(Q_load[node] for node in next_ring)
                ringP[i, k] = ring_P
                ringQ[i, k] = ring_Q
                
                current_ring = next_ring
                if not current_ring:
                    break
        
        return {
            'ringP': ringP,
            'ringQ': ringQ,
            'ring_meta': {'K': K, 'decay': decay, 'mode': 'rings'}
        }
    
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def generate_stable_sample_id(
    case_id: str,
    k: int,
    tie_buses: np.ndarray,
    tie_corridors: np.ndarray,
    node_pq: np.ndarray,
    pq_prior: Optional[np.ndarray] = None
) -> str:
    """
    生成稳定的样本ID，用于去重和互斥校验
    
    Args:
        case_id: 案例ID
        k: 分区数
        tie_buses: 边界母线
        tie_corridors: 走廊定义
        node_pq: 节点P/Q负荷
        pq_prior: 可选的PQ先验
        
    Returns:
        样本ID的SHA1哈希值
    """
    # 构造唯一标识符
    components = [
        case_id.encode('utf-8'),
        str(k).encode('utf-8'),
        np.asarray(tie_buses).tobytes(),
        np.asarray(tie_corridors).tobytes(),
        np.asarray(node_pq).tobytes(),
    ]
    
    if pq_prior is not None:
        components.append(np.asarray(pq_prior).tobytes())
    
    # 计算SHA1哈希
    hasher = hashlib.sha1()
    for component in components:
        hasher.update(component)
    
    return hasher.hexdigest()


def build_khop_summaries(
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    tie_buses: np.ndarray,
    K: int = 3,
    decay: float = 0.5,
    mode: str = 'split'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    构建K跳邻域"无功能力摘要"特征 - 节点汇总式（node semantics）
    
    按照A.2伪代码实现：
    1. 先把每条边的指标（B_edge = 1/|X_eff|、S_edge = S_max）各分0.5到两端节点
    2. 以每个边界母线为源，做BFS得到dist[v] ∈ {0..K}
    3. 环r的值 = Σ_{v: dist[v]=r} B_node[v]（同理S_node）
    
    Args:
        edge_index: [2, E] 边索引
        edge_attr: [E, 8] 边属性，其中 [:, 1] 是 X_eff，[:, 2] 是 S_max
        tie_buses: [B] 边界母线ID列表
        K: 跳数
        decay: 衰减因子
        mode: 'split'（推荐）或 'decayed'
        
    Returns:
        (feat_B, feat_S): 电纳富集和容量富集特征 [B, K+1] 或 [B, 1]
    """
    from collections import defaultdict, deque
    
    N = max(edge_index.max() + 1, max(tie_buses) + 1) if len(tie_buses) > 0 else edge_index.max() + 1
    B = len(tie_buses)
    
    if B == 0:
        return np.zeros((0, K+1), dtype=np.float32), np.zeros((0, K+1), dtype=np.float32)
    
    # 1) 边 → 节点（各分0.5）
    u, v = edge_index[0].astype(np.int64), edge_index[1].astype(np.int64)
    Xeff = np.abs(edge_attr[:, 1]).astype(np.float32)
    Smax = np.maximum(edge_attr[:, 2], 0.0).astype(np.float32)
    B_edge = 1.0 / np.maximum(Xeff, 1e-9)  # 电纳富集：1/|X_eff|
    
    # 初始化节点特征
    B_node = np.zeros(N, dtype=np.float32)
    S_node = np.zeros(N, dtype=np.float32)
    
    # 各分0.5到两端节点
    for i in range(len(u)):
        B_node[u[i]] += 0.5 * B_edge[i]
        B_node[v[i]] += 0.5 * B_edge[i]
        S_node[u[i]] += 0.5 * Smax[i]
        S_node[v[i]] += 0.5 * Smax[i]
    
    # 2) 建邻接（无向）
    adj = defaultdict(list)
    for i in range(len(u)):
        adj[u[i]].append(v[i])
        adj[v[i]].append(u[i])
    
    def bfs_hop_dist(adj, src, K):
        """BFS计算到源点的hop距离，返回dict: node -> hop (0..K)"""
        dist = {}
        queue = deque([(src, 0)])
        visited = {src}
        
        while queue:
            node, hop = queue.popleft()
            if hop > K:
                continue
            dist[node] = hop
            
            for neighbor in adj[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, hop + 1))
        
        return dist
    
    # 3) 对每个tie_bus做BFS，统计分环
    if mode == 'split':
        B_feat = np.zeros((B, K+1), dtype=np.float32)
        S_feat = np.zeros((B, K+1), dtype=np.float32)
        
        for i, src in enumerate(tie_buses):
            dist = bfs_hop_dist(adj, int(src), K)
            rings_B = [0.0] * (K+1)
            rings_S = [0.0] * (K+1)
            
            for node, d in dist.items():
                if d <= K:  # 仅遍历0..K层
                    rings_B[d] += float(B_node[node])
                    rings_S[d] += float(S_node[node])
            
            B_feat[i, :] = np.array(rings_B)
            S_feat[i, :] = np.array(rings_S)
        
        return B_feat, S_feat
    
    elif mode == 'decayed':
        B_feat = np.zeros((B, 1), dtype=np.float32)
        S_feat = np.zeros((B, 1), dtype=np.float32)
        
        for i, src in enumerate(tie_buses):
            dist = bfs_hop_dist(adj, int(src), K)
            rings_B = [0.0] * (K+1)
            rings_S = [0.0] * (K+1)
            
            for node, d in dist.items():
                if d <= K:  # 仅遍历0..K层
                    rings_B[d] += float(B_node[node])
                    rings_S[d] += float(S_node[node])
            
            # 按decay加权
            B_feat[i, 0] = sum((decay**r) * rings_B[r] for r in range(K+1))
            S_feat[i, 0] = sum((decay**r) * rings_S[r] for r in range(K+1))
        
        return B_feat, S_feat
    
    else:
        raise ValueError(f"Unsupported mode: {mode}")


# Q先验函数已移除 - 当前实现效果不佳


def add_precomputed_fields(
    data: Any,  # PyG Data对象
    edge_attr: np.ndarray,
    tie_edge_corridor: np.ndarray,
    num_corridors: int,
    edge_index: np.ndarray,
    node_features: np.ndarray,
    tie_buses: np.ndarray,
    case_id: str,
    k: int,
    ring_K: int = 3,
    ring_decay: float = 0.5,
    ring_mode: str = 'decayed'
) -> Any:
    """
    为PyG Data对象添加预计算字段
    
    Args:
        data: PyG Data对象
        其他参数用于预计算
        
    Returns:
        添加了预计算字段的Data对象
    """
    device = data.x.device
    
    # 1. 预计算走廊容量
    Smax_corr = precompute_corridor_capacity(edge_attr, tie_edge_corridor, num_corridors)
    data.Smax_corr = torch.tensor(Smax_corr, dtype=torch.float32).to(device)
    
    # 2. 预计算环形汇总，并直接拼接到 x（按节点维度对齐，非边界母线补零）
    ring_results = precompute_ring_sums(
        edge_index, node_features, tie_buses, 
        K=ring_K, decay=ring_decay, mode=ring_mode
    )
    N = int(node_features.shape[0])
    if ring_mode == 'decayed':
        # decayed: [B,2] -> 映射到节点矩阵 [N,2]
        ring_decayed = torch.tensor(ring_results['ring_decayed'], dtype=torch.float32, device=device)
        data.ring_decayed = ring_decayed
        extra = torch.zeros((N, 2), dtype=data.x.dtype, device=device)
        if len(tie_buses) > 0:
            idx = torch.as_tensor(tie_buses, dtype=torch.long, device=device)
            extra[idx] = ring_decayed
        # 拼接到 x
        data.x = torch.cat([data.x, extra], dim=1)
    else:
        # rings: ringP/ringQ [B,K+1] -> [B, 2*(K+1)]，再按节点映射
        ringP = torch.tensor(ring_results['ringP'], dtype=torch.float32, device=device)
        ringQ = torch.tensor(ring_results['ringQ'], dtype=torch.float32, device=device)
        data.ringP = ringP
        data.ringQ = ringQ
        B = ringP.size(0)
        if B > 0:
            flat = torch.cat([ringP, ringQ], dim=1)  # [B, 2*(K+1)]
            extra = torch.zeros((N, flat.size(1)), dtype=data.x.dtype, device=device)
            idx = torch.as_tensor(tie_buses, dtype=torch.long, device=device)
            extra[idx] = flat
            data.x = torch.cat([data.x, extra], dim=1)
        else:
            # 无边界母线，拼接零占位（与 decayed 模式保持一致性，避免维度不确定）
            data.x = torch.cat([data.x, torch.zeros((N, 2*(ring_K+1)), dtype=data.x.dtype, device=device)], dim=1)
    
    data.ring_meta = ring_results['ring_meta']
    
    # 3. 新增：邻域"无功能力摘要"特征（Q先验已移除）
    # 读取环境变量配置
    cap_K = int(os.getenv("OPFDATA_CAP_K", "3"))
    cap_decay = float(os.getenv("OPFDATA_CAP_DECAY", "0.5"))
    add_qref = os.getenv("OPFDATA_ADD_QREF", "1") == "1"
    ring_mode_cap = os.getenv("OPFDATA_RING_MODE", "split")  # 推荐使用split模式
    
    if add_qref and len(tie_buses) > 0:
        # 构建邻域摘要（使用split模式获得更高分辨率）
        feat_B, feat_S = build_khop_summaries(
            edge_index, edge_attr, tie_buses, 
            K=cap_K, decay=cap_decay, mode=ring_mode_cap
        )
        
        # 生成邻域摘要特征（Q先验已移除）
        if ring_mode_cap == 'split':
            # 对于split模式，使用decayed版本作为Q_ref
            B_decayed = np.sum(feat_B * (cap_decay ** np.arange(cap_K+1)), axis=1, keepdims=True)
            S_decayed = np.sum(feat_S * (cap_decay ** np.arange(cap_K+1)), axis=1, keepdims=True)
        else:
            B_decayed = feat_B
            S_decayed = feat_S
        
        # 把B/S分环也拼进x（仅 tie_buses 行非零）
        B = len(tie_buses)
        
        # Q先验已移除 - 当前实现效果不佳，让网络从基础特征学习
        # Q_ref_bus = scale_B * B_decayed + scale_S * S_decayed
        Q_ref_bus = np.zeros((B, 1), dtype=np.float32)  # 占位符，保持接口兼容
        if ring_mode_cap == 'split':
            # split模式：拼接完整的B/S分环特征
            x_extra_rings = torch.zeros((N, 2*(cap_K+1)), dtype=data.x.dtype, device=device)
            idx = torch.as_tensor(tie_buses, dtype=torch.long, device=device)
            x_extra_rings[idx, 0:(cap_K+1)] = torch.tensor(feat_B, dtype=torch.float32, device=device)
            x_extra_rings[idx, (cap_K+1):2*(cap_K+1)] = torch.tensor(feat_S, dtype=torch.float32, device=device)
        else:
            # decayed模式：拼接decayed版本
            x_extra_rings = torch.zeros((N, 2), dtype=data.x.dtype, device=device)
            idx = torch.as_tensor(tie_buses, dtype=torch.long, device=device)
            x_extra_rings[idx, 0] = torch.tensor(B_decayed.flatten(), dtype=torch.float32, device=device)
            x_extra_rings[idx, 1] = torch.tensor(S_decayed.flatten(), dtype=torch.float32, device=device)
        
        # Q先验已移除，不再拼接到x中
        # 拼接到 x：只拼接B/S分环特征
        # 注意：data.x现在应该是10列（V1.2），而不是7列（V1.1）
        data.x = torch.cat([data.x, x_extra_rings], dim=1)
        
        # 保存原始特征用于调试
        data.feat_B = torch.tensor(feat_B, dtype=torch.float32, device=device)
        data.feat_S = torch.tensor(feat_S, dtype=torch.float32, device=device)
        # Q_ref_bus已移除
    
    # 3. 生成稳定样本ID
    node_pq = node_features[:, 0:2]
    pq_prior = data.pq_prior.cpu().numpy() if hasattr(data, 'pq_prior') else None
    
    sample_id = generate_stable_sample_id(
        case_id, k, tie_buses, data.tie_corridors.cpu().numpy(), 
        node_pq, pq_prior
    )
    data.sample_id = sample_id
    
    return data
