"""
BC-GNN（V-θ 版本）：预测边界母线电压和联络线相角差，通过AC潮流方程重构功率。

要点
- V-θ 路线：预测电压V和相角差Δθ，通过物理方程保证一致性
- V head：输出边界母线电压 [B] p.u.，范围 [0.9, 1.1]
- θ head：输出走廊相角差 [C, 2] = [sin(Δθ), cos(Δθ)]，单位圆表示
- 重构层：通过AC潮流方程 (V, Δθ) → (P, Q)，损耗自动包含

输入（V1.1 对齐）
- x[N,20] = [P_load, Q_load, theta_dcpf, is_coupling_bus, P_gen, Q_gen, P_gen_min, P_gen_max, Q_gen_min, Q_gen_max, ...预计算特征]
- edge_attr[E,10] = [R, X_eff, S_max, is_tie, edge_type, shift, Pdc, Pdc_ratio, b_fr, b_to]
- tie_buses/tie_corridors/tie_edge_corridor 必须存在

输出（推理字典）
- V_pred [B]: 边界母线电压 (p.u.)
- sincos_pred [C,2]: 走廊相角差 [sin(Δθ), cos(Δθ)]
- corridor_pfqt [C,4]: 重构的功率 [Pf_u, Pt_v, Qf_u, Qt_v] (用于ACOPF)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool, JumpingKnowledge
from torch_geometric.utils import add_self_loops, degree, scatter
import torch_geometric.nn as pyg_nn
from collections import defaultdict, deque


def compute_min_hop_to_any_tie_bus(edge_index, tie_buses, K=3):
    """
    计算每个节点到最近边界母线的hop距离
    
    Args:
        edge_index: [2, E] 边索引
        tie_buses: [B] 边界母线ID列表
        K: 最大跳数
        
    Returns:
        dist_to_boundary: [N] 每个节点到最近边界母线的hop距离，值在{0..K}
    """
    if len(tie_buses) == 0:
        # 没有边界母线，所有节点距离设为K+1
        N = edge_index.max().item() + 1
        return torch.full((N,), K+1, dtype=torch.long)
    
    N = edge_index.max().item() + 1
    B = len(tie_buses)
    
    # 构建邻接表（无向图）
    adj = defaultdict(list)
    for i in range(edge_index.size(1)):
        u, v = edge_index[0, i].item(), edge_index[1, i].item()
        adj[u].append(v)
        adj[v].append(u)
    
    # 多源BFS：从所有边界母线同时开始
    dist = torch.full((N,), K+1, dtype=torch.long)  # 初始化为K+1
    queue = deque()
    visited = set()
    
    # 将所有边界母线加入队列，距离为0
    for bus_id in tie_buses:
        bus_id = bus_id.item() if hasattr(bus_id, 'item') else bus_id
        dist[bus_id] = 0
        queue.append((bus_id, 0))
        visited.add(bus_id)
    
    # BFS遍历
    while queue:
        node, hop = queue.popleft()
        if hop >= K:
            continue
            
        for neighbor in adj[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                new_hop = hop + 1
                dist[neighbor] = new_hop
                queue.append((neighbor, new_hop))
    
    # 截断到K
    dist = torch.clamp(dist, 0, K)
    return dist


class ResNetBlock(nn.Module):
    """ResNet残差块"""
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim)
        )
    
    def forward(self, x):
        return x + self.net(x)


class RingBiasGATv2Layer(nn.Module):
    """带边界注意力（Ring-Bias）的GATv2层"""
    def __init__(self, in_dim, out_dim, edge_dim, heads=4, dropout=0.1, K=3):
        super().__init__()
        self.gat_conv = pyg_nn.GATv2Conv(
            in_dim, out_dim, heads=heads, dropout=dropout,
            edge_dim=edge_dim, add_self_loops=False
        )
        self.norm = nn.LayerNorm(out_dim * heads)
        self.residual = nn.Linear(in_dim, out_dim * heads) if in_dim != out_dim * heads else nn.Identity()
        
        # 每个hop距离一个可学习偏置
        self.bias_per_hop = nn.Parameter(torch.zeros(K+1))  # b[0..K]
        self.K = K

    def forward(self, x, edge_index, edge_attr, dist_to_boundary=None):
        """
        Args:
            x: 节点特征 [N, in_dim]
            edge_index: 边索引 [2, E]
            edge_attr: 边特征 [E, edge_dim]
            dist_to_boundary: 每个节点到最近边界母线的hop距离 [N], 值在{0..K}
        """
        res = self.residual(x)
        
        if dist_to_boundary is not None:
            # 计算标准GATv2注意力分数
            # 这里我们需要手动实现带偏置的注意力机制
            # 由于GATv2Conv内部实现复杂，我们采用简化方案：在消息传递后应用hop偏置
            
            # 标准GATv2前向传播
            out = self.gat_conv(x, edge_index, edge_attr)
            
            # 应用hop偏置：对来自不同hop距离节点的消息进行偏置调整
            if dist_to_boundary is not None:
                # 获取发送端节点的hop距离
                sender_hop = dist_to_boundary[edge_index[1]]  # [E]
                # 将hop距离截断到[0, K]范围
                sender_hop = torch.clamp(sender_hop, 0, self.K).long()
                # 获取对应的偏置
                hop_bias = self.bias_per_hop[sender_hop]  # [E]
                
                # 对边特征应用偏置（简化实现）
                # 这里我们通过调整边特征来间接影响注意力
                edge_attr_biased = edge_attr + hop_bias.unsqueeze(-1) * 0.1
                # 重新计算带偏置的输出
                out = self.gat_conv(x, edge_index, edge_attr_biased)
        else:
            # 没有hop距离信息时，使用标准GATv2
            out = self.gat_conv(x, edge_index, edge_attr)
        
        out = self.norm(out)
        return out + res


class GATLayerWithResidual(nn.Module):
    """带残差连接和层归一化的GATv2层（保持向后兼容）"""
    def __init__(self, in_dim, out_dim, edge_dim, heads=4, dropout=0.1):
        super().__init__()
        self.gat_conv = pyg_nn.GATv2Conv(
            in_dim, out_dim, heads=heads, dropout=dropout,
            edge_dim=edge_dim, add_self_loops=False
        )
        self.norm = nn.LayerNorm(out_dim * heads)
        self.residual = nn.Linear(in_dim, out_dim * heads) if in_dim != out_dim * heads else nn.Identity()

    def forward(self, x, edge_index, edge_attr, dist_to_boundary=None):
        res = self.residual(x)
        out = self.gat_conv(x, edge_index, edge_attr)
        out = self.norm(out)
        return out + res


class BCRefinement(nn.Module):
    """边界-内容细化模块"""
    def __init__(self, hidden_dim, n_iterations=2):  # 减少到两轮
        super().__init__()
        self.n_iterations = n_iterations
        self.weight_mode = 'b_smooth'  # 'mean' or 'b_smooth' - 便于消融对比
        
        # 边更新网络
        self.edge_refine = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # 节点更新网络
        self.node_refine = nn.GRUCell(hidden_dim, hidden_dim)
        
        # 内部邻居聚合网络
        self.node_neigh_fc = nn.Linear(hidden_dim, hidden_dim)
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        # 初始化门偏置为负，默认更保守
        with torch.no_grad():
            self.gate_mlp[-1].bias.fill_(-2.0)
    
    def forward(self, h, e, edge_index, is_tie, edge_attr_raw):
        """
        h: 节点特征 [N, hidden]
        e: 边特征 [E, hidden]
        edge_index: 边连接 [2,E]
        is_tie: 联络线标记（来自 edge_attr[:,3]）[E]
        edge_attr_raw: 原始边特征（V1.1 列位）[E,10]
            [R, X_eff, S_max, is_tie, edge_type, shift, Pdc, Pdc_ratio, b_fr, b_to]
        """
        # 识别联络线和耦合母线
        tie_mask = is_tie.bool()
        if not tie_mask.any():
            return h, e
        
        tie_idx = tie_mask.nonzero(as_tuple=True)[0]
        tie_edges = edge_index[:, tie_idx]
        coupling_buses = torch.unique(tie_edges.flatten())
        
        # 优化：只在第一次迭代时克隆，避免重复克隆
        h_work = h.clone()
        e_work = e.clone()
        
        # 迭代细化
        for _ in range(self.n_iterations):
            # 边更新：联络线吸收两端信息
            edge_input = torch.cat([
                e_work[tie_idx],
                h_work[tie_edges[0]],
                h_work[tie_edges[1]]
            ], dim=-1)
            e_refined = self.edge_refine(edge_input)
            e_work[tie_idx] = e_work[tie_idx] + e_refined  # 残差连接，直接in-place
            
            # 节点更新：耦合母线聚合"联络线边信息" + "内部邻居摘要"（门控融合）
            
            row, col = edge_index.long()  # 确保long类型
            internal_mask = ~is_tie.bool()
            
            # 收集“内部边，且指向耦合母线”的消息索引
            is_cpl = torch.zeros(h.size(0), dtype=torch.bool, device=h.device)
            is_cpl[coupling_buses] = True
            to_cpl = internal_mask & is_cpl[col]
            
            if to_cpl.any():
                # 邻居节点向量 → 线性投影
                m_node_msg = self.node_neigh_fc(h_work[row[to_cpl]])  # [M,d]
                # 温和版电纳权重 1/sqrt(X²+ε) - 数值稳定
                X = edge_attr_raw[to_cpl, 1].abs()  # X_eff 绝对值
                # 权重模式选择（便于消融对比）
                if self.weight_mode == 'mean':
                    w = torch.ones_like(X, dtype=h_work.dtype)
                else:  # 'b_smooth'
                    w = 1.0 / torch.sqrt(X * X + 1e-6)  # 温和版电纳权重
                # 归一化权重
                w_norm = scatter(w, col[to_cpl], dim=0, dim_size=h_work.size(0), reduce='sum')
                w_norm = w / (w_norm[col[to_cpl]] + 1e-6)
                # 加权聚合
                m_node_all = scatter(m_node_msg * w_norm.unsqueeze(-1), col[to_cpl],
                                     dim=0, dim_size=h_work.size(0), reduce='sum')
                m_node = m_node_all[coupling_buses]  # [K,d]
            else:
                m_node = torch.zeros(len(coupling_buses), h_work.size(1), device=h_work.device)
            
            # 联络线边路的聚合
            end_flat = torch.cat([tie_edges[0], tie_edges[1]], dim=0)  # [2T]
            msg_edge = torch.cat([e_work[tie_idx], e_work[tie_idx]], dim=0)  # [2T,d]
            # 计算度数并平均
            ones = torch.ones_like(end_flat, dtype=h_work.dtype)
            # 修复：使用clamp_min而非clamp_min_（避免inplace操作）
            deg = scatter(ones, end_flat, dim=0, dim_size=h_work.size(0), reduce='sum').clamp_min(1.0)
            m_edge_all = scatter(msg_edge, end_flat, dim=0, dim_size=h_work.size(0), reduce='sum')
            m_edge_all = m_edge_all / deg.unsqueeze(-1)
            m_edge = m_edge_all[coupling_buses]  # [K,d]
            
            # 门控融合 + GRU 更新
            gate = torch.sigmoid(self.gate_mlp(torch.cat([m_edge, m_node, h_work[coupling_buses]], dim=-1)))
            m = m_edge + gate * m_node
            h_cpl_new = self.node_refine(m, h_work[coupling_buses])
            # 保持dtype一致，避免AMP下的类型不匹配
            if h_cpl_new.dtype != h_work.dtype:
                h_cpl_new = h_cpl_new.to(h_work.dtype)
            h_work[coupling_buses] = h_cpl_new
        
        return h_work, e_work


class BCGNN(nn.Module):
    """BC-GNN主模型"""
    def __init__(self, node_features=20, edge_features=8, hidden_dim=64,
                 use_global=False, voltage_stats=None, n_mpnn_layers=4, jk_mode='cat',
                 use_voltage_prior=True,
                 ring_k: int = 0, ring_decay = None, ring_use_decayed: bool = True,
                 use_ring_bias: bool = True, ring_K: int = 3):
        super().__init__()
        self.use_global = use_global
        self.use_voltage_prior = use_voltage_prior
        # 环形汇总与先验设置
        self.ring_k = int(ring_k or 0)
        self.ring_decay = ring_decay if (ring_decay is not None and ring_k) else None
        self.ring_use_decayed = bool(ring_use_decayed)
        
        # 自适应电压映射参数
        if voltage_stats is not None:
            self.v_min = voltage_stats['v_min']
            self.v_max = voltage_stats['v_max']
        else:
            # 默认范围
            self.v_min = 0.9
            self.v_max = 1.1
        
        # 节点特征编码（V1.1 基础为 4D，可附加先验/环形汇总等附加通道）
        self.node_encoder = nn.Linear(node_features, hidden_dim)

        # 分类型边编码（V1.1+ edge_attr 列位：[R, X_eff, S_max, is_tie, edge_type, shift, Pdc, Pdc_ratio, b_fr, b_to]）
        # 注：b_fr和b_to用于AC潮流重构，不参与边编码
        self.ac_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),  # AC: [R, X_eff, S_max]
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        self.trafo_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim // 2),  # 变压器: [R, X_eff, S_max, shift]
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        self.edge_type_emb = nn.Embedding(2, hidden_dim // 4)  # edge_type: 0=AC, 1=Trafo
        self.edge_proj = nn.Linear(hidden_dim // 2 + hidden_dim // 4, hidden_dim)
        
        # 全局条件接口
        if self.use_global:
            self.global_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), 
                nn.ReLU()
            )
        
        # 4层GAT + JK融合（使用RingBiasGATv2Layer）
        self.n_gat_layers = n_mpnn_layers
        self.use_ring_bias = use_ring_bias  # 启用边界注意力
        self.ring_K = ring_K  # hop距离截断
        
        # Hop-Bias嵌入（轻量实现）
        self.hop_K = int(ring_K) if ring_K is not None else 0
        if self.hop_K > 0:
            self.hop_bias = nn.Embedding(self.hop_K + 1, hidden_dim)  # Add-Bias
            nn.init.zeros_(self.hop_bias.weight)  # 从0开始训练更稳
        
        if self.use_ring_bias:
            self.gat_layers = nn.ModuleList([
                RingBiasGATv2Layer(hidden_dim, hidden_dim // 4, hidden_dim, heads=4, K=self.ring_K)
                for _ in range(self.n_gat_layers)
            ])
        else:
            self.gat_layers = nn.ModuleList([
                GATLayerWithResidual(hidden_dim, hidden_dim // 4, hidden_dim, heads=4)
                for _ in range(self.n_gat_layers)
            ])

        # JK融合
        self.jk_mode = jk_mode
        if self.jk_mode == 'cat':
            self.jk_proj = nn.Linear(hidden_dim * self.n_gat_layers, hidden_dim)

        # BC细化（2轮）
        self.bc_refine = BCRefinement(hidden_dim, n_iterations=2)

        self.hidden_dim = hidden_dim  # 保存以便在forward中使用

        # V-θ 预测头
        # V head: 边界母线电压预测 [B] → [0.9, 1.1] p.u.
        self.V_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )

        # θ head: 走廊相角差预测 [C] → [sin(Δθ), cos(Δθ)]
        self.theta_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 3),  # 输入: [h_sym, h_antisym, e_corr]
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            ResNetBlock(self.hidden_dim),
            nn.Linear(self.hidden_dim, 2)  # 输出 [sin(Δθ), cos(Δθ)]
        )
    
    def encode_edges(self, edge_attr):
        """分类型边编码（V1.1 列位：见上）。"""
        edge_type = edge_attr[:, 4].long()  # edge_type: 0=AC, 1=Trafo
        # 分别处理 AC 和变压器特征
        ac_feat = self.ac_encoder(edge_attr[:, [0, 1, 2]])        # [R, X_eff, S_max]
        trafo_feat = self.trafo_encoder(edge_attr[:, [0, 1, 2, 5]])  # [R, X_eff, S_max, shift]
        
        # 根据类型选择特征
        edge_feat = torch.where(edge_type.unsqueeze(1) == 0, ac_feat, trafo_feat)
        
        # 添加类型嵌入
        type_emb = self.edge_type_emb(edge_type)
        
        # 最终投影
        return self.edge_proj(torch.cat([edge_feat, type_emb], dim=-1))
    
    def gat_forward_with_jk(self, x, edge_index, edge_attr, dist_to_boundary=None):
        """4层GAT + JK融合（支持边界注意力）"""
        layer_outputs = []
        h = x

        for gat_layer in self.gat_layers:
            # 应用hop-bias（轻量实现）
            if (dist_to_boundary is not None) and (self.hop_K > 0):
                # 修复：使用clamp而非clamp_（避免inplace操作破坏梯度）
                d = dist_to_boundary.clamp(0, self.hop_K).long()
                h = h + self.hop_bias(d)
            
            if self.use_ring_bias and dist_to_boundary is not None:
                h = gat_layer(h, edge_index, edge_attr, dist_to_boundary)
            else:
                h = gat_layer(h, edge_index, edge_attr)
            layer_outputs.append(h)
        
        if self.jk_mode == 'cat':
            x_cat = torch.cat(layer_outputs, dim=-1)  # [N, 4*hidden_dim]
            x = self.jk_proj(x_cat)  # [N, hidden_dim]
        elif self.jk_mode == 'max':
            x = torch.max(torch.stack(layer_outputs, dim=0), dim=0)[0]
        else:  # 'last'
            x = layer_outputs[-1]
        
        return x
    
    def power_flow_reconstruction(self, V_pred, sincos_pred, data):
        """
        AC潮流方程重构：(V, Δθ) → (P, Q)（π模型：包含充电电纳）

        对于走廊 (u,v)，聚合所有并联线路的导纳参数后：
        P_u = V_u² G - V_u V_v [G cos(Δθ) + B sin(Δθ)]
        Q_u = -V_u² (B + b_fr) - V_u V_v [G sin(Δθ) - B cos(Δθ)]
        P_v = V_v² G - V_v V_u [G cos(Δθ) - B sin(Δθ)]
        Q_v = -V_v² (B + b_to) - V_v V_u [-G sin(Δθ) - B cos(Δθ)]

        Args:
            V_pred: [B] 边界母线电压预测 (p.u.)
            sincos_pred: [C, 2] = [sin(Δθ), cos(Δθ)]
            data: PyG Batch，包含 tie_corridors, tie_buses, edge_attr, tie_edge_corridor

        Returns:
            corridor_pfqt: [C, 4] = [Pf_u, Pt_v, Qf_u, Qt_v] (p.u.)
        """
        device = V_pred.device
        tie_corridors = data.tie_corridors.to(device)  # [C, 2] (u, v)
        C = tie_corridors.size(0)

        if C == 0:
            return torch.zeros(0, 4, device=device)

        # 1. 从 edge_attr 提取所有联络线的 R, X, b_fr, b_to
        tie_edges_mask = data.tie_edge_corridor >= 0
        if not tie_edges_mask.any():
            return torch.zeros(C, 4, device=device)

        R = data.edge_attr[tie_edges_mask, 0]  # [E_tie]
        X_eff = data.edge_attr[tie_edges_mask, 1]  # [E_tie]
        b_fr = data.edge_attr[tie_edges_mask, 8]  # [E_tie] 充电电纳 from
        b_to = data.edge_attr[tie_edges_mask, 9]  # [E_tie] 充电电纳 to
        corr_id = data.tie_edge_corridor[tie_edges_mask]  # [E_tie]

        # 2. 计算各线路的 G, B（串联导纳）
        Z2 = R**2 + X_eff**2 + 1e-12  # 避免除零
        G = R / Z2  # [E_tie]
        B = -X_eff / Z2  # [E_tie]

        # 3. 按走廊聚合 G, B, b_fr, b_to（并联导纳相加）
        G_corr = scatter(G, corr_id, dim=0, dim_size=C, reduce='sum')  # [C]
        B_corr = scatter(B, corr_id, dim=0, dim_size=C, reduce='sum')  # [C]
        b_fr_corr = scatter(b_fr, corr_id, dim=0, dim_size=C, reduce='sum')  # [C]
        b_to_corr = scatter(b_to, corr_id, dim=0, dim_size=C, reduce='sum')  # [C]

        # 4. 获取走廊两端电压
        # 需要建立 tie_buses 索引到 V_pred 的映射
        bus_to_v_idx = torch.zeros(data.x.size(0), dtype=torch.long, device=device)
        bus_to_v_idx[data.tie_buses] = torch.arange(len(data.tie_buses), device=device)

        u_idx = bus_to_v_idx[tie_corridors[:, 0]]
        v_idx = bus_to_v_idx[tie_corridors[:, 1]]

        V_u = V_pred[u_idx]  # [C]
        V_v = V_pred[v_idx]  # [C]

        # 5. sin/cos
        sin_dth = sincos_pred[:, 0]  # [C]
        cos_dth = sincos_pred[:, 1]  # [C]

        # 6. AC潮流方程（π模型：包含充电电纳）
        # u端发出
        Pu = V_u**2 * G_corr - V_u * V_v * (G_corr * cos_dth + B_corr * sin_dth)
        Qu = -V_u**2 * (B_corr + b_fr_corr) - V_u * V_v * (G_corr * sin_dth - B_corr * cos_dth)

        # v端接收（注意符号）
        Pv = V_v**2 * G_corr - V_v * V_u * (G_corr * cos_dth - B_corr * sin_dth)
        Qv = -V_v**2 * (B_corr + b_to_corr) - V_v * V_u * (-G_corr * sin_dth - B_corr * cos_dth)

        # 7. 组装为 [Pf_u, Pt_v, Qf_u, Qt_v]
        corridor_pfqt = torch.stack([Pu, Pv, Qu, Qv], dim=1)  # [C, 4]

        return corridor_pfqt

    def power_flow_reconstruction_per_edge(self, V_pred, sincos_pred, data, tie_edge_indices):
        """
        AC潮流方程重构（按线版本）：为每条联络线独立计算功率

        Args:
            V_pred: [B] 边界母线电压预测 (p.u.)
            sincos_pred: [E_tie, 2] 每条线的 [sin(Δθ), cos(Δθ)]
            data: PyG Batch
            tie_edge_indices: [E_tie] 联络线在edge_index中的索引

        Returns:
            edge_pq: [E_tie, 4] = [Pf, Pt, Qf, Qt] (p.u.)
        """
        device = V_pred.device
        E_tie = len(tie_edge_indices)

        if E_tie == 0:
            return torch.zeros(0, 4, device=device)

        # 1. 提取线路参数 R, X, b_fr, b_to
        R = data.edge_attr[tie_edge_indices, 0]  # [E_tie]
        X_eff = data.edge_attr[tie_edge_indices, 1]  # [E_tie]
        b_fr = data.edge_attr[tie_edge_indices, 8]  # [E_tie] 充电电纳 from
        b_to = data.edge_attr[tie_edge_indices, 9]  # [E_tie] 充电电纳 to

        # 2. 计算导纳 G, B
        Z2 = R**2 + X_eff**2 + 1e-12
        G = R / Z2  # [E_tie] 串联电导
        B = -X_eff / Z2  # [E_tie] 串联电纳

        # 3. 获取两端节点
        u_nodes = data.edge_index[0, tie_edge_indices]
        v_nodes = data.edge_index[1, tie_edge_indices]

        # 4. 映射到 V_pred 的索引
        bus_to_vidx = torch.zeros(data.x.size(0), dtype=torch.long, device=device)
        bus_to_vidx[data.tie_buses] = torch.arange(len(data.tie_buses), device=device)

        u_vidx = bus_to_vidx[u_nodes]
        v_vidx = bus_to_vidx[v_nodes]

        V_u = V_pred[u_vidx]  # [E_tie]
        V_v = V_pred[v_vidx]  # [E_tie]

        # 5. sin/cos
        sin_dth = sincos_pred[:, 0]  # [E_tie]
        cos_dth = sincos_pred[:, 1]  # [E_tie]

        # 6. AC潮流方程（π模型：包含充电电纳）
        # P不受充电电纳影响
        Pf = V_u**2 * G - V_u * V_v * (G * cos_dth + B * sin_dth)
        Pt = V_v**2 * G - V_v * V_u * (G * cos_dth - B * sin_dth)

        # Q包含充电电纳：Qf受b_fr影响，Qt受b_to影响
        Qf = -V_u**2 * (B + b_fr) - V_u * V_v * (G * sin_dth - B * cos_dth)
        Qt = -V_v**2 * (B + b_to) - V_v * V_u * (-G * sin_dth - B * cos_dth)

        # 7. 组装 [Pf, Pt, Qf, Qt]
        # ⚠️ 修复方向定义不一致：
        # - AC潮流公式计算的Pf/Pt与标签中的Pf/Pt定义顺序相反
        # - 本公式Pf对应标签Pt，本公式Pt对应标签Pf（Q同理）
        # - 已验证：交换后误差为0，损耗(Pf+Pt)一致
        edge_pq = torch.stack([Pt, Pf, Qt, Qf], dim=1)  # [E_tie, 4] = [Pf_label, Pt_label, Qf_label, Qt_label]

        return edge_pq

    def forward(self, data, apply_capacity_projection=False, projection_alpha=0.95):
        """
        输入: PyG Data 对象（V1.1）
        - x: [N, 20]，基础4列 + 发电机6列 + 预计算特征10列
        - edge_index: [2, E]
        - edge_attr: [E, 10] = [R, X_eff, S_max, is_tie, edge_type, shift, Pdc, Pdc_ratio, b_fr, b_to]
        - tie_buses/tie_corridors/tie_edge_corridor 等边界对象

        输出: 预测字典（V-θ 路线）
        - V_pred [B]: 边界母线电压 (p.u.)
        - sincos_pred [C,2]: 走廊相角差 [sin(Δθ), cos(Δθ)]
        - corridor_pfqt [C,4]: 重构的功率 [Pf_u, Pt_v, Qf_u, Qt_v]
        """
        # 1. 特征编码
        # 使用20D节点特征（基础4列+发电机6列+预计算特征10列，Q先验已移除）
        h = self.node_encoder(data.x)
        
        # 分类型边编码（数值稳定化在编码器内部处理）
        e = self.encode_edges(data.edge_attr)
        
        # 2. 计算hop距离（用于边界注意力）
        dist_to_boundary = None
        if hasattr(data, 'dist_to_tie'):
            # 优先使用预计算的hop距离
            dist_to_boundary = data.dist_to_tie
        elif (self.use_ring_bias or self.hop_K > 0) and hasattr(data, 'tie_buses') and data.tie_buses is not None:
            # 在线计算hop距离
            dist_to_boundary = compute_min_hop_to_any_tie_bus(
                data.edge_index, data.tie_buses, K=max(self.ring_K, self.hop_K)
            )
        
        # 3. 4层GAT + JK融合（支持边界注意力）
        h = self.gat_forward_with_jk(h, data.edge_index, e, dist_to_boundary)
        
        # 全局条件接口
        if self.use_global and hasattr(data, 'batch'):
            import torch_geometric.nn as pyg_nn
            g = pyg_nn.global_mean_pool(h, data.batch)  # [B,d]
            g = self.global_mlp(g)
            g_tiled = g[data.batch]  # [N,d]
        else:
            g_tiled = None
        
        # 3. 动态识别边界（优先用 data.tie_corridors 和 data.tie_buses）
        device = h.device
        if hasattr(data, 'tie_corridors') and data.tie_corridors is not None and data.tie_corridors.numel() > 0:
            # data.tie_corridors: [T,2] 已经是无向去重后固定方向（u<v）
            tie_edges = data.tie_corridors.to(device).t().contiguous()  # [2,T]
        else:
            # 回退：从 edge_attr 的 is_tie 里抽、再做无向唯一化
            is_tie_edge = data.edge_attr[:, 3] > 0.5
            if not is_tie_edge.any():
                # 没有联络线的情况（PQ-only：返回空走廊集）
                return {
                    'coupling_buses': h.new_zeros(0, dtype=torch.long),
                    'tie_edges': h.new_zeros(2, 0, dtype=torch.long)
                }
            u, v = data.edge_index[0, is_tie_edge], data.edge_index[1, is_tie_edge]
            u, v = torch.minimum(u, v), torch.maximum(u, v)
            tie_uv = torch.stack([u, v], dim=1)
            tie_uv = torch.unique(tie_uv, dim=0)     # [T,2]
            tie_edges = tie_uv.t().contiguous()      # [2,T]
        
        # 使用data.tie_buses保持顺序对齐（重要！）
        if hasattr(data, 'tie_buses') and data.tie_buses is not None:
            coupling_buses = data.tie_buses.to(device).long()
        else:
            coupling_buses = torch.unique(tie_edges.flatten())
        
        if tie_edges.size(1) == 0:
            # 无走廊，返回空集
            return {
                'coupling_buses': h.new_zeros(0, dtype=torch.long),
                'tie_edges': h.new_zeros(2, 0, dtype=torch.long)
            }
        
        # 4. BC细化
        is_tie_edge = data.edge_attr[:, 3]  # 联络线标记
        h_refined, e_refined = self.bc_refine(h, e, data.edge_index, is_tie_edge, data.edge_attr)
        
        # 5. 走廊级边特征聚合 e_corr
        T = tie_edges.size(1)  # 走廊数
        if hasattr(data, 'tie_edge_corridor') and T > 0:
            corr_id = data.tie_edge_corridor.to(device)  # [E], -1=非联络线
            valid = corr_id >= 0
            if valid.any():
                e_corr = scatter(e_refined[valid], corr_id[valid],
                               dim=0, dim_size=T, reduce='mean')  # [T, hidden_dim]
            else:
                e_corr = torch.zeros(T, e_refined.size(1), device=device)
        else:
            e_corr = torch.zeros(T, self.hidden_dim, device=device)  # 用零补齐，保证维度一致
        
        # 6. V-θ 预测（按线预测版本）
        result = {
            'coupling_buses': coupling_buses,
            'tie_edges': tie_edges
        }

        # 6.1 电压预测：边界母线 → [v_min, v_max] p.u.
        h_tie_bus = h_refined[coupling_buses]  # [B, hidden_dim]
        V_logits = self.V_head(h_tie_bus).squeeze(-1)  # [B]
        v_range = self.v_max - self.v_min
        V_pred = torch.sigmoid(V_logits) * v_range + self.v_min
        result['V_pred'] = V_pred

        # 6.2 相角差预测：按线预测 → [sin(Δθ), cos(Δθ)]
        # 优先使用 data.tie_edge_indices（按线预测）
        if hasattr(data, 'tie_edge_indices') and data.tie_edge_indices is not None:
            # 按线预测模式
            tie_edge_indices = data.tie_edge_indices.to(device)
            E_tie = len(tie_edge_indices)

            if E_tie > 0:
                # 提取联络线的两端节点
                u_nodes = data.edge_index[0, tie_edge_indices]
                v_nodes = data.edge_index[1, tie_edge_indices]

                # 提取节点和边特征
                h_u = h_refined[u_nodes]  # [E_tie, hidden_dim]
                h_v = h_refined[v_nodes]  # [E_tie, hidden_dim]
                e_tie = e_refined[tie_edge_indices]  # [E_tie, hidden_dim]

                # 构造输入：[h_u, h_v, e_tie]
                edge_in = torch.cat([h_u, h_v, e_tie], dim=-1)  # [E_tie, 3*hidden_dim]

                # 预测 sin/cos
                sincos_raw = self.theta_head(edge_in)  # [E_tie, 2]
                sincos_pred = F.normalize(sincos_raw, p=2, dim=1)  # L2归一化到单位圆
                result['sincos_pred'] = sincos_pred
                result['tie_edge_indices'] = tie_edge_indices

                # 6.3 AC潮流方程重构：按线重构功率
                edge_pq = self.power_flow_reconstruction_per_edge(
                    V_pred, sincos_pred, data, tie_edge_indices
                )
                result['edge_pq'] = edge_pq  # [E_tie, 4] = [Pf, Pt, Qf, Qt]

                # 聚合到走廊级输出（保持与旧接口兼容）
                if hasattr(data, 'tie_edge_corridor') and data.tie_edge_corridor is not None:
                    corr_ids = data.tie_edge_corridor[tie_edge_indices]
                    valid = corr_ids >= 0
                    if valid.any() and hasattr(data, 'tie_corridors') and data.tie_corridors is not None:
                        C = data.tie_corridors.size(0)
                        sincos_corr = scatter(
                            sincos_pred[valid],
                            corr_ids[valid],
                            dim=0,
                            dim_size=C,
                            reduce='mean'
                        )
                        sincos_corr = F.normalize(sincos_corr, p=2, dim=1)
                        result['sincos_corridor'] = sincos_corr
                        corridor_pfqt = self.power_flow_reconstruction(V_pred, sincos_corr, data)
                        result['corridor_pfqt'] = corridor_pfqt
            else:
                # 无联络线
                result['sincos_pred'] = torch.zeros(0, 2, device=device)
                result['edge_pq'] = torch.zeros(0, 4, device=device)
                result['tie_edge_indices'] = torch.zeros(0, dtype=torch.long, device=device)

        else:
            # 回退：走廊预测模式（向后兼容）
            h_u = h_refined[tie_edges[0]]
            h_v = h_refined[tie_edges[1]]
            h_sym = h_u + h_v
            h_antisym = h_u - h_v
            corr_in = torch.cat([h_sym, h_antisym, e_corr], dim=-1)

            sincos_raw = self.theta_head(corr_in)
            sincos_pred = F.normalize(sincos_raw, p=2, dim=1)
            result['sincos_pred'] = sincos_pred

            # 走廊级重构
            corridor_pfqt = self.power_flow_reconstruction(V_pred, sincos_pred, data)
            result['corridor_pfqt'] = corridor_pfqt

        return result


if __name__ == "__main__":
    # 无内联测试。请使用真实处理后的样本进行验证，或运行训练脚本里的评测。
    pass
