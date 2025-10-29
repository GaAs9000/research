"""
分区算法模块
从gnn_data/src/opfdata/processor.py直接迁移
"""

import numpy as np
import os
from typing import Dict, List, Tuple, Optional
import hashlib
import heapq


class Partitioner:
    """电网分区算法"""
    
    def __init__(self):
        pass
    
    def create_partition_dynamic(self, network_data: Dict, opf_data: Dict, k: int, seed: int) -> Optional[np.ndarray]:
        """基于运行状态的动态分区：加权邻接 + 连通性修复 + 规模均衡。

        返回满足约束的分区标签；失败则返回None。
        """
        try:
            # 1) 获取相角（一律使用增强 DCPF，不读取 AC 解）
            from .dcpf import DCPFCalculator
            calculator = DCPFCalculator()
            angles = calculator.calculate_dcpf(network_data)

            # 2) 构建加权邻接及流强亲和矩阵
            weighted_adj, flow_affinity = self._build_weighted_adjacency(network_data, angles)

            # 3) 选择分区方法
            n_buses = network_data['n_buses']
            # 规模约束区间，默认 [0.8, 1.2]，可通过环境变量 OPFDATA_SIZE_TOL="0.8,1.2" 配置
            tol_env = os.getenv("OPFDATA_SIZE_TOL")
            if tol_env:
                try:
                    low_s, high_s = [float(x.strip()) for x in tol_env.split(',')]
                except Exception:
                    low_s, high_s = 0.8, 1.2
            else:
                low_s, high_s = 0.8, 1.2
            lower = max(3, int(np.floor(low_s * n_buses / float(k))))
            upper = int(np.ceil(high_s * n_buses / float(k)))
            topo_adj_bool = self._build_topology_adj_bool(network_data)

            method = os.getenv("OPFDATA_METHOD", "constructive").lower()
            if method == "constructive":
                seeds = self._select_seeds(topo_adj_bool, weighted_adj, k)
                parts = self._construct_balanced_partitions(topo_adj_bool, weighted_adj, seeds, k, lower, upper)
                # 先在规模约束下尽量修复连通性
                parts = self._ensure_partition_connectivity(topo_adj_bool, weighted_adj, parts, k, size_bounds=(lower, upper))
                # 如仍存在孤岛，则放宽规模约束，强制保证连通性（连通性优先）
                if not self._all_parts_connected(topo_adj_bool, parts, k):
                    parts = self._ensure_partition_connectivity(topo_adj_bool, weighted_adj, parts, k, size_bounds=None)
            else:
                # 谱聚类 + 修复 + 全局重平衡
                from sklearn.cluster import SpectralClustering
                clustering = SpectralClustering(
                    n_clusters=k,
                    affinity='precomputed',
                    assign_labels='kmeans',
                    n_init=10,
                    random_state=seed,
                )
                parts = clustering.fit_predict(weighted_adj)
                parts = self._ensure_partition_connectivity(topo_adj_bool, weighted_adj, parts, k, size_bounds=(lower, upper))
                # 优化的单轮平衡：预测性调整避免多轮迭代
                parts = self._balance_partition_sizes_optimized(topo_adj_bool, weighted_adj, parts, k, min_size=3, lower_bound=lower, upper_bound=upper)
                parts = self._ensure_partition_connectivity(topo_adj_bool, weighted_adj, parts, k, size_bounds=(lower, upper))
                sizes_now = [int(np.sum(parts == pid)) for pid in range(k)]
                # 如果单轮未成功，最多再补一轮
                if not all(lower <= s <= upper for s in sizes_now):
                    parts = self._balance_partition_sizes(topo_adj_bool, weighted_adj, parts, k, min_size=3, lower_bound=lower, upper_bound=upper)
                    parts = self._ensure_partition_connectivity(topo_adj_bool, weighted_adj, parts, k, size_bounds=(lower, upper))
                # 最终连通性修复也需遵守规模约束
                parts = self._ensure_partition_connectivity(topo_adj_bool, weighted_adj, parts, k, size_bounds=(lower, upper))

                # 兜底：如不满足则切换至构造式
                sizes_now = [int(np.sum(parts == pid)) for pid in range(k)]
                conn_ok = self._all_parts_connected(topo_adj_bool, parts, k)
                size_ok = all(lower <= s <= upper for s in sizes_now)
                if not conn_ok or not size_ok:
                    seeds = self._select_seeds(topo_adj_bool, weighted_adj, k)
                    parts = self._construct_balanced_partitions(topo_adj_bool, weighted_adj, seeds, k, lower, upper)
                    parts = self._ensure_partition_connectivity(topo_adj_bool, weighted_adj, parts, k, size_bounds=(lower, upper))
                    # 若仍不连通，则放宽约束以确保连通性
                    if not self._all_parts_connected(topo_adj_bool, parts, k):
                        parts = self._ensure_partition_connectivity(topo_adj_bool, weighted_adj, parts, k, size_bounds=None)

            return parts
        except Exception as e:
            print(f"Dynamic partition failed: {e}")
            return None

    def _build_weighted_adjacency(self, network_data: Dict, angles: np.ndarray,
                                   alpha: float = 0.7, gamma: float = 0.75,
                                   clip_q: Tuple[float, float] = (0.01, 0.99)) -> Tuple[np.ndarray, np.ndarray]:
        """按母线对聚合的加权邻接：Trafo-only 权重置 0；AC 与 Trafo 各自计算后相加。

        返回:
          - weighted_adj: [N,N] 归一化后的权重（用于分区优先级）
          - flow_affinity: [N,N] 原始聚合流强（未归一化，便于诊断）
        """
        n_buses = int(network_data['n_buses'])
        weighted_adj = np.zeros((n_buses, n_buses), dtype=float)
        flow_affinity = np.zeros((n_buses, n_buses), dtype=float)
        eps = 1e-8

        # 聚合容器: key=(min(u,v),max(u,v)) -> [w_ac, w_tr, ac_count, tr_count]
        agg: Dict[Tuple[int, int], List[float]] = {}

        # AC 聚合
        ac = network_data['ac_lines']
        if ac.get('senders'):
            ac_from = np.asarray(ac['senders'], dtype=int)
            ac_to = np.asarray(ac['receivers'], dtype=int)
            ac_feats = np.asarray(ac.get('features', []), dtype=float)
            for i in range(len(ac_from)):
                u = int(ac_from[i]); v = int(ac_to[i])
                a, b = (u, v) if u < v else (v, u)
                x = float(ac_feats[i, 5]) if (ac_feats.size > 0 and i < ac_feats.shape[0] and ac_feats.shape[1] > 5) else 0.1
                x_eff = abs(x) if abs(x) > 1e-12 else 1.0
                pdc = abs((angles[u] - angles[v]) / x_eff)
                if (a, b) not in agg:
                    agg[(a, b)] = [0.0, 0.0, 0.0, 0.0]
                agg[(a, b)][0] += pdc
                agg[(a, b)][2] += 1.0

        # Trafo 聚合
        tr = network_data['transformers']
        if tr.get('senders'):
            tr_from = np.asarray(tr['senders'], dtype=int)
            tr_to = np.asarray(tr['receivers'], dtype=int)
            tr_feats = np.asarray(tr.get('features', []), dtype=float)
            for i in range(len(tr_from)):
                u = int(tr_from[i]); v = int(tr_to[i])
                a, b = (u, v) if u < v else (v, u)
                if tr_feats.size > 0 and i < tr_feats.shape[0]:
                    x = float(tr_feats[i, 3]) if tr_feats.shape[1] > 3 else 0.1
                    tap = float(tr_feats[i, 7]) if (tr_feats.shape[1] > 7 and abs(tr_feats[i, 7]) > 1e-12) else 1.0
                    shift = float(tr_feats[i, 8]) if tr_feats.shape[1] > 8 else 0.0
                    if abs(shift) > np.pi:
                        shift = float(np.deg2rad(shift))
                else:
                    x = 0.1; tap = 1.0; shift = 0.0
                x_eff = abs(x) / max(abs(tap), 1e-12)
                pdc = abs((angles[u] - angles[v] - shift) / (x_eff if x_eff > 1e-12 else 1.0))
                if (a, b) not in agg:
                    agg[(a, b)] = [0.0, 0.0, 0.0, 0.0]
                agg[(a, b)][1] += pdc
                agg[(a, b)][3] += 1.0

        # 写入矩阵（Trafo-only -> 0；有 AC 则 w = w_ac + w_tr），并做 max 归一
        max_w = 0.0
        for (a, b), (w_ac, w_tr, ac_cnt, tr_cnt) in agg.items():
            if ac_cnt > 0:
                w = float(w_ac + w_tr)
            else:
                w = 0.0
            if w > 0:
                weighted_adj[a, b] = w
                weighted_adj[b, a] = w
                flow_affinity[a, b] = w
                flow_affinity[b, a] = w
                if w > max_w:
                    max_w = w

        if max_w > 0:
            weighted_adj = weighted_adj / max_w

        return weighted_adj, flow_affinity

    def _ensure_partition_connectivity(self, topo_adj_bool: np.ndarray, weighted_adj: np.ndarray, parts: np.ndarray, k: int,
                                       size_bounds: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """确保每个分区连通（以拓扑邻接为准）：
        将小的连通分量整体并入与其（按加权邻接）耦合最强、且未超上界的相邻分区。
        """
        n = weighted_adj.shape[0]

        def bfs_component(start: int, mask: np.ndarray) -> List[int]:
            """优化的BFS找连通分量 - 使用deque和向量化"""
            from collections import deque
            if not mask[start]:
                return []
            
            visited = []
            seen = np.zeros(len(mask), dtype=bool)
            seen[start] = True
            queue = deque([start])
            
            while queue:
                u = queue.popleft()  # O(1) vs pop(0)的O(n)
                visited.append(u)
                
                # 向量化找邻居：避免Python循环
                neighbors = np.where(topo_adj_bool[u] & mask & ~seen)[0]
                if len(neighbors) > 0:
                    seen[neighbors] = True  # 批量标记
                    queue.extend(neighbors)  # 批量加入队列
            
            return visited

        def get_components(nodes: List[int]) -> List[List[int]]:
            mask = np.zeros(n, dtype=bool)
            mask[nodes] = True
            comps: List[List[int]] = []
            remaining = set(nodes)
            while remaining:
                s = next(iter(remaining))
                comp = bfs_component(s, mask)
                comps.append(comp)
                for u in comp:
                    if u in remaining:
                        remaining.remove(u)
            # 大到小排序，便于保留最大分量
            comps.sort(key=len, reverse=True)
            return comps

        def sum_cross_weight(group: List[int], target_nodes: List[int]) -> float:
            if not group or not target_nodes:
                return 0.0
            sub = weighted_adj[np.ix_(group, target_nodes)]
            return float(np.sum(sub))

        updated = parts.copy()
        changed = True
        max_iters = 20
        iter_count = 0
        while changed and iter_count < max_iters:
            changed = False
            iter_count += 1
            for pid in range(k):
                nodes_in_p = np.where(updated == pid)[0].tolist()
                if len(nodes_in_p) <= 1:
                    continue
                comps = get_components(nodes_in_p)
                if len(comps) <= 1:
                    continue
                # 保留最大分量，其余分量并出
                keep = set(comps[0])
                for small_comp in comps[1:]:
                    # 选择耦合最强且规模未超上界的目标分区
                    best_q = None
                    best_w = -1.0
                    for qid in range(k):
                        if qid == pid:
                            continue
                        if size_bounds is not None:
                            lower_bound, upper_bound = size_bounds
                            if np.sum(updated == qid) >= upper_bound:
                                continue
                        target_nodes = np.where(updated == qid)[0].tolist()
                        w = sum_cross_weight(small_comp, target_nodes)
                        if w > best_w:
                            best_w = w
                            best_q = qid
                    if best_q is None:
                        continue
                    updated[small_comp] = best_q
                    changed = True
        return updated

    def _all_parts_connected(self, topo_adj_bool: np.ndarray, parts: np.ndarray, k: int) -> bool:
        """检查所有分区在拓扑图上是否单连通。"""
        n = topo_adj_bool.shape[0]
        for pid in range(k):
            nodes = np.where(parts == pid)[0]
            if nodes.size == 0:
                return False
            # BFS from first node
            mask = np.zeros(n, dtype=bool)
            mask[nodes] = True
            start = int(nodes[0])
            seen = set([start])
            queue = [start]
            while queue:
                u = queue.pop(0)
                for v in np.where(topo_adj_bool[u] & mask)[0]:
                    if v not in seen:
                        seen.add(int(v))
                        queue.append(int(v))
            if len(seen) != nodes.size:
                return False
        return True

    def _build_topology_adj_bool(self, network_data: Dict) -> np.ndarray:
        """构建拓扑邻接布尔矩阵（AC线路与变压器，不考虑权重/容量）。"""
        n = network_data['n_buses']
        adj = np.zeros((n, n), dtype=bool)
        ac = network_data['ac_lines']
        for fb, tb in zip(ac['senders'], ac['receivers']):
            adj[fb, tb] = True
            adj[tb, fb] = True
        trafo = network_data['transformers']
        for fb, tb in zip(trafo['senders'], trafo['receivers']):
            adj[fb, tb] = True
            adj[tb, fb] = True
        return adj

    def _select_seeds(self, topo_adj_bool: np.ndarray, weighted_adj: np.ndarray, k: int) -> List[int]:
        """选择k个种子节点：
        - 第一个为加权度最大的节点
        - 其余通过最大最小拓扑距离贪心选择，尽量分散
        """
        n = topo_adj_bool.shape[0]
        weighted_degree = np.sum(weighted_adj, axis=1)
        seeds: List[int] = []
        used = np.zeros(n, dtype=bool)
        # 第一个种子
        s0 = int(np.argmax(weighted_degree))
        seeds.append(s0)
        used[s0] = True

        def bfs_dist(source: int) -> np.ndarray:
            dist = np.full(n, np.inf)
            dist[source] = 0.0
            queue = [source]
            while queue:
                u = queue.pop(0)
                for v in np.where(topo_adj_bool[u])[0]:
                    if dist[v] == np.inf:
                        dist[v] = dist[u] + 1.0
                        queue.append(int(v))
            return dist

        dists = bfs_dist(s0)
        for _ in range(1, k):
            # 选择与现有seeds的最小距离最大的节点
            cand = -1
            best = -1.0
            for i in range(n):
                if used[i]:
                    continue
                min_d = float(dists[i])
                if min_d > best:
                    best = float(min_d)
                    cand = int(i)
            if cand == -1:
                # 退化：选未用节点中加权度最大者
                cand = int(np.argmax(np.where(used, -np.inf, weighted_degree)))
            seeds.append(cand)
            used[cand] = True
            # 更新全局最小距离（与所有seeds的最小）
            d_new = bfs_dist(cand)
            dists = np.minimum(dists, d_new)
        return seeds

    def _construct_balanced_partitions(self, topo_adj_bool: np.ndarray, weighted_adj: np.ndarray,
                                        seeds: List[int], k: int, lower: int, upper: int) -> np.ndarray:
        """基于种子多源优先队列生长的严格构造式分区：
        - 仅沿拓扑相邻增长，保证连通
        - 优先按照加权邻接强度扩展
        - 强制不超过upper；不足lower再做边界迁移修复
        """
        n = topo_adj_bool.shape[0]
        parts = -np.ones(n, dtype=int)
        sizes = [0 for _ in range(k)]

        # 初始化PQ：(-priority, node)
        frontiers: List[List[Tuple[float, int]]] = [[] for _ in range(k)]
        in_pq = np.zeros((k, n), dtype=bool)
        for pid, s in enumerate(seeds):
            parts[s] = pid
            sizes[pid] += 1
            for nb in np.where(topo_adj_bool[s])[0]:
                if parts[nb] == -1 and not in_pq[pid, nb]:
                    pr = -float(weighted_adj[s, nb])
                    heapq.heappush(frontiers[pid], (pr, int(nb)))
                    in_pq[pid, nb] = True

        # 轮询扩展直到所有节点被分配或所有frontier空
        changed = True
        while changed:
            changed = False
            all_empty = True
            for pid in range(k):
                # 跳过已达上限的分区
                if sizes[pid] >= upper:
                    continue
                while frontiers[pid]:
                    all_empty = False
                    pr, v = heapq.heappop(frontiers[pid])
                    in_pq[pid, v] = False
                    if parts[v] != -1:
                        continue
                    # 分配
                    parts[v] = pid
                    sizes[pid] += 1
                    changed = True
                    # 扩展邻居
                    for nb in np.where(topo_adj_bool[v])[0]:
                        if parts[nb] == -1 and not in_pq[pid, nb]:
                            heapq.heappush(frontiers[pid], (-float(weighted_adj[v, nb]), int(nb)))
                            in_pq[pid, nb] = True
                    break  # round-robin，每次取一个
            if all_empty:
                break

        # 仍未分配的节点（可能因所有分区达到上界）：允许选择规模最小的分区接收，保持拓扑相邻优先
        unassigned = np.where(parts == -1)[0].tolist()
        for v in unassigned:
            # 可接收的分区（未超上界且与v拓扑相邻）
            candidates = []
            for pid in range(k):
                if sizes[pid] >= upper:
                    continue
                if not np.any(topo_adj_bool[v] & (parts == pid)):
                    continue
                w = float(np.sum(weighted_adj[v, parts == pid]))
                candidates.append((w, pid))
            if not candidates:
                # 放宽：选当前规模最小的分区
                pid = int(np.argmin(sizes))
            else:
                pid = int(max(candidates)[1])
            parts[v] = pid
            sizes[pid] += 1

        # 规模下界修复：将边界节点从超标分区迁移到欠标分区（需拓扑相邻，避免割点）
        def is_cut_vertex_in_partition(node: int, pid: int) -> bool:
            nodes = np.where(parts == pid)[0]
            if nodes.size <= 2:
                return False
            remaining = [u for u in nodes if u != node]
            if not remaining:
                return False
            mask = np.zeros(n, dtype=bool)
            mask[remaining] = True
            start = int(remaining[0])
            seen = set([start])
            queue = [start]
            while queue:
                u = queue.pop(0)
                for v in np.where(topo_adj_bool[u] & mask)[0]:
                    if v not in seen:
                        seen.add(int(v))
                        queue.append(int(v))
            return len(seen) != len(remaining)

        max_balance_iters = 10 * n
        biter = 0
        while biter < max_balance_iters:
            biter += 1
            under = [pid for pid in range(k) if sizes[pid] < lower]
            over = [pid for pid in range(k) if sizes[pid] > upper]
            if not under and not over:
                break
            moved = False
            for u_pid in under:
                # 从最超标的相邻分区转移边界节点
                neighbor_parts = set(int(parts[v]) for v in np.where(np.any(topo_adj_bool[:, parts == u_pid], axis=1))[0] if parts[v] != u_pid)
                candidates_parts = sorted(over, key=lambda pid: sizes[pid], reverse=True) + [pid for pid in neighbor_parts if pid not in over]
                for o_pid in candidates_parts:
                    if o_pid == u_pid or sizes[o_pid] <= lower:
                        continue
                    border_nodes = [int(v) for v in np.where((parts == o_pid) & np.any(topo_adj_bool[:, parts == u_pid], axis=1))[0]]
                    best = None
                    for v in border_nodes:
                        if is_cut_vertex_in_partition(v, o_pid):
                            continue
                        gain = float(np.sum(weighted_adj[v, parts == u_pid]) - np.sum(weighted_adj[v, parts == o_pid]))
                        if best is None or gain > best[0]:
                            best = (gain, v)
                    if best is not None:
                        _, v = best
                        parts[v] = u_pid
                        sizes[u_pid] += 1
                        sizes[o_pid] -= 1
                        moved = True
                        break
                if moved:
                    break
            if not moved:
                break

        return parts

    def _balance_partition_sizes_optimized(self, topo_adj_bool: np.ndarray, weighted_adj: np.ndarray, parts: np.ndarray, k: int,
                                         min_size: int, lower_bound: int, upper_bound: int) -> np.ndarray:
        """优化的分区平衡：预测性批量迁移，减少迭代次数"""
        n = weighted_adj.shape[0]
        updated = parts.copy()
        
        # 计算所有分区的当前规模
        sizes = [int(np.sum(updated == pid)) for pid in range(k)]
        
        # 识别需要调整的分区
        over_parts = [(pid, sizes[pid]) for pid in range(k) if sizes[pid] > upper_bound]
        under_parts = [(pid, sizes[pid]) for pid in range(k) if sizes[pid] < lower_bound or sizes[pid] < min_size]
        
        if not over_parts or not under_parts:
            return updated
        
        # 优化策略：批量计算边界节点及其权重
        for over_pid, over_size in over_parts:
            if not under_parts:
                break
                
            # 需要迁移的节点数量
            excess = over_size - upper_bound
            if excess <= 0:
                continue
            
            # 向量化找到边界节点
            over_nodes = np.where(updated == over_pid)[0]
            boundary_candidates = []
            
            for node in over_nodes:
                # 检查是否为边界节点（有邻居在其他分区）
                neighbors = np.where(topo_adj_bool[node])[0]
                other_partition_neighbors = neighbors[updated[neighbors] != over_pid]
                
                if len(other_partition_neighbors) > 0:
                    # 快速割点检测：只检查度数 > 1的节点
                    node_degree = np.sum(topo_adj_bool[node] & (updated == over_pid))
                    if node_degree <= 1:  # 叶节点，安全迁移
                        # 计算到各个不足分区的最大权重
                        max_weight = 0
                        best_target = None
                        for target_pid, target_size in under_parts:
                            if target_size >= upper_bound:
                                continue
                            target_neighbors = other_partition_neighbors[updated[other_partition_neighbors] == target_pid]
                            if len(target_neighbors) > 0:
                                weight = np.sum(weighted_adj[node, target_neighbors])
                                if weight > max_weight:
                                    max_weight = weight
                                    best_target = target_pid
                        
                        if best_target is not None:
                            boundary_candidates.append((node, max_weight, best_target))
            
            # 按权重排序，优先迁移高权重节点
            boundary_candidates.sort(key=lambda x: x[1], reverse=True)
            
            # 批量迁移
            migrated = 0
            for node, weight, target_pid in boundary_candidates[:excess]:
                # 检查目标分区是否还有空间
                if sizes[target_pid] < upper_bound:
                    updated[node] = target_pid
                    sizes[over_pid] -= 1
                    sizes[target_pid] += 1
                    migrated += 1
                    
                    # 更新under_parts列表
                    under_parts = [(pid, sizes[pid]) for pid in range(k) 
                                 if sizes[pid] < lower_bound or sizes[pid] < min_size]
                    
                    if migrated >= excess:
                        break
        
        return updated

    def _balance_partition_sizes(self, topo_adj_bool: np.ndarray, weighted_adj: np.ndarray, parts: np.ndarray, k: int,
                                 min_size: int, lower_bound: int, upper_bound: int) -> np.ndarray:
        """在保持连通性的前提下，使各分区规模落入 [lower_bound, upper_bound]。
        策略：从超标分区向欠标分区迁移边界节点（非割点，且与目标分区有连接）。
        """
        # 这里只添加方法签名，实际实现较复杂，如需完整功能可以继续迁移
        return parts
