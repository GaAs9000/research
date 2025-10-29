import json
import numpy as np
try:
    import torch  # 可选依赖：仅在需要GPU DCPF和构建PyG Data对象时使用
except ImportError:
    torch = None
try:
    from torch_geometric.data import Data  # 可选依赖：仅在构建PyG Data对象时使用
except ImportError:
    Data = None
from typing import List, Dict, Tuple, Optional, Any
# pandapower 未在本数据生成路径中使用

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver import OPFUtils
from precompute_utils import add_precomputed_fields
import hashlib
import heapq


class OPFDataProcessor:
    """
    核心处理器，负责将 OPFData JSON 文件转换为 PyTorch Geometric 的 Data 对象。
    """
    
    def __init__(self):
        """初始化处理器，实例化分区器和工具类。"""
        self.utils = OPFUtils()
        # 轻量日志等级：ERROR > WARN > INFO > DEBUG
        # 控制台默认 ERROR，避免逐条噪声；文件日志可单独设置
        self._log_level = os.getenv("OPFDATA_LOG_LEVEL", "ERROR").upper()
        self._file_log_level = os.getenv("OPFDATA_FILE_LOG_LEVEL", "INFO").upper()
        self._log_file_path: Optional[str] = None
        # 运行期统计（按 JSON 文件级别聚合）
        self._stats = {
            'attempts': 0,            # 分区尝试次数（= k*seeds）
            'repairs': 0,             # 触发修复次数
            'repair_failures': 0,     # 修复后仍断连次数
            'size_violations': 0,     # 最终越界分区计数（under+over）
        }
        # 不再使用固定归一化器（V1.1 去工程化归一化）
        # 可以通过环境变量 OPFDATA_K_VALUES (例如 "2,3,4") 来覆盖默认的分区数 k 的列表
        env_k_values = os.getenv("OPFDATA_K_VALUES")
        if env_k_values:
            try:
                self.k_values = [int(x.strip()) for x in env_k_values.split(',') if x.strip()]
            except (ValueError, TypeError):
                self.k_values = [2, 3, 4, 5]
        else:
            self.k_values = [2, 3, 4, 5]
        
        # 也可以通过环境变量 OPFDATA_K 来指定一个固定的 k 值
        self.k_fixed = int(os.getenv("OPFDATA_K", "3"))
        # 通过环境变量 OPFDATA_MULTI_K 控制是否启用多 k 值处理模式 (默认开启)
        self.multi_k = os.getenv("OPFDATA_MULTI_K", "1") == "1"
        # 尺寸均衡开关（默认关闭，严格优先连通性修复）
        # 设 OPFDATA_BALANCE_SIZES=1 可启用叶节点均衡
        self.balance_sizes = os.getenv("OPFDATA_BALANCE_SIZES", "0") == "1"

    def _log(self, level: str, msg: str) -> None:
        order = {"ERROR": 40, "WARN": 30, "INFO": 20, "DEBUG": 10}
        cur = order.get(self._log_level, 30)
        lev = order.get(level.upper(), 30)
        if lev >= cur:
            print(f"[{level.upper()}] {msg}")
        # 文件日志
        file_cur = order.get(self._file_log_level, 20)
        if self._log_file_path and lev >= file_cur:
            try:
                from datetime import datetime
                ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                with open(self._log_file_path, 'a', encoding='utf-8') as f:
                    f.write(f"{ts} [{level.upper()}] {msg}\n")
            except Exception:
                pass

    def set_log_file(self, path: str) -> None:
        # 确保目录存在
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._log_file_path = path

    def reset_run_stats(self) -> None:
        self._stats = {
            'attempts': 0,
            'repairs': 0,
            'repair_failures': 0,
            'size_violations': 0,
        }

    def get_run_stats(self) -> dict:
        return dict(self._stats)

    def _connectivity_summary(self, topo_adj_bool: np.ndarray, parts: np.ndarray, k: int) -> List[dict]:
        """返回每个分区的连通性摘要: 节点数/分量数/最大分量规模。"""
        n = topo_adj_bool.shape[0]
        from collections import deque
        def comp_count(nodes: np.ndarray) -> tuple[int, int]:
            if nodes.size == 0:
                return 0, 0
            mask = np.zeros(n, dtype=bool)
            mask[nodes] = True
            seen = np.zeros(n, dtype=bool)
            comps = 0
            largest = 0
            for s in nodes:
                if seen[s]:
                    continue
                comps += 1
                size = 0
                dq = deque([int(s)])
                seen[int(s)] = True
                while dq:
                    u = dq.popleft()
                    size += 1
                    for v in np.where(topo_adj_bool[u] & mask & ~seen)[0]:
                        seen[int(v)] = True
                        dq.append(int(v))
                largest = max(largest, size)
            return comps, largest

        out = []
        for pid in range(k):
            nodes = np.where(parts == pid)[0]
            comps, largest = comp_count(nodes)
            out.append({
                "pid": pid,
                "size": int(nodes.size),
                "components": int(comps),
                "largest": int(largest),
            })
        return out

    def _size_summary(self, parts: np.ndarray, k: int, lower: int, upper: int) -> List[dict]:
        """返回每个分区的规模摘要与是否越界（软约束，仅汇报）。"""
        out = []
        for pid in range(k):
            sz = int(np.sum(parts == pid))
            out.append({
                "pid": pid,
                "size": sz,
                "under": bool(sz < lower),
                "over": bool(sz > upper),
            })
        return out
        
    def process_single_json(self, json_path: str) -> List[Data]:
        """
        处理单个 JSON 文件，为每个有效的分区配置生成一个 PyG Data 对象。

        Args:
            json_path (str): OPFData JSON 文件的路径。

        Returns:
            List[Data]: 一个包含多个 PyTorch Geometric Data 对象的列表。
        """
        # 从文件名中提取case_id
        import os
        filename = os.path.basename(json_path)
        case_id = os.path.splitext(filename)[0]  # 去掉.json扩展名
        
        # 加载 JSON 数据
        with open(json_path, 'r') as f:
            opf_data = json.load(f)
            
        # 提取网络拓扑和参数，构造成一个结构化的字典
        network_data = self._extract_network_data(opf_data)
        
        samples = []
        
        # 基于文件路径生成一个确定性的随机种子，确保结果可复现
        seed_int = int(hashlib.md5(json_path.encode('utf-8')).hexdigest(), 16) % (2**31 - 1)

        if self.multi_k:
            # 多 k 值模式：为 self.k_values 中的每个 k 都尝试生成一个样本
            for k in self.k_values:
                try:
                    # 创建动态分区
                    partition = self._create_partition_dynamic(network_data, opf_data, int(k), seed_int)
                    if partition is None:
                        continue
                    # 创建 PyG 数据对象
                    pyg_data = self._create_pyg_data(opf_data, network_data, partition, int(k), case_id)
                    if pyg_data is not None:
                        samples.append(pyg_data)
                except Exception as e:
                    print(f"处理分区 k={k} 失败: {e}")
                    continue
        else:
            # 固定 k 值模式
            k = int(self.k_fixed)
            try:
                partition = self._create_partition_dynamic(network_data, opf_data, k, seed_int)
                if partition is not None:
                    pyg_data = self._create_pyg_data(opf_data, network_data, partition, k, case_id)
                    if pyg_data is not None:
                        samples.append(pyg_data)
            except Exception as e:
                print(f"处理分区 k={k} 失败: {e}")
                
        return samples
    
    def _extract_network_data(self, opf_data: Dict) -> Dict:
        """
        从 OPFData JSON 中提取网络拓扑和参数。

        Args:
            opf_data (Dict): 解析后的 OPFData JSON 数据。

        Returns:
            Dict: 一个包含结构化网络信息的字典。
        """
        grid = opf_data['grid']

        # 提取节点信息
        buses = grid['nodes']['bus']
        loads = grid['nodes']['load']
        generators = grid['nodes']['generator']  # 新增：发电机信息

        # 提取边信息
        ac_lines = grid['edges']['ac_line']
        transformers = grid['edges']['transformer']
        load_links = grid['edges']['load_link']
        gen_links = grid['edges'].get('generator_link', {'senders': [], 'receivers': []})

        # 构建母线索引到负荷 (P, Q) 的映射
        load_map: Dict[int, Tuple[float, float]] = {}
        for load_idx, bus_idx in zip(load_links['senders'], load_links['receivers']):
            p_load, q_load = loads[load_idx]
            if bus_idx in load_map:
                load_map[bus_idx] = (
                    load_map[bus_idx][0] + float(p_load),
                    load_map[bus_idx][1] + float(q_load),
                )
            else:
                load_map[bus_idx] = (float(p_load), float(q_load))

        # 构建母线索引到发电机功率约束的映射（向量化实现）
        gen_map: Dict[int, Tuple[float, float, float, float, float, float]] = {}
        # 格式: bus_id -> (P_gen, Q_gen, P_min, P_max, Q_min, Q_max)
        
        # 向量化提取发电机特征
        gen_senders = gen_links['senders']
        gen_receivers = gen_links['receivers']
        
        for gen_idx, bus_idx in zip(gen_senders, gen_receivers):
            if gen_idx < len(generators):
                gen_data = generators[gen_idx]
                # 按数据指南提取特征：[mbase, pg, pmin, pmax, qg, qmin, qmax, vg, cost_squared, cost_linear, cost_offset]
                pg = float(gen_data[1])      # 初始有功发电功率
                pmin = float(gen_data[2])    # 最小有功发电功率
                pmax = float(gen_data[3])    # 最大有功发电功率
                qg = float(gen_data[4])      # 初始无功发电功率
                qmin = float(gen_data[5])    # 最小无功发电功率
                qmax = float(gen_data[6])    # 最大无功发电功率
                
                if bus_idx in gen_map:
                    # 累加多个发电机的功率和约束
                    old_pg, old_qg, old_pmin, old_pmax, old_qmin, old_qmax = gen_map[bus_idx]
                    gen_map[bus_idx] = (
                        old_pg + pg,           # 累加P_gen
                        old_qg + qg,           # 累加Q_gen
                        old_pmin + pmin,       # 累加P_min
                        old_pmax + pmax,       # 累加P_max
                        old_qmin + qmin,       # 累加Q_min
                        old_qmax + qmax        # 累加Q_max
                    )
                else:
                    gen_map[bus_idx] = (pg, qg, pmin, pmax, qmin, qmax)

        # 根据母线类型 (bus_type==3) 确定参考母线
        slack_bus = 0
        try:
            for i, bus_data in enumerate(buses):
                bus_type = int(round(bus_data[1]))
                if bus_type == 3:
                    slack_bus = i
                    break
        except (ValueError, IndexError):
            pass

        # 获取基准功率 (baseMVA)
        ctx = grid.get('context', {})
        if isinstance(ctx, dict) and 'baseMVA' in ctx:
            base_mva = float(ctx['baseMVA'])
        elif isinstance(ctx, list):
            try:
                base_mva = float(ctx[0][0][0])
            except Exception:
                base_mva = 100.0
        else:
            base_mva = 100.0

        return {
            'n_buses': len(buses),
            'buses': buses,
            'loads': loads,
            'generators': generators,  # 新增：发电机信息
            'load_map': load_map,
            'gen_map': gen_map,  # 更新：现在包含功率约束信息
            'ac_lines': ac_lines,
            'transformers': transformers,
            'load_links': load_links,
            'gen_links': gen_links,
            'slack_bus': slack_bus,
            'baseMVA': base_mva,
        }
    
    # 旧版静态谱聚类接口已废弃

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

    def _build_weighted_adjacency(self, network_data: Dict, angles: np.ndarray,
                                   alpha: float = 0.7, gamma: float = 0.75,
                                   clip_q: Tuple[float, float] = (0.01, 0.99)) -> Tuple[np.ndarray, np.ndarray]:
        """按母线对聚合的加权邻接：Trafo-only 权重置 0；AC 与 Trafo 各自计算后相加。

        返回:
          - weighted_adj: [N,N] 权重矩阵（= 聚合 |Pdc| 归一化，Trafo-only 为 0）
          - flow_affinity: [N,N] 聚合 |Pdc|（未归一化）
        """
        n_buses = int(network_data['n_buses'])
        weighted_adj = np.zeros((n_buses, n_buses), dtype=float)
        flow_affinity = np.zeros((n_buses, n_buses), dtype=float)
        eps = 1e-8

        # 聚合: (min(u,v),max(u,v)) -> [w_ac, w_tr, ac_cnt, tr_cnt]
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

        # 写入矩阵：Trafo-only -> 0；有 AC 则 w = w_ac + w_tr
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

    def _build_adjacency_lists(self, weighted_adj: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """构建邻接表，避免稠密矩阵切片开销"""
        n = weighted_adj.shape[0]
        neighbors = []
        edge_weights = []
        
        for u in range(n):
            # 找到有权重的邻居
            nbs_mask = weighted_adj[u] > 1e-12
            nbs = np.where(nbs_mask)[0]
            weights = weighted_adj[u, nbs]
            
            neighbors.append(nbs)
            edge_weights.append(weights)
        
        return neighbors, edge_weights

    def _select_seeds_weighted_dijkstra(self, neighbors: List[np.ndarray], edge_weights: List[np.ndarray], k: int, seed: int = 0) -> List[int]:
        """增量式 Dijkstra 最远优先选种子：一次多源裁剪传播，复杂度更低。

        Args:
            neighbors: 邻接表，neighbors[u] = 节点u的邻居数组
            edge_weights: 边权表，edge_weights[u] = 节点u到邻居的权重数组
            k: 种子数量
            seed: 随机种子

        - 第一个种子：加权度最大的节点
        - 后续：基于当前 dmin（到已选种子集合的最短距离）选择 argmax dmin
        - 裁剪传播：仅当新源能改善 dmin[v] 时才继续 relax
        """
        import heapq as _heapq
        rng = np.random.RandomState(seed)
        n = len(neighbors)
        eps = 1e-8

        # 计算加权度
        weighted_degree = np.array([np.sum(edge_weights[u]) for u in range(n)])
        s0 = int(np.argmax(weighted_degree))

        # 初始化 dmin：从 s0 的一次 Dijkstra
        dmin = np.full(n, np.inf)
        dmin[s0] = 0.0
        pq = [(0.0, s0)]
        while pq:
            du, u = _heapq.heappop(pq)
            if du > dmin[u]:
                continue
            nbs = neighbors[u]
            weights = edge_weights[u]
            if nbs.size == 0:
                continue
            w_inv = 1.0 / (eps + weights)
            for idx, v in enumerate(nbs):
                nd = du + float(w_inv[idx])
                if nd < dmin[v]:
                    dmin[v] = nd
                    _heapq.heappush(pq, (nd, int(v)))

        seeds = [s0]
        for _ in range(1, k):
            # 避免选择已选过的种子
            available_mask = np.ones(n, dtype=bool)
            available_mask[seeds] = False
            
            # 在可用节点中选择距离最远的
            available_dmin = np.where(available_mask, dmin, -np.inf)
            scores = available_dmin + 1e-9 * rng.rand(n)
            cand = int(np.argmax(scores))
            seeds.append(cand)
            # 增量传播：仅改进 dmin 才扩展
            pq = [(0.0, cand)]
            while pq:
                du, u = _heapq.heappop(pq)
                if du >= dmin[u]:
                    continue
                nbs = neighbors[u]
                weights = edge_weights[u]
                if nbs.size == 0:
                    continue
                w_inv = 1.0 / (eps + weights)
                for idx, v in enumerate(nbs):
                    nd = du + float(w_inv[idx])
                    if nd < dmin[v]:
                        dmin[v] = nd
                        _heapq.heappush(pq, (nd, int(v)))
        return seeds

    def _grow_partitions_priority_queues(self, topo_adj_bool: np.ndarray, neighbors: List[np.ndarray], edge_weights: List[np.ndarray],
                                         weight_map: List[dict], seeds: List[int], k: int, lower: int, upper: int) -> np.ndarray:
        """全局堆公平生长：只允许邻接扩张，天然保持连通。"""
        import heapq
        n = len(neighbors)
        target = max(1, int(round(n / float(k))))
        beta = float(os.getenv("OPFDATA_GROW_BETA", "0.8"))

        topo_neighbors = [np.where(topo_adj_bool[u])[0] for u in range(n)]

        def has_neighbor_in_pid(v: int, pid: int, parts_arr: np.ndarray) -> bool:
            nbs = topo_neighbors[v]
            if nbs.size == 0:
                return False
            return bool(np.any(parts_arr[nbs] == pid))

        def push_frontier(heap, pid: int, v_from: int, parts_arr: np.ndarray):
            # 使用拓扑邻接作为扩张候选，优先级由权重决定
            topo_nbs = topo_neighbors[v_from]
            wmap = weight_map[v_from]
            if topo_nbs.size == 0:
                return
            gap = max(0, target - sizes[pid]) / max(target, 1)
            pr_scale = (1.0 + beta * gap)
            for v in topo_nbs:
                if assigned[v]:
                    continue
                # O(1) 查权重
                w = float(wmap.get(int(v), 0.0))
                if w <= 0.0:
                    continue
                priority = - w * pr_scale
                heapq.heappush(heap, (priority, pid, int(v)))

        parts = -np.ones(n, dtype=int)
        sizes = [0 for _ in range(k)]
        assigned = np.zeros(n, dtype=bool)
        heap = []

        for pid, s in enumerate(seeds):
            parts[s] = pid
            sizes[pid] += 1
            assigned[s] = True
        for pid, s in enumerate(seeds):
            push_frontier(heap, pid, s, parts)

        while heap:
            pr, pid, v = heapq.heappop(heap)
            if assigned[v]:
                continue
            # 软约束：当分区接近上界时，跳过此次分配，让其他分区有机会生长
            if sizes[pid] >= upper:
                continue
            if not has_neighbor_in_pid(v, pid, parts):
                continue
            parts[v] = pid
            sizes[pid] += 1
            assigned[v] = True
            push_frontier(heap, pid, v, parts)

        if not np.all(assigned):
            remaining = np.where(~assigned)[0]
            for v in remaining:
                best_pid = -1
                best_w = -1.0
                nbs = topo_neighbors[v]
                if nbs.size == 0:
                    continue
                sums = {}
                for nb in nbs:
                    pid_nb = int(parts[nb])
                    if pid_nb < 0:
                        continue
                    w = float(weight_map[v].get(int(nb), 0.0))
                    if w <= 0.0:
                        continue
                    sums[pid_nb] = sums.get(pid_nb, 0.0) + w
                for pid_cand, wsum in sums.items():
                    if wsum > best_w:
                        best_w = wsum
                        best_pid = pid_cand
                if best_pid == -1:
                    best_pid = int(np.argmin(sizes))
                parts[v] = best_pid
                sizes[best_pid] += 1
                assigned[v] = True

        return parts

    def _create_partition_dynamic(self, network_data: Dict, opf_data: Dict, k: int, seed: int) -> Optional[np.ndarray]:
        """改进版动态分区（唯一实现）：增强 DCPF + 多源优先队列生长。

        返回满足约束的分区标签；失败返回 None。
        """
        try:
            # 1) 相角（一律使用增强 DCPF）
            angles = self.utils.get_bus_angles(network_data, opf_data)

            # 2) 权重构建
            weighted_adj, _ = self._build_weighted_adjacency(network_data, angles)

            # 3) 构建邻接表（避免稠密矩阵切片开销）+ O(1) 权重映射
            neighbors, edge_weights = self._build_adjacency_lists(weighted_adj)
            weight_map = [{int(v): float(w) for v, w in zip(neighbors[u].tolist(), edge_weights[u].tolist())} for u in range(len(neighbors))]

            # 4) 规模与拓扑
            n_buses = int(network_data['n_buses'])
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

            # 4) 改进种子（带权 Dijkstra）
            seeds = self._select_seeds_weighted_dijkstra(neighbors, edge_weights, k, seed)

            # 5) 多源前沿生长
            parts = self._grow_partitions_priority_queues(topo_adj_bool, neighbors, edge_weights, weight_map, seeds, k, lower, upper)

            # 6) 可选：尺寸均衡（默认关闭，避免破坏连通性）
            if self.balance_sizes:
                parts = self._balance_partition_sizes(
                    topo_adj_bool, neighbors, edge_weights, weight_map,
                    parts, k, min_size=3, lower_bound=lower, upper_bound=upper
                )

            # 7) 连通性检查 + 可选修复
            if not self._all_parts_connected(topo_adj_bool, parts, k):
                mode = os.getenv("OPFDATA_CONNECTIVITY_MODE", "repair")
                self._log("INFO", f"Disconnected partition detected (k={k}, seed={seed}). Mode={mode}")
                # 输出摘要诊断
                summary = self._connectivity_summary(topo_adj_bool, parts, k)
                for s in summary:
                    if s["components"] > 1:
                        self._log("INFO", f" pid={s['pid']} size={s['size']} comps={s['components']} largest={s['largest']}")
                if mode == "repair":
                    self._stats['repairs'] += 1
                    parts = self._repair_connectivity(topo_adj_bool, neighbors, edge_weights, weight_map, parts, k, lower, upper)
                    # 修复后可选再次均衡（默认关闭，避免再度引入断连）
                    if self.balance_sizes:
                        parts = self._balance_partition_sizes(
                            topo_adj_bool, neighbors, edge_weights, weight_map,
                            parts, k, min_size=3, lower_bound=lower, upper_bound=upper
                        )
                    if not self._all_parts_connected(topo_adj_bool, parts, k):
                        # 输出修复后摘要，但不终止：保留样本，交由训练忽略极端点
                        summary2 = self._connectivity_summary(topo_adj_bool, parts, k)
                        for s in summary2:
                            if s["components"] > 1:
                                self._log("INFO", f" after-repair pid={s['pid']} size={s['size']} comps={s['components']} largest={s['largest']}")
                        self._stats['repair_failures'] += 1
                        self._log("INFO", "Partition remains disconnected after repair; proceeding with best-effort assignment.")
                    else:
                        self._log("DEBUG", f"Repair succeeded (k={k}, seed={seed}).")
                else:
                    raise RuntimeError("Partitioning produced a disconnected part; aborting per strict mode.")

            # 8) 最终规模软约束计数（仅统计，不报错）
            sz_sum_final = self._size_summary(parts, k, lower, upper)
            self._stats['size_violations'] += sum(1 for s in sz_sum_final if s['under'] or s['over'])
            self._stats['attempts'] += 1

            return parts
        except Exception as e:
            print(f"Dynamic partition failed: {e}")
            return None


        def _build_weighted_adjacency(self, network_data: Dict, angles: np.ndarray,
                                       alpha: float = 0.7, gamma: float = 0.75,
                                       clip_q: Tuple[float, float] = (0.01, 0.99)) -> Tuple[np.ndarray, np.ndarray]:
            """按母线对聚合的加权邻接：Trafo-only 权重置 0；AC 与 Trafo 各自计算后相加。

            返回:
              - weighted_adj: [N,N] 权重矩阵（= 聚合 |Pdc| 归一化，Trafo-only 为 0）
              - flow_affinity: [N,N] 聚合 |Pdc|（未归一化）
            """
            n_buses = int(network_data['n_buses'])
            weighted_adj = np.zeros((n_buses, n_buses), dtype=float)
            flow_affinity = np.zeros((n_buses, n_buses), dtype=float)
            eps = 1e-8

            # 聚合: (min(u,v),max(u,v)) -> [w_ac, w_tr, ac_cnt, tr_cnt]
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

            # 写入矩阵：Trafo-only -> 0；有 AC 则 w = w_ac + w_tr
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

    def _repair_connectivity(self, topo_adj_bool: np.ndarray, neighbors: List[np.ndarray], edge_weights: List[np.ndarray], weight_map: List[dict],
                              parts: np.ndarray, k: int, lower: int, upper: int) -> np.ndarray:
        """将分区内的小连通分量并入相邻、耦合最大的分区，尽量满足上下界。

        - 对每个分区，保留其最大连通分量，其余分量视为待合并对象
        - 目标分区选择：与该分量拓扑相邻，且跨边权重和最大；若多个并列，选当前规模最小者
        - 若无相邻目标（极少数），退化为对全图中与该分量跨权重和最大的分区
        - 合并后不立即硬性裁剪上界，后续由叶节点均衡拉回
        """
        n = topo_adj_bool.shape[0]
        updated = parts.copy()

        # 辅助：取节点集合在拓扑图上的连通分量（降序）
        def get_components(nodes_list: List[int]) -> List[List[int]]:
            nodes = np.array(nodes_list, dtype=int)
            comps: List[List[int]] = []
            if nodes.size == 0:
                return comps
            mask = np.zeros(n, dtype=bool)
            mask[nodes] = True
            remaining = set(nodes.tolist())
            from collections import deque
            while remaining:
                s = remaining.pop()
                comp = [s]
                q = deque([s])
                seen = set([s])
                while q:
                    u = q.popleft()
                    nbs = np.where(topo_adj_bool[u] & mask)[0]
                    for v in nbs:
                        if v not in seen:
                            seen.add(int(v))
                            comp.append(int(v))
                            q.append(int(v))
                for u in comp:
                    if u in remaining:
                        remaining.remove(u)
                comps.append(comp)
            comps.sort(key=len, reverse=True)
            return comps

        sizes = [int(np.sum(updated == pid)) for pid in range(k)]

        for pid in range(k):
            nodes_in_p = np.where(updated == pid)[0].tolist()
            if len(nodes_in_p) <= 1:
                continue
            comps = get_components(nodes_in_p)
            if len(comps) <= 1:
                continue
            keep = set(comps[0])
            # 对每个小分量选择目标分区
            for comp in comps[1:]:
                self._log("DEBUG", f"Repair: pid={pid} comp_size={len(comp)}")
                # 候选目标分区（拓扑相邻）
                best_q = None
                best_w = -1.0
                comp_set = set(comp)
                # 预取跨边候选：使用拓扑邻接（包含 AC+Trafo）
                boundary_targets = set()
                for u in comp:
                    # 所有与 u 拓扑相邻的节点（不限制为 AC）
                    for nb in np.where(topo_adj_bool[u])[0]:
                        if int(nb) not in comp_set:
                            boundary_targets.add(int(nb))
                # 统计每个分区的跨权重和
                cross_by_q = {}
                for bt in boundary_targets:
                    qid = int(updated[bt])
                    if qid == pid:
                        continue
                    # u->bt 的权重（邻接表）
                    wsum = 0.0
                    for u in comp:
                        wsum += float(weight_map[u].get(int(bt), 0.0))
                    cross_by_q[qid] = cross_by_q.get(qid, 0.0) + wsum
                if cross_by_q:
                    # 选跨权重和最大的分区；若并列，选规模最小者
                    cand = sorted(cross_by_q.items(), key=lambda x: (x[1], -sizes[x[0]]), reverse=True)[0][0]
                    best_q = int(cand)
                else:
                    # 极端退化：无相邻，选择全局跨权重和最大的分区
                    best_q = None
                    best_w = -1.0
                    for qid in range(k):
                        if qid == pid:
                            continue
                        # 累计 comp 到 qid 的权重
                        wsum = 0.0
                        q_nodes = np.where(updated == qid)[0]
                        for u in comp:
                            wmap_u = weight_map[u]
                            for qn in q_nodes:
                                wsum += float(wmap_u.get(int(qn), 0.0))
                        if wsum > best_w:
                            best_w = wsum
                            best_q = qid
                if best_q is None:
                    # 最后兜底：选规模最小者
                    best_q = int(np.argmin(sizes))
                # 合并
                updated[np.array(comp, dtype=int)] = best_q
                sizes[best_q] += len(comp)
                sizes[pid] -= len(comp)
                self._log("INFO", f"Repair: move comp(size={len(comp)}) from pid={pid} to pid={best_q}")

        # Final strict connectivity pass (best-effort -> must-connect):
        # Ignore size bounds; for any remaining small component, merge it into the
        # topologically adjacent partition with maximum cross-weight. If none found,
        # merge into the globally most coupled partition. Iterate a few rounds.
        max_final_iters = 5
        for _ in range(max_final_iters):
            changed = False
            for pid in range(k):
                nodes_in_p = np.where(updated == pid)[0].tolist()
                if len(nodes_in_p) <= 1:
                    continue
                comps = get_components(nodes_in_p)
                if len(comps) <= 1:
                    continue
                keep = set(comps[0])
                for comp in comps[1:]:
                    # Find adjacent partitions by topology
                    comp_set = set(comp)
                    boundary_targets = set()
                    for u in comp:
                        for nb in np.where(topo_adj_bool[u])[0]:
                            if int(nb) not in comp_set:
                                boundary_targets.add(int(nb))
                    # Prefer attaching to the smallest adjacent partition by size
                    best_q = None
                    if boundary_targets:
                        adj_parts = set(int(updated[bt]) for bt in boundary_targets if int(updated[bt]) != pid)
                        if adj_parts:
                            best_q = min(adj_parts, key=lambda q: int(np.sum(updated == q)))
                    # If no adjacent partition found, attach to globally smallest partition (excluding pid)
                    if best_q is None:
                        candidates = [q for q in range(k) if q != pid]
                        best_q = min(candidates, key=lambda q: int(np.sum(updated == q))) if candidates else int((pid + 1) % k)
                    # Merge component (ignore size bounds for strict connectivity)
                    updated[np.array(comp, dtype=int)] = best_q
                    changed = True
            if not changed:
                break
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

    def _balance_partition_sizes(self, topo_adj_bool: np.ndarray, neighbors: List[np.ndarray], edge_weights: List[np.ndarray], weight_map: List[dict], parts: np.ndarray, k: int,
                                 min_size: int, lower_bound: int, upper_bound: int) -> np.ndarray:
        """叶节点均衡：仅迁移边界叶节点（deg_in<=1），避免割点 BFS，高效稳定。"""
        n = len(neighbors)
        updated = parts.copy()

        def part_sizes() -> List[int]:
            return [int(np.sum(updated == pid)) for pid in range(k)]

        # 主循环（有限步，防止死循环）
        max_steps = 3 * n
        step = 0
        while step < max_steps:
            step += 1
            sizes = part_sizes()
            over = [pid for pid, s in enumerate(sizes) if s > upper_bound]
            under = [pid for pid, s in enumerate(sizes) if s < lower_bound]
            # 单个分区过小（小于min_size），也视为欠标
            for pid, s in enumerate(sizes):
                if s < min_size and pid not in under:
                    under.append(pid)

            if not over and not under:
                break

            moved = False
            # 逐个欠标分区尝试从最超标者拉“叶节点”（deg_in<=1）
            under_sorted = sorted(under, key=lambda pid: sizes[pid])
            over_sorted = sorted(over, key=lambda pid: sizes[pid], reverse=True)
            # 预计算 deg_in（同分区内部度），使用邻接表
            deg_in = np.zeros(n, dtype=int)
            for v in range(n):
                nbs_v = neighbors[v]
                if nbs_v.size == 0:
                    continue
                mask = (updated[nbs_v] == updated[v])
                deg_in[v] = int(np.sum(mask))
            for u_pid in under_sorted:
                if sizes[u_pid] >= lower_bound:
                    continue
                u_nodes = np.where(updated == u_pid)[0]
                donors = over_sorted if over_sorted else [pid for pid in range(k) if pid != u_pid]
                best_move = None  # (gain, v_node, from_pid)
                for o_pid in donors:
                    if o_pid == u_pid or sizes[o_pid] <= max(lower_bound, min_size):
                        continue
                    o_nodes = np.where(updated == o_pid)[0]
                    if len(o_nodes) <= 1:
                        continue
                    # 候选：o_pid 的边界叶节点，且与 u_pid 拓扑相邻
                    topo_to_u = np.sum(topo_adj_bool[np.ix_(o_nodes, u_nodes)], axis=1)
                    candidate_indices = [int(o_nodes[i]) for i in np.where((topo_to_u > 0) & (deg_in[o_nodes] <= 1))[0]]
                    for v in candidate_indices:
                        # 使用权重映射计算增益
                        wmap = weight_map[v]
                        gain_u = 0.0
                        for u_node in u_nodes:
                            gain_u += float(wmap.get(int(u_node), 0.0))
                        gain_o = 0.0
                        for o_node in o_nodes:
                            gain_o += float(wmap.get(int(o_node), 0.0))
                        
                        gain = gain_u - gain_o
                        if best_move is None or gain > best_move[0]:
                            best_move = (gain, v, o_pid)
                if best_move is not None:
                    _, v_node, from_pid = best_move
                    updated[v_node] = u_pid
                    moved = True
                    break

            if not moved:
                # 早停
                break

        return updated

    # 旧版组件与种子选择已移除（使用增量式Dijkstra与全局堆路径）

    # 旧版 cut_flow 评估仅用于调试，已废弃
    
    def _create_pyg_data(self, opf_data: Dict, network_data: Dict, partition: np.ndarray, k: int, case_id: str = None) -> Any:
        """
        创建 PyTorch Geometric 的 Data 对象。

        Args:
            opf_data (Dict): 原始的 OPFData JSON 数据。
            network_data (Dict): 提取出的结构化网络数据。
            partition (np.ndarray): 节点的分区标签数组。
            k (int): 分区数量。

        Returns:
            一个 PyTorch Geometric Data 对象，如果创建失败则返回 None。
        """
        try:
            # 在多进程环境中，数据创建应始终在 CPU 上进行，以避免 CUDA 初始化冲突。
            device = torch.device('cpu')
            
            # 注意：不读取 AC 相角/电压作为输入；仅用于提取监督标签（见下）。
            
            # 识别边界元素（新格式）
            tie_corridors, tie_lines, tie_line2corridor, tie_buses = self._identify_boundary_elements(network_data, partition)
            
            if len(tie_corridors) == 0:
                return None  # 如果没有联络线，则认为该分区无效
            
            # 计算母线相角（增强版 DCPF）
            theta_dcpf = self.utils.get_bus_angles(network_data, opf_data)

            # 构建边索引和边特征矩阵（V1.1）
            edge_index, edge_attr = self._build_edge_features(network_data, partition, theta_dcpf)

            # 构建节点特征矩阵（V1.1）
            node_features = self._build_node_features(network_data, partition, tie_buses, theta_dcpf)
            
            # 构建边映射
            tie_edge_line, tie_edge_corridor, tie_edge_sign = self._build_edge_mappings(
                edge_index, tie_corridors, tie_lines, tie_line2corridor
            )

            # 构建联络线列表（按线预测版本）
            # ⚠️ 重要：只保留正向边（与JSON定义的sender→receiver方向一致）
            # edge_index包含双向边（为了GNN消息传递），但我们只对正向边预测和提取标签
            tie_edges = []
            tie_edge_indices = []  # 记录联络线在edge_index中的索引
            seen_edges = set()  # 用于去重（确保只保留一个方向）

            for i in range(edge_index.shape[1]):
                if edge_attr[i, 3] > 0.5:  # is_tie 标记
                    u, v = int(edge_index[0, i]), int(edge_index[1, i])
                    # 只保留正向边（u < v 的方向，或与JSON中sender→receiver一致的方向）
                    # 使用规范化的边key来去重
                    edge_key = (min(u, v), max(u, v))

                    if edge_key not in seen_edges:
                        # 优先保留 u < v 的方向（与大多数JSON定义一致）
                        if u < v:
                            tie_edges.append((u, v))
                            tie_edge_indices.append(i)
                            seen_edges.add(edge_key)
                        else:
                            # 如果当前是 u > v，跳过，等待后面的 v < u
                            # 但为了防止遗漏，我们也记录下来
                            pass

            # 第二遍：处理那些只有反向边的情况（u > v 但没有对应的 v < u）
            for i in range(edge_index.shape[1]):
                if edge_attr[i, 3] > 0.5:
                    u, v = int(edge_index[0, i]), int(edge_index[1, i])
                    edge_key = (min(u, v), max(u, v))

                    if edge_key not in seen_edges:
                        # 这条边只有反向方向，也保留
                        tie_edges.append((u, v))
                        tie_edge_indices.append(i)
                        seen_edges.add(edge_key)

            # 提取 V-θ 标签（按线预测版本）
            y_bus_V, y_edge_sincos, y_edge_pq = self._extract_vtheta_labels(
                opf_data, tie_edges, tie_buses, edge_index
            )

            # 计算 DCPF 相角先验（可选，用于模型输入或loss regularization）
            # theta_dcpf 已在前面计算，这里提取边界母线的 DCPF 相角作为先验
            if len(tie_buses) > 0:
                theta_prior = theta_dcpf[np.array(tie_buses, dtype=int)]
            else:
                theta_prior = np.zeros(0, dtype=np.float32)
            
            # 创建 PyG Data 对象（按线预测版本）
            if Data is None:
                raise RuntimeError("torch_geometric is not installed, cannot create PyG Data object.")
            data = Data(
                # 核心图结构
                x=torch.tensor(node_features, dtype=torch.float32).to(device),
                edge_index=torch.tensor(edge_index, dtype=torch.long).to(device),
                edge_attr=torch.tensor(edge_attr, dtype=torch.float32).to(device),

                # 分区元数据
                partition=torch.tensor(partition, dtype=torch.long).to(device),
                k=k,
                num_partitions=len(set(partition)),

                # 边界对象（保留用于兼容）
                tie_corridors=torch.tensor(tie_corridors, dtype=torch.long).to(device),
                tie_lines=torch.tensor(tie_lines, dtype=torch.long).to(device),
                tie_line2corridor=torch.tensor(tie_line2corridor, dtype=torch.long).to(device),
                tie_buses=torch.tensor(tie_buses, dtype=torch.long).to(device),

                # 按线预测的边界对象
                tie_edges=torch.tensor(tie_edges, dtype=torch.long).to(device),  # [E_tie, 2]
                tie_edge_indices=torch.tensor(tie_edge_indices, dtype=torch.long).to(device),  # [E_tie]

                # 监督标签（V-θ 路线，按线预测）
                y_bus_V=torch.tensor(y_bus_V, dtype=torch.float32).to(device),  # [B]
                y_edge_sincos=torch.tensor(y_edge_sincos, dtype=torch.float32).to(device),  # [E_tie, 2]
                y_edge_pq=torch.tensor(y_edge_pq, dtype=torch.float32).to(device),  # [E_tie, 4]

                # 相角先验（DCPF）
                theta_prior=torch.tensor(theta_prior, dtype=torch.float32).to(device),

                # 边映射（保留用于兼容）
                tie_edge_line=torch.tensor(tie_edge_line, dtype=torch.long).to(device),
                tie_edge_corridor=torch.tensor(tie_edge_corridor, dtype=torch.long).to(device),
                tie_edge_sign=torch.tensor(tie_edge_sign, dtype=torch.int8).to(device),
            )

            # 源追踪元信息（便于校验与溯源）
            data.case_id = str(case_id or "unknown_case")
            data.partition_k = int(k)
            
            # 添加预计算字段（优化训练性能）
            data = add_precomputed_fields(
                data=data,
                edge_attr=edge_attr,
                tie_edge_corridor=tie_edge_corridor,
                num_corridors=len(tie_corridors),
                edge_index=edge_index,
                node_features=node_features,
                tie_buses=tie_buses,
                case_id=case_id or "unknown_case",
                k=k,
                ring_K=3,  # 可配置
                ring_decay=0.5,  # 可配置
                ring_mode='decayed'  # 可配置
            )
            
            # 验证生成的数据对象的内部一致性（V1.1）
            self._validate_pyg_data(data)

            return data
            
        except Exception as e:
            print(f"创建 PyG 数据失败: {e}")
            return None
    
    def _identify_boundary_elements(self, network_data: Dict, partition: np.ndarray) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int, int]], List[int], List[int]]:
        """
        根据分区结果识别走廊、线路实例和耦合母线。

        Args:
            network_data (Dict): 结构化网络数据。
            partition (np.ndarray): 节点的分区标签数组。

        Returns:
            一个元组 (tie_corridors, tie_lines, tie_line2corridor, tie_buses)。
        """
        corridors = []  # [(u,v)] where u < v
        corridor_map = {}  # {(u,v): corridor_id}
        tie_lines = []  # [(u,v,cid)] 线路实例
        tie_line2corridor = []  # 线路实例到走廊的映射
        
        # 收集所有跨分区线路
        all_tie_connections = []
        
        # 处理交流线路
        ac_lines = network_data['ac_lines']
        for from_bus, to_bus in zip(ac_lines['senders'], ac_lines['receivers']):
            if partition[from_bus] != partition[to_bus]:
                all_tie_connections.append((int(from_bus), int(to_bus), 'ac'))
        
        # 处理变压器
        transformers = network_data['transformers']
        for from_bus, to_bus in zip(transformers['senders'], transformers['receivers']):
            if partition[from_bus] != partition[to_bus]:
                all_tie_connections.append((int(from_bus), int(to_bus), 'trafo'))
        
        # 构建走廊和线路实例
        corridor_line_types = {}  # 记录每个走廊包含的线路类型
        
        for fb, tb, line_type in all_tie_connections:
            key = (min(fb, tb), max(fb, tb))
            
            # 走廊处理
            if key not in corridor_map:
                corridor_map[key] = len(corridors)
                corridors.append(key)
                corridor_line_types[key] = []
            
            corridor_id = corridor_map[key]
            corridor_line_types[key].append(line_type)
            
            # 计算该走廊内的线路编号
            line_count_in_corridor = sum(1 for i, line in enumerate(tie_lines) 
                                        if tie_line2corridor[i] == corridor_id)
            
            # 添加线路实例（保持原始方向）
            tie_lines.append((fb, tb, line_count_in_corridor))
            tie_line2corridor.append(corridor_id)
        
        # 【核心修复】过滤掉纯变压器走廊，只保留AC联络走廊
        filtered_corridors = []
        filtered_corridor_map = {}  # 新的走廊映射
        filtered_tie_lines = []
        filtered_tie_line2corridor = []
        
        # 为每个保留的走廊按出现顺序重新编号其回线索引
        per_corr_local_idx: Dict[int, int] = {}
        for i, corridor in enumerate(corridors):
            line_types = corridor_line_types[corridor]
            # 检查是否为纯变压器走廊
            is_pure_transformer = all(lt == 'trafo' for lt in line_types)
            
            if not is_pure_transformer:  # 保留非纯变压器走廊
                new_corridor_id = len(filtered_corridors)
                filtered_corridors.append(corridor)
                filtered_corridor_map[corridor] = new_corridor_id
                per_corr_local_idx[new_corridor_id] = 0
                
                # 重新映射该走廊的线路实例
                for j, (line, old_cid) in enumerate(zip(tie_lines, tie_line2corridor)):
                    if old_cid == i:  # 属于当前走廊
                        fb, tb, line_idx = line
                        local_idx = per_corr_local_idx[new_corridor_id]
                        filtered_tie_lines.append((fb, tb, local_idx))
                        per_corr_local_idx[new_corridor_id] = local_idx + 1
                        filtered_tie_line2corridor.append(new_corridor_id)
        
        # 收集耦合母线（基于过滤后的线路）
        tie_buses = set()
        for u, v, _ in filtered_tie_lines:
            tie_buses.update([u, v])
        
        # 记录过滤统计
        original_count = len(corridors)
        filtered_count = len(filtered_corridors)
        if original_count > filtered_count:
            self._log("INFO", f"边界过滤: {original_count}个走廊 -> {filtered_count}个走廊 (排除{original_count-filtered_count}个纯变压器走廊)")
        
        return filtered_corridors, filtered_tie_lines, filtered_tie_line2corridor, sorted(list(tie_buses))
    
    def _build_edge_mappings(self, edge_index: np.ndarray, corridors: List[Tuple[int, int]], 
                            tie_lines: List[Tuple[int, int, int]], tie_line2corridor: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        正确区分同一走廊的多回线实例；保证每个实例的两条方向边（若存在）映射到同一 line_id。
        
        Args:
            edge_index: 边索引 [2, n_edges]
            corridors: 走廊列表
            tie_lines: 线路实例列表
            tie_line2corridor: 线路到走廊的映射
            
        Returns:
            (tie_edge_line, tie_edge_corridor, tie_edge_sign)
        """
        n_edges = edge_index.shape[1]
        tie_edge_line = np.full(n_edges, -1, dtype=np.int64)
        tie_edge_corridor = np.full(n_edges, -1, dtype=np.int64)
        tie_edge_sign = np.zeros(n_edges, dtype=np.int8)

        # 1) 构建 (u<v) -> corridor_id
        cid_of = {uv: i for i, uv in enumerate(corridors)}

        # 2) 按走廊+方向分桶边索引
        buckets = {}  # (u,v) -> {"uv":[edge_idx...], "vu":[edge_idx...]}
        for edge_idx in range(n_edges):
            fb = int(edge_index[0, edge_idx])
            tb = int(edge_index[1, edge_idx])
            a, b = (fb, tb) if fb < tb else (tb, fb)
            if (a, b) not in cid_of:
                # 非联络线（本函数只用于 tie 的映射；无走廊则保持 -1/0）
                continue
            if (a, b) not in buckets:
                buckets[(a, b)] = {"uv": [], "vu": []}
            if fb == a and tb == b:
                buckets[(a, b)]["uv"].append(edge_idx)
            else:
                buckets[(a, b)]["vu"].append(edge_idx)

        # 3) 扫描 tie_lines，按实例顺序从桶里"配对"出该实例的两条方向边
        #    注意：不同实例的顺序 = 它们在 tie_lines 出现的顺序（与上游构建保持一致）
        line_idx = 0
        per_corr_counters = {}  # (u,v) -> {"uv":0, "vu":0}
        for (u, v, cid), corr_id in zip(tie_lines, tie_line2corridor):
            a, b = (u, v) if u < v else (v, u)
            if (a, b) not in per_corr_counters:
                per_corr_counters[(a, b)] = {"uv": 0, "vu": 0}
            counters = per_corr_counters[(a, b)]
            bucket = buckets.get((a, b), {"uv": [], "vu": []})

            # 与规范方向一致的一条
            if counters["uv"] < len(bucket["uv"]):
                k_uv = bucket["uv"][counters["uv"]]
                tie_edge_line[k_uv] = line_idx
                tie_edge_corridor[k_uv] = corr_id
                tie_edge_sign[k_uv] = +1
                counters["uv"] += 1

            # 反方向的一条
            if counters["vu"] < len(bucket["vu"]):
                k_vu = bucket["vu"][counters["vu"]]
                tie_edge_line[k_vu] = line_idx
                tie_edge_corridor[k_vu] = corr_id
                tie_edge_sign[k_vu] = -1
                counters["vu"] += 1

            line_idx += 1

        return tie_edge_line, tie_edge_corridor, tie_edge_sign
    
    
    def _build_node_features(self, network_data: Dict, partition: np.ndarray, tie_buses: List[int],
                              theta_dcpf: np.ndarray) -> np.ndarray:
        """
        节点特征（V1.2）：[P_load, Q_load, theta_dcpf, is_coupling, P_gen, Q_gen, P_gen_min, P_gen_max, Q_gen_min, Q_gen_max]
        全部为 p.u.，不做统计归一化。
        """
        n_buses = int(network_data['n_buses'])
        load_map = network_data['load_map']
        gen_map = network_data['gen_map']  # 新增：发电机功率约束映射
        tie_set = set(int(b) for b in tie_buses)
        X = np.zeros((n_buses, 10), dtype=float)  # 扩展到10列
        
        for i in range(n_buses):
            # 负荷信息
            p_pu, q_pu = load_map.get(i, (0.0, 0.0))
            X[i, 0] = float(p_pu)
            X[i, 1] = float(q_pu)
            
            # 相角和边界标识
            X[i, 2] = float(theta_dcpf[i])
            X[i, 3] = 1.0 if i in tie_set else 0.0
            
            # 发电机功率约束信息
            gen_data = gen_map.get(i, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
            X[i, 4] = float(gen_data[0])  # P_gen
            X[i, 5] = float(gen_data[1])  # Q_gen
            X[i, 6] = float(gen_data[2])  # P_gen_min
            X[i, 7] = float(gen_data[3])  # P_gen_max
            X[i, 8] = float(gen_data[4])  # Q_gen_min
            X[i, 9] = float(gen_data[5])  # Q_gen_max
        
        # 温和裁剪角度
        X[:, 2] = np.clip(X[:, 2], -1.5, 1.5)
        return X
    
    # 旧版重组函数已废弃（V1.1 不再使用电压先验）
    
    
    def _build_edge_features(self, network_data: Dict, partition: np.ndarray,
                              theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        边特征（V1.1+）：[R, X_eff, S_max, is_tie, edge_type, shift, Pdc, Pdc_ratio, b_fr, b_to]
        - AC 线：edge_type=0, shift=0, X_eff=|x|, b_fr/b_to为充电电纳
        - 变压器：edge_type=1, shift=原值（弧度，度值已兜底转换），X_eff=|x|/|tap|, b_fr/b_to为充电电纳
        - Pdc 来自增强 DCPF（solver），反向边取相反数
        - b_fr, b_to: 线路对地充电电纳 (p.u.)，用于完整π模型AC潮流计算
        """
        base_mva = float(network_data.get('baseMVA', 100.0))
        edges = []
        attrs = []

        # 预先计算物理 Pdc（只计算一次正向列表）
        ac_from, ac_to, ac_pdc, tr_from, tr_to, tr_pdc = self.utils.compute_branch_pdc(network_data, theta)

        # AC 线路
        ac = network_data['ac_lines']
        for idx, (fb, tb, feat) in enumerate(zip(ac['senders'], ac['receivers'], ac['features'])):
            fb = int(fb); tb = int(tb)
            b_fr = float(feat[2]) if len(feat) > 2 else 0.0  # 充电电纳 from
            b_to = float(feat[3]) if len(feat) > 3 else 0.0  # 充电电纳 to
            r = float(feat[4]) if len(feat) > 4 else 0.0
            x = float(feat[5]) if len(feat) > 5 else 1e-8
            # S_max 已为 p.u.（与 OPFData V1.1 对齐），不再除以 baseMVA
            smax = float(feat[6]) if len(feat) > 6 else 1.0
            is_tie = 1.0 if partition[fb] != partition[tb] else 0.0
            x_eff = abs(x)
            pdc = float(ac_pdc[idx])
            pdc_ratio = min(abs(pdc) / max(smax, 1e-8), 2.0)
            # forward fb->tb
            edges.append([fb, tb])
            attrs.append([r, x_eff, smax, is_tie, 0.0, 0.0, pdc, pdc_ratio, b_fr, b_to])
            # reverse tb->fb (b_fr和b_to交换)
            edges.append([tb, fb])
            attrs.append([r, x_eff, smax, is_tie, 0.0, 0.0, -pdc, pdc_ratio, b_to, b_fr])

        # 变压器
        tr = network_data['transformers']
        for idx, (fb, tb, feat) in enumerate(zip(tr['senders'], tr['receivers'], tr['features'])):
            fb = int(fb); tb = int(tb)
            r = float(feat[2]) if len(feat) > 2 else 0.0
            x = float(feat[3]) if len(feat) > 3 else 1e-8
            # S_max 已为 p.u.（与 OPFData V1.1 对齐），不再除以 baseMVA
            smax = float(feat[4]) if len(feat) > 4 else 1.0
            tap = float(feat[7]) if len(feat) > 7 and abs(feat[7]) > 1e-8 else 1.0
            shift = float(feat[8]) if len(feat) > 8 else 0.0
            b_fr = float(feat[9]) if len(feat) > 9 else 0.0  # 变压器充电电纳 from
            b_to = float(feat[10]) if len(feat) > 10 else 0.0  # 变压器充电电纳 to
            if abs(shift) > np.pi:
                shift = float(np.deg2rad(shift))
            x_eff = abs(x) / max(abs(tap), 1e-8)
            is_tie = 0.0  # 【AC-only修正】变压器边永远不是联络边
            pdc = float(tr_pdc[idx])
            pdc_ratio = min(abs(pdc) / max(smax, 1e-8), 2.0)
            # forward fb->tb （shift 正向）
            edges.append([fb, tb])
            attrs.append([r, x_eff, smax, is_tie, 1.0, +shift, pdc, pdc_ratio, b_fr, b_to])
            # reverse tb->fb （shift 反向，Pdc 取反，b_fr/b_to交换）
            edges.append([tb, fb])
            attrs.append([r, x_eff, smax, is_tie, 1.0, -shift, -pdc, pdc_ratio, b_to, b_fr])

        if not edges:
            return np.empty((2, 0), dtype=int), np.empty((0, 10), dtype=float)
        edge_index = np.array(edges, dtype=int).T
        edge_attr = np.array(attrs, dtype=float)
        assert edge_attr.shape[1] == 10, f"边特征维度错误: {edge_attr.shape[1]}，应该是10 (包含b_fr和b_to)"
        return edge_index, edge_attr
    
    def _extract_vtheta_labels(self, opf_data: Dict,
                               tie_edges: List[Tuple[int, int]],
                               tie_buses: List[int],
                               edge_index: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        从 OPFData 的 AC-OPF solution 中提取 V-θ 标签（按线预测版本）：
        - y_bus_V [B]: 边界母线电压幅值 (p.u.)
        - y_edge_sincos [E_tie, 2]: 每条联络线的相角差 [sin(Δθ), cos(Δθ)]
        - y_edge_pq [E_tie, 4]: 每条联络线的功率 [Pf, Pt, Qf, Qt]

        Args:
            opf_data: OPFData JSON
            tie_edges: 联络线列表 [(u,v), ...]（有向，按edge_index中的顺序）
            tie_buses: 边界母线列表 [b1, b2, ...]
            edge_index: [2, E] 完整的边索引

        Returns:
            (y_bus_V, y_edge_sincos, y_edge_pq)

        Note:
            solution.nodes.bus 格式为 [[theta, V_mag], ...]
            solution.edges.ac_line/transformer.features[i] = [Pf, Qf, Pt, Qt]
        """
        try:
            # ========== 1. 提取母线电压和相角 ==========
            sol_bus = np.array(opf_data['solution']['nodes']['bus'], dtype=float)  # [N, 2]
            if sol_bus.shape[1] != 2:
                raise ValueError(f"Expected solution.nodes.bus to have 2 columns [theta, V], got {sol_bus.shape[1]}")

            theta_all = sol_bus[:, 0]  # [N] 相角 (弧度)
            V_all = sol_bus[:, 1]      # [N] 电压幅值 (p.u.)

            # 提取边界母线电压
            y_bus_V = V_all[tie_buses]  # [B]

            # ========== 2. 提取每条联络线的相角差 ==========
            E_tie = len(tie_edges)
            y_edge_sincos = np.zeros((E_tie, 2), dtype=float)

            for i, (u, v) in enumerate(tie_edges):
                dtheta = theta_all[u] - theta_all[v]  # Δθ = θ_u - θ_v
                y_edge_sincos[i, 0] = np.sin(dtheta)
                y_edge_sincos[i, 1] = np.cos(dtheta)

            # ========== 3. 提取每条联络线的功率 ==========
            # 从 solution.edges 中提取 AC线路和变压器的功率
            grid_edges = opf_data.get('grid', {}).get('edges', {})
            sol_edges = opf_data.get('solution', {}).get('edges', {})

            ac_grid = grid_edges.get('ac_line', {})
            tr_grid = grid_edges.get('transformer', {})
            ac_sol = np.asarray(sol_edges.get('ac_line', {}).get('features', []), dtype=float)
            tr_sol = np.asarray(sol_edges.get('transformer', {}).get('features', []), dtype=float)

            # 建立 (u,v) → 功率的映射
            # AC线路
            ac_senders = ac_grid.get('senders', [])
            ac_receivers = ac_grid.get('receivers', [])
            ac_power_map = {}  # {(u,v): (Pf, Qf, Pt, Qt)}
            for i in range(len(ac_senders)):
                u, v = int(ac_senders[i]), int(ac_receivers[i])
                if i < len(ac_sol) and ac_sol.shape[1] >= 4:
                    pf, qf, pt, qt = ac_sol[i, 0], ac_sol[i, 1], ac_sol[i, 2], ac_sol[i, 3]
                else:
                    pf = qf = pt = qt = 0.0
                ac_power_map[(u, v)] = (float(pf), float(qf), float(pt), float(qt))

            # 变压器
            tr_senders = tr_grid.get('senders', [])
            tr_receivers = tr_grid.get('receivers', [])
            tr_power_map = {}
            for i in range(len(tr_senders)):
                u, v = int(tr_senders[i]), int(tr_receivers[i])
                if i < len(tr_sol) and tr_sol.shape[1] >= 4:
                    pf, qf, pt, qt = tr_sol[i, 0], tr_sol[i, 1], tr_sol[i, 2], tr_sol[i, 3]
                else:
                    pf = qf = pt = qt = 0.0
                tr_power_map[(u, v)] = (float(pf), float(qf), float(pt), float(qt))

            # 为每条联络线提取功率
            # ⚠️ tie_edges现在只包含正向边，方向与JSON定义一致
            y_edge_pq = np.zeros((E_tie, 4), dtype=float)
            for i, (u, v) in enumerate(tie_edges):
                # 先查找正向 (u, v)
                if (u, v) in ac_power_map:
                    pf, qf, pt, qt = ac_power_map[(u, v)]
                elif (u, v) in tr_power_map:
                    pf, qf, pt, qt = tr_power_map[(u, v)]
                # 如果找不到，尝试反向（处理JSON中可能的方向不一致）
                elif (v, u) in ac_power_map:
                    # 反向：交换 from/to 功率
                    pt_rev, qt_rev, pf_rev, qf_rev = ac_power_map[(v, u)]
                    pf, qf, pt, qt = pf_rev, qf_rev, pt_rev, qt_rev
                elif (v, u) in tr_power_map:
                    pt_rev, qt_rev, pf_rev, qf_rev = tr_power_map[(v, u)]
                    pf, qf, pt, qt = pf_rev, qf_rev, pt_rev, qt_rev
                else:
                    # 未找到，使用零值
                    pf = qf = pt = qt = 0.0
                    self._log("WARN", f"联络线 ({u},{v}) 和 ({v},{u}) 都未在solution中找到功率数据")

                y_edge_pq[i] = [pf, pt, qf, qt]  # [Pf, Pt, Qf, Qt]

            return y_bus_V, y_edge_sincos, y_edge_pq

        except Exception as e:
            self._log("ERROR", f"Failed to extract V-θ labels: {e}")
            # 返回零标签作为fallback
            B = len(tie_buses)
            E_tie = len(tie_edges)
            return np.zeros(B, dtype=float), np.zeros((E_tie, 2), dtype=float), np.zeros((E_tie, 4), dtype=float)

    def _extract_end_pq_labels(self, opf_data: Dict,
                               corridors: List[Tuple[int, int]],
                               base_mva: float) -> np.ndarray:
        """
        【已废弃】基于 OPFData 的 solution 边功率，提取端口级 y_end_pq [C,4]：
        每个走廊 (u<v) 的 (Pu, Pv, Qu, Qv)，端口按 (u端, v端) 聚合所有回线。
        - 若回线方向与 (u<v) 一致：Pu+=pf, Pv+=pt, Qu+=qf, Qv+=qt
        - 若记录为 (v->u)：Pu+=pt, Pv+=pf, Qu+=qt, Qv+=qf
        - 单位：OPFData solution.edges.* 已为 p.u.，无需再除 baseMVA

        Note: 本函数保留用于向后兼容，V-θ路线请使用 _extract_vtheta_labels
        """
        grid_edges = opf_data.get('grid', {}).get('edges', {})
        sol_edges = opf_data.get('solution', {}).get('edges', {})
        ac_grid = grid_edges.get('ac_line', {})
        tr_grid = grid_edges.get('transformer', {})
        ac_sol = np.asarray(sol_edges.get('ac_line', {}).get('features', []), dtype=float)
        tr_sol = np.asarray(sol_edges.get('transformer', {}).get('features', []), dtype=float)
        # 端口级累计（内部先按 [Pu, Pv, Qu, Qv] 组装，最后重排为 [pf_u, pt_v, qf_u, qt_v]）
        C = len(corridors)
        y_end = np.zeros((C, 4), dtype=float)
        corr_map = { (int(min(u,v)), int(max(u,v))): idx for idx, (u,v) in enumerate(corridors) }

        # AC 线路
        ac_senders = ac_grid.get('senders', [])
        ac_receivers = ac_grid.get('receivers', [])
        for i in range(len(ac_senders)):
            u = int(ac_senders[i]); v = int(ac_receivers[i])
            a, b = (u, v) if u < v else (v, u)
            key = (a, b)
            if key not in corr_map:
                continue
            cid = corr_map[key]
            if i < len(ac_sol) and ac_sol.shape[1] >= 4:
                pf, qf, pt, qt = float(ac_sol[i, 0]), float(ac_sol[i, 1]), float(ac_sol[i, 2]), float(ac_sol[i, 3])
            else:
                pf = qf = pt = qt = 0.0
            if u == a and v == b:
                y_end[cid, 0] += pf  # Pu (p.u.)
                y_end[cid, 1] += pt  # Pv (p.u.)
                y_end[cid, 2] += qf  # Qu (p.u.)
                y_end[cid, 3] += qt  # Qv (p.u.)
            else:
                y_end[cid, 0] += pt
                y_end[cid, 1] += pf
                y_end[cid, 2] += qt
                y_end[cid, 3] += qf

        # 变压器
        tr_senders = tr_grid.get('senders', [])
        tr_receivers = tr_grid.get('receivers', [])
        for i in range(len(tr_senders)):
            u = int(tr_senders[i]); v = int(tr_receivers[i])
            a, b = (u, v) if u < v else (v, u)
            key = (a, b)
            if key not in corr_map:
                continue
            cid = corr_map[key]
            if i < len(tr_sol) and tr_sol.shape[1] >= 4:
                pf, qf, pt, qt = float(tr_sol[i, 0]), float(tr_sol[i, 1]), float(tr_sol[i, 2]), float(tr_sol[i, 3])
            else:
                pf = qf = pt = qt = 0.0
            if u == a and v == b:
                y_end[cid, 0] += pf
                y_end[cid, 1] += pt
                y_end[cid, 2] += qf
                y_end[cid, 3] += qt
            else:
                y_end[cid, 0] += pt
                y_end[cid, 1] += pf
                y_end[cid, 2] += qt
                y_end[cid, 3] += qf
        # 重排为 [pf_u, pt_v, qf_u, qt_v]
        pfptq = np.zeros_like(y_end)
        pfptq[:, 0] = y_end[:, 0]  # pf_u (Pu)
        pfptq[:, 1] = y_end[:, 1]  # pt_v (Pv)
        pfptq[:, 2] = y_end[:, 2]  # qf_u (Qu)
        pfptq[:, 3] = y_end[:, 3]  # qt_v (Qv)
        return pfptq
    
    def _validate_pyg_data(self, data) -> None:
        """V-θ 路线数据一致性检查（按线预测版本）。"""
        # 1) 标签维度检查
        assert data.y_bus_V.shape[0] == len(data.tie_buses), \
            f"y_bus_V 长度不匹配: 实际={data.y_bus_V.shape[0]}, 期望={len(data.tie_buses)}"

        if hasattr(data, 'y_edge_sincos') and data.y_edge_sincos is not None:
            assert data.y_edge_sincos.shape[0] == len(data.tie_edge_indices), \
                f"y_edge_sincos 长度不匹配: 实际={data.y_edge_sincos.shape[0]}, 期望={len(data.tie_edge_indices)}"
            assert data.y_edge_sincos.shape[1] == 2, "y_edge_sincos 应为 [E_tie,2] (sin(Δθ), cos(Δθ))"

        if hasattr(data, 'y_edge_pq') and data.y_edge_pq is not None:
            assert data.y_edge_pq.shape[0] == len(data.tie_edge_indices), \
                f"y_edge_pq 长度不匹配: 实际={data.y_edge_pq.shape[0]}, 期望={len(data.tie_edge_indices)}"
            assert data.y_edge_pq.shape[1] == 4, "y_edge_pq 应为 [E_tie,4] (Pf, Pt, Qf, Qt)"

        # 2) 电压范围检查
        V_cpu = data.y_bus_V.cpu().numpy() if data.y_bus_V.is_cuda else data.y_bus_V.numpy()
        assert np.all((V_cpu >= 0.8) & (V_cpu <= 1.2)), \
            f"电压超出合理范围 [0.8, 1.2]: min={V_cpu.min():.4f}, max={V_cpu.max():.4f}"

        # 3) sin/cos 单位圆检查（允许小误差）
        if hasattr(data, 'y_edge_sincos') and data.y_edge_sincos is not None:
            sincos_cpu = data.y_edge_sincos.cpu().numpy() if data.y_edge_sincos.is_cuda else data.y_edge_sincos.numpy()
            if sincos_cpu.size > 0:
                norms = np.sqrt(sincos_cpu[:, 0]**2 + sincos_cpu[:, 1]**2)
                assert np.allclose(norms, 1.0, atol=1e-6), \
                    f"sin/cos 不在单位圆上: norm范围 [{norms.min():.6f}, {norms.max():.6f}]"

        # 4) 验证tie_edges的唯一性（不检查走廊）
        edges_set = set()
        if hasattr(data, 'tie_edges') and data.tie_edges is not None:
            for u, v in data.tie_edges.tolist():
                edge_key = (min(u, v), max(u, v))
                if edge_key in edges_set:
                    self._log("WARN", f"发现重复的tie_edge: {edge_key}")
                edges_set.add(edge_key)

        # 保留走廊验证（向后兼容）
        corridors_set = set()
        for u, v in data.tie_corridors.tolist():
            assert u < v, f"走廊 ({u},{v}) 不满足 u<v 约束"
            assert (u, v) not in corridors_set, f"重复的走廊 ({u},{v})"
            corridors_set.add((u, v))

        # 5) 验证线路到走廊的映射
        assert len(data.tie_line2corridor) == len(data.tie_lines), \
            f"tie_line2corridor 长度不匹配: {len(data.tie_line2corridor)} vs {len(data.tie_lines)}"

        # 6) 验证边映射的有效性
        n_edges = data.edge_index.shape[1]
        assert data.tie_edge_corridor.shape[0] == n_edges, \
            f"tie_edge_corridor 长度应该等于边数"
        assert data.tie_edge_sign.shape[0] == n_edges, \
            f"tie_edge_sign 长度应该等于边数"

        # 7) 检查 S_max 范围
        edge_attr_cpu = data.edge_attr.cpu().numpy() if data.edge_attr.is_cuda else data.edge_attr.numpy()
        smax = edge_attr_cpu[:, 2]
        assert np.all(smax >= 0), "S_max 应≥0"
    
    def process_batch(self, json_dir: str, batch_size: int = 100) -> List[Data]:
        """
        批量处理一个目录下的多个 JSON 文件。

        Args:
            json_dir (str): 包含 JSON 文件的目录。
            batch_size (int): 要处理的文件数量。

        Returns:
            List[Data]: 所有成功处理的 PyG Data 对象列表。
        """
        import os
        
        json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
        json_files = sorted(json_files)[:batch_size]
        
        all_samples = []
        
        for json_file in json_files:
            json_path = os.path.join(json_dir, json_file)
            samples = self.process_single_json(json_path)
            all_samples.extend(samples)
            
            if len(all_samples) % 100 == 0:
                print(f"已处理 {len(all_samples)} 个样本...")
        
        return all_samples

    # 旧版电压先验计算已废弃（V1.1 不再使用）
