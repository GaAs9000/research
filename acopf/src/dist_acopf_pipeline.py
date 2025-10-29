
"""
dist_acopf_pipeline.py

End-to-end (PQ-only): OPFData JSON -> OPFDataProcessor (gnn_data, V1.1) ->
BC-GNN corridor port P/Q prediction -> per-area subnet modeling via
"cut inter-area links + boundary bus injections" (no ghost buses) -> per-area OPF.

Notes:
- Canonical index domain is CANON (0-based contiguous), same as OPFData JSON.
- Dependencies: pandapower, numpy; partition building via gnn_data OPFDataProcessor.
- Usage examples:
    python dist_acopf_pipeline.py --json /path/to/case.json --k 3 --seed 42 --boundary predicted
    # Or oracle boundary for quick self-check:
    python dist_acopf_pipeline.py --json /path/to/case.json --k 3 --seed 42 --boundary oracle
"""
import json, math, argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import numpy as np

# --- 外部依赖 ---
import pandapower as pp
from pandapower.converter import from_ppc
from concurrent.futures import ProcessPoolExecutor, as_completed

# ========== 数据结构 ==========
@dataclass
class CaseCANON:
    baseMVA: float
    n_bus: int
    # nodes
    bus: Dict[int, Dict]         # i -> {base_kv, vmin, vmax, type}
    gen: Dict[int, Dict]         # gid -> {bus, pmin,pmax,qmin,qmax, vg, c2,c1,c0}
    load: Dict[int, Dict]        # lid -> {bus, pd, qd}
    shunt: Dict[int, Dict]       # sid -> {bus, gs, bs}
    # edges
    ac_lines: List[Dict]         # [{f,t, r_pu, x_pu, b_fr, b_to, rateA}, ...]
    trafos: List[Dict]           # [{h,l, r_pu, x_pu, tap, shift, b_fr, b_to, rateA}, ...]
    # links
    gen_link: List[Tuple[int,int]]   # (gid, bus)
    load_link: List[Tuple[int,int]]  # (lid, bus)
    shunt_link: List[Tuple[int,int]] # (sid, bus)

@dataclass
class PartitionPack:
    part_of_bus: Dict[int, int]               # CANON: bus -> area
    tie_corridors: List[Tuple[int,int]]       # CANON 无向 (i<j)
    tie_buses: List[int]                      # CANON

# Legacy interfaces related to voltage/angle predictions were removed; PQ-only remains.

# ========== NEW: Baseline建网方案 ==========
def _safe_base_mva(grid):
    """安全提取baseMVA"""
    ctx = grid.get("context", None)
    # 兼容 2 种结构：{"context":{"baseMVA":100}} 或 OPFData 的 [[[100.0]]]
    if isinstance(ctx, dict) and "baseMVA" in ctx:
        return float(ctx["baseMVA"])
    if isinstance(ctx, list):
        return float(ctx[0][0][0])
    return 100.0

def opfdata_to_ppc(opf_json: dict) -> dict:
    """OPFData JSON → MATPOWER PPC格式转换"""
    g = opf_json["grid"]
    base_mva = _safe_base_mva(g)

    buses = g["nodes"]["bus"]
    loads = g["nodes"].get("load", [])
    gens = g["nodes"].get("generator", [])
    shunts = g["nodes"].get("shunt", [])

    ac = g["edges"].get("ac_line", {})
    tra = g["edges"].get("transformer", {})
    ll = g["edges"].get("load_link", {})
    gl = g["edges"].get("generator_link", {})
    sl = g["edges"].get("shunt_link", {})

    nb = len(buses)
    ng = len(gens)
    nl = len(ac.get("senders", [])) + len(tra.get("senders", []))

    # bus: [BUS_I, TYPE, PD, QD, GS, BS, BUS_AREA, VM, VA, BASE_KV, ZONE, VMAX, VMIN]
    bus = np.zeros((max(1, nb), 13))
    for i, b in enumerate(buses):
        base_kv, btype, vmin, vmax = float(b[0]), int(round(b[1])), float(b[2]), float(b[3])
        bus[i, 0] = i + 1
        bus[i, 1] = 3 if btype == 3 else (2 if btype == 2 else 1)
        bus[i, 2] = 0.0      # PD (MW) 后面累加负荷
        bus[i, 3] = 0.0      # QD (MVAr)
        bus[i, 4] = 0.0      # GS (MW at V=1)
        bus[i, 5] = 0.0      # BS (MVAr at V=1)
        bus[i, 7] = 1.0      # VM init
        bus[i, 8] = 0.0      # VA init (deg)
        bus[i, 9] = base_kv
        bus[i, 11] = vmax
        bus[i, 12] = vmin

    # 负荷累加到母线（MW / MVAr）
    for lid, bidx in zip(ll.get("senders", []), ll.get("receivers", [])):
        if lid < len(loads):
            pd, qd = loads[lid]
            bus[int(bidx), 2] += float(pd) * base_mva
            bus[int(bidx), 3] += float(qd) * base_mva

    # 并联等效到母线（GS/BS）
    for sid, bidx in zip(sl.get("senders", []), sl.get("receivers", [])):
        if sid < len(shunts):
            bs, gs = shunts[sid]  # 注意文件字段顺序: [bs, gs]
            bus[int(bidx), 4] += float(gs) * base_mva
            bus[int(bidx), 5] += float(bs) * base_mva

    # gen: [BUS, PG, QG, QMAX, QMIN, VG, MBASE, STATUS, PMAX, PMIN]
    gen = np.zeros((max(1, ng), 10))
    for gi, (g_row, bidx) in enumerate(zip(gens, gl.get("receivers", []))):
        pg, pmin, pmax = float(g_row[1])*base_mva, float(g_row[2])*base_mva, float(g_row[3])*base_mva
        qg, qmin, qmax = float(g_row[4])*base_mva, float(g_row[5])*base_mva, float(g_row[6])*base_mva
        vg = float(g_row[7])
        gen[gi, :] = [int(bidx)+1, pg, qg, qmax, qmin, vg, base_mva, 1, pmax, pmin]

    # branch: [F_BUS, T_BUS, BR_R, BR_X, BR_B, RATE_A, RATE_B, RATE_C, TAP, SHIFT, BR_STATUS, ANGMIN, ANGMAX]
    branch = np.zeros((max(1, nl), 13))
    row = 0
    # AC 线路：合并 b_fr + b_to 为 BR_B（p.u.）
    for f, t, feat in zip(ac.get("senders", []), ac.get("receivers", []), ac.get("features", [])):
        angmin, angmax, b_fr, b_to, br_r, br_x, rate_a = feat[:7]
        br_b = float(b_fr) + float(b_to)
        branch[row, :] = [
            int(f)+1, int(t)+1, float(br_r), float(br_x), br_b,
            float(rate_a)*base_mva, 0.0, 0.0,
            0.0, 0.0, 1.0, float(angmin)*180/np.pi, float(angmax)*180/np.pi
        ]
        row += 1
    # 变压器：用 branch 表示，含 TAP/SHIFT
    for f, t, feat in zip(tra.get("senders", []), tra.get("receivers", []), tra.get("features", [])):
        # feat: [angmin, angmax, br_r, br_x, rate_a, rate_b, rate_c, tap, shift, b_fr, b_to]
        br_r, br_x = float(feat[2]), float(feat[3])
        rate_a = float(feat[4]) * base_mva
        tap = float(feat[7]) if len(feat) > 7 else 1.0
        shift = float(feat[8])*180/np.pi if len(feat) > 8 else 0.0
        branch[row, :] = [
            int(f)+1, int(t)+1, br_r, br_x, 0.0,
            rate_a, 0.0, 0.0, tap, shift, 1.0, -360.0, 360.0
        ]
        row += 1

    # gencost: [MODEL, STARTUP, SHUTDOWN, NCOST, c2, c1, c0]（注意单位从 p.u. → MW）
    gencost = np.zeros((max(1, ng), 7))
    for gi, g_row in enumerate(gens):
        c2 = float(g_row[8]) / (base_mva**2)
        c1 = float(g_row[9]) / base_mva
        c0 = float(g_row[10])
        gencost[gi, :] = [2, 0, 0, 3, max(0.0, c2), max(1e-6, c1), max(0.0, c0)]

    # 确保至少一个 slack
    if not np.any(bus[:,1] == 3):
        if ng > 0:
            slack_bus = int(gen[0,0]) - 1
        else:
            slack_bus = 0
        bus[slack_bus, 1] = 3

    ppc = {"version": '2', "baseMVA": base_mva, "bus": bus, "gen": gen, "branch": branch, "gencost": gencost}
    return ppc

def build_full_net_from_json(json_path: str):
    """使用baseline方式构建完整网络：JSON → PPC → from_ppc"""
    try:
        with open(json_path, "r") as f:
            j = json.load(f)
        ppc = opfdata_to_ppc(j)
        net = from_ppc(ppc, f_hz=50)
        
        # 验证网络完整性
        if len(net.bus) == 0:
            raise ValueError("生成的网络缺少母线")
            
        print(f"✅ Baseline网络构建成功: {len(net.bus)}母线, {len(net.gen)}发电机, "
              f"{len(net.sgen) if hasattr(net, 'sgen') else 0}静态发电机")
        return net
    except Exception as e:
        print(f"❌ Baseline建网失败: {e}")
        raise RuntimeError(f"无法使用baseline方法构建网络: {e}")

# ========== 1) 解析 OPFData JSON ==========
def load_opfdata_json(path: str) -> CaseCANON:
    with open(path, "r") as f:
        j = json.load(f)
    g = j["grid"]
    # baseMVA
    # OPFData 的 grid.context 结构是 [[[baseMVA]]]
    baseMVA = float(g["context"][0][0][0])

    # buses
    bus_rows = g["nodes"]["bus"]
    bus: Dict[int, Dict] = {}
    for i, row in enumerate(bus_rows):
        base_kv, btype, vmin, vmax = row
        bus[i] = {"base_kv": float(base_kv), "type": int(btype), "vmin": float(vmin), "vmax": float(vmax)}
    n_bus = len(bus_rows)

    # generators
    gen_rows = g["nodes"]["generator"]
    gen: Dict[int, Dict] = {}
    for gid, row in enumerate(gen_rows):
        # 列序 (见 OPFData 文档)：mbase, pg, pmin, pmax, qg, qmin, qmax, vg, c2, c1, c0
        _, _, pmin, pmax, _, qmin, qmax, vg, c2, c1, c0 = row
        gen[gid] = {"pmin": float(pmin), "pmax": float(pmax), "qmin": float(qmin), "qmax": float(qmax),
                    "vg": float(vg), "c2": float(c2), "c1": float(c1), "c0": float(c0)}

    # loads
    load_rows = g["nodes"]["load"]
    load: Dict[int, Dict] = {lid: {"pd": float(r[0]), "qd": float(r[1])} for lid, r in enumerate(load_rows)}

    # shunts
    shunt_rows = g["nodes"]["shunt"]
    shunt: Dict[int, Dict] = {sid: {"bs": float(r[0]), "gs": float(r[1])} for sid, r in enumerate(shunt_rows)}

    # edges: ac_line
    L = g["edges"]["ac_line"]
    ac_lines = []
    for s, r, feat in zip(L["senders"], L["receivers"], L["features"]):
        angmin, angmax, b_fr, b_to, br_r, br_x, rate_a, *_ = feat
        ac_lines.append({"f": int(s), "t": int(r),
                         "r_pu": float(br_r), "x_pu": float(br_x),
                         "b_fr": float(b_fr), "b_to": float(b_to),
                         "rateA": float(rate_a)})

    # edges: transformer
    T = g["edges"]["transformer"]
    trafos = []
    for s, r, feat in zip(T["senders"], T["receivers"], T["features"]):
        # 列序 (见 OPFData 文档)：angmin, angmax, br_r, br_x, rate_a, rate_b, rate_c, tap, shift, b_fr, b_to
        # 有些字段在不同系统可能缺省，这里做健壮处理
        angmin, angmax, br_r, br_x, rate_a, *rest = feat
        tap   = rest[2] if len(rest) >= 3 else 1.0
        shift = rest[3] if len(rest) >= 4 else 0.0
        b_fr  = rest[4] if len(rest) >= 5 else 0.0
        b_to  = rest[5] if len(rest) >= 6 else 0.0
        trafos.append({"h": int(s), "l": int(r),
                       "r_pu": float(br_r), "x_pu": float(br_x),
                       "tap": float(tap), "shift": float(shift),
                       "b_fr": float(b_fr), "b_to": float(b_to),
                       "rateA": float(rate_a)})

    # links
    gl = g["edges"]["generator_link"]
    gen_link = list(zip(map(int, gl["senders"]), map(int, gl["receivers"])))
    ll = g["edges"]["load_link"]
    load_link = list(zip(map(int, ll["senders"]), map(int, ll["receivers"])))
    sl = g["edges"]["shunt_link"]
    shunt_link = list(zip(map(int, sl["senders"]), map(int, sl["receivers"])))

    # attach bus indices to gens/loads/shunts
    for gid, b in gen_link:
        gen[gid]["bus"] = int(b)
    for lid, b in load_link:
        load[lid]["bus"] = int(b)
    for sid, b in shunt_link:
        shunt[sid]["bus"] = int(b)

    return CaseCANON(baseMVA=baseMVA, n_bus=n_bus,
                     bus=bus, gen=gen, load=load, shunt=shunt,
                     ac_lines=ac_lines, trafos=trafos,
                     gen_link=gen_link, load_link=load_link, shunt_link=shunt_link)

# ========== 2) 谱分区 + PyG 适配 ==========
def build_pyg_and_partition(json_path: str, k: int, seed: int):
    """
    使用 gnn_data 的 OPFDataProcessor（V1.1，PQ-only）构建与训练侧完全一致的 PyG Data。
    返回：
      - pyg_data: torch_geometric.data.Data（含 .partition, .tie_corridors, .tie_buses, .tie_edge_corridor, .edge_attr[8] 等）
      - canon_bus_of_pyg_bus: Dict[pyg_node] -> CANON bus（恒等映射）
      - pyg_bus_of_canon_bus: Dict[CANON bus] -> pyg_node（恒等映射）
    """
    # 读取 JSON
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    # 引入 gnn_data 处理器与仓库根路径（供 `gnn.*` 导入）
    import sys, os
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    gd_src = os.path.join(repo_root, 'gnn_data', 'src')
    if gd_src not in sys.path:
        sys.path.insert(0, gd_src)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    try:
        from opfdata.processor import OPFDataProcessor  # type: ignore
    except Exception as e:
        raise RuntimeError(f"无法导入 OPFDataProcessor，请确认 gnn_data/src 在路径上: {e}")

    processor = OPFDataProcessor()
    network_data = processor._extract_network_data(json_data)
    partition = processor._create_partition_dynamic(network_data, json_data, k, seed)
    if partition is None:
        raise RuntimeError("OPFDataProcessor 动态分区失败")
    pyg_data = processor._create_pyg_data(json_data, network_data, partition, k)
    if pyg_data is None:
        raise RuntimeError("OPFDataProcessor 创建 PyG Data 失败")

    # CANON ↔ PyG 映射（索引一致）
    n_bus = len(json_data["grid"]["nodes"]["bus"])
    canon_bus_of_pyg_bus = {i: i for i in range(n_bus)}
    pyg_bus_of_canon_bus = {i: i for i in range(n_bus)}
    return pyg_data, canon_bus_of_pyg_bus, pyg_bus_of_canon_bus

def extract_partition_pack(case: CaseCANON, pyg_data, canon_bus_of_pyg_bus) -> PartitionPack:
    """从 PyG 数据构建分区信息（AC-only 走廊）。

    - 分区标签来源：pyg_data.partition
    - 走廊/耦合母线优先使用 gnn_data 产生的 AC-only 对象：pyg_data.tie_corridors / pyg_data.tie_buses
      如缺失则回退为仅基于 AC 线计算（严格排除纯变压器走廊）。
    """
    # 取出每个 CANON bus 的 area
    part_of_bus: Dict[int, int] = {}
    if hasattr(pyg_data, "partition"):
        for pyg_node, area in enumerate(pyg_data.partition.tolist()):
            b = canon_bus_of_pyg_bus[pyg_node]
            part_of_bus[b] = int(area)
    else:
        raise RuntimeError("pyg_data.partition 缺失，请在 PyG 构建时返回 partition 标签")

    # 优先使用 gnn_data 的 AC-only 走廊/耦合母线
    if hasattr(pyg_data, "tie_corridors") and hasattr(pyg_data, "tie_buses"):
        tc = getattr(pyg_data, "tie_corridors").cpu().tolist()
        corridors = set((int(u), int(v)) if int(u) < int(v) else (int(v), int(u)) for u, v in tc)
        tie_buses = sorted(int(b) for b in getattr(pyg_data, "tie_buses").cpu().tolist())
    else:
        # 回退：仅用 AC 线路构造跨区走廊（严格 i<j），不加入变压器
        corridors = set()
        def add_ac_edge(i, j):
            ai, aj = part_of_bus[i], part_of_bus[j]
            if ai != aj:
                u, v = (i, j) if i < j else (j, i)
                corridors.add((u, v))
        for br in case.ac_lines:
            add_ac_edge(br["f"], br["t"])
        tie_buses = sorted(list({i for e in corridors for i in e}))

    return PartitionPack(part_of_bus=part_of_bus,
                         tie_corridors=sorted(list(corridors)),
                         tie_buses=tie_buses)

# ========== 3) 预测适配（只输出 CANON） ==========
def predict_boundary_CANON(pyg_data, canon_bus_of_pyg_bus, model_path: str = None):
    """使用BC-GNN（PQ-only）预测边界：返回母线级边界PQ与走廊PQ。

    Returns dict: { 'boundary_pq': {bus_id: (P,Q)}, 'corridor_pfqt': [[pf_u,pt_v,qf_u,qt_v], ...] }
    """
    from bcgnn_predictor import BCGNNPredictor
    import os
    if model_path is None:
        model_path = os.getenv("BCGNN_MODEL_PATH", "checkpoints/best_model.pt")
    model = BCGNNPredictor(model_path)
    raw = model.predict(pyg_data)
    if not raw.get('success', False):
        raise RuntimeError(f"BC-GNN预测失败: {raw.get('error', 'Unknown error')}")
    # 映射到 CANON bus
    boundary_pq_canon = {}
    for pyg_bus, (P, Q) in raw['boundary_pq'].items():
        pyg_bus = int(pyg_bus)
        canon_bus = int(canon_bus_of_pyg_bus.get(pyg_bus, pyg_bus))
        boundary_pq_canon[canon_bus] = (float(P), float(Q))
    return {'boundary_pq': boundary_pq_canon, 'corridor_pfqt': raw.get('corridor_pfqt', [])}

# (Removed) Angle synchronization utilities from legacy flows are no longer used in PQ-only.

# ========== 5) 分区子网(Pandapower)构建：M1 幽灵母线 - 修复版 ==========
def pick_area_slack_bus(case: CaseCANON, pack: PartitionPack, area: int) -> int:
    """选择区域唯一主参考母线，消除多Slack冲突"""
    # 1) 优先选原 JSON 中的 reference bus (bus_type == 3)
    cand_ref = [b for b, meta in case.bus.items()
                if pack.part_of_bus[b] == area and int(meta["type"]) == 3]
    if cand_ref:
        return cand_ref[0]
    # 2) 否则选本区 pmax 最大的发电机所在母线
    best = None; best_cap = -1.0
    for gid, g in case.gen.items():
        b = g.get("bus", None)
        if b is None or pack.part_of_bus[b] != area: 
            continue
        cap = float(g["pmax"])
        if cap > best_cap:
            best_cap, best = cap, b
    # 3) 再否则，任取一个该区母线（保证存在）
    if best is not None:
        return best
    for b, a in pack.part_of_bus.items():
        if a == area:
            return b
    raise RuntimeError(f"No bus found for area {area}")

# 物理走廊主键：保证AC/变压器 + (i<j) 的唯一性
def _corridor_key(kind: str, u: int, v: int):
    return (str(kind), int(u) if u < v else int(v), int(v) if u < v else int(u))

# 旧的按走廊逐线调度逻辑已移除，统一改为母线边界注入口径。

def _deprecated():
    return None

# --- 新增：集中处理 PP ↔ CANON 索引映射 ---
def _make_pp_indexers(net_full):
    """
    返回 (canon2pp, pp2canon, offset)，统一解决 bus 索引是否 0-based/1-based 的问题。
    - canon2pp: CANON bus -> net_full.bus.index
    - pp2canon: net_full.bus.index -> CANON bus
    """
    idx = np.asarray(list(net_full.bus.index))
    n = len(idx)
    sidx = set(idx.tolist())

    if sidx == set(range(n)):          # 0,1,2,...,n-1
        canon2pp = lambda b: int(b)
        pp2canon = lambda b: int(b)
        offset = 0
    elif sidx == set(range(1, n + 1)): # 1,2,3,...,n
        canon2pp = lambda b: int(b) + 1
        pp2canon = lambda b: int(b) - 1
        offset = 1
    else:
        raise RuntimeError(
            f"Unexpected net_full.bus.index: min={idx.min()}, max={idx.max()}, "
            f"count={n}. Expect contiguous 0..n-1 or 1..n."
        )
    return canon2pp, pp2canon, offset

# === 修复版：基于全网from_ppc再分区的方案 ===
def _removed_ghost_builder(*args, **kwargs):
    """Legacy ghost-bus builder removed in bus-injection mode."""
    raise NotImplementedError("ghost-bus builder removed; use build_area_nets_bus_injection")
    
    

# ========== 5.x) 分区子网构建：切断联络线 + 边界母线注入（无幽灵母线） ==========
def build_area_nets_bus_injection(net_full,
                                  pack: PartitionPack,
                                  boundary_bus_pq_pu: Dict[int, Tuple[float, float]],
                                  baseMVA: float,
                                  enable_balancer: bool = True,
                                  balancer_limit_mw: float = 2000.0,
                                  balancer_cost_eur_per_mw2: float = 2000.0,
                                  vm_boundary: Dict[int, float] | None = None,
                                  va_boundary: Dict[int, float] | None = None):
    """
    构建每个分区的独立子网：切断所有跨区线路/变压器；在本区的每个边界母线上，直接施加预测的母线边界注入 (P_bd, Q_bd)。
    - 优点：避免跨区线路损耗与相移带来的偏差，严格遵循“母线注入口径”的预测。
    - 缺点：丢失跨区支路对电压/无功的耦合作用（作为近似）。

    Args:
        net_full: 全网 pandapower 网络
        pack: 分区与边界定义
        boundary_bus_pq_pu: {bus_id: (P_bd_pu, Q_bd_pu)} 预测的母线边界注入（p.u.）
        baseMVA: 基准 MVA，用于转换至 MW / MVAr
        enable_balancer: 是否加入应急平衡源（高二次成本）
    Returns:
        area_nets: 每个分区的 pandapower 网络
        bus_maps: {area: {canon_bus: pp_bus_idx}}
    """
    import pandapower as pp
    areas = sorted(set(pack.part_of_bus.values()))

    # 便捷映射：PP ↔ CANON
    canon2pp, pp2canon, _ = _make_pp_indexers(net_full)

    # 初始化
    area_nets: Dict[int, Any] = {}
    bus_maps: Dict[int, Dict[int, int]] = {}

    for a in areas:
        net = pp.create_empty_network(sn_mva=net_full.sn_mva)
        area_nets[a] = net
        bus_maps[a] = {}

        # 1) 复制本区母线
        area_buses = [b for b, aa in pack.part_of_bus.items() if aa == a]
        for b_canon in area_buses:
            b_pp = canon2pp(b_canon)
            nb = pp.create_bus(
                net,
                vn_kv=float(net_full.bus.at[b_pp, "vn_kv"]),
                min_vm_pu=float(net_full.bus.at[b_pp, "min_vm_pu"]),
                max_vm_pu=float(net_full.bus.at[b_pp, "max_vm_pu"]),
                in_service=bool(net_full.bus.at[b_pp, "in_service"]))
            bus_maps[a][b_canon] = nb

        # 2) 复制区内线路与变压器（切断跨区）
        for idx in net_full.line.index:
            fb_pp = int(net_full.line.at[idx, "from_bus"]) ; tb_pp = int(net_full.line.at[idx, "to_bus"]) 
            fb_c = pp2canon(fb_pp); tb_c = pp2canon(tb_pp)
            if pack.part_of_bus[fb_c] == a and pack.part_of_bus[tb_c] == a:
                pp.create_line_from_parameters(
                    net,
                    from_bus=bus_maps[a][fb_c], to_bus=bus_maps[a][tb_c],
                    length_km=float(net_full.line.at[idx, "length_km"]),
                    r_ohm_per_km=float(net_full.line.at[idx, "r_ohm_per_km"]),
                    x_ohm_per_km=float(net_full.line.at[idx, "x_ohm_per_km"]),
                    c_nf_per_km=float(net_full.line.at[idx, "c_nf_per_km"]),
                    max_i_ka=float(net_full.line.at[idx, "max_i_ka"]))

        for idx in net_full.trafo.index:
            hv_pp = int(net_full.trafo.at[idx, "hv_bus"]) ; lv_pp = int(net_full.trafo.at[idx, "lv_bus"]) 
            hv_c = pp2canon(hv_pp); lv_c = pp2canon(lv_pp)
            if pack.part_of_bus[hv_c] == a and pack.part_of_bus[lv_c] == a:
                pp.create_transformer_from_parameters(
                    net,
                    hv_bus=bus_maps[a][hv_c], lv_bus=bus_maps[a][lv_c],
                    sn_mva=float(net_full.trafo.at[idx, "sn_mva"]),
                    vn_hv_kv=float(net_full.bus.at[hv_pp, "vn_kv"]),
                    vn_lv_kv=float(net_full.bus.at[lv_pp, "vn_kv"]),
                    vk_percent=float(net_full.trafo.at[idx, "vk_percent"]),
                    vkr_percent=float(net_full.trafo.at[idx, "vkr_percent"]),
                    pfe_kw=float(net_full.trafo.at[idx, "pfe_kw"]),
                    i0_percent=float(net_full.trafo.at[idx, "i0_percent"]))

        # 3) 复制内部负荷/机组/并联，继承成本
        gen_id_map = {}
        for idx in net_full.gen.index:
            b_pp = int(net_full.gen.at[idx, "bus"]) ; b_c = pp2canon(b_pp)
            if pack.part_of_bus[b_c] == a:
                gi = pp.create_gen(
                    net, bus_maps[a][b_c], 
                    p_mw=float(net_full.gen.at[idx, "p_mw"]),
                    vm_pu=float(net_full.gen.at[idx, "vm_pu"]))
                # 上下界
                net.gen.loc[gi, ["min_p_mw","max_p_mw"]] = [
                    float(net_full.gen.at[idx, "min_p_mw"]), float(net_full.gen.at[idx, "max_p_mw"]) ]
                net.gen.loc[gi, ["min_q_mvar","max_q_mvar"]] = [
                    float(net_full.gen.at[idx, "min_q_mvar"]), float(net_full.gen.at[idx, "max_q_mvar"]) ]
                gen_id_map[idx] = gi

        sgen_id_map = {}
        if hasattr(net_full, "sgen") and len(net_full.sgen.index):
            for idx in net_full.sgen.index:
                b_pp = int(net_full.sgen.at[idx, "bus"]) ; b_c = pp2canon(b_pp)
                if pack.part_of_bus[b_c] == a:
                    si = pp.create_sgen(
                        net, bus_maps[a][b_c],
                        p_mw=float(net_full.sgen.at[idx, "p_mw"]),
                        q_mvar=float(net_full.sgen.at[idx, "q_mvar"]),
                        controllable=bool(net_full.sgen.at[idx, "controllable"]) if "controllable" in net_full.sgen.columns else False,
                        in_service=bool(net_full.sgen.at[idx, "in_service"]))
                    sgen_id_map[idx] = si

        for idx in net_full.load.index:
            b_pp = int(net_full.load.at[idx, "bus"]) ; b_c = pp2canon(b_pp)
            if pack.part_of_bus[b_c] == a:
                pp.create_load(
                    net, bus_maps[a][b_c],
                    p_mw=float(net_full.load.at[idx, "p_mw"]),
                    q_mvar=float(net_full.load.at[idx, "q_mvar"]))

        if hasattr(net_full, "shunt") and len(net_full.shunt.index):
            for idx in net_full.shunt.index:
                b_pp = int(net_full.shunt.at[idx, "bus"]) ; b_c = pp2canon(b_pp)
                if pack.part_of_bus[b_c] == a:
                    pp.create_shunt(
                        net, bus_maps[a][b_c],
                        p_mw=float(net_full.shunt.at[idx, "p_mw"]) if "p_mw" in net_full.shunt.columns else 0.0,
                        q_mvar=float(net_full.shunt.at[idx, "q_mvar"]) if "q_mvar" in net_full.shunt.columns else 0.0)

        # 继承成本函数（排除 sgen boundary 注入）
        if hasattr(net_full, "poly_cost") and len(net_full.poly_cost.index):
            for _, row in net_full.poly_cost.iterrows():
                et = row["et"] ; element = int(row["element"]) 
                if et == "gen" and element in gen_id_map:
                    pp.create_poly_cost(net, element=gen_id_map[element], et="gen",
                                        cp0_eur=float(row["cp0_eur"]),
                                        cp1_eur_per_mw=float(row["cp1_eur_per_mw"]),
                                        cp2_eur_per_mw2=float(row["cp2_eur_per_mw2"]))
                elif et == "sgen" and element in sgen_id_map:
                    pp.create_poly_cost(net, element=sgen_id_map[element], et="sgen",
                                        cp0_eur=float(row["cp0_eur"]),
                                        cp1_eur_per_mw=float(row["cp1_eur_per_mw"]),
                                        cp2_eur_per_mw2=float(row["cp2_eur_per_mw2"]))

        # 4) 单一角参考 ExtGrid（仅角参考，不供电）
        tie_buses_in_area = [b for b in pack.tie_buses if pack.part_of_bus[b] == a]
        if len(tie_buses_in_area) > 0:
            ref_b = tie_buses_in_area[0]
        else:
            ref_b = next(iter(bus_maps[a].keys()))
        vm_ref = 1.0 if vm_boundary is None else float(vm_boundary.get(ref_b, 1.0))
        va_ref = 0.0 if va_boundary is None else float(va_boundary.get(ref_b, 0.0))
        pp.create_ext_grid(net, bus_maps[a][ref_b], vm_pu=vm_ref, va_degree=np.degrees(va_ref),
                           min_p_mw=-0.1, max_p_mw=0.1,
                           min_q_mvar=-100.0, max_q_mvar=100.0)

        # 5) 边界母线注入：直接以 sgen 固定注入（不参与优化）
        for b in tie_buses_in_area:
            if b in boundary_bus_pq_pu:
                P_pu, Q_pu = boundary_bus_pq_pu[b]
                p_mw = float(P_pu) * baseMVA
                q_mvar = float(Q_pu) * baseMVA
                # 固定注入：controllable=False
                pp.create_sgen(net, bus_maps[a][b], p_mw=p_mw, q_mvar=q_mvar, controllable=False)

        # 6) 可选兜底平衡源（sgen，二次成本）
        if enable_balancer:
            bal = pp.create_sgen(net, bus_maps[a][ref_b], p_mw=0.0, q_mvar=0.0, controllable=True)
            net.sgen.loc[bal, ["min_p_mw","max_p_mw"]] = [-balancer_limit_mw, balancer_limit_mw]
            pp.create_poly_cost(net, element=int(bal), et="sgen",
                                cp0_eur=0.0, cp1_eur_per_mw=0.0,
                                cp2_eur_per_mw2=float(balancer_cost_eur_per_mw2))

        # 清理缓存，避免OPF使用旧PPC
        if hasattr(net, "_ppc"):
            net._ppc = None
        if hasattr(net, "_ppc_opf"):
            net._ppc_opf = None

        print(f"[Bus-Injection][Area {a}] buses={len(net.bus)}, lines={len(net.line)}, trafos={len(net.trafo)}, loads={len(net.load)}, gens={len(net.gen)}, sgens={len(net.sgen) if hasattr(net,'sgen') else 0}")

    return area_nets, bus_maps

def _removed_m1_fallback(*args, **kwargs):
    """Legacy M1 ghost-bus fallback removed."""
    raise NotImplementedError("M1 fallback removed; use build_area_nets_bus_injection")

# ADMM 共识实现已移除

def run_parallel_opf(area_nets: Dict, workers: int = 0) -> List:
    """执行并行OPF求解（不含ADMM项）"""
    items = list(area_nets.items())
    results = []
    if workers and workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(solve_area_opf, item) for item in items]
            for fu in as_completed(futs):
                results.append(fu.result())
    else:
        for item in items:
            results.append(solve_area_opf(item))
    return sorted(results, key=lambda x: x[0])

# ========== 6) 并行 OPF ==========
def solve_area_opf(area_id_net_tuple):
    area, net = area_id_net_tuple
    try:
        # 预热：先做潮流初始化
        try:
            pp.runpp(net, calculate_voltage_angles=True, init="flat")
        except Exception:
            pass  # 预热失败继续进行OPF
        
        # OPF求解
        pp.runopp(net, calculate_voltage_angles=True, verbose=False, init='results')
        
        # 🛠️ Fix 4: 后处理成本检查，确保非负成本
        if hasattr(net, 'res_cost') and net.res_cost < 0:
            print(f"⚠️ Area {area} negative cost {net.res_cost:.2f}€ detected!")
            
            # 计算成本组件用于调试
            if hasattr(net, 'poly_cost') and len(net.poly_cost):
                gen_costs = []
                for _, row in net.poly_cost.iterrows():
                    if row['et'] == 'gen':
                        gi = int(row['element'])
                        p_mw = net.res_gen.at[gi, 'p_mw'] if gi in net.res_gen.index else 0.0
                        cost = row['cp0_eur'] + row['cp1_eur_per_mw']*p_mw + row['cp2_eur_per_mw2']*(p_mw**2)
                        gen_costs.append((gi, p_mw, cost))
                
                # 报告负成本的发电机
                negative_gens = [(gi, p, c) for gi, p, c in gen_costs if c < 0]
                if negative_gens:
                    print(f"  Negative cost generators: {negative_gens[:3]}")  # 显示前3个
            
            # 保留负成本用于调试，不强制修正
            print(f"  Keeping negative cost for debugging...")
        
        # 诊断信息
        n_ext_grid = len(net.ext_grid)
        n_gen = len(net.gen)
        n_bus = len(net.bus)
        print(f"Area {area}: SUCCESS - {n_bus} buses, {n_gen} gens, {n_ext_grid} ext_grids, cost={net.res_cost:.2f}")
        
        return area, True, float(net.res_cost)
    except Exception as e:
        # 更详细的错误诊断
        n_ext_grid = len(net.ext_grid) if hasattr(net, 'ext_grid') else 0
        n_gen = len(net.gen) if hasattr(net, 'gen') else 0  
        n_bus = len(net.bus) if hasattr(net, 'bus') else 0
        print(f"Area {area}: FAILED - {n_bus} buses, {n_gen} gens, {n_ext_grid} ext_grids, error: {str(e)[:100]}")
        return area, False, str(e)

# ========== 7) 主流程 ==========
def run_pipeline(json_path: str, k: int, seed: int, boundary: str = "predicted",
                 workers:int=0, model_path: str=None,
                 oracle_tol: float = 100.0, soft_oracle: bool = True,
                 balancer: bool = True, balancer_limit: float = 2000.0,
                 balancer_cost: float = 2000.0, oracle_anchor_w: float = 0.05):
    """
    boundary: 'predicted' 使用 BC-GNN 预测；'oracle' 使用 JSON 真解作为边界(用于自检)
    """
    # 载入 OPFData
    case = load_opfdata_json(json_path)

    # 谱分区 + PyG（单入口）
    pyg_data, canon_bus_of_pyg_bus, pyg_bus_of_canon_bus = build_pyg_and_partition(json_path, k, seed)
    pack = extract_partition_pack(case, pyg_data, canon_bus_of_pyg_bus)

    # 边界相量与跨区线路功率调度
    if boundary == "predicted":
        pred = predict_boundary_CANON(pyg_data, canon_bus_of_pyg_bus, model_path)
        # 直接采用母线级边界注入（p.u.）作为子图边界条件
        vm_boundary = {b: 1.0 for b in pack.tie_buses}
        va_boundary = {b: 0.0 for b in pack.tie_buses}
        boundary_bus_pq = pred['boundary_pq']  # {bus_id: (P_pu, Q_pu)}
        print("使用BC-GNN预测母线边界注入（p.u.），切断联络线并在边界母线直接注入")
    elif boundary == "oracle":
        # 直接使用 PyG Data 的 y_bus_pq（若存在）作为母线边界真值（p.u.）；否则回退到等值聚合
        vm_boundary = {b: 1.0 for b in pack.tie_buses}
        va_boundary = {b: 0.0 for b in pack.tie_buses}
        boundary_bus_pq = {}
        if hasattr(pyg_data, 'y_bus_pq') and pyg_data.y_bus_pq is not None and pyg_data.y_bus_pq.numel() > 0:
            y = pyg_data.y_bus_pq.cpu().numpy()
            for i, b in enumerate(getattr(pyg_data, 'tie_buses').cpu().tolist()):
                boundary_bus_pq[int(b)] = (float(y[i][0]), float(y[i][1]))
        else:
            # 回退：从 solution[edges] 聚合到 tie_buses（与 gnn_data 标签构造一致）
            with open(json_path, 'r') as f:
                j = json.load(f)
            grid = j.get('grid', {})
            sol = j.get('solution', {})
            edges_grid = grid.get('edges', {})
            edges_sol = sol.get('edges', {})
            base_mva = float(case.baseMVA)
            bus_acc = {int(b): [0.0, 0.0] for b in pack.tie_buses}
            def add_block(gb, sb):
                send, recv = gb.get('senders', []), gb.get('receivers', [])
                feats = sb.get('features', [])
                for i in range(min(len(send), len(recv), len(feats))):
                    u, v = int(send[i]), int(recv[i])
                    if pack.part_of_bus[u] == pack.part_of_bus[v]:
                        continue
                    pf,qf,pt,qt = float(feats[i][0]), float(feats[i][1]), float(feats[i][2]), float(feats[i][3])
                    if u in bus_acc:
                        bus_acc[u][0] += pf / base_mva ; bus_acc[u][1] += qf / base_mva
                    if v in bus_acc:
                        bus_acc[v][0] += pt / base_mva ; bus_acc[v][1] += qt / base_mva
            if 'ac_line' in edges_grid and 'ac_line' in edges_sol:
                add_block(edges_grid['ac_line'], edges_sol['ac_line'])
            # 纯变压器不作为联络线，Oracle聚合亦不计入跨区边界注入
            for b,(P,Q) in bus_acc.items():
                boundary_bus_pq[b] = (float(P), float(Q))
        print("使用Oracle母线边界注入（p.u.），切断联络线并在边界母线直接注入")
    else:
        raise ValueError("boundary 必须是 'predicted' 或 'oracle'")

    # 构建分区子网（切断联络线 + 边界母线直接注入）
    net_full = build_full_net_from_json(json_path)
    area_nets, bus_maps = build_area_nets_bus_injection(
        net_full, pack, boundary_bus_pq, case.baseMVA,
        enable_balancer=balancer,
        balancer_limit_mw=balancer_limit,
        balancer_cost_eur_per_mw2=balancer_cost,
        vm_boundary=vm_boundary,
        va_boundary=va_boundary
    )
    
    # 为兼容原有返回格式，构建ppidx映射
    ppidx = {}
    for area, net in area_nets.items():
        ppidx[area] = {}
        area_buses = [b for b, a in pack.part_of_bus.items() if a == area]
        for i, canon_bus in enumerate(area_buses):
            ppidx[area][canon_bus] = i

    # 传统独立区域OPF求解
    print("使用独立区域OPF求解")
    results = run_parallel_opf(area_nets, workers)
    ok = all(r[1] for r in results)
    total_cost = sum(r[2] for r in results if r[1])
    
    return ok, total_cost, sorted(results, key=lambda x: x[0])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="OPFData JSON 路径")
    ap.add_argument("--k", type=int, default=3, help="分区数")
    ap.add_argument("--seed", type=int, default=42, help="分区随机种子")
    ap.add_argument("--boundary", type=str, default="predicted", choices=["predicted","oracle"],
                    help="使用BC-GNN预测或使用真解边界(自检)")
    ap.add_argument("--workers", type=int, default=0, help="并行进程数，0=串行")
    ap.add_argument("--model", type=str, default=None, help="BC-GNN权重路径（如你的预测器需要）")

    # === 新增：软化 Oracle + 平衡源开关与强度 ===
    ap.add_argument("--oracle_tol", type=float, default=100.0, help="Oracle硬界容差MW（原20，建议≥100）")
    ap.add_argument("--soft_oracle", action="store_true", help="把Oracle硬界改为二次罚软约束")
    ap.add_argument("--no-soft_oracle", dest="soft_oracle", action="store_false")
    ap.set_defaults(soft_oracle=True)

    ap.add_argument("--balancer", action="store_true", help="启用区域应急平衡源（高成本sgen）")
    ap.add_argument("--no-balancer", dest="balancer", action="store_false")
    ap.set_defaults(balancer=True)
    ap.add_argument("--balancer_limit", type=float, default=2000.0, help="应急源功率上下界(MW)")
    ap.add_argument("--balancer_cost", type=float, default=2000.0, help="应急源线性成本(€/MW)")
    ap.add_argument("--oracle_anchor_w", type=float, default=0.05, help="Oracle软锚权重(€/MW^2)")

    # 已移除 ADMM 相关参数

    args = ap.parse_args()

    ok, total_cost, details = run_pipeline(
        args.json, args.k, args.seed,
        boundary=args.boundary, workers=args.workers, model_path=args.model,
        # === 传给主流程 ===
        oracle_tol=args.oracle_tol,
        soft_oracle=args.soft_oracle,
        balancer=args.balancer,
        balancer_limit=args.balancer_limit,
        balancer_cost=args.balancer_cost,
        oracle_anchor_w=args.oracle_anchor_w
    )
    print("PIPELINE_OK:", ok)
    print("TOTAL_COST:", total_cost)
    print("AREA_DETAILS:", details)

if __name__ == "__main__":
    main()
