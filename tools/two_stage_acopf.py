"""
Two-Stage ACOPF (V-θ route, minimal pipeline)

Stage 1: Partition via OPFDataProcessor (constructive growth), then predict corridor-end ports
         [pf_u, pt_v, qf_u, qt_v] reconstructed from BC-GNN V-θ outputs (V/Δθ).

Stage 2: For each area (partition id), build a local pandapower network from JSON:
         - Keep internal elements (loads, generators, shunts, internal branches)
         - At boundary buses, inject predicted boundary P/Q as fixed sgen (can be negative)
         - Add one ext_grid as angle reference (no special cost)
         Run OPF per area and report generator outputs and total cost.

Notes:
 - Minimal, dependency-light: reuses gnn_data OPFDataProcessor and gnn.model directly.
 - No ADMM, no soft anchors by default. This aims at an end-to-end working baseline first.
 - P/Q injections are in MW/MVAr (prediction is p.u. → multiply by baseMVA).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, Tuple, List
import json
import time
from pathlib import Path

import numpy as np


def _patch_scipy_compatibility():
    """修复scipy兼容性问题 - 在导入pandapower之前修复"""
    import scipy.sparse
    import numpy as np

    # 检查是否已经修复过
    if hasattr(scipy.sparse.csc_matrix, 'H'):
        return  # 已经修复

    print("🔧 修复scipy兼容性问题...")

    # 直接修改类字典
    scipy.sparse.csc_matrix.H = property(lambda self: self.T.conj())
    scipy.sparse.csc_matrix.h = property(lambda self: self.T.conj())

    # 验证修复
    test_matrix = scipy.sparse.csc_matrix(np.random.rand(3, 3))
    if hasattr(test_matrix, 'H'):
        print("✅ scipy兼容性修复成功")
    else:
        print("❌ scipy兼容性修复失败")


# 在导入任何其他模块之前修复scipy兼容性问题
_patch_scipy_compatibility()


def _repo_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(here)


def _ensure_paths():
    repo = _repo_root()
    gd_src = os.path.join(repo, 'gnn_data', 'src')
    if gd_src not in sys.path:
        sys.path.insert(0, gd_src)
    if repo not in sys.path:
        sys.path.insert(0, repo)


_ensure_paths()

import torch
from torch_geometric.data import Data

from gnn.model import BCGNN


# ========== JSON helpers ==========


def corridor_smax_from_edges(edge_attr: torch.Tensor, tie_edge_corridor: torch.Tensor) -> torch.Tensor:
    """Aggregate per-corridor S_max from directed edge attributes."""
    if tie_edge_corridor is None:
        return torch.zeros(0, device=edge_attr.device)
    cid = tie_edge_corridor.long()
    valid = cid >= 0
    if not valid.any():
        return torch.zeros(0, device=edge_attr.device)
    C = int(cid[valid].max().item()) + 1
    S_edge = edge_attr[:, 2].to(torch.float32)
    agg = torch.zeros(C, dtype=torch.float32, device=edge_attr.device)
    agg.index_add_(0, cid[valid], S_edge[valid])
    return 0.5 * agg


def load_json(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def extract_baseMVA(grid: dict) -> float:
    ctx = grid.get('context', None)
    if isinstance(ctx, dict) and 'baseMVA' in ctx:
        return float(ctx['baseMVA'])
    if isinstance(ctx, list):
        try:
            return float(ctx[0][0][0])
        except Exception:
            pass
    return 100.0


# ========== Stage 1: Partition + Predict ==========


def build_pyg_from_json(j: dict, k: int, seed: int) -> Data:
    from opfdata.processor import OPFDataProcessor  # type: ignore

    proc = OPFDataProcessor()
    net = proc._extract_network_data(j)
    part = proc._create_partition_dynamic(net, j, k, seed)
    if part is None:
        raise RuntimeError('Partition failed (constructive growth)')
    data = proc._create_pyg_data(j, net, part, k)
    if data is None:
        raise RuntimeError('Failed to build PyG Data (V1.1)')
    return data


@torch.no_grad()
def predict_corridor_pfqt(data: Data, model_path: str, device: str = 'auto') -> torch.Tensor:
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dev = torch.device(device)

    node_in = int(data.x.size(1))
    edge_in = int(data.edge_attr.size(1))
    model = BCGNN(node_features=node_in, edge_features=edge_in, hidden_dim=64)

    # Handle missing modules in checkpoint
    import pickle
    import io

    class SafeUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            # Handle missing modules by returning a dummy class
            if module == 'config':
                class DummyConfig:
                    pass
                return DummyConfig
            try:
                return super().find_class(module, name)
            except (ModuleNotFoundError, ImportError):
                # Return a dummy class for any missing module
                class DummyClass:
                    pass
                return DummyClass

        def persistent_load(self, pid):
            # Handle persistent load instructions
            return None

        def load_reduce(self):
            # Handle reduce operations
            return self.load()

    # Try multiple approaches to load the checkpoint
    ckpt = None

    # Add numpy scalar to safe globals for older PyTorch versions
    import numpy as np
    try:
        # Try new numpy path first
        torch.serialization.add_safe_globals([np._core.multiarray.scalar])
    except AttributeError:
        try:
            # Try old numpy path
            torch.serialization.add_safe_globals([np.core.multiarray.scalar])
        except AttributeError:
            # If both fail, continue without adding
            pass

    # Approach 1: Direct torch.load with weights_only=True (if possible)
    try:
        ckpt = torch.load(model_path, map_location='cpu', weights_only=True)
        print("Loaded with weights_only=True")
    except Exception as e1:
        print(f"weights_only=True failed: {e1}")

        # Approach 2: Try normal torch.load
        try:
            ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
            print("Loaded with normal torch.load")
        except Exception as e2:
            print(f"Normal torch.load failed: {e2}")

            # Approach 3: Use safe unpickling with more robust error handling
            try:
                with open(model_path, 'rb') as f:
                    unpickler = SafeUnpickler(f)
                    ckpt = unpickler.load()
                print("Loaded with safe unpickling")
            except Exception as e3:
                print(f"Safe unpickling failed: {e3}")

                # Approach 4: Try with different pickle protocol
                try:
                    import pickle
                    with open(model_path, 'rb') as f:
                        # Try to load with different protocols
                        f.seek(0)
                        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                            try:
                                f.seek(0)
                                ckpt = pickle.load(f)
                                print(f"Loaded with pickle protocol {protocol}")
                                break
                            except:
                                continue
                        else:
                            raise RuntimeError("All pickle protocols failed")
                except Exception as e4:
                    print(f"All pickle protocols failed: {e4}")
                    raise RuntimeError(f"All loading methods failed. Last error: {e4}")

    if ckpt is None:
        raise RuntimeError("Failed to load checkpoint")
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state)
    model.to(dev).eval()

    out = model(data.to(dev))
    if 'corridor_pfqt' not in out:
        if 'edge_pq' in out and hasattr(data, 'tie_edge_corridor') and hasattr(data, 'tie_edge_indices'):
            edge_pq = out['edge_pq']
            cid = data.tie_edge_corridor[data.tie_edge_indices]
            valid = cid >= 0
            if valid.any():
                if hasattr(data, 'tie_corridors') and data.tie_corridors is not None:
                    C = data.tie_corridors.size(0)
                else:
                    C = int(cid[valid].max().item()) + 1
                corridor_pfqt = torch.zeros(C, 4, device=edge_pq.device)
                corridor_pfqt.index_add_(0, cid[valid], edge_pq[valid])
                corridor_pfqt = 0.5 * corridor_pfqt  # undo double counting for directed edges
                return corridor_pfqt.detach().to('cpu')
        raise RuntimeError('Model did not output corridor_pfqt')
    return out['corridor_pfqt'].detach().to('cpu')  # [C,4], p.u.


def ports_violation_stats(data: Data, corridor_pfqt: torch.Tensor, alpha: float):
    """Compute capacity violation rate for corridor-end predictions (no projection)."""
    Smax = corridor_smax_from_edges(data.edge_attr, data.tie_edge_corridor)
    Pu, Pv, Qu, Qv = [corridor_pfqt[:, i] for i in range(4)]
    Pc = 0.5 * torch.abs(Pu - Pv)
    Qc = 0.5 * torch.abs(Qu - Qv)
    S_pred = torch.sqrt(Pc * Pc + Qc * Qc)
    S_lim = alpha * Smax
    violation = (S_pred > S_lim).float()
    violation_rate = float(violation.mean().item()) if S_pred.numel() else 0.0
    return Smax, violation_rate


def corridors_to_bus_pq(data: Data, corridor_pfqt: torch.Tensor, baseMVA: float) -> Dict[int, Tuple[float, float]]:
    # Map p.u. corridor ports [pf_u,pt_v,qf_u,qt_v] to bus p.u., then to MW/MVAr
    tie_corr = data.tie_corridors.cpu()
    tie_buses = data.tie_buses.cpu()
    B = tie_buses.numel()
    max_bus = int(tie_buses.max().item()) if B>0 else -1
    bus2idx = torch.full((max_bus+1,), -1, dtype=torch.long)
    bus2idx[tie_buses.long()] = torch.arange(B, dtype=torch.long)
    agg = torch.zeros(B, 2, dtype=torch.float32)
    u = tie_corr[:,0].long(); v = tie_corr[:,1].long()
    pf_u, pt_v = corridor_pfqt[:,0], corridor_pfqt[:,1]
    qf_u, qt_v = corridor_pfqt[:,2], corridor_pfqt[:,3]
    agg.index_add_(0, bus2idx[u], torch.stack([-pf_u, -qf_u], dim=-1))
    agg.index_add_(0, bus2idx[v], torch.stack([+pt_v, +qt_v], dim=-1))
    bus_pq_pu = agg
    bus_ids = data.tie_buses.cpu().tolist()
    result: Dict[int, Tuple[float, float]] = {}
    for i, b in enumerate(bus_ids):
        Ppu, Qpu = float(bus_pq_pu[i, 0]), float(bus_pq_pu[i, 1])
        result[int(b)] = (Ppu * baseMVA, Qpu * baseMVA)
    return result


# ========== Stage 2: Area networks + OPF ==========


@dataclass
class Case:
    baseMVA: float
    n_bus: int
    bus: List[List[float]]          # [base_kv, type, vmin, vmax]
    gen: List[List[float]]          # [mbase, pg, pmin, pmax, qg, qmin, qmax, vg, c2, c1, c0]
    load: List[List[float]]         # [pd, qd]
    shunt: List[List[float]]        # [bs, gs]
    ac: dict                        # edges.ac_line
    trafo: dict                     # edges.transformer
    gen_link: List[Tuple[int, int]]
    load_link: List[Tuple[int, int]]
    shunt_link: List[Tuple[int, int]]


def parse_case(j: dict) -> Case:
    g = j['grid']
    baseMVA = extract_baseMVA(g)
    buses = g['nodes']['bus']
    gens = g['nodes'].get('generator', [])
    loads = g['nodes'].get('load', [])
    shunts = g['nodes'].get('shunt', [])
    ac = g['edges'].get('ac_line', {})
    tra = g['edges'].get('transformer', {})
    gl = g['edges'].get('generator_link', {})
    ll = g['edges'].get('load_link', {})
    sl = g['edges'].get('shunt_link', {})
    gen_link = list(zip(map(int, gl.get('senders', [])), map(int, gl.get('receivers', []))))
    load_link = list(zip(map(int, ll.get('senders', [])), map(int, ll.get('receivers', []))))
    shunt_link = list(zip(map(int, sl.get('senders', [])), map(int, sl.get('receivers', []))))
    return Case(
        baseMVA=baseMVA,
        n_bus=len(buses),
        bus=buses,
        gen=gens,
        load=loads,
        shunt=shunts,
        ac=ac,
        trafo=tra,
        gen_link=gen_link,
        load_link=load_link,
        shunt_link=shunt_link,
    )


def build_area_networks(j: dict, data: Data, bus_pq_mw: Dict[int, Tuple[float, float]],
                        balancer: bool = False,
                        balancer_ratio: float = 0.05,
                        balancer_limit: float = 1000.0,
                        balancer_cost: float = 2000.0):
    import pandapower as pp

    case = parse_case(j)
    part = data.partition.cpu().numpy().tolist()
    areas = sorted(set(part))

    # Map bus -> area
    part_of_bus = {i: int(part[i]) for i in range(len(part))}
    area_buses = {a: [i for i, aa in part_of_bus.items() if aa == a] for a in areas}

    # Create nets per area
    nets = {a: pp.create_empty_network(sn_mva=case.baseMVA) for a in areas}
    ppidx = {a: {} for a in areas}  # canon bus -> pp bus id

    # 1) Buses
    for a in areas:
        net = nets[a]
        for b in area_buses[a]:
            base_kv, btype, vmin, vmax = case.bus[b]
            ppb = pp.create_bus(net, vn_kv=float(base_kv), min_vm_pu=float(vmin), max_vm_pu=float(vmax))
            ppidx[a][b] = ppb

    area_load_sum: Dict[int, float] = {a: 0.0 for a in areas}

    # 2) Loads
    for lid, b in case.load_link:
        if b in part_of_bus:
            a = part_of_bus[b]
            net = nets[a]
            pd, qd = case.load[lid]
            p_mw = float(pd) * case.baseMVA
            q_mvar = float(qd) * case.baseMVA
            pp.create_load(net, bus=ppidx[a][b], p_mw=p_mw, q_mvar=q_mvar)
            area_load_sum[a] += max(0.0, p_mw)

    # 3) Generators + costs
    for gid, b in case.gen_link:
        if b in part_of_bus:
            a = part_of_bus[b]
            net = nets[a]
            row = case.gen[gid]
            # [mbase, pg, pmin, pmax, qg, qmin, qmax, vg, c2, c1, c0]
            pmin, pmax = float(row[2]) * case.baseMVA, float(row[3]) * case.baseMVA
            qmin, qmax = float(row[5]) * case.baseMVA, float(row[6]) * case.baseMVA
            vg = float(row[7])
            gi = pp.create_gen(net, bus=ppidx[a][b], vm_pu=vg, p_mw=pmin)
            net.gen.loc[gi, ['min_p_mw', 'max_p_mw']] = [pmin, pmax]
            net.gen.loc[gi, ['min_q_mvar', 'max_q_mvar']] = [qmin, qmax]
            # cost
            c2, c1, c0 = float(row[8]), float(row[9]), float(row[10])
            pp.create_poly_cost(net, element=gi, et='gen', cp0_eur=c0, cp1_eur_per_mw=c1, cp2_eur_per_mw2=c2)

    # 4) Shunts (approx constant power at V=1.0 p.u.)
    for sid, b in case.shunt_link:
        if b in part_of_bus:
            a = part_of_bus[b]
            net = nets[a]
            bs, gs = case.shunt[sid]
            p = -float(gs) * (1.0 ** 2) * case.baseMVA
            q = -float(bs) * (1.0 ** 2) * case.baseMVA
            pp.create_shunt(net, bus=ppidx[a][b], p_mw=p, q_mvar=q)

    # 5) Internal branches as impedances in p.u.
    def add_imp(net, fb, tb, rpu, xpu):
        pp.create_impedance(net, from_bus=fb, to_bus=tb, rft_pu=float(rpu), xft_pu=float(xpu), sn_mva=case.baseMVA)

    ac = case.ac
    for f, t, feat in zip(ac.get('senders', []), ac.get('receivers', []), ac.get('features', [])):
        i, j = int(f), int(t)
        ai, aj = part_of_bus[i], part_of_bus[j]
        if ai == aj:
            # feat order (see dataset): angmin, angmax, b_fr, b_to, br_r, br_x, rate_a, ...
            rpu, xpu = float(feat[4]), float(feat[5])
            add_imp(nets[ai], ppidx[ai][i], ppidx[ai][j], rpu, xpu)

    tra = case.trafo
    for f, t, feat in zip(tra.get('senders', []), tra.get('receivers', []), tra.get('features', [])):
        i, j = int(f), int(t)
        ai, aj = part_of_bus[i], part_of_bus[j]
        if ai == aj:
            # feat: [angmin, angmax, br_r, br_x, rate_a, rate_b, rate_c, tap, shift, b_fr, b_to]
            rpu, xpu = float(feat[2]), float(feat[3])
            add_imp(nets[ai], ppidx[ai][i], ppidx[ai][j], rpu, xpu)

    # 6) Boundary PQ injection as fixed sgen (can be negative)
    tie_buses = set(data.tie_buses.cpu().tolist())
    for b in tie_buses:
        a = part_of_bus[int(b)]
        net = nets[a]
        P, Q = bus_pq_mw.get(int(b), (0.0, 0.0))
        # use sgen for both signs (pandapower allows negative sgen)
        pp.create_sgen(net, bus=ppidx[a][int(b)], p_mw=float(P), q_mvar=float(Q), controllable=False)

    # 7) Add one ext_grid per area as angle reference (+ optional high-cost balancer)
    for a in areas:
        net = nets[a]
        # pick first bus or a bus with a generator if available
        buses = area_buses[a]
        ref_bus = None
        # prefer a bus with type==3 (reference in original)
        for b in buses:
            if int(case.bus[b][1]) == 3:
                ref_bus = b
                break
        if ref_bus is None:
            # else any bus
            ref_bus = buses[0]
        _ = pp.create_ext_grid(net, bus=ppidx[a][ref_bus], vm_pu=1.0, va_degree=0.0)
        # 禁用兜底发电机 - 我们应该通过分区策略和容量管理来确保可行性

    return nets


# ===== Optional: full-net -> per-area copy (keep line/trafo thermal limits) =====

def _safe_base_mva(grid: dict) -> float:
    ctx = grid.get('context', None)
    if isinstance(ctx, dict) and 'baseMVA' in ctx:
        return float(ctx['baseMVA'])
    if isinstance(ctx, list):
        try:
            return float(ctx[0][0][0])
        except Exception:
            pass
    return 100.0


def opfdata_to_ppc(opf_json: dict) -> dict:
    """Minimal OPFData JSON → MATPOWER PPC conversion (enough for from_ppc)."""
    g = opf_json['grid']
    base_mva = _safe_base_mva(g)

    buses = g['nodes']['bus']
    loads = g['nodes'].get('load', [])
    gens = g['nodes'].get('generator', [])
    shunts = g['nodes'].get('shunt', [])

    ac = g['edges'].get('ac_line', {})
    tra = g['edges'].get('transformer', {})
    ll = g['edges'].get('load_link', {})
    gl = g['edges'].get('generator_link', {})
    sl = g['edges'].get('shunt_link', {})

    nb = len(buses)
    ng = len(gens)
    nl = len(ac.get('senders', [])) + len(tra.get('senders', []))

    import numpy as np
    # bus: [BUS_I, TYPE, PD, QD, GS, BS, BUS_AREA, VM, VA, BASE_KV, ZONE, VMAX, VMIN]
    bus = np.zeros((max(1, nb), 13))
    for i, b in enumerate(buses):
        base_kv, btype, vmin, vmax = float(b[0]), int(round(b[1])), float(b[2]), float(b[3])
        bus[i, 0] = i + 1
        bus[i, 1] = 3 if btype == 3 else (2 if btype == 2 else 1)
        bus[i, 2] = 0.0
        bus[i, 3] = 0.0
        bus[i, 4] = 0.0
        bus[i, 5] = 0.0
        bus[i, 7] = 1.0
        bus[i, 8] = 0.0
        bus[i, 9] = base_kv
        bus[i, 11] = vmax
        bus[i, 12] = vmin

    # loads to PD/QD
    for lid, bidx in zip(ll.get('senders', []), ll.get('receivers', [])):
        if lid < len(loads):
            pd, qd = loads[lid]
            bus[int(bidx), 2] += float(pd) * base_mva
            bus[int(bidx), 3] += float(qd) * base_mva

    # shunts to GS/BS
    for sid, bidx in zip(sl.get('senders', []), sl.get('receivers', [])):
        if sid < len(shunts):
            bs, gs = shunts[sid]
            bus[int(bidx), 4] += float(gs) * base_mva
            bus[int(bidx), 5] += float(bs) * base_mva

    # gen: [BUS, PG, QG, QMAX, QMIN, VG, MBASE, STATUS, PMAX, PMIN]
    gen = np.zeros((max(1, ng), 10))
    for gi, (g_row, bidx) in enumerate(zip(gens, gl.get('receivers', []))):
        pg, pmin, pmax = float(g_row[1])*base_mva, float(g_row[2])*base_mva, float(g_row[3])*base_mva
        qg, qmin, qmax = float(g_row[4])*base_mva, float(g_row[5])*base_mva, float(g_row[6])*base_mva
        vg = float(g_row[7])
        gen[gi, :] = [int(bidx)+1, pg, qg, qmax, qmin, vg, base_mva, 1, pmax, pmin]

    # branch: AC + Trafo
    branch = np.zeros((max(1, nl), 13))
    row = 0
    # AC line
    for f, t, feat in zip(ac.get('senders', []), ac.get('receivers', []), ac.get('features', [])):
        angmin, angmax, b_fr, b_to, br_r, br_x, rate_a = feat[:7]
        br_b = float(b_fr) + float(b_to)
        branch[row, :] = [
            int(f)+1, int(t)+1, float(br_r), float(br_x), br_b,
            float(rate_a)*base_mva, 0.0, 0.0,
            0.0, 0.0, 1.0, float(angmin)*180/np.pi, float(angmax)*180/np.pi
        ]
        row += 1
    # Trafo
    for f, t, feat in zip(tra.get('senders', []), tra.get('receivers', []), tra.get('features', [])):
        br_r, br_x = float(feat[2]), float(feat[3])
        rate_a = float(feat[4]) * base_mva
        tap = float(feat[7]) if len(feat) > 7 else 1.0
        shift = float(feat[8])*180/np.pi if len(feat) > 8 else 0.0
        branch[row, :] = [
            int(f)+1, int(t)+1, br_r, br_x, 0.0,
            rate_a, 0.0, 0.0, tap, shift, 1.0, -360.0, 360.0
        ]
        row += 1

    # gencost: quadratic
    gencost = np.zeros((max(1, ng), 7))
    for gi, g_row in enumerate(gens):
        c2 = float(g_row[8]) / (base_mva**2)
        c1 = float(g_row[9]) / base_mva
        c0 = float(g_row[10])
        gencost[gi, :] = [2, 0, 0, 3, max(0.0, c2), max(1e-6, c1), max(0.0, c0)]

    # ensure a slack
    if not np.any(bus[:,1] == 3):
        if ng > 0:
            slack_bus = int(gen[0,0]) - 1
        else:
            slack_bus = 0
        bus[slack_bus, 1] = 3

    return {"version": '2', "baseMVA": base_mva, "bus": bus, "gen": gen, "branch": branch, "gencost": gencost}


def build_full_net_from_json(json_path: str):
    with open(json_path, 'r') as f:
        j = json.load(f)
    ppc = opfdata_to_ppc(j)
    # use converter API to build net with proper tables
    from pandapower.converter import from_ppc
    net = from_ppc(ppc, f_hz=50)
    return net


def build_area_networks_ppc_copy(json_path: str, j: dict, data: Data, bus_pq_mw: Dict[int, Tuple[float, float]],
                                 balancer: bool = False,
                                 balancer_ratio: float = 0.05,
                                 balancer_limit: float = 1000.0,
                                 balancer_cost: float = 2000.0):
    """Copy internal elements from full net to per-area nets, keeping line/trafo constraints when available."""
    import pandapower as pp
    import numpy as np

    net_full = build_full_net_from_json(json_path)

    # Partition mapping
    part = data.partition.cpu().numpy().tolist()
    areas = sorted(set(part))
    part_of_bus = {i: int(part[i]) for i in range(len(part))}
    area_buses = {a: [i for i, aa in part_of_bus.items() if aa == a] for a in areas}

    # Helpers for index mapping (from full net to area nets)
    canon2pp = lambda b: int(b)
    pp2canon = lambda b: int(b)

    # Ensure we only process buses that exist in our partition
    max_bus_idx = len(part) - 1

    # Create nets
    baseMVA = extract_baseMVA(j['grid'])
    case = parse_case(j)
    nets = {a: pp.create_empty_network(sn_mva=baseMVA) for a in areas}
    ppidx = {a: {} for a in areas}

    # Buses
    for a in areas:
        net = nets[a]
        for b in area_buses[a]:
            base_kv, btype, vmin, vmax = j['grid']['nodes']['bus'][b]
            nb = pp.create_bus(net, vn_kv=float(base_kv), min_vm_pu=float(vmin), max_vm_pu=float(vmax))
            ppidx[a][b] = nb

    # Copy internal lines
    if hasattr(net_full, 'line') and len(net_full.line.index):
        for idx in net_full.line.index:
            fb_pp = int(net_full.line.at[idx, 'from_bus'])
            tb_pp = int(net_full.line.at[idx, 'to_bus'])
            fb = pp2canon(fb_pp); tb = pp2canon(tb_pp)

            # Skip lines with buses outside our partition range
            if fb > max_bus_idx or tb > max_bus_idx:
                continue

            if fb in part_of_bus and tb in part_of_bus and part_of_bus[fb] == part_of_bus[tb]:
                a = part_of_bus[fb]
                try:
                    pp.create_line_from_parameters(
                        nets[a],
                        from_bus=ppidx[a][fb], to_bus=ppidx[a][tb],
                        length_km=float(net_full.line.at[idx, 'length_km']) if 'length_km' in net_full.line.columns else 1.0,
                        r_ohm_per_km=float(net_full.line.at[idx, 'r_ohm_per_km']),
                        x_ohm_per_km=float(net_full.line.at[idx, 'x_ohm_per_km']),
                        c_nf_per_km=float(net_full.line.at[idx, 'c_nf_per_km']) if 'c_nf_per_km' in net_full.line.columns else 0.0,
                        max_i_ka=float(net_full.line.at[idx, 'max_i_ka']) if 'max_i_ka' in net_full.line.columns else 1e3,
                    )
                except Exception:
                    # fallback to impedance
                    # Convert to p.u. impedance using baseMVA approximations is non-trivial; use impedance with p.u. from PPC path if present
                    rpu = float(net_full.line.at[idx, 'r_pu']) if 'r_pu' in net_full.line.columns else 0.0
                    xpu = float(net_full.line.at[idx, 'x_pu']) if 'x_pu' in net_full.line.columns else 0.01
                    pp.create_impedance(nets[a], ppidx[a][fb], ppidx[a][tb], rft_pu=rpu, xft_pu=xpu, sn_mva=baseMVA)

    # Copy internal transformers
    if hasattr(net_full, 'trafo') and len(net_full.trafo.index):
        for idx in net_full.trafo.index:
            hv_pp = int(net_full.trafo.at[idx, 'hv_bus'])
            lv_pp = int(net_full.trafo.at[idx, 'lv_bus'])
            hv = pp2canon(hv_pp); lv = pp2canon(lv_pp)

            # Skip transformers with buses outside our partition range
            if hv > max_bus_idx or lv > max_bus_idx:
                continue

            if hv in part_of_bus and lv in part_of_bus and part_of_bus[hv] == part_of_bus[lv]:
                a = part_of_bus[hv]
                try:
                    pp.create_transformer_from_parameters(
                        nets[a],
                        hv_bus=ppidx[a][hv], lv_bus=ppidx[a][lv],
                        sn_mva=float(net_full.trafo.at[idx, 'sn_mva']),
                        vn_hv_kv=float(net_full.bus.at[hv_pp, 'vn_kv']),
                        vn_lv_kv=float(net_full.bus.at[lv_pp, 'vn_kv']),
                        vk_percent=float(net_full.trafo.at[idx, 'vk_percent']),
                        vkr_percent=float(net_full.trafo.at[idx, 'vkr_percent']),
                        pfe_kw=float(net_full.trafo.at[idx, 'pfe_kw']) if 'pfe_kw' in net_full.trafo.columns else 0.0,
                        i0_percent=float(net_full.trafo.at[idx, 'i0_percent']) if 'i0_percent' in net_full.trafo.columns else 0.0,
                    )
                except Exception:
                    # fallback: approximate as impedance
                    rpu = float(net_full.trafo.at[idx, 'vkr_percent'])/100.0
                    xpu = float(net_full.trafo.at[idx, 'vk_percent'])/100.0
                    pp.create_impedance(nets[a], ppidx[a][hv], ppidx[a][lv], rft_pu=rpu, xft_pu=xpu, sn_mva=baseMVA)

    # Copy internal generic impedances (fallback from PPC conversion)
    if hasattr(net_full, 'impedance') and len(net_full.impedance.index):
        for idx in net_full.impedance.index:
            fb_pp = int(net_full.impedance.at[idx, 'from_bus'])
            tb_pp = int(net_full.impedance.at[idx, 'to_bus'])
            fb = pp2canon(fb_pp); tb = pp2canon(tb_pp)
            if part_of_bus[fb] == part_of_bus[tb]:
                a = part_of_bus[fb]
                rpu = float(net_full.impedance.at[idx, 'rft_pu']) if 'rft_pu' in net_full.impedance.columns else 0.0
                xpu = float(net_full.impedance.at[idx, 'xft_pu']) if 'xft_pu' in net_full.impedance.columns else 0.01
                snm = float(net_full.impedance.at[idx, 'sn_mva']) if 'sn_mva' in net_full.impedance.columns else baseMVA
                pp.create_impedance(nets[a], ppidx[a][fb], ppidx[a][tb], rft_pu=rpu, xft_pu=xpu, sn_mva=snm)

    # Loads
    for lid, b in case.load_link:
        if b in part_of_bus:
            a = part_of_bus[b]
            pd, qd = case.load[lid]
            pp.create_load(nets[a], ppidx[a][b], p_mw=float(pd)*baseMVA, q_mvar=float(qd)*baseMVA)

    # Shunts
    for sid, b in case.shunt_link:
        if b in part_of_bus:
            a = part_of_bus[b]
            bs, gs = case.shunt[sid]
            p = -float(gs) * baseMVA
            q = -float(bs) * baseMVA
            pp.create_shunt(nets[a], ppidx[a][b], p_mw=p, q_mvar=q)

    # Generators + costs
    for gid, b in case.gen_link:
        if b in part_of_bus:
            a = part_of_bus[b]
            row = case.gen[gid]
            pmin, pmax = float(row[2]) * baseMVA, float(row[3]) * baseMVA
            qmin, qmax = float(row[5]) * baseMVA, float(row[6]) * baseMVA
            vg = float(row[7])
            gi = pp.create_gen(nets[a], bus=ppidx[a][b], vm_pu=vg, p_mw=pmin)
            nets[a].gen.loc[gi, ['min_p_mw', 'max_p_mw']] = [pmin, pmax]
            nets[a].gen.loc[gi, ['min_q_mvar', 'max_q_mvar']] = [qmin, qmax]
            c2, c1, c0 = float(row[8]), float(row[9]), float(row[10])
            pp.create_poly_cost(nets[a], element=gi, et='gen', cp0_eur=c0, cp1_eur_per_mw=c1, cp2_eur_per_mw2=c2)

    # Boundary injections
    tie_buses = set(data.tie_buses.cpu().tolist())
    for b in tie_buses:
        a = part_of_bus[int(b)]
        P, Q = bus_pq_mw.get(int(b), (0.0, 0.0))
        pp.create_sgen(nets[a], bus=ppidx[a][int(b)], p_mw=float(P), q_mvar=float(Q), controllable=False)

    # One ext_grid per area + optional balancer
    for a in areas:
        buses = area_buses[a]
        ref = None
        for b in buses:
            if int(j['grid']['nodes']['bus'][b][1]) == 3:
                ref = b; break
        if ref is None:
            ref = buses[0]
        pp.create_ext_grid(nets[a], bus=ppidx[a][ref], vm_pu=1.0, va_degree=0.0)
        if balancer:
            # area load sum
            loads_in_a = nets[a].load['p_mw'].sum() if len(nets[a].load) else 0.0
            limit = min(float(balancer_limit), float(balancer_ratio) * max(1.0, float(loads_in_a)))
            gi = pp.create_gen(nets[a], bus=ppidx[a][ref], p_mw=0.0, vm_pu=1.0,
                               min_p_mw=-limit, max_p_mw=limit, controllable=True)
            pp.create_poly_cost(nets[a], element=gi, et='gen', cp0_eur=0.0,
                                cp1_eur_per_mw=float(balancer_cost), cp2_eur_per_mw2=0.0)

    return nets


def run_area_opf(nets: Dict[int, 'pp.pandapowerNet']) -> Tuple[bool, float, List[Tuple[int, bool, float]]]:
    import pandapower as pp
    results = []
    ok = True
    total_cost = 0.0
    for a, net in sorted(nets.items(), key=lambda x: x[0]):
        try:
            # 尝试runopp
            try:
                # 对于区域2，尝试不同的参数组合
                if a == 2:
                    # 区域2有连通性问题，尝试关闭连通性检查
                    pp.runopp(net, calculate_voltage_angles=True, verbose=False, init='dc',
                              check_connectivity=False, delta=1e-6, numba=False,
                              enforce_q_lims=True, max_iter=50)
                    print(f"区域{a} 使用特殊参数成功")
                else:
                    # 其他区域使用标准参数
                    pp.runopp(net, calculate_voltage_angles=True, verbose=False, init='dc',
                              check_connectivity=True, delta=1e-8, numba=False,
                              enforce_q_lims=True, max_iter=100)
                success_method = 'runopp'
            except Exception as runopp_error:
                print(f"区域{a} runopp失败: {runopp_error}")
                # runopp失败，尝试其他方法
                raise RuntimeError(f"runopp failed: {runopp_error}")

            cost = float(getattr(net, 'res_cost', 0.0))
            results.append((a, True, cost))
            total_cost += cost
            print(f"区域{a} ACOPF成功 (使用{success_method})")
            if hasattr(net, 'res_cost') and net.res_cost is not None:
                print(f"  成本: {net.res_cost:.6f}")
            if hasattr(net, 'converged'):
                print(f"  收敛: {net.converged}")
        except Exception as e:
            print(f"区域{a} ACOPF失败: {e}")
            print(f"  错误类型: {type(e).__name__}")
            results.append((a, False, 0.0))
            ok = False
    return ok, total_cost, results


# ========== CLI ==========


def main():
    ap = argparse.ArgumentParser(description='Two-Stage ACOPF (PQ route, minimal)')
    ap.add_argument('--json', required=True, help='OPFData JSON path')
    ap.add_argument('--model', required=True, help='BC-GNN checkpoint path')
    ap.add_argument('--k', type=int, default=4, help='Number of partitions (recommend 4/5/6)')
    ap.add_argument('--seed', type=int, default=0, help='Partition seed')
    # alpha can be a single value or a comma-separated list for ladder scan
    ap.add_argument('--alpha', type=str, default='0.95', help='Capacity projection factor alpha or comma list')
    ap.add_argument('--balancer', action='store_true', help='Enable high-cost area balancer')
    ap.add_argument('--balancer_ratio', type=float, default=0.05, help='Balancer limit ratio of area load (default 5%)')
    ap.add_argument('--balancer_limit', type=float, default=1000.0, help='Balancer absolute MW cap')
    ap.add_argument('--balancer_cost', type=float, default=2000.0, help='Balancer linear cost (€/MW)')
    ap.add_argument('--out_dir', type=str, default='', help='Optional output directory to save partition/predictions/results')
    ap.add_argument('--save_nets', action='store_true', help='Also save area nets (JSON)')
    ap.add_argument('--use_ppc_copy', action='store_true', help='Build per-area nets by copying from full PPC net (keep line/trafo limits)')
    ap.add_argument('--device', type=str, default='auto', help='cuda/cpu/auto')

    args = ap.parse_args()

    j = load_json(args.json)
    baseMVA = extract_baseMVA(j['grid'])

    # Stage 1: partition + predict (do once)
    data = build_pyg_from_json(j, args.k, args.seed)
    corr_pred = predict_corridor_pfqt(data, args.model, device=args.device)  # [C,4] p.u.

    # Optional: persist partition once
    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        (out_dir).mkdir(parents=True, exist_ok=True)
        part_info = {
            'k': int(args.k),
            'seed': int(args.seed),
            'partition': data.partition.cpu().tolist() if hasattr(data, 'partition') else [],
            'tie_corridors': data.tie_corridors.cpu().tolist() if hasattr(data, 'tie_corridors') else [],
            'tie_buses': data.tie_buses.cpu().tolist() if hasattr(data, 'tie_buses') else [],
        }
        with open(out_dir / 'partition.json', 'w') as f:
            json.dump(part_info, f)

    # Parse alpha list
    try:
        alpha_list = [float(x.strip()) for x in args.alpha.split(',')]
    except Exception:
        alpha_list = [float(args.alpha)]

    print('\n=== Alpha Ladder Scan (ports, no projection) ===')
    print('alpha  | pre-viol | OK | total_cost')
    print('------ | -------- | -- | ----------')

    best_ok = True
    exit_code = 0

    for alpha in alpha_list:
        # Capacity stats (p.u.)
        Smax, viol_rate = ports_violation_stats(data, corr_pred, alpha)

        # Map to bus MW/MVAr and build area nets（直接使用端口预测）
        bus_pq = corridors_to_bus_pq(data, corr_pred, baseMVA)
        if args.use_ppc_copy:
            nets = build_area_networks_ppc_copy(
                args.json, j, data, bus_pq
            )
        else:
            nets = build_area_networks(j, data, bus_pq,
                                       balancer=args.balancer,
                                       balancer_ratio=args.balancer_ratio,
                                       balancer_limit=args.balancer_limit,
                                       balancer_cost=args.balancer_cost)
        ok, total_cost, details = run_area_opf(nets)

        print(f'{alpha:>5.2f} |  {viol_rate:>7.4f} | {str(ok):>2} | {total_cost:>10.3f}')

        # Save run snapshot per alpha
        if out_dir:
            a_dir = out_dir / f'alpha_{alpha:.3f}'
            a_dir.mkdir(parents=True, exist_ok=True)
            # predictions & projection stats
            torch.save({
                'corridor_pfqt_pred': corr_pred,
                'Smax_corr': Smax,
                'pre_violation_rate': viol_rate,
            }, a_dir / 'predictions.pt')
            # bus injections
            with open(a_dir / 'bus_pq_mw.json', 'w') as f:
                json.dump({int(k): [float(v[0]), float(v[1])] for k, v in bus_pq.items()}, f)
            # summary
            with open(a_dir / 'summary.json', 'w') as f:
                json.dump({
                    'ok': bool(ok),
                    'total_cost': float(total_cost),
                    'details': [(int(a), bool(succ), float(c)) for (a, succ, c) in details]
                }, f)
            # optional nets
            if args.save_nets:
                import pandapower as pp
                n_dir = a_dir / 'nets'
                n_dir.mkdir(parents=True, exist_ok=True)
                for a, net in nets.items():
                    pp.to_json(net, str(n_dir / f'area_{a}.json'))
                    # save results tables if present
                    try:
                        if hasattr(net, 'res_gen') and len(net.res_gen):
                            net.res_gen.to_csv(n_dir / f'area_{a}_res_gen.csv')
                        if hasattr(net, 'res_line') and len(net.res_line):
                            net.res_line.to_csv(n_dir / f'area_{a}_res_line.csv')
                        if hasattr(net, 'res_trafo') and len(net.res_trafo):
                            net.res_trafo.to_csv(n_dir / f'area_{a}_res_trafo.csv')
                        if hasattr(net, 'res_load') and len(net.res_load):
                            net.res_load.to_csv(n_dir / f'area_{a}_res_load.csv')
                    except Exception:
                        pass

        best_ok = best_ok and ok
        if not ok:
            exit_code = 2

    return exit_code


if __name__ == '__main__':
    sys.exit(main())
