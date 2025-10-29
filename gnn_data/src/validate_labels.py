#!/usr/bin/env python3
"""
Validate processed V1.1 PQ-only samples for label and capacity consistency.

Checks per-sample:
- Corridor->Bus consistency: sum corridor (u->v) contributions to buses equals y_bus_pq
- Capacity consistency: precomputed Smax_corr equals aggregation from edge_attr/tie_edge_corridor
- Optional raw JSON recompute: rebuild y_corridor_pq/y_bus_pq from solution pf/qf/pt/qt

Usage:
  python gnn_data/src/validate_labels.py --processed-dir DATA_DIR [--raw-dir RAW_JSON_DIR] [--max-samples 50] [--atol 1e-6]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
try:
    from tqdm import tqdm
except Exception:  # tqdm is optional; fall back to plain loop
    tqdm = None


def pfqt_to_bus_mapping(corridor_pfqt: torch.Tensor,
                        tie_corridors: torch.Tensor,
                        tie_buses: torch.Tensor,
                        device: torch.device) -> torch.Tensor:
    """Map corridor-end [pf_u,pt_v,qf_u,qt_v] to bus boundary injections (P_bd,Q_bd)."""
    B = tie_buses.numel()
    bus_pq = torch.zeros(B, 2, dtype=corridor_pfqt.dtype, device=device)
    max_bus = int(tie_buses.max().item())
    bus2idx = torch.full((max_bus + 1,), -1, dtype=torch.long, device=device)
    bus2idx[tie_buses.long()] = torch.arange(B, device=device)
    u = tie_corridors[:, 0].long().to(device)
    v = tie_corridors[:, 1].long().to(device)
    u_idx = bus2idx[u]
    v_idx = bus2idx[v]
    pf_u = corridor_pfqt[:, 0].to(device)
    pt_v = corridor_pfqt[:, 1].to(device)
    qf_u = corridor_pfqt[:, 2].to(device)
    qt_v = corridor_pfqt[:, 3].to(device)
    bus_pq.index_add_(0, u_idx, torch.stack([-pf_u, -qf_u], dim=1))
    bus_pq.index_add_(0, v_idx, torch.stack([pt_v, qt_v], dim=1))
    return bus_pq


def corridor_Smax_from_edges(edge_attr: torch.Tensor,
                             tie_edge_corridor: torch.Tensor,
                             C_hint: int | None = None) -> torch.Tensor:
    """Aggregate corridor S_max from edge attributes; divide by 2 to undo directed duplication."""
    cid = tie_edge_corridor.long()
    valid = cid >= 0
    if C_hint is None:
        C = int(cid[valid].max().item()) + 1 if valid.any() else 0
    else:
        C = int(C_hint)
    S_e = edge_attr[:, 2].to(torch.float32)
    Scorr = torch.zeros(C, dtype=torch.float32, device=edge_attr.device)
    if valid.any():
        Scorr.index_add_(0, cid[valid], S_e[valid])
    return 0.5 * Scorr


def glob_pt_files(processed_dir: Path, max_files: int | None) -> List[Path]:
    files: List[Path] = []
    # Collect .pt under processed_dir and its common children (train/val/test)
    for pattern in ["*.pt", "train/*.pt", "val/*.pt", "test/*.pt"]:
        files.extend(sorted(processed_dir.glob(pattern)))
    # De-duplicate while keeping order
    seen = set()
    unique = []
    for p in files:
        if p in seen:
            continue
        seen.add(p)
        unique.append(p)
        if max_files and len(unique) >= max_files:
            break
    return unique


def load_samples(pt_path: Path) -> List[torch.Tensor]:
    # PyTorch 2.6+: default weights_only=True breaks loading custom classes (PyG Data)
    # We generated these files locally, so it's safe to disable weights_only.
    obj = torch.load(pt_path, map_location='cpu', weights_only=False)
    if isinstance(obj, list):
        return obj
    return [obj]


def recompute_from_raw(json_path: Path,
                       tie_corridors: torch.Tensor,
                       tie_buses: torch.Tensor,
                       base_mva: float | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
    import json
    import numpy as np
    with open(json_path, 'r') as f:
        j = json.load(f)
    grid = j.get('grid', {})
    sol = j.get('solution', {})
    edges_grid = grid.get('edges', {})
    edges_sol = sol.get('edges', {})
    # baseMVA
    if base_mva is None:
        ctx = grid.get('context')
        if isinstance(ctx, dict) and 'baseMVA' in ctx:
            base_mva = float(ctx['baseMVA'])
        else:
            base_mva = 100.0
    # Corridor map
    corridors = tie_corridors.cpu().numpy()
    corr_map = {(int(u), int(v)): i for i, (u, v) in enumerate(corridors)}
    C = corridors.shape[0]
    y_corr = np.zeros((C, 4), dtype=float)
    # Helper to add a block
    def add_block(block_grid, block_sol):
        senders = block_grid.get('senders', [])
        receivers = block_grid.get('receivers', [])
        feats = block_sol.get('features', [])
        feats = np.asarray(feats, dtype=float)
        for i in range(len(senders)):
            u = int(senders[i]); v = int(receivers[i])
            a, b = (u, v) if u < v else (v, u)
            key = (a, b)
            if key not in corr_map:
                continue
            cid = corr_map[key]
            pf=qf=pt=qt=0.0
            if i < len(feats) and feats.shape[1] >= 4:
                pf, qf, pt, qt = feats[i, 0], feats[i, 1], feats[i, 2], feats[i, 3]
            # OPFData solution.edges.* 已为 p.u.，无需再除 baseMVA
            if (u == a and v == b):
                y_corr[cid, 0] += pf  # pf_u
                y_corr[cid, 1] += pt  # pt_v
                y_corr[cid, 2] += qf  # qf_u
                y_corr[cid, 3] += qt  # qt_v
            else:
                y_corr[cid, 0] += pt
                y_corr[cid, 1] += pf
                y_corr[cid, 2] += qt
                y_corr[cid, 3] += qf
    # AC and Trafo blocks
    add_block(edges_grid.get('ac_line', {}), edges_sol.get('ac_line', {}))
    add_block(edges_grid.get('transformer', {}), edges_sol.get('transformer', {}))
    # 由端口映射得到母线注入
    tb = tie_buses.cpu().numpy().astype(int)
    y_bus = np.zeros((tb.size, 2), dtype=float)
    bus2idx = {int(b): i for i, b in enumerate(tb.tolist())}
    for (u, v), (pf_u, pt_v, qf_u, qt_v) in zip(corridors, y_corr):
        iu = bus2idx.get(int(u), -1)
        iv = bus2idx.get(int(v), -1)
        if iu >= 0:
            y_bus[iu, 0] -= float(pf_u)
            y_bus[iu, 1] -= float(qf_u)
        if iv >= 0:
            y_bus[iv, 0] += float(pt_v)
            y_bus[iv, 1] += float(qt_v)
    return torch.tensor(y_corr, dtype=torch.float32), torch.tensor(y_bus, dtype=torch.float32)


def loads_from_raw(json_path: Path) -> torch.Tensor:
    """Aggregate bus loads (P,Q) from raw JSON's load_link.
    Returns [N,2] in p.u., ordered by bus index.
    """
    import json
    with open(json_path, 'r') as f:
        j = json.load(f)
    buses = j['grid']['nodes']['bus']
    loads = j['grid']['nodes']['load']
    ll = j['grid']['edges']['load_link']
    import numpy as np
    N = len(buses)
    pq = np.zeros((N, 2), dtype=float)
    for li, bi in zip(ll['senders'], ll['receivers']):
        p, q = loads[li]
        pq[int(bi), 0] += float(p)
        pq[int(bi), 1] += float(q)
    return torch.tensor(pq, dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--processed-dir', required=True, help='Processed dir (contains .pt or split subdirs)')
    ap.add_argument('--raw-dir', default=None, help='Raw JSON dir to recompute labels (optional)')
    ap.add_argument('--max-samples', type=int, default=50)
    ap.add_argument('--atol', type=float, default=1e-6)
    ap.add_argument('--no-progress', action='store_true', help='Disable progress bar')
    args = ap.parse_args()

    processed_dir = Path(args.processed_dir)
    raw_dir = Path(args.raw_dir) if args.raw_dir else None
    files = glob_pt_files(processed_dir, args.max_samples)
    if not files:
        print('No .pt files found in', processed_dir)
        return 2

    n_ok = 0
    n_total = 0
    max_bus_diff = 0.0
    max_cap_diff = 0.0
    max_bus_net = 0.0
    zero_corr_total = 0
    zero_corr_samples = 0
    pure_trafo_corr_samples = 0
    json_checks = 0
    json_bus_max = 0.0
    load_max = 0.0
    pair_incons = 0
    json_corr_max = 0.0

    iterator = files
    use_pbar = (tqdm is not None) and (not args.no_progress)
    if use_pbar:
        iterator = tqdm(files, desc='Validating', unit='file')

    for pt in iterator:
        samples = load_samples(pt)
        for data in samples:
            n_total += 1
            device = torch.device('cpu')
            # Basic fields
            has_core = all([hasattr(data, k) for k in ('tie_corridors', 'tie_buses')])
            has_labels = all([hasattr(data, k) for k in ('y_corridor_pfqt', 'y_bus_pq')])
            if not (has_core and has_labels):
                print(f'[skip] {pt.name}: missing core/labels')
                continue
            # 1) corridor->bus consistency（注：本数据设计中 y_bus_pq 包含 Trafo；此项可能不为0）
            bus_pred = pfqt_to_bus_mapping(data.y_corridor_pfqt, data.tie_corridors, data.tie_buses, device)
            diff_bus = torch.abs(bus_pred - data.y_bus_pq).max().item() if data.y_bus_pq.numel() > 0 else 0.0
            max_bus_diff = max(max_bus_diff, diff_bus)
            # 1.1) 边界母线净注入守恒：sum_b y_bus_pq ≈ 0
            if data.y_bus_pq.numel() > 0:
                net_bd = torch.abs(data.y_bus_pq.sum(dim=0)).max().item()
                max_bus_net = max(max_bus_net, net_bd)
            # 2) capacity consistency
            diff_cap = 0.0
            if hasattr(data, 'Smax_corr') and hasattr(data, 'tie_edge_corridor'):
                S_re = corridor_Smax_from_edges(data.edge_attr, data.tie_edge_corridor, data.tie_corridors.size(0))
                if data.Smax_corr.numel() == S_re.numel():
                    diff_cap = torch.abs(S_re - data.Smax_corr).max().item()
                    max_cap_diff = max(max_cap_diff, diff_cap)
            # 2.1) 纯变压器走廊检测：任一走廊是否仅由 edge_type==1 组成
            if hasattr(data, 'tie_edge_corridor'):
                cid = data.tie_edge_corridor.long()
                valid = cid >= 0
                if valid.any():
                    etype = data.edge_attr[:, 4].to(torch.float32)
                    C = int(data.tie_corridors.size(0))
                    pure_flag = False
                    for i in range(C):
                        mask = valid & (cid == i)
                        if mask.any():
                            et_i = etype[mask]
                            if torch.all(et_i > 0.5):  # 全为变压器
                                pure_flag = True
                                break
                    pure_trafo_corr_samples += int(pure_flag)
            # 3) 零标签走廊统计
            if hasattr(data, 'y_corridor_pfqt'):
                zero_mask = (torch.abs(data.y_corridor_pfqt).max(dim=1).values <= args.atol)
                cnt_zero = int(zero_mask.sum().item())
                zero_corr_total += cnt_zero
                if cnt_zero > 0:
                    zero_corr_samples += 1
            # 3) raw recompute (optional)
            if raw_dir is not None:
                # Prefer per-sample case_id mapping (most reliable)
                json_path = None
                if hasattr(data, 'case_id') and isinstance(data.case_id, str):
                    cand = raw_dir / f"{data.case_id}.json"
                    if cand.exists():
                        json_path = cand
                # Fallback: try infer from chunk filename (legacy, often fails)
                if json_path is None and pt.suffix == '.pt' and '_k' in pt.stem:
                    base = pt.stem
                    file_base = base.split('_k')[0]
                    cand = raw_dir / f'{file_base}.json'
                    if cand.exists():
                        json_path = cand
                if json_path is not None and json_path.exists():
                    # 3.0) loads vs node features
                    try:
                        loads_ref = loads_from_raw(json_path)
                        if data.x.size(0) == loads_ref.size(0):
                            load_max = max(load_max, torch.abs(data.x[:, 0:2] - loads_ref).max().item())
                    except Exception:
                        pass
                    y_corr_ref, y_bus_ref = recompute_from_raw(json_path, data.tie_corridors, data.tie_buses)
                    if data.y_corridor_pfqt.size() == y_corr_ref.size():
                        json_corr_max = max(json_corr_max, torch.abs(data.y_corridor_pfqt - y_corr_ref).max().item())
                    if data.y_bus_pq.size() == y_bus_ref.size():
                        json_bus_max = max(json_bus_max, torch.abs(data.y_bus_pq - y_bus_ref).max().item())
                    json_checks += 1
            # 4) edge pair sanity (no raw needed): forward/reverse pairs should mirror
            try:
                E = data.edge_index.size(1)
                ea = data.edge_attr
                ei = data.edge_index
                # Expect forward/reverse constructed consecutively
                if E % 2 == 0:
                    for i in range(0, E, 2):
                        u1, v1 = int(ei[0, i].item()), int(ei[1, i].item())
                        u2, v2 = int(ei[0, i+1].item()), int(ei[1, i+1].item())
                        a1, a2 = ea[i], ea[i+1]
                        # topology reversed
                        ok_pair = (u1 == v2) and (v1 == u2)
                        # invariant attrs
                        ok_pair = ok_pair and torch.allclose(a1[[0,1,2,4]], a2[[0,1,2,4]], atol=1e-6)
                        # tie flag same
                        ok_pair = ok_pair and abs(float(a1[3].item()) - float(a2[3].item())) < 1e-6
                        # shift mirrored
                        ok_pair = ok_pair and abs(float(a1[5].item()) + float(a2[5].item())) < 1e-6
                        # Pdc mirrored
                        ok_pair = ok_pair and abs(float(a1[6].item()) + float(a2[6].item())) < 1e-6
                        if not ok_pair:
                            pair_incons += 1
                            break
            except Exception:
                pass
            ok = (diff_bus <= args.atol) and (diff_cap <= args.atol)
            n_ok += int(ok)
        if use_pbar:
            # show running counts in postfix
            iterator.set_postfix({'samples': n_total, 'ok': n_ok})
    # Summary
    print("\n=== Validation Summary ===")
    print(f"Samples checked: {n_total}")
    print(f"Corridor->Bus max abs diff: {max_bus_diff:.3e} (strict check)")
    print(f"Capacity Smax max abs diff: {max_cap_diff:.3e} (strict check)")
    print(f"Boundary net injection max |sum|: {max_bus_net:.3e} (info: equals total corridor losses; not expected ≈ 0)")
    print(f"Zero-labeled corridors: total={zero_corr_total}, in {zero_corr_samples} samples")
    print(f"Pure-Trafo corridor samples (should be 0): {pure_trafo_corr_samples}")
    if json_checks:
        print(f"Raw recompute checks: {json_checks}")
        print(f"  y_corridor_pfqt vs JSON max abs diff: {json_corr_max:.3e}")
        print(f"  y_bus_pq (mapped) vs JSON max abs diff: {json_bus_max:.3e}")
        print(f"  node loads vs JSON max abs diff: {load_max:.3e}")
    print(f"Edge forward/reverse pair inconsistencies: {pair_incons}")
    print(f"Pass (bus&capacity <= atol): {n_ok}/{n_total}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
