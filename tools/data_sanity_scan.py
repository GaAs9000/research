"""
Quick sanity scanner for processed V1.1 (V-θ route) datasets.

Checks per sample (non-exhaustive):
- NaN/Inf in core tensors (x, edge_attr, y_bus_V, y_edge_sincos, y_edge_pq)
- AC-only tie-lines: no transformer edges should have is_tie>0
- AC shift≈0: edge_type=AC should have near-zero shift
- Smax integrity: edge Smax should be > 0
- sin/cos labels lie on the unit circle within tolerance
- Ring features alignment (if present)

Usage:
  python tools/data_sanity_scan.py --processed-dir /path/to/processed \
      --max-samples 100 --alpha 0.95 --cap-ratio 1.25 --atol 1e-6
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import torch


def _glob_pt(processed_dir: Path, max_files: int | None = None) -> List[Path]:
    pats = ["train/*.pt", "val/*.pt", "test/*.pt", "*.pt"]
    files: List[Path] = []
    for pat in pats:
        files.extend(sorted((processed_dir).glob(pat)))
    # unique and trim
    out = []
    seen = set()
    for p in files:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
        if max_files and len(out) >= max_files:
            break
    return out


def _iter_samples(pt_path: Path):
    obj = torch.load(pt_path, map_location='cpu', weights_only=False)
    if isinstance(obj, list):
        for d in obj:
            yield d
    else:
        yield obj


def scan_sample(d, atol: float) -> Dict[str, List[str]]:
    issues: Dict[str, List[str]] = {}

    def add(kind: str, msg: str):
        issues.setdefault(kind, []).append(msg)

    # 1) Finite checks
    for name in ['x', 'edge_attr', 'y_bus_V', 'y_edge_sincos', 'y_edge_pq']:
        if hasattr(d, name):
            t = getattr(d, name)
            if isinstance(t, torch.Tensor):
                if not torch.isfinite(t).all():
                    add('non_finite', f'{name} has NaN/Inf')

    # 2) AC-only tie-lines & AC shift≈0
    if hasattr(d, 'edge_attr'):
        ea = d.edge_attr
        is_tie = ea[:, 3] > 0.5
        edge_type = ea[:, 4]
        shift = ea[:, 5]
        # transformer edges (edge_type>=0.5) must not be ties
        if (is_tie & (edge_type >= 0.5)).any():
            add('tie_mask', 'transformer edge marked as tie')
        # AC edges should have near-zero shift
        if (edge_type < 0.5).any():
            bad_shift = torch.abs(shift[edge_type < 0.5]) > 1e-6
            if bad_shift.any():
                add('ac_shift', f'AC edges with |shift|>1e-6: {int(bad_shift.sum())}')
        # Smax > 0 on edges
        if (ea[:, 2] <= 0).any():
            add('smax_edge', f'edge Smax<=0 count={int((ea[:,2]<=0).sum())}')

    # 3) sin/cos unit circle check
    if hasattr(d, 'y_edge_sincos') and d.y_edge_sincos is not None:
        norms = torch.linalg.vector_norm(d.y_edge_sincos.to(torch.float32), dim=-1)
        if (torch.abs(norms - 1.0) > 5e-3).any():
            max_dev = float(torch.abs(norms - 1.0).max().item())
            add('sincos_norm', f'sin/cos not unit norm (max dev {max_dev:.2e})')

    # 4) Ring features in x (if present)
    if hasattr(d, 'ring_decayed') and d.ring_decayed is not None and hasattr(d, 'x') and hasattr(d, 'tie_buses'):
        B = int(d.tie_buses.numel())
        if B > 0 and d.ring_decayed.size(0) == B and d.x.size(1) >= 6:
            ring_cols = slice(d.x.size(1) - 2, d.x.size(1))
            ring_mat = d.x[:, ring_cols]
            tb = d.tie_buses.long()
            max_bus = int(tb.max().item())
            nz_mask = (ring_mat.abs() > 1e-8).any(dim=1)
            # ring should be nonzero on tie buses only
            # Check if ring features are non-zero only on tie buses
            tie_mask = torch.zeros(d.x.size(0), dtype=torch.bool)
            if max_bus < d.x.size(0):
                tie_mask[tb] = True
            bad_non_tie = int((nz_mask & ~tie_mask).sum().item())
            if bad_non_tie > 0:
                add('ring_align', f'ring features non-zero on non-tie buses: {bad_non_tie}')
            diff = (ring_mat[tb] - d.ring_decayed).abs().max().item()
            if diff > 1e-5:
                add('ring_align', f'ring_decayed vs x misalign max={diff:.2e}')

    return issues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--processed-dir', required=True)
    ap.add_argument('--max-samples', type=int, default=50)
    ap.add_argument('--atol', type=float, default=1e-6)
    args = ap.parse_args()

    pdir = Path(args.processed_dir)
    files = _glob_pt(pdir, args.max_samples)
    if not files:
        print('No .pt files under', pdir)
        return 2

    summary: Dict[str, int] = {}
    examples: Dict[str, List[str]] = {}
    total = 0

    for f in files:
        for d in _iter_samples(f):
            total += 1
            issues = scan_sample(d, args.atol)
            for k, msgs in issues.items():
                summary[k] = summary.get(k, 0) + 1
                if k not in examples:
                    examples[k] = []
                # record first few examples with file name
                if len(examples[k]) < 5:
                    sid = getattr(d, 'sample_id', 'unknown')
                    examples[k].append(f'{f.name} :: {sid} :: ' + ' | '.join(msgs))

    print('\n=== Data Sanity Summary ===')
    print(f'samples checked: {total}')
    if not summary:
        print('No anomalies detected in scanned samples.')
        return 0
    for k, v in sorted(summary.items(), key=lambda x: -x[1]):
        print(f'- {k}: {v}')
        for ex in examples.get(k, [])[:3]:
            print(f'  * {ex}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

