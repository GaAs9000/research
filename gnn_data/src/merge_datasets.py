"""
Merge multiple processed datasets (train/val/test chunk_*.pt files) into one directory.

- Creates split folders (train/val/test) under output_dir
- For each input processed_dir, iterates split/chunk_*.pt and links (or copies) into output with new indices
"""

import argparse
import os
from pathlib import Path
import shutil

def merge_splits(input_dirs, output_dir, split, link=True):
    out_split = Path(output_dir) / split
    out_split.mkdir(parents=True, exist_ok=True)
    counter = 0
    for d in input_dirs:
        in_split = Path(d) / split
        if not in_split.exists():
            continue
        files = sorted(in_split.glob('chunk_*.pt'))
        for f in files:
            dst = out_split / f"chunk_{counter:04d}.pt"
            if link:
                try:
                    # Prefer hard link for performance; fallback to copy
                    os.link(f, dst)
                except Exception:
                    shutil.copy2(f, dst)
            else:
                shutil.copy2(f, dst)
            counter += 1
    return counter

def main():
    ap = argparse.ArgumentParser(description='Merge processed datasets into one')
    ap.add_argument('--inputs', nargs='+', required=True, help='Input processed dirs (each contains train/val/test)')
    ap.add_argument('--output', required=True, help='Output merged processed dir')
    ap.add_argument('--no-link', action='store_true', help='Copy files instead of linking')
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    for split in ['train', 'val', 'test']:
        n = merge_splits(args.inputs, out, split, link=not args.no_link)
        print(f"Merged {n} chunks into {out/ split}")

    print('Done.')

if __name__ == '__main__':
    main()

