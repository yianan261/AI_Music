#!/usr/bin/env python3
"""
Prepare MAESTRO dataset: select N pieces, copy to raw_audio/, write mapping metadata.
Run after extracting the MAESTRO dataset.
"""
import argparse

from ai_music.data.prepare_maestro import prepare_maestro


def main():
    parser = argparse.ArgumentParser(description="Prepare MAESTRO subset")
    parser.add_argument("--n-pieces", type=int, default=100, help="Number of pieces to copy (ignored with --all)")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Copy every MAESTRO WAV listed in the CSV that exists on disk (~1.2k pieces)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subset selection")
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["train", "validation", "test"],
        help="Filter by MAESTRO split (default: all)",
    )
    args = parser.parse_args()
    prepare_maestro(n_pieces=args.n_pieces, seed=args.seed, split=args.split, use_all=args.all)


if __name__ == "__main__":
    main()
