#!/usr/bin/env python3
"""
Advisory GPU status helper for shared servers.

Shows memory usage per GPU, marks "likely free" (used < 500 MiB), and suggests a GPU.
This is advisory only—"likely free" does not mean guaranteed. On shared servers:

  1. Run this script (or nvidia-smi) to inspect
  2. Choose a GPU
  3. Run with: CUDA_VISIBLE_DEVICES=<id> python scripts/run_mert_retrieval.py ...

Do not assume low memory now means safe to use—others may launch on the same GPU.
"""

from ai_music.utils.device import _print_advisory_status

if __name__ == "__main__":
    _print_advisory_status()
