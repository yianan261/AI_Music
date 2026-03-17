#!/usr/bin/env python3
"""Check GPU status and which device would be auto-selected for model inference."""

from ai_music.utils.device import print_gpu_status

if __name__ == "__main__":
    print_gpu_status()
