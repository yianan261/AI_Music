#!/usr/bin/env python3
"""
MERT-v1-330M expects 24kHz.
Preprocess audio: mono, normalized. Outputs to processed_16k/ (CQT) and processed_24k/ (MERT).
"""
from ai_music.data.preprocess import preprocess

if __name__ == "__main__":
    preprocess()
