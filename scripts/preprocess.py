#!/usr/bin/env python3
"""
Basic preprocessing: standardize audio to 16kHz, mono, normalized.
MERT-v1-330M expects 24kHz; we keep 16kHz for CQT baseline compatibility.
MERT extraction will resample to 24kHz on the fly.
"""
from ai_music.data.preprocess import preprocess

if __name__ == "__main__":
    preprocess()
