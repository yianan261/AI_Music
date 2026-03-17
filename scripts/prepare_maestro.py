#!/usr/bin/env python3
"""
Prepare MAESTRO dataset for MVP: select 100 pieces and copy audio to data/raw_audio/.
Run this after extracting the MAESTRO dataset.
"""
from ai_music.data.prepare_maestro import prepare_maestro

if __name__ == "__main__":
    prepare_maestro()
