# AI Music - ID and Accuracy Project
# GPU image for MERT + FAISS
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system deps for audio
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Python deps & package
COPY pyproject.toml .
COPY requirements.txt .
COPY src/ ./src/
RUN pip install --no-cache-dir -e .

# App code
COPY scripts/ ./scripts/
COPY notebooks/ ./notebooks/
COPY requirements.txt README.md .

# Default: bash (override in compose/run)
CMD ["bash"]
