# Use lightweight Python 3.8 image (matches conda.yaml specification)
FROM python:3.8-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install minimal system dependencies for building packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Upgrade pip (dependencies are installed after container creation)
RUN pip install --upgrade pip setuptools wheel

# Default command
CMD ["/bin/bash"]
