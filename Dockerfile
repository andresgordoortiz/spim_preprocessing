# Multi-stage build for microscopy analysis

# Stage 1: Base image with Python/Conda environment
FROM mambaorg/micromamba:1.5.10 AS base

USER root

# 1. Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget unzip ca-certificates openjdk-17-jre-headless \
        libgl1 libglib2.0-0 libxrender1 libxtst6 libxi6 libxext6 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 2. Setup working directory
WORKDIR /app

# 3. Copy and create conda environment
COPY --chown=mambauser:mambauser microscopy_env.yml .
RUN --mount=type=cache,target=/opt/conda/pkgs \
    micromamba create -f microscopy_env.yml -y

# 4. Download and setup FIJI directly
RUN wget --progress=dot:giga -O fiji-nojre.zip https://downloads.imagej.net/fiji/latest/fiji-nojre.zip && \
    ls -lh fiji-nojre.zip && \
    file fiji-nojre.zip && \
    unzip fiji-nojre.zip -d /opt && \
    ls -la /opt/ && \
    rm fiji-nojre.zip && \
    chmod +x /opt/Fiji.app/ImageJ-linux64

# Set FIJI environment variable
ENV FIJI_ROOT=/opt/Fiji.app
ENV PATH="${FIJI_ROOT}:${PATH}"

# 5. Switch to non-root user
USER mambauser

# 6. Set up the micromamba environment activation
ARG MAMBA_DOCKERFILE_ACTIVATE=1
ENV ENV_NAME=microscopy_env

# 7. Copy application code
COPY --chown=mambauser:mambauser . /app

# 8. Default command
CMD ["/bin/bash"]