# Multi-stage build for microscopy analysis
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

# 4. Download and setup FIJI
# We unzip to /opt. The zip usually extracts a folder named 'Fiji' or 'Fiji.app'.
# We verify the directory name with a wildcard move to ensure it ends up at /opt/fiji.
RUN wget --progress=dot:giga -O fiji.zip \
        https://downloads.imagej.net/fiji/latest/fiji-latest-portable-nojava.zip && \
    unzip -q fiji.zip -d /opt && \
    rm fiji.zip && \
    # Consolidate whatever the zip folder was named (Fiji or Fiji.app) into /opt/fiji
    mv /opt/Fiji* /opt/fiji && \
    # Make the binary executable (it is at the root, per your ls output)
    chmod +x /opt/fiji/fiji-linux-x64 && \
    # Create symlink for easy access
    ln -s /opt/fiji/fiji-linux-x64 /usr/local/bin/fiji

# Set FIJI environment variables
# FIJI_ROOT must point to the folder containing 'jars' and 'plugins'
ENV FIJI_ROOT=/opt/fiji
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