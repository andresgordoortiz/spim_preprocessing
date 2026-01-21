FROM mambaorg/micromamba:1.5.1
USER root

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies (use openjdk-17 for Debian Bookworm)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget \
        unzip \
        ca-certificates \
        openjdk-17-jre-headless \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# --- 1. Setup Python Environment ---
WORKDIR /app
COPY microscopy_env.yml .

# Create environment with micromamba (much faster than conda)
RUN micromamba create -f microscopy_env.yml -y && \
    micromamba clean -afy

# Activate environment and install cellpose
RUN /bin/bash -c "eval \"\$(micromamba shell hook --shell bash)\" && \
    micromamba activate microscopy_env && \
    pip install --no-cache-dir cellpose"

# --- 2. Setup FIJI and TrackMate ---
# Download and extract Fiji
RUN wget https://downloads.imagej.net/fiji/latest/fiji-latest-linux64-jdk.zip && \
    unzip fiji-latest-linux64-jdk.zip && \
    rm fiji-latest-linux64-jdk.zip && \
    chmod +x /app/Fiji.app/ImageJ-linux64 || chmod +x /app/Fiji.app/fiji-linux64 || true

ENV PATH="/app/Fiji.app:$PATH"

# --- 3. Configuration ---
RUN mkdir -p /data
WORKDIR /data

# Set up entrypoint to activate conda environment
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]
CMD ["/bin/bash"]