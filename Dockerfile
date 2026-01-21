# --- Stage 1: Source Fiji ---
FROM fiji/fiji:latest AS fiji_source

# --- Stage 2: Final Build ---
FROM mambaorg/micromamba:1.5.1

USER root

# 1. Install System Dependencies
# Includes Java 17, X11 libraries for headless Fiji, and GL support
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget \
        unzip \
        ca-certificates \
        openjdk-17-jre-headless \
        libgl1 \
        libglib2.0-0 \
        libxrender1 \
        libxtst6 \
        libxi6 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

# 2. Setup Python Environment
WORKDIR /app
COPY microscopy_env.yml .

# Create environment and install cellpose as specified
RUN micromamba create -f microscopy_env.yml -y && \
    micromamba clean -afy

# Ensure Cellpose and core tools are in the environment
RUN /opt/conda/bin/micromamba run -n microscopy_env pip install --no-cache-dir cellpose

# 3. Setup FIJI (ImageJ)
COPY --from=fiji_source /opt/fiji /opt/fiji
ENV PATH="/opt/fiji:$PATH"
# Alias for easier calling
ENV FIJI_EXE="/opt/fiji/ImageJ-linux64"

# 4. Enable Fiji Update Sites for Cellpose & TrackMate
# This updates Fiji and adds the sites required for the pipeline
RUN $FIJI_EXE --update update && \
    $FIJI_EXE --update add-update-site TrackMate-Helper https://sites.imagej.net/TrackMate-Helper/ && \
    $FIJI_EXE --update add-update-site CSBDeep https://sites.imagej.net/CSBDeep/ && \
    $FIJI_EXE --update apply-changes

# 5. Runtime Configuration
RUN mkdir -p /data
WORKDIR /data

# Use micromamba's entrypoint to ensure 'micromamba run' works
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]

# Default command: Start a bash shell within the activated environment
CMD ["micromamba", "run", "-n", "microscopy_env", "/bin/bash"]