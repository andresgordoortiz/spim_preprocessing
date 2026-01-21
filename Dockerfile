# --- Stage 1: Source Fiji ---
FROM fiji/fiji:latest AS fiji_source

# --- Stage 2: Final Build ---
FROM mambaorg/micromamba:1.5.1

USER root

# 1. Install System Dependencies
# Added essential libraries for Fiji's headless mode and TrackMate's logic
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget unzip ca-certificates openjdk-17-jre-headless \
        libgl1 libglib2.0-0 libxrender1 libxtst6 libxi6 libxext6 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 2. Setup Python Environment
WORKDIR /app
COPY --chown=$MAMBA_USER:$MAMBA_USER microscopy_env.yml .

# FIX: Removed 'micromamba clean' because it conflicts with the cache mount.
# The cache mount automatically keeps these files out of your final image.
RUN --mount=type=cache,target=/opt/conda/pkgs \
    micromamba create -f microscopy_env.yml -y

# 3. Setup FIJI (ImageJ)
COPY --from=fiji_source /opt/fiji /opt/fiji
ENV PATH="/opt/fiji:$PATH"
ARG FIJI_BIN="/opt/fiji/ImageJ-linux64"

# 4. Update Fiji and Enable TrackMate/Cellpose sites
# Using 'micromamba run' ensures we use the environment's Java if needed,
# though we've installed a system-level openjdk-17 as well.
RUN $FIJI_BIN --update update && \
    $FIJI_BIN --update add-update-site TrackMate-Helper https://sites.imagej.net/TrackMate-Helper/ && \
    $FIJI_BIN --update add-update-site CSBDeep https://sites.imagej.net/CSBDeep/ && \
    $FIJI_BIN --update apply-changes

# 5. Final Configuration
WORKDIR /data
# Ensure the entrypoint is executable (standard in this base image)
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]
CMD ["micromamba", "run", "-n", "microscopy_env", "bash"]