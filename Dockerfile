# --- Stage 1: Source Fiji ---
FROM fiji/fiji:latest AS fiji_source

# --- Stage 2: Final Build ---
FROM mambaorg/micromamba:1.5.1

USER root

# 1. Install System Dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget unzip ca-certificates openjdk-17-jre-headless \
        libgl1 libglib2.0-0 libxrender1 libxtst6 libxi6 libxext6 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 2. Setup Python Environment
WORKDIR /app
COPY --chown=$MAMBA_USER:$MAMBA_USER microscopy_env.yml .

# Cache mount for space-saving on GitHub Actions
RUN --mount=type=cache,target=/opt/conda/pkgs \
    micromamba create -f microscopy_env.yml -y

# 3. Setup FIJI (ImageJ)
# The official fiji/fiji image keeps the app in /fiji
COPY --from=fiji_source /fiji /opt/fiji
RUN chmod +x /opt/fiji/ImageJ-linux64

# Set Environment Variables
ENV PATH="/opt/fiji:$PATH"
ENV FIJI_BIN="/opt/fiji/ImageJ-linux64"

# 4. Update Fiji and Enable TrackMate/Cellpose sites
# We run these as a single layer to save space.
# '--headless' is added to ensure it doesn't try to open a window during build.
RUN $FIJI_BIN --headless --update update && \
    $FIJI_BIN --headless --update add-update-site TrackMate-Helper https://sites.imagej.net/TrackMate-Helper/ && \
    $FIJI_BIN --headless --update add-update-site CSBDeep https://sites.imagej.net/CSBDeep/ && \
    $FIJI_BIN --headless --update apply-changes

# 5. Final Configuration
WORKDIR /data
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]
CMD ["micromamba", "run", "-n", "microscopy_env", "bash"]