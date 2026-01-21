# --- Stage 1: Source Fiji ---
FROM fiji/fiji:latest AS fiji_source

# --- Stage 2: Final Build ---
FROM mambaorg/micromamba:1.5.1

USER root

# 1. Install System Dependencies (Added X11 libs for headless Fiji/TrackMate)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget unzip ca-certificates openjdk-17-jre-headless \
        libgl1 libglib2.0-0 libxrender1 libxtst6 libxi6 libxext6 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 2. Setup Python Environment
WORKDIR /app
COPY microscopy_env.yml .

# Micromamba will install Python + Pip packages (Cellpose) in one go
# Using cache mounts to save space on GitHub Actions
RUN --mount=type=cache,target=/opt/conda/pkgs \
    micromamba create -f microscopy_env.yml -y && \
    micromamba clean -afy

# 3. Setup FIJI (ImageJ)
COPY --from=fiji_source /opt/fiji /opt/fiji
ENV PATH="/opt/fiji:$PATH"
# Use a variable for the Fiji executable for clarity
ARG FIJI_BIN="/opt/fiji/ImageJ-linux64"

# 4. Update Fiji and Enable TrackMate/Cellpose sites
# We use 'micromamba run' just to ensure we are in a clean shell environment
RUN $FIJI_BIN --update update && \
    $FIJI_BIN --update add-update-site TrackMate-Helper https://sites.imagej.net/TrackMate-Helper/ && \
    $FIJI_BIN --update add-update-site CSBDeep https://sites.imagej.net/CSBDeep/ && \
    $FIJI_BIN --update apply-changes

# 5. Final Configuration
WORKDIR /data
# This entrypoint is provided by the base image and handles environment activation
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]
CMD ["micromamba", "run", "-n", "microscopy_env", "bash"]