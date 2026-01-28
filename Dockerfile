# --- Stage 1: Build TrackMate from Source ---
FROM maven:3.9.6-eclipse-temurin-17 AS builder

WORKDIR /build
# Download the specific source version
RUN wget https://github.com/trackmate-sc/TrackMate/archive/refs/tags/TrackMate-7.13.0.tar.gz && \
    tar -xzf TrackMate-7.13.0.tar.gz

# Compile and package (skipping tests to speed up the build)
WORKDIR /build/TrackMate-TrackMate-7.13.0
RUN mvn clean package -DskipTests


# --- Stage 2: Final Image ---
FROM mambaorg/micromamba:1.5.10 AS base

USER root

# 1. Install system dependencies (Adding maven dependencies if needed)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget unzip ca-certificates openjdk-17-jdk-headless \
        libgl1 libglib2.0-0 libxrender1 libxtst6 libxi6 libxext6 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 2. Setup working directory
WORKDIR /app

# 3. Setup conda environment
COPY --chown=mambauser:mambauser microscopy_env.yml .
RUN --mount=type=cache,target=/opt/conda/pkgs \
    micromamba create -f microscopy_env.yml -y

# 4. Download and setup FIJI
RUN wget --progress=dot:giga -O fiji.zip \
        https://downloads.imagej.net/fiji/latest/fiji-latest-portable-nojava.zip && \
    unzip -q fiji.zip -d /opt && \
    rm fiji.zip && \
    mv /opt/Fiji* /opt/fiji && \
    chmod +x /opt/fiji/fiji-linux-x64 && \
    ln -s /opt/fiji/fiji-linux-x64 /usr/local/bin/fiji

# --- New Step: Install the custom TrackMate build ---
# Remove any existing TrackMate jars to prevent version conflicts
RUN rm -f /opt/fiji/plugins/TrackMate_-*.jar /opt/fiji/jars/TrackMate_-*.jar

# Copy the newly compiled jar from the builder stage
# Note: Maven usually outputs to the 'target' directory
COPY --from=builder /build/TrackMate-TrackMate-7.13.0/target/TrackMate_-*.jar /opt/fiji/plugins/

# Set FIJI environment variables
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