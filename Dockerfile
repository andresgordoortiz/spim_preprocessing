# Use a more specific, stable version of the base image
FROM continuumio/miniconda3:latest

# Set non-interactive to prevent prompts from hanging the build
ENV DEBIAN_FRONTEND=noninteractive

# Added --fix-missing and specific update logic
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    wget \
    unzip \
    openjdk-11-jdk \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
# --- 1. Setup Python Environment (microscopy_env) ---
WORKDIR /app
COPY microscopy_env.yml .

# Create the conda environment
# We use 'clean' to reduce image size
RUN conda env create -f microscopy_env.yml && conda clean -afy

# Add conda environment to PATH so we don't need to 'conda activate' every time
# Replace 'microscopy_env' with the actual 'name:' from your yml file
ENV PATH /opt/conda/envs/microscopy_env/bin:$PATH

# Ensure Cellpose is installed (if not already in your yml)
# Note: If your .yml already has cellpose, this line can be removed.
RUN pip install cellpose

# --- 2. Setup FIJI and TrackMate ---
# Download the latest Linux 64-bit FIJI (includes TrackMate)
RUN wget https://downloads.imagej.net/fiji/latest/fiji-linux64.zip && \
    unzip fiji-linux64.zip && \
    rm fiji-linux64.zip

# Add Fiji to the PATH
ENV PATH /app/Fiji.app:$PATH

# (Optional) Update Fiji and enable specific update sites if needed
# The ImageJ-linux64 executable is the headless launcher
RUN ImageJ-linux64 --headless --update update

# --- 3. Configuration ---
# Create a directory for data
RUN mkdir /data

WORKDIR /data

# Default command (can be overridden)
CMD ["/bin/bash"]