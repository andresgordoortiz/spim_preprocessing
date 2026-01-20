FROM continuumio/miniconda3:23.5.2-0

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget \
        unzip \
        ca-certificates \
        openjdk-11-jre-headless \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# --- 1. Setup Python Environment ---
WORKDIR /app
COPY microscopy_env.yml .

RUN conda env create -f microscopy_env.yml && conda clean -afy

ENV PATH /opt/conda/envs/microscopy_env/bin:$PATH

RUN pip install --no-cache-dir cellpose

# --- 2. Setup FIJI and TrackMate ---
RUN wget https://downloads.imagej.net/fiji/latest/fiji-linux64.zip && \
    unzip fiji-linux64.zip && \
    rm fiji-linux64.zip

ENV PATH /app/Fiji.app:$PATH

RUN ImageJ-linux64 --headless --update update

# --- 3. Configuration ---
RUN mkdir /data
WORKDIR /data

CMD ["/bin/bash"]
