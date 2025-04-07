FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="/usr/local/bin:$PATH"

# -- Install OS & build dependencies --
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    libssl-dev \
    zlib1g-dev \
    libncurses5-dev \
    libncursesw5-dev \
    libreadline-dev \
    libsqlite3-dev \
    libgdbm-dev \
    libdb5.3-dev \
    libbz2-dev \
    libexpat1-dev \
    liblzma-dev \
    tk-dev \
    ninja-build \
    libffi-dev \
    libgl1-mesa-glx libglib2.0-0 \
    ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# -- Build & install Python 3.10 --
RUN cd /tmp && \
    curl -O https://www.python.org/ftp/python/3.10.14/Python-3.10.14.tgz && \
    tar -xvf Python-3.10.14.tgz && \
    cd Python-3.10.14 && \
    ./configure --enable-optimizations && \
    make -j$(nproc) && \
    make altinstall && \
    ln -s /usr/local/bin/python3.10 /usr/bin/python && \
    ln -s /usr/local/bin/pip3.10 /usr/bin/pip

# -- Install core Python packages --
RUN pip install --upgrade pip
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip install packaging ninja cmake wheel
RUN pip install flash-attn --no-build-isolation
RUN pip install transformers==4.49.0
RUN pip install accelerate autoawq jupyter notebook ipykernel matplotlib Pillow

# -- Setup workspace --
WORKDIR /workspace
COPY ./code ./code
WORKDIR /workspace/code

CMD ["bash"]
