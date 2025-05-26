FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# 1) System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3-dev python3-pip build-essential git tmux libgl1 libglib2.0-0 \
      ffmpeg net-tools && \
    rm -rf /var/lib/apt/lists/*

# 2) Python + pip
RUN ln -sf /usr/bin/python3 /usr/bin/python \
 && pip3 install --no-cache-dir --upgrade pip

RUN python3 -m pip install --upgrade pip
RUN ln -sf /usr/bin/python3 /usr/bin/python

# 3) Constrain numpy to <2.0 to avoid ABI issues
RUN pip install --no-cache-dir "numpy<2.0"

RUN pip install --no-cache-dir packaging ninja pybind11

RUN pip install --no-cache-dir       torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1       --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir triton==3.3.0

# # 2) FlashAttention wheel for torch2.3/cu12
RUN pip install --no-cache-dir flash-attn==2.5.9.post1     --no-deps --no-build-isolation

# # 6) Transformers + Qwen-VL utilities + decord
RUN pip install --no-cache-dir \
      transformers==4.49.0 qwen-vl-utils==0.0.11 decord



RUN pip install --no-cache-dir einops \
    git+https://github.com/casper-hansen/AutoAWQ.git@main \
    opencv-python matplotlib

ENV CUDA_HOME=/usr/local/cuda

# (optional, but highly recommended) install cmake and upgrade setuptools/wheel
RUN apt-get update && apt-get install -y cmake \
     && pip install --no-cache-dir --upgrade setuptools wheel

WORKDIR /workspace/GroundingDINO
RUN git clone --depth 1 https://github.com/IDEA-Research/GroundingDINO.git .

RUN pip install --no-cache-dir notebook jupyter flask

# 8) Verification (to run after build):
#    docker run --gpus all <image> \
#      python -c "import torch, flash_attn, groundingdino; \
#                 print(torch.__version__, torch.version.cuda, \
#                       flash_attn.__version__, \
#                       'OK' if groundingdino else 'fail')"
