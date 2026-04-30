FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# Override at build time, for example:
#   --build-arg TORCH_CUDA_ARCH_LIST="9.0"
# A100: 8.0, RTX 4090/L40: 8.9, H100/H200: 9.0
ARG TORCH_CUDA_ARCH_LIST="9.0"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

# Use bash so venv/path behavior is predictable
SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    ninja-build \
    wget \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Create isolated Python environment to avoid Ubuntu 24.04 PEP 668 error
RUN python3 -m venv /opt/venv

ENV PATH="/opt/venv/bin:${PATH}"
ENV VIRTUAL_ENV="/opt/venv"

RUN python -m pip install --upgrade pip setuptools wheel

ARG VORTEX_TORCH_REF=v0.3

WORKDIR /workspace

RUN git clone -b "${VORTEX_TORCH_REF}" --recursive \
    https://github.com/Infini-AI-Lab/vortex_torch.git

# Install SGLang first
WORKDIR /workspace/vortex_torch/third_party/sglang

RUN bash install.sh

# Install vortex_torch
WORKDIR /workspace/vortex_torch

RUN pip install --no-cache-dir -e .

# Optional: verify Python environment
RUN which python && python --version && which pip && pip --version

CMD ["/bin/bash"]