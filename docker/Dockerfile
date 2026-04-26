FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda-12.8
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Override at build time for your cluster, for example:
#   --build-arg TORCH_CUDA_ARCH_LIST="9.0"
# A100: 8.0, RTX 4090/L40: 8.9, H100/H200: 9.0
ARG TORCH_CUDA_ARCH_LIST="8.0 8.9 9.0"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-pip \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    ninja-build \
    wget \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/vortex_torch

# Install SGLang first so Docker can reuse this expensive layer when only
# vortex_torch source files change.
COPY third_party/sglang ./third_party/sglang
WORKDIR /workspace/vortex_torch/third_party/sglang
RUN bash install.sh

WORKDIR /workspace/vortex_torch
COPY . .
RUN if grep -RIl "/home/""xun" . >/tmp/local_path_matches; then \
    echo "Refusing to build image because copied files contain local /home/""xun paths:"; \
    cat /tmp/local_path_matches; \
    exit 1; \
    fi
RUN pip install --no-cache-dir -e .

CMD ["/bin/bash"]
