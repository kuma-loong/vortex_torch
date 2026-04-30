FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

ARG TORCH_CUDA_ARCH_LIST="9.0"
ARG VORTEX_TORCH_REF="v0.3"
ARG AWS_EFA_INSTALLER_URL="https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz"

ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
    VIRTUAL_ENV="/opt/venv" \
    PATH="/opt/venv/bin:/opt/amazon/efa/bin:${PATH}" \
    LD_LIBRARY_PATH="/opt/amazon/efa/lib:/opt/amazon/ofi-nccl/lib/x86_64-linux-gnu:/opt/amazon/ofi-nccl/lib:/opt/aws-ofi-nccl/lib:${LD_LIBRARY_PATH}"

# Base system dependencies
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
    tar \
    unzip \
    cpio \
    iproute2 \
    pciutils \
    environment-modules \
    tcl \
    libnuma1 \
    libnuma-dev \
    libhwloc15 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Optional AWS kernel headers/packages; don't fail build if unavailable
RUN apt-get update && apt-get install -y --no-install-recommends \
    linux-aws \
    linux-headers-aws \
    || true \
    && rm -rf /var/lib/apt/lists/*

# Python virtual environment
RUN python3 -m venv /opt/venv \
    && python -m pip install --upgrade pip setuptools wheel

WORKDIR /workspace

# Clone vortex_torch
RUN git clone -b "${VORTEX_TORCH_REF}" --recursive \
    https://github.com/Infini-AI-Lab/vortex_torch.git

# Install SGLang
WORKDIR /workspace/vortex_torch/third_party/sglang
RUN bash install.sh

# Install vortex_torch
WORKDIR /workspace/vortex_torch
RUN pip install --no-cache-dir -e .

# Install AWS EFA without kernel module / MPI
WORKDIR /workspace
RUN curl -fsSL -o aws-efa-installer-latest.tar.gz "${AWS_EFA_INSTALLER_URL}" \
    && tar --no-same-owner -xf aws-efa-installer-latest.tar.gz \
    && cd aws-efa-installer \
    && ./efa_installer.sh -y --skip-kmod --disable-ngc --skip-mpi \
    && cd /workspace \
    && rm -rf aws-efa-installer aws-efa-installer-latest.tar.gz

# Install AWS CLI v2
RUN curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" \
    -o "/tmp/awscliv2.zip" \
    && unzip -q /tmp/awscliv2.zip -d /tmp \
    && /tmp/aws/install \
    && rm -rf /tmp/aws /tmp/awscliv2.zip

# Runtime checks
RUN which python \
    && python --version \
    && which pip \
    && pip --version \
    && python - <<'PY'
import ctypes
ctypes.CDLL("libnuma.so.1")
print("OK: libnuma.so.1 loaded")
PY

# Verify NCCL OFI plugin if installed
RUN plugin_path="$(find /opt/amazon /opt/aws-ofi-nccl -name 'libnccl-net.so' -print -quit 2>/dev/null || true)" \
    && if [[ -n "${plugin_path}" ]]; then \
        ldconfig; \
        for required_lib in libhwloc.so.15 libfabric.so.1 libefa.so.1 libibverbs.so.1; do \
            ldconfig -p | grep -q "${required_lib}" || exit 1; \
        done; \
        ldd "${plugin_path}"; \
        ! ldd "${plugin_path}" | grep -q "not found"; \
        echo "Found NCCL net plugin at ${plugin_path}"; \
    else \
        echo "WARNING: libnccl-net.so not found"; \
    fi

WORKDIR /workspace/vortex_torch

CMD ["/bin/bash"]