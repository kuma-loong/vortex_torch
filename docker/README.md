# Vortex Torch Docker

Build from the repository root:

```bash
docker build -f docker/Dockerfile -t vortex-torch:cu128 .
```

For a specific GPU architecture, pass `TORCH_CUDA_ARCH_LIST`:

```bash
docker build -f docker/Dockerfile -t vortex-torch:h100 \
  --build-arg TORCH_CUDA_ARCH_LIST="9.0" .
```

Common architecture values:

- `8.0`: A100
- `8.9`: RTX 4090, L40
- `9.0`: H100, H200

Run with GPU access:

```bash
docker run --gpus all -it --rm vortex-torch:cu128
```

For development with your local checkout mounted into the container:

```bash
docker run --gpus all -it --rm \
  -v "$PWD":/workspace/vortex_torch \
  vortex-torch:cu128
```

Inside the mounted container, rerun this after editing CUDA/C++ extension code:

```bash
pip install -e .
```
