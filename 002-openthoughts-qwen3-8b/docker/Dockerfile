# Build context: this docker/ directory. Candidate; not yet built/GPU-validated.
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04@sha256:520292dbb4f755fd360766059e62956e9379485d9e073bbd2f6e3c20c270ed66

ARG LF_COMMIT=d20b86665a2753efdcc0f4adb7503ce87f8bdb80
ARG USER_ID=223974
ARG GROUP_ID=11166
ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/sft/bin:$PATH \
    CUDA_HOME=/usr/local/cuda \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    MAX_JOBS=4 \
    NVCC_THREADS=4 \
    TORCH_CUDA_ARCH_LIST=9.0 \
    DS_BUILD_OPS=0

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.12 python3.12-venv python3.12-dev \
      gcc g++ build-essential git ca-certificates curl libaio-dev libnuma-dev \
      libsndfile1 ffmpeg pkg-config && \
    rm -rf /var/lib/apt/lists/* && \
    python3.12 -m venv /opt/sft && \
    /opt/sft/bin/pip install --no-cache-dir uv==0.13.6 setuptools==75.8.0 wheel==0.45.1

COPY requirements.lock /opt/image-manifest/requirements.lock
RUN uv pip install --python /opt/sft/bin/python -r /opt/image-manifest/requirements.lock && \
    uv pip install --python /opt/sft/bin/python --no-deps --no-build-isolation deepspeed==0.17.6 && \
    python -c "import torch; assert torch._C._GLIBCXX_USE_CXX11_ABI is True" && \
    curl -fL --retry 3 'https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl' \
      -o /tmp/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl && \
    echo 'f25da18657a87fc83dc1bfb8b7751b82246e9db355510226b674fd437c34b5fb  /tmp/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl' | sha256sum -c - && \
    uv pip install --python /opt/sft/bin/python --no-deps \
      /tmp/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl && \
    rm /tmp/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

# Use the exact gitlink selected by OpenThoughts-Agent, not the moving branch.
RUN git init /opt/llamafactory && \
    git -C /opt/llamafactory remote add origin https://github.com/mlfoundations/llama-factory.git && \
    git -C /opt/llamafactory fetch --depth 1 origin ${LF_COMMIT} && \
    git -C /opt/llamafactory checkout --detach FETCH_HEAD && \
    test "$(git -C /opt/llamafactory rev-parse HEAD)" = "${LF_COMMIT}" && \
    uv pip install --python /opt/sft/bin/python --no-deps --no-build-isolation /opt/llamafactory && \
    uv pip check --python /opt/sft/bin/python && \
    uv pip freeze --python /opt/sft/bin/python > /opt/image-manifest/pip-freeze.txt

# Build-time import checks do not exercise CUDA kernels or distributed training.
RUN python -c "import torch, transformers, deepspeed, flash_attn; from liger_kernel.transformers import apply_liger_kernel_to_qwen3; from llamafactory.data.template import TEMPLATES; assert 'qwen3' in TEMPLATES; assert torch.__version__.startswith('2.8.0'); assert torch.version.cuda == '12.8'" && \
    groupadd --gid ${GROUP_ID} research && \
    useradd --uid ${USER_ID} --gid ${GROUP_ID} --create-home --shell /bin/bash shuli && \
    mkdir -p /workspace && chown ${USER_ID}:${GROUP_ID} /workspace

USER shuli
WORKDIR /workspace
CMD ["bash"]
