FROM nvidia/cuda:12.8.1-devel-ubuntu24.04@sha256:520292dbb4f755fd360766059e62956e9379485d9e073bbd2f6e3c20c270ed66
ENV DEBIAN_FRONTEND=noninteractive PATH=/opt/datasense/bin:$PATH PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false UV_CONCURRENT_DOWNLOADS=4 UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_INSTALLS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
RUN apt-get update && apt-get install -y --no-install-recommends python3.12 python3.12-venv python3.12-dev git curl ca-certificates build-essential libsndfile1 ffmpeg && rm -rf /var/lib/apt/lists/* && python3.12 -m venv /opt/datasense && /opt/datasense/bin/pip install --no-cache-dir uv==0.13.6
COPY requirements.txt /opt/image-manifest/requirements.txt
RUN uv pip install --python /opt/datasense/bin/python -r /opt/image-manifest/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match && uv pip check --python /opt/datasense/bin/python && uv pip freeze --python /opt/datasense/bin/python > /opt/image-manifest/pip-freeze.txt
COPY bfcl /opt/bfcl
RUN python /opt/bfcl/prepare_eval.py --output /opt/bfcl-data && python -c "import torch, transformers, vllm, bfcl_eval"
RUN groupadd --gid 11166 research && useradd --uid 223974 --gid 11166 --create-home --shell /bin/bash shuli && mkdir -p /workspace && chown 223974:11166 /workspace
USER 223974:11166
WORKDIR /workspace
CMD ["bash"]
