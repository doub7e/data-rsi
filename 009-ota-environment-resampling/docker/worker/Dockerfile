FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates curl bash && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir uv==0.9.5
COPY requirements.lock.txt /opt/worker-manifest/requirements.lock.txt
RUN uv venv --python /usr/local/bin/python3.12 /opt/harbor && uv pip install --python /opt/harbor/bin/python -r /opt/worker-manifest/requirements.lock.txt && uv pip freeze --python /opt/harbor/bin/python > /opt/worker-manifest/pip-freeze.txt
ENV PATH="/opt/harbor/bin:/usr/local/bin:/usr/bin:/bin"
ENV HARBOR_DISABLE_TELEMETRY=1 OTEL_SDK_DISABLED=true
WORKDIR /workspace
