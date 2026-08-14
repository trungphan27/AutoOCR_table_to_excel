# syntax=docker/dockerfile:1.7

ARG APP_HOME=/app

FROM python:3.12-slim-bookworm AS cpu
ARG APP_HOME
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_WAIT_POLICY=PASSIVE \
    MALLOC_ARENA_MAX=2
WORKDIR ${APP_HOME}
RUN apt-get update \
    && apt-get install --no-install-recommends -y libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system app \
    && useradd --system --gid app --home-dir ${APP_HOME} app
COPY requirements-deploy-common.txt requirements-deploy-cpu.txt ./
RUN python -m pip install --no-cache-dir \
    -r requirements-deploy-common.txt -r requirements-deploy-cpu.txt
COPY . .
RUN mkdir -p ${APP_HOME}/output/deploy \
    && chown -R app:app ${APP_HOME}/output
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=5)"]
CMD ["python", "-m", "uvicorn", "deploy.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 AS gpu
ARG APP_HOME
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH=/opt/venv/bin:${PATH} \
    OMP_NUM_THREADS=1 \
    OMP_WAIT_POLICY=PASSIVE \
    MALLOC_ARENA_MAX=2
WORKDIR ${APP_HOME}
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        python3.12 python3.12-venv libglib2.0-0 libgomp1 \
    && python3.12 -m venv /opt/venv \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system app \
    && useradd --system --gid app --home-dir ${APP_HOME} app
COPY requirements-deploy-common.txt requirements-deploy-gpu.txt ./
RUN python -m pip install --no-cache-dir \
    -r requirements-deploy-common.txt -r requirements-deploy-gpu.txt
COPY . .
RUN mkdir -p ${APP_HOME}/output/deploy \
    && chown -R app:app ${APP_HOME}/output
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=5)"]
CMD ["python", "-m", "uvicorn", "deploy.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
