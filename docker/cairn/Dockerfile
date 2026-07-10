# Cairn agent image for Cage — runs the whole Cairn engine (Server + Dispatcher
# + Claude-Code workers) inside ONE Cage trial container via Docker-in-Docker.
#
# Why DinD: Cairn's Dispatcher drives a real Docker daemon to spawn one worker
# container per project and `docker exec`s the agent CLI into it. We run a
# *private* dockerd inside the (privileged) trial container so:
#   * Cairn's source stays 100% unchanged — it really drives Docker.
#   * the worker container uses Cairn's default `network_mode: host`, which under
#     the inner daemon resolves to the TRIAL container's network namespace — the
#     Cage per-trial bridge. So the worker reaches the target (by alias) and the
#     Cage sidecar proxy (localhost) for free, and stays hard-isolated from other
#     trials (it is the trial's netns, not the real host).  [verified empirically]
#
# The thin orchestrator (entrypoint) is NOT baked here — Cage `docker cp`s the
# custom-agent source dir in at trial start, so you edit it with no rebuild. Only
# the heavy, stable deps (dockerd, the vendored Cairn engine, node/claude) live
# in this image.
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# China mirror (matches the other docker/*.Dockerfile in this repo).
RUN sed -i 's|http://archive.ubuntu.com|http://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list \
    && sed -i 's|http://security.ubuntu.com|http://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list

# Base tooling + Docker engine (dockerd + CLI) for the inner daemon + sudo so the
# `agent` user (the hardcoded trial launch user) can start dockerd / chmod its
# socket without a framework change.
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    jq \
    git \
    sudo \
    iproute2 \
    iptables \
    uidmap \
    python3 \
    python3-pip \
    docker.io \
    nmap \
    netcat-openbsd \
    openssh-client \
    sshpass \
    iputils-ping \
    dnsutils \
    net-tools \
    && rm -rf /var/lib/apt/lists/*

# Proxy deps (httpx/h2) + Cairn engine runtime deps (from cairn/pyproject.toml).
RUN pip3 install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    httpx h2 \
    "fastapi>=0.115" "uvicorn[standard]>=0.34" "click>=8.1" "pyyaml>=6.0" \
    "docker>=7.1.0" "requests>=2.32.3"

# Node.js 20 + Claude Code — so this image can ALSO serve as the inner worker
# image (Cairn's `claudecode` driver runs `claude ... -p`), not only the engine.
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*
ARG CLAUDE_CODE_VERSION=2.1.150
RUN npm install -g @anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}

# Cage sidecar proxy (standalone, httpx-only) — same path every Cage agent uses.
COPY cage/proxy/sidecar.py /opt/cage-proxy/container_proxy.py

# Cairn engine from the pinned submodule third_party/Cairn — copied VERBATIM,
# never edited. Importable as `cairn`. (Build needs the submodule checked out:
# `git submodule update --init third_party/Cairn`.)
COPY third_party/Cairn/cairn/src/cairn /opt/cairn/cairn
ENV PYTHONPATH=/opt/cairn

# `cairn` CLI shim (Cairn's pyproject uses the uv_build backend; avoid installing
# it as a package — just expose its click entrypoint). No Cairn source change.
RUN printf '#!/bin/sh\nexec python3 -c "from cairn.cli import main; main()" "$@"\n' \
        > /usr/local/bin/cairn \
    && chmod +x /usr/local/bin/cairn

# Inner dockerd config. storage-driver=vfs: the trial container's own rootfs is
# overlay, and overlay-on-overlay fails ("driver not supported: overlay2") unless
# /var/lib/docker is a separate host-backed fs (the official dind trick). vfs has
# no such dependency — it works in any nested/privileged context. It copies whole
# layers (slower, more disk), but we load ONE worker image + run ONE worker
# container per trial, so the cost is bounded. registry-mirror lets the inner
# daemon pull a worker image when no baked tar is shipped.
RUN mkdir -p /etc/docker \
    && printf '{\n  "registry-mirrors": ["https://docker.m.daocloud.io"],\n  "storage-driver": "vfs"\n}\n' \
        > /etc/docker/daemon.json

# Agent user (Cage's setup_workspace/launch run as `agent`) + passwordless sudo
# (for `dockerd` / socket chmod) + docker group. Pre-create /opt/cage-agent so
# the custom-agent `setup_container` (docker cp + chown, run as the image's
# default user) works. The image's DEFAULT user stays root on purpose — Cage
# execs the launch as `--user agent` explicitly, while `docker cp`/`chown` in
# setup_container need root. The non-root *worker* runs from cairn_worker.Dockerfile.
RUN useradd -m -s /bin/bash agent \
    && mkdir -p /home/agent/workspace /opt/cage-agent \
    && chown -R agent:agent /home/agent /opt/cage-agent \
    && groupadd -f docker \
    && usermod -aG docker agent \
    && echo 'agent ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/agent \
    && chmod 0440 /etc/sudoers.d/agent

ENV HOME=/home/agent

CMD ["sleep", "infinity"]
