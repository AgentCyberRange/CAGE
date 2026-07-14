FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Replace apt sources with Tsinghua mirror (HTTP before ca-certificates installed)
RUN sed -i 's|http://archive.ubuntu.com|http://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list \
    && sed -i 's|http://security.ubuntu.com|http://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list

# Install essentials + Python3
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    jq \
    git \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for cage proxy (using Tsinghua PyPI mirror)
RUN pip3 install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple httpx h2

# Install Node.js 20 (for Codex CLI)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install cage proxy (standalone, no cage package dependency)
COPY cage/proxy/sidecar.py /opt/cage-proxy/container_proxy.py

# Create agent user
RUN useradd -m -s /bin/bash agent \
    && mkdir -p /home/agent/workspace \
    && chown -R agent:agent /home/agent

# Pre-install Codex CLI. Kept as the LAST build layer so `cage agent build
# --version <v>` (or a --build-arg) only re-runs this step — everything above
# stays cached. Runs before `ENV HOME` so npm's cache lands in /root, not the
# agent's home. Default `latest`; pin a version for a reproducible image.
ARG CODEX_VERSION=latest
RUN npm install -g @openai/codex@${CODEX_VERSION}

ENV HOME=/home/agent

CMD ["sleep", "infinity"]
