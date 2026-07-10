FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Replace apt sources with Tsinghua mirror (HTTP before ca-certificates installed)
RUN sed -i 's|http://archive.ubuntu.com|http://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list \
    && sed -i 's|http://security.ubuntu.com|http://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list

# Install essentials + Python3 + the binary/PoC inspection & build tools agents
# reflexively reach for across benchmarks (empirically: xxd is by far the most-
# attempted missing command, then file/nasm/gdb/strace/autotools/cmake). These
# are basic dev tools — NOT the target binaries — so they belong in the base
# image and benefit every task, not just cybergym.
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    jq \
    git \
    python3 \
    python3-pip \
    file \
    xxd \
    bsdmainutils \
    nasm \
    gdb \
    strace \
    ltrace \
    build-essential \
    cmake \
    pkg-config \
    autoconf \
    automake \
    libtool \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for cage proxy (using Tsinghua PyPI mirror)
RUN pip3 install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple httpx h2

# Install Node.js 20 (for Claude Code)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Pre-install Claude Code CLI
ARG CLAUDE_CODE_VERSION=2.1.150
RUN npm install -g @anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}

# Install cage proxy (standalone, no cage package dependency)
COPY cage/proxy/sidecar.py /opt/cage-proxy/container_proxy.py

# Create agent user
RUN useradd -m -s /bin/bash agent \
    && mkdir -p /home/agent/workspace \
    && chown -R agent:agent /home/agent

ENV HOME=/home/agent

CMD ["sleep", "infinity"]
