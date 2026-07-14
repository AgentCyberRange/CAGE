# Claude Code + CyberGym dynamic-analysis sandbox — white-box vuln reproduction.
#
# Builds on the oss-fuzz runner runtime (ABI-compatible with the prebuilt target
# binaries) and adds:
#   - a dynamic-analysis toolchain (gdb, strace, ltrace) + binary-PoC
#     view/build tools (xxd, hexdump; python3/od/perl already in the base)
#   - the (base-agnostic) claude_code agent layer (Node + Claude Code CLI + the
#     httpx-only cage proxy sidecar)
#
# With CyberGym's ``dynamic_sandbox: true`` the benchmark stages the prebuilt
# vulnerable target into this image's /out so the agent can run/gdb/strace the
# crash inside its OWN workspace container instead of submitting blind.
#
# Build:
#   docker build -f docker/claude_code_cyberdebug.Dockerfile -t cage/claude-code:cyberdebug .
FROM cybergym/oss-fuzz-base-runner:latest

ENV DEBIAN_FRONTEND=noninteractive

# Dynamic-analysis toolchain + binary-PoC view/build tools + claude_code apt deps
# in one layer. xxd (vim-common) and hexdump (bsdmainutils) are what agents
# reflexively reach for to inspect/craft binary PoCs; python3/od/perl/binutils are
# already in the oss-fuzz base. The oss-fuzz base ships clang but NOT gcc/make in
# PATH, and agents also reach for file/nasm/cmake/autotools when they try to build
# a repro — add them so white-box runs aren't blocked on a missing basic tool.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdb strace ltrace \
    xxd bsdmainutils file \
    nasm build-essential cmake pkg-config \
    autoconf automake libtool \
    curl ca-certificates jq git \
    && rm -rf /var/lib/apt/lists/*

# cage proxy sidecar deps MUST land in the python cage invokes in-container,
# which is PATH `python3` = /usr/local/bin/python3 (oss-fuzz's 3.10), NOT the
# system 3.8. Use `python3 -m pip` so it targets that interpreter.
RUN python3 -m pip install --no-cache-dir httpx h2

# Node.js 20 (Claude Code CLI is installed last — see the tail of this file).
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

COPY cage/proxy/sidecar.py /opt/cage-proxy/container_proxy.py

RUN useradd -m -s /bin/bash agent \
    && mkdir -p /home/agent/workspace \
    && chown -R agent:agent /home/agent

# Pre-install Claude Code CLI. Kept as the LAST build layer so `cage agent
# build --version <v>` (or a --build-arg) only re-runs this step — everything
# above stays cached. Runs before `ENV HOME` so npm's cache lands in /root,
# not the agent's home. Default `latest`; pin a version for a reproducible image.
ARG CLAUDE_CODE_VERSION=latest
RUN npm install -g @anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}

ENV HOME=/home/agent
CMD ["sleep", "infinity"]
