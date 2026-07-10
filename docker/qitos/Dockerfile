# QitOS CyberGym agent image for Cage.
#
# Bakes the vendored QitOS harness (the pinned submodules third_party/qitos +
# third_party/cybergym_agent, copied VERBATIM, never edited) + its deps + the
# Cage proxy. The thin orchestrator (cage/agents/custom/qitos/qitos_cage_entry.py)
# is NOT baked — Cage `docker cp`s the custom-agent source dir in at trial start,
# so you edit it with no rebuild. Build needs the submodules checked out:
#   git submodule update --init third_party/qitos third_party/cybergym_agent
#   docker build -f docker/qitos.Dockerfile -t cage/qitos:latest .
#
# Traceability: the exact upstream commit each image was built from is the
# submodule gitlink recorded in Cage's tree (`git submodule status`).
#
# What runs: `python -m cybergym_agent.run_local` (HostEnv — runs its tools in
# THIS container, no nested docker). `cybergym_agent` must import as a TOP-LEVEL
# package (run_local uses absolute `from cybergym_agent...` imports), and `qitos`
# (the framework it builds on) must be installed.
FROM pursu1ng/ctfenv:latest
ENV DEBIAN_FRONTEND=noninteractive

# qitos needs Python >=3.10; ctfenv's default `python3` is miniconda 3.9, so use
# the system /usr/bin/python3 (3.10) for the harness AND the cage proxy deps.
ARG PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple

# qitos framework (from the pinned submodule third_party/qitos) + its model extra
# (openai) and the cage proxy / agent runtime deps. tree_sitter backs the agent's
# constraint extractor.
COPY third_party/qitos /opt/qitos-src
RUN set -eux; \
    /usr/bin/python3 -m pip install --no-cache-dir -i "${PIP_INDEX}" \
        httpx h2 openai tree_sitter; \
    /usr/bin/python3 -m pip install --no-cache-dir -i "${PIP_INDEX}" /opt/qitos-src

# cybergym_agent (pinned submodule) as a TOP-LEVEL package on PYTHONPATH.
COPY third_party/cybergym_agent /opt/cyber-agent/cybergym_agent
ENV PYTHONPATH=/opt/cyber-agent
RUN /usr/bin/python3 -c "import qitos, cybergym_agent; from cybergym_agent import run_local; print('qitos + cybergym_agent import OK')"

# Cage in-container proxy (standalone, httpx-only) — same container path Cage's
# host side copies the sidecar to at trial start (baked for parity/offline).
COPY cage/proxy/sidecar.py /opt/cage-proxy/container_proxy.py

# Agent user (ctfenv base runs as root by default; Cage launches as --user agent).
# Pre-create /opt/cage-agent so the custom-agent setup_container (docker cp the
# entrypoint + chown, as the image's default user) works.
ARG AGENT_SUDO_PASSWORD=cage
RUN useradd -m -s /bin/bash agent \
    && mkdir -p /home/agent/workspace /opt/cage-agent \
    && chown -R agent:agent /home/agent /opt/cage-agent \
    && apt-get update \
    && apt-get install -y --no-install-recommends sudo \
    && echo "agent:${AGENT_SUDO_PASSWORD}" | chpasswd \
    && usermod -aG sudo agent \
    && rm -rf /var/lib/apt/lists/*

ENV HOME=/home/agent
CMD ["sleep", "infinity"]
