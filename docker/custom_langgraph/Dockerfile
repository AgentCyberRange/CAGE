# Runtime base for custom LangGraph/LangChain agents (image: cage/custom-langgraph:base).
#
# Carries python + the LangGraph stack + the Cage sidecar runtime, but NOT any
# agent's code — that is copied in at container start (CustomAgent.setup_container),
# so editing the agent needs no image rebuild. Same pentest-env base as the
# other agents, so an agent's shell tools and the sidecar proxy work unchanged
# on the benchmarks Cage ships (e.g. CyberGym's submit.sh / workspace).
#
# Build:  cage agent build  is for built-in agents; build this base directly:
#   docker build -f docker/custom_langgraph.Dockerfile -t cage/custom-langgraph:base .

FROM pursu1ng/cage-images:pentest-env

ENV DEBIAN_FRONTEND=noninteractive

# The pentest-env base pins apt + pip to Chinese mirrors (TUNA), unreachable
# outside mainland China. Reset both to global upstreams so this builds anywhere.
RUN { \
      . /etc/os-release; \
      echo "deb http://archive.ubuntu.com/ubuntu/ $VERSION_CODENAME main restricted universe multiverse"; \
      echo "deb http://archive.ubuntu.com/ubuntu/ $VERSION_CODENAME-updates main restricted universe multiverse"; \
      echo "deb http://archive.ubuntu.com/ubuntu/ $VERSION_CODENAME-backports main restricted universe multiverse"; \
      echo "deb http://security.ubuntu.com/ubuntu/ $VERSION_CODENAME-security main restricted universe multiverse"; \
    } > /etc/apt/sources.list \
    && printf '[global]\nindex-url = https://pypi.org/simple\ntimeout = 120\n' > /etc/pip.conf

# The custom agent runs as the unprivileged `agent` user, whose `python3` is the
# system /usr/bin/python3 (3.10 in this base) — NOT root's miniconda python. So
# the LangGraph stack + OpenAI client (what agentic_poc/llm.py imports) MUST be
# installed into /usr/bin/python3, or `python3 -m <agent>` can't import langgraph.
# 3.10 takes langchain-core 1.x; pin <2 as the only guard (the trace hook's
# public register_configure_hook API is stable across majors, and LangGraph stamps
# metadata['langgraph_node'] on 1.x).
RUN /usr/bin/python3 -m pip install --no-cache-dir \
      langgraph 'langchain-core<2' langchain-openai

# The sidecar proxy is run (as root) via the PATH python3; give BOTH interpreters
# its lean runtime deps (httpx + h2 — the sidecar invariant), independent of the
# agent stack above.
RUN set -eux; \
    for py in python3 /usr/bin/python3; do \
      if command -v "$py" >/dev/null 2>&1; then \
        "$py" -m pip install --no-cache-dir httpx h2; \
      fi; \
    done

# Cage sidecar proxy (also copied in at runtime by the host; baked for parity).
COPY cage/proxy/sidecar.py /opt/cage-proxy/container_proxy.py

# Cage's LangChain auto-trace runtime: a `cage_trace` package + a `sitecustomize`
# that imports it at interpreter startup. Put it on PYTHONPATH so EVERY python in
# the container auto-loads it; when CAGE_TRACE is set (CustomAgent.env_vars) it
# attaches a global LangChain callback that records per-node spans — zero code in
# the agent. With CAGE_TRACE unset it returns before importing langchain (cheap
# no-op), so the lean sidecar process is unaffected.
COPY cage/agents/custom/trace_runtime/ /opt/cage-trace/
ENV PYTHONPATH=/opt/cage-trace

# Where CustomAgent copies the agent's source; pre-create so the runtime
# `docker cp` lands cleanly and is owned by the agent.
RUN mkdir -p /opt/cage-agent/src

# Unprivileged agent user (pentest-env runs as root by default), matching the
# other agent images: known sudo password + raw-socket caps.
ARG AGENT_SUDO_PASSWORD=cage
RUN useradd -m -s /bin/bash agent \
    && mkdir -p /home/agent/workspace \
    && chown -R agent:agent /home/agent /opt/cage-agent \
    && apt-get update \
    && apt-get install -y --no-install-recommends sudo libcap2-bin \
    && echo "agent:${AGENT_SUDO_PASSWORD}" | chpasswd \
    && usermod -aG sudo agent \
    && for bin in nmap fscan masscan hping3 ping arping tcpdump; do \
         path=$(command -v "$bin" 2>/dev/null || true); \
         if [ -n "$path" ]; then setcap cap_net_raw,cap_net_admin+eip "$(readlink -f "$path")" || true; fi; \
       done \
    && rm -rf /var/lib/apt/lists/*

ENV HOME=/home/agent

CMD ["sleep", "infinity"]
