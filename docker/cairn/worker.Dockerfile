# Cairn *worker* image — the container Cairn's Dispatcher spawns (inside the
# trial's inner Docker daemon) and `docker exec`s `claude ... -p` into.
#
# It is the engine image plus `USER agent`: claude-code refuses
# `--dangerously-skip-permissions` when uid==0, and Cairn's `claudecode` driver
# always passes that flag, so the worker's DEFAULT user must be non-root. The
# engine image keeps root as default (its `setup_container` does docker cp +
# chown); only the worker variant flips the default user.
#
# For capability parity with native Cairn, point the agent's `worker_image`
# param at the full Cairn Kali worker (ghcr.io/oritera/cairn-worker-container)
# instead — it already ships `USER kali` + the full toolchain. This lean variant
# (claude + nmap/nc/ssh/sshpass) is the self-contained default that needs no
# external pull.
# Built FROM the un-baked engine tag (see docker/build_cairn.sh) to avoid a
# chicken-and-egg with the final `cage/cairn:latest` (which bakes THIS image's
# tar in). Override with --build-arg BASE=... for the full Cairn Kali worker.
ARG BASE=cage/cairn:engine
FROM ${BASE}

# claude-code: stay headless/quiet inside the sealed worker container.
ENV CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1

# Cairn's worker prompt/AGENTS context expects a writable workspace.
RUN mkdir -p /home/agent/workspace && chown -R agent:agent /home/agent/workspace
WORKDIR /home/agent/workspace

# claude-code refuses --dangerously-skip-permissions as root; the worker's
# DEFAULT user must be non-root (the engine keeps root for Cage setup_container).
USER agent
CMD ["sleep", "infinity"]
