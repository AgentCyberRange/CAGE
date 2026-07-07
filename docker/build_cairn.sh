#!/usr/bin/env bash
# Build the Cairn agent images for Cage.
#
# The inner Docker daemon (DinD) inside each trial container starts EMPTY, so the
# Claude-Code worker image must arrive as a tar baked into the engine image
# (no registry holds our local image). This produces three tags:
#
#   cage/cairn:engine        — engine: dockerd + vendored Cairn + claude (root default)
#   cage/cairn-worker:latest — worker: engine + USER agent (non-root claude)
#   cage/cairn:latest        — engine + the worker tar baked at /opt/cairn/worker-image.tar
#
# The agent manifest uses cage/cairn:latest; the entrypoint `docker load`s the
# baked tar into the inner daemon. For full Cairn Kali capability parity, set
# worker_image=ghcr.io/oritera/cairn-worker-container in the run yaml and make
# the inner daemon able to pull it instead (then this bake is unnecessary).
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root (build context)

ENGINE_BASE=cage/cairn:engine
WORKER=cage/cairn-worker:latest
ENGINE=cage/cairn:latest

echo "==> [1/3] engine base: ${ENGINE_BASE}"
docker build -f docker/cairn.Dockerfile -t "${ENGINE_BASE}" .

echo "==> [2/3] worker (engine + USER agent): ${WORKER}"
docker build -f docker/cairn_worker.Dockerfile --build-arg BASE="${ENGINE_BASE}" -t "${WORKER}" .

echo "==> [3/3] bake worker tar into final engine: ${ENGINE}"
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT
docker save "${WORKER}" -o "${tmp}/worker-image.tar"
cat > "${tmp}/Dockerfile" <<EOF
FROM ${ENGINE_BASE}
RUN mkdir -p /opt/cairn
COPY worker-image.tar /opt/cairn/worker-image.tar
EOF
docker build -t "${ENGINE}" -f "${tmp}/Dockerfile" "${tmp}"

echo "==> done:"
docker images | grep -E 'cage/cairn' | head
