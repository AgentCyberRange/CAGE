#!/usr/bin/env bash
# Build the QitOS CyberGym agent image for Cage.
#
# ONE build path for every agent: `cage agent build --agent qitos` runs this
# (declared as build.script in cage/agents/custom/qitos/agent.yml). Unlike cairn
# (which bakes multiple images), qitos needs only a single `docker build` — but
# it must first ensure the pinned submodules the Dockerfile copies from are
# checked out, so this thin script guards that and then builds.
set -euo pipefail
cd "$(dirname "$0")/.."   # cage repo root

TAG="${QITOS_IMAGE:-cage/qitos:latest}"

# The Dockerfile COPYs the harness from the pinned submodules; a fresh clone may
# not have them checked out. Fetch them if missing (idempotent).
for sm in third_party/qitos third_party/cybergym_agent; do
  if [ ! -e "$sm/.git" ] && [ -z "$(ls -A "$sm" 2>/dev/null)" ]; then
    echo "==> submodule $sm not checked out — fetching"
    git submodule update --init "$sm"
  fi
done
if [ ! -d third_party/qitos/qitos ] || [ ! -f third_party/cybergym_agent/run_local.py ]; then
  echo "ERROR: submodules missing expected files; run:" >&2
  echo "       git submodule update --init third_party/qitos third_party/cybergym_agent" >&2
  exit 1
fi

echo "==> submodule pins (traceability):"
git submodule status third_party/qitos third_party/cybergym_agent | sed 's/^/    /'

echo "==> docker build $TAG"
docker build -f docker/qitos/Dockerfile -t "$TAG" .
echo "==> done: $TAG"
