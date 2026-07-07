#!/usr/bin/env bash
# Download missing arvo-external **fix** images and turn them into binary-only
# grading data, so vul-only tasks become both-builds (binary-gradable).
#
# Why this exists: the arvo-external fix images are not in any local cache and
# the CN docker mirrors throttle n132/arvo blobs to failure. The only route that
# works here is crane pulling **direct from Docker Hub through the LAN proxy**
# (~3.6 MB/s/stream — slow but reliable). Per task: crane-pull the fix image to a
# transient tar, extract its arvo/out/libs (and the vul side from the LOCAL vul
# cache), then delete the tar. Disk stays bounded; only the ~tens-of-MB binary
# set grows. Resumable (skips tasks already extracted) and parallel (-P JOBS).
#
# Env knobs: OUT (binary_dir to populate), JOBS (parallel pulls), PROXY.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASETS="$HERE/../datasets"
# crane binary: expected on PATH by default. Point CRANE at an explicit path if you
# keep it elsewhere — but NOT on a scratch/tmp disk that gets wiped, or every pull
# fails with "No such file or directory" (→ instant pull-FAIL).
CRANE="${CRANE:-crane}"
EXTRACT="$HERE/extract_server_binaries.py"
# VULCACHE: dir of local n132/arvo *-vul* image tars (the vul side of each pair).
VULCACHE="${VULCACHE:-$DATASETS/image-cache}"
# DLCACHE (transient fix tars) and OUT (extracted binaries) should live on a large
# persistent disk, NOT a small root/scratch disk that fills fast. Defaults populate
# the benchmark's own datasets/ (server-binary-arvo == default_arvo.yml binary_dir).
DLCACHE="${DLCACHE:-$DATASETS/.dlcache}"
OUT="${OUT:-$DATASETS/server-binary-arvo}"
# Target list: need-fix tasks whose LOCAL vul tar is REAL (so the pull completes
# a both-builds pair). Smallest-vul-first for quick wins. LIMIT caps the run.
IDS_FILE="${IDS_FILE:-$HERE/../datasets/arvo_needfix_realvul.txt}"
LIMIT="${LIMIT:-0}"
JOBS="${JOBS:-4}"
PROXY="${PROXY:-http://127.0.0.1:7890}"
MIN_FREE_GB="${MIN_FREE_GB:-80}"     # pause a worker if /tmp filesystem dips under this
export HTTPS_PROXY="$PROXY" HTTP_PROXY="$PROXY"

mkdir -p "$DLCACHE/n132_arvo" "$OUT"
LOG="$OUT/download_fix.log"

process() {
  local id="$1"
  local fixmark="$OUT/arvo/$id/fix/arvo"
  if [ -f "$fixmark" ]; then echo "skip-done $id"; return; fi
  # disk guard: don't start a multi-GB pull if /tmp is low
  local freegb; freegb=$(df -BG --output=avail "$DLCACHE" 2>/dev/null | tail -1 | tr -dc '0-9')
  if [ -n "$freegb" ] && [ "$freegb" -lt "$MIN_FREE_GB" ]; then echo "lowdisk-skip $id (${freegb}G)"; return; fi

  local tar="$DLCACHE/n132_arvo/$id-fix.tar" t0; t0=$(date +%s)
  if ! timeout 1800 "$CRANE" pull "n132/arvo:$id-fix" "$tar" >/dev/null 2>&1; then
    echo "pull-FAIL $id ($(( $(date +%s)-t0 ))s)"; rm -f "$tar"; return
  fi
  # extract fix/ from the freshly pulled tar; vul/ from the local vul cache
  python3 "$EXTRACT" --image-cache "$DLCACHE"  --task-ids "$id" --out "$OUT" --allow-partial >/dev/null 2>&1
  python3 "$EXTRACT" --image-cache "$VULCACHE" --task-ids "$id" --out "$OUT" --allow-partial >/dev/null 2>&1
  rm -f "$tar"
  if [ -f "$fixmark" ] && [ -f "$OUT/arvo/$id/vul/arvo" ]; then
    echo "OK $id ($(( $(date +%s)-t0 ))s)"
  else
    echo "partial $id (fix=$([ -f "$fixmark" ] && echo y || echo n) vul=$([ -f "$OUT/arvo/$id/vul/arvo" ] && echo y || echo n))"
  fi
}
export -f process
export CRANE EXTRACT VULCACHE DLCACHE OUT MIN_FREE_GB HTTPS_PROXY HTTP_PROXY

# ids: lines like "n132/arvo:888-fix" -> "888"
mapfile -t IDS < <(cut -d: -f2 "$IDS_FILE" | sed 's/-fix//' | grep -E '^[0-9]+$')
[ "$LIMIT" -gt 0 ] && IDS=("${IDS[@]:0:$LIMIT}")
echo "=== download_fix_binaries: ${#IDS[@]} ids, JOBS=$JOBS, OUT=$OUT, proxy=$PROXY ===" | tee -a "$LOG"
printf '%s\n' "${IDS[@]}" \
  | xargs -P "$JOBS" -I{} bash -c 'process "$@"' _ {} \
  | tee -a "$LOG" \
  | stdbuf -oL grep -E '^(OK|pull-FAIL|partial|lowdisk)' \
  | awk '{print} END{}'
echo "=== done; tally: ===" | tee -a "$LOG"
grep -cE '^OK ' "$LOG" | xargs echo "OK total (cumulative in log):" | tee -a "$LOG"
