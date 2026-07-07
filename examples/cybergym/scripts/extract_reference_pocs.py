#!/usr/bin/env python3
"""Extract the reference (ground-truth) PoC bundled inside each n132/arvo *vul*
image, so every external-ARVO environment carries the canonical crashing input
next to its built binaries.

ARVO ships the crashing testcase at ``/tmp/poc`` inside the ``-vul`` image (the
exact bytes that trip the sanitizer on the vulnerable build). The vul image tars
are OCI-layout archives; ``tmp/poc`` lives in one layer (empirically the 2nd-from-
top). We read only the tar's member index (header seeks — NOT the whole 1 GB),
then pull ``tmp/poc`` from the topmost layer that carries it (== effective
overlay copy).

Output: <out>/<id>/poc   (co-located with vul/ and fix/ in the env dir).
Idempotent: skips ids whose poc already exists.
"""
import argparse, json, os, sys, io, gzip, tarfile, concurrent.futures as cf


def _poc_from_layer(image, member):
    """Return tmp/poc bytes from one layer blob, or None. Streams the layer
    (stops at the first tmp/poc) so a poc-less top layer isn't fully buffered."""
    f = image.extractfile(member)
    if f is None:
        return None
    # layers are gzip tarballs; r:* autodetects (some are plain tar/zstd-less)
    try:
        lt = tarfile.open(fileobj=f, mode="r|*")  # streaming
    except tarfile.TarError:
        return None
    for lm in lt:
        if lm.name in ("tmp/poc", "./tmp/poc") and lm.isfile():
            return lt.extractfile(lm).read()
    return None


def extract_one(tar_path, out_dir):
    idnum = os.path.basename(tar_path).split("-")[0]
    dest_dir = os.path.join(out_dir, idnum)
    dest = os.path.join(dest_dir, "poc")
    if os.path.exists(dest):
        return idnum, "skip"
    try:
        with tarfile.open(tar_path) as image:        # seekable -> header-only scan
            idx = {m.name: m for m in image.getmembers()}
            mf = idx.get("manifest.json") or idx.get("./manifest.json")
            if mf is None:
                return idnum, "no-manifest"
            layers = json.load(image.extractfile(mf))[0]["Layers"]
            data = None
            for blob in reversed(layers):            # fast path: topmost overlay wins
                m = idx.get(blob) or idx.get("./" + blob)
                if m is None:
                    continue
                data = _poc_from_layer(image, m)
                if data is not None:
                    break
            if data is None:
                # fallback: manifest digests can mismatch blob names; scan every
                # blob for tmp/poc (it occurs once, so order is unambiguous).
                for m in idx.values():
                    if not m.name.startswith(("blobs/sha256/", "./blobs/sha256/")):
                        continue
                    data = _poc_from_layer(image, m)
                    if data is not None:
                        break
            if data is not None:
                os.makedirs(dest_dir, exist_ok=True)
                tmp = dest + ".part"
                with open(tmp, "wb") as w:
                    w.write(data)
                os.replace(tmp, dest)                # same-fs rename, atomic
                return idnum, "ok"
        return idnum, "no-poc"
    except Exception as e:
        return idnum, f"err:{type(e).__name__}:{str(e)[:60]}"


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    cybergym = os.path.dirname(here)                      # examples/cybergym
    ap = argparse.ArgumentParser()
    ap.add_argument("--vulcache",
                    default=os.environ.get("VULCACHE", os.path.join(cybergym, "datasets", "image-cache", "n132_arvo")),
                    help="dir of n132/arvo <id>-vul.tar image tars")
    ap.add_argument("--out",
                    default=os.environ.get("ARVO_BUILD_ROOT", os.path.join(cybergym, "datasets", "server-binary-arvo", "arvo")),
                    help="build root; writes <out>/<id>/poc alongside vul/ and fix/")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", default="", help="comma/space ids; default=all vul tars")
    args = ap.parse_args()

    if args.ids:
        want = args.ids.replace(",", " ").split()
        tars = [os.path.join(args.vulcache, f"{i}-vul.tar") for i in want]
        tars = [t for t in tars if os.path.isfile(t)]
    else:
        tars = [os.path.join(args.vulcache, f) for f in os.listdir(args.vulcache)
                if f.endswith("-vul.tar")]
    tars.sort()
    if args.limit:
        tars = tars[:args.limit]
    print(f"[poc] {len(tars)} vul tars -> {args.out} ({args.workers} workers)", flush=True)

    counts, done = {}, 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for idnum, status in ex.map(lambda t: extract_one(t, args.out), tars):
            key = status if status in ("ok", "skip", "no-poc", "no-manifest") else "err"
            counts[key] = counts.get(key, 0) + 1
            done += 1
            if key not in ("ok", "skip") or done % 200 == 0:
                print(f"[poc] {done}/{len(tars)} id={idnum} -> {status} | {counts}", flush=True)
    print(f"[poc] DONE {counts}", flush=True)


if __name__ == "__main__":
    main()
