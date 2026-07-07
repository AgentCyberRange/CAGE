#!/usr/bin/env python3
"""Classify the BUILT external-ARVO environments with the subset100 scheme.

"External ARVO" = the 3480-task ARVO catalog NOT in CyberGym-1507. We have
*built* (vul+fix binaries extracted) a large subset of them into
  arvo-external-server-data/arvo/<id>/{vul,fix}/{arvo,out,libs}
This script classifies every BUILT environment by the SAME vuln/input buckets
used in sample.py for the 1507->100 subset, so the two are directly comparable.

Accuracy notes — what makes this more accurate than a pure arvo.json lookup:
 * input-type (`inp`) keys off the REAL fuzz harness name, recovered from each
   built env's out/ dir (e.g. ping_ttf_fuzzer -> font), not just the project.
   arvo.json ships no harness field, so this signal only exists post-build.
 * vuln-type (`vuln`) keys off arvo.json crash_type (sanitizer string).
 * project/crash_type come from datasets/arvo.json (covers all 3480 ext tasks).
"""
import json, os, collections, io, sys

HERE     = os.path.dirname(os.path.abspath(__file__))                 # examples/cybergym/external_arvo
CYBERGYM = os.path.dirname(HERE)                                       # examples/cybergym
OUT_ROOT = os.environ.get("ARVO_BUILD_ROOT", f"{CYBERGYM}/datasets/server-binary-arvo/arvo")

arvo = json.load(open(f"{CYBERGYM}/datasets/arvo.json"))

# ---- buckets: verbatim from sample.py (the repo's classification scheme) -----
PROJECT_MAP={
 'c-blosc2':'archive','miniz':'archive',
 'gpac':'av_media','libxaac':'av_media','libavc':'av_media','libhevc':'av_media','libsndfile':'av_media',
 'faad2':'av_media','gstreamer':'av_media',
 'gdal':'image','leptonica':'image','libjxl':'image','libraw':'image','libheif':'image','exiv2':'image',
 'lcms':'image','openexr':'image','libvips':'image','libultrahdr':'image','skcms':'image','stb':'image',
 'assimp':'3d_model','tinygltf':'3d_model',
 'upx':'binary_exec','radare2':'binary_exec','sleuthkit':'binary_exec','libbpf':'binary_exec',
 'util-linux':'binary_exec','file':'binary_exec','libgit2':'binary_exec','yara':'binary_exec',
 'selinux':'lang_config','sudoers':'lang_config','hunspell':'lang_config','libucl':'lang_config',
 'liblouis':'lang_config','pcre2':'lang_config',
 'libplist':'markup_data','icu':'markup_data','readstat':'markup_data','matio':'markup_data','libical':'markup_data',
 'htslib':'data','arrow':'data','hdf5':'data','perfetto':'data','fluent-bit':'data',
 'openthread':'network','curl':'network','open62541':'network','mosquitto':'network','usrsctp':'network',
 'freeradius':'network','h2o':'network','libzmq':'network','wpantund':'network','zeek':'network',
 'libssh2':'network','krb5':'network','lwan':'network',
 'h3':'geo','proj4':'geo','geos':'geo','mapserver':'geo','igraph':'geo',
 'libtpms':'crypto_cert','libspectre':'document',
}
# --- accuracy extension: projects seen in the BUILT external set whose WHOLE
# domain is one input-modality (so a project->bucket map is correct regardless
# of which harness). Multi-modal projects (serenity/qtbase/glib/...) are left
# OUT on purpose so per-harness keyword bucketing decides them. -------------
PROJECT_MAP.update({
 # network / protocol parsers
 'uwebsockets':'network','envoy':'network','grpc':'network','mongoose':'network',
 'tor':'network','openvswitch':'network','httpd':'network','nginx':'network',
 'trafficserver':'network','dropbear':'network','dovecot':'network','ntpsec':'network',
 'open5gs':'network','lldpd':'network','libiec61850':'network','s2opc':'network',
 'wget2':'network','libpsl':'network','irssi':'network','freeradius-server':'network',
 # crypto / certificates / security protocols
 'rnp':'crypto_cert','p11-kit':'crypto_cert','libspdm':'crypto_cert',
 # image codecs / raster
 'opencv':'image','graphicsmagick':'image','libavif':'image','brunsli':'image','simd':'image',
 # 3D / point-cloud / scene
 'pcl':'3d_model','alembic':'3d_model',
 # geospatial
 'proj.4':'geo','postgis':'geo','s2geometry':'geo',
 # language / script / bytecode parsers
 'oniguruma':'lang_script','tint':'lang_script','moddable':'lang_script','re2':'lang_script',
 'wabt':'lang_script','node':'lang_script','dawn':'lang_script','llvm-project':'lang_script',
 'glog':'lang_script',
 # markup / serialization / unicode
 'libsass':'markup_data','flatbuffers':'markup_data','firebase-ios-sdk':'markup_data',
 'fribidi':'markup_data','simdutf':'markup_data',
 # databases
 'mdbtools':'db',
 # structured scientific / columnar data
 'rdkit':'data','croaring':'data',
 # config / dictionaries
 'lxc':'lang_config','aspell':'lang_config',
 # audio / video / speech / subtitles
 'vlc':'av_media','libass':'av_media','espeak-ng':'av_media',
 # executables / filesystems / binary formats
 'clamav-devel':'binary_exec','e2fsprogs':'binary_exec',
 # --- fix systematic keyword false-matches (highest-priority project map) ---
 # 'json' contains substring 'js' -> wrongly hit the javascript keyword;
 # these are data-serialization parsers, not script engines.
 'json':'markup_data','simdjson':'markup_data','valijson':'markup_data',
 'jsoncons':'markup_data','rapidjson':'markup_data','expat':'markup_data',
 # '*_parse_*' harnesses wrongly hit the generic 'parse' keyword; these are
 # network/protocol or config-file domains, not scripting languages.
 'c-ares':'network','kamailio':'network','libcoap':'network','spice-usbredir':'network',
 'systemd':'lang_config',
})
BUCKETS=[('image',['tiff','png','jpeg','jpg','gif','bmp','webp','image','coder','rawspeed','skia','pixel','exif','openjp','jbig','jp2','ico','raw']),
 ('document',['pdf','postscript','gstoraster','ghostscript','mupdf','poppler','gs_','docx','rtf','spectre']),
 ('font',['font','freetype','harfbuzz','ots','woff','ttf','otf','sfnt']),
 ('av_media',['ffmpeg','codec','vpx','opus','flac','aom','audio','video','av1','h264','hevc','matroska','wav','sndfile','aac']),
 ('network',['packet','wireshark','fuzzshark','pcap','dns','rtp','sip','snmp','process_packet','dissect','ndpi','tcp','quic','mqtt','sctp','thread']),
 ('crypto_cert',['crypto','cert','x509','pkcs','asn1','openssl','rsa','tls','ssl','opensc','iasecc','gnutls','botan','key','tpm','krb']),
 ('archive',['zip','tar','gzip','zlib','brotli','zstd','archive','7z','unrar','lz4','xz','cab','bzip','blosc','miniz']),
 ('lang_script',['mruby','ruby','lua','js','javascript','wasm','php','python','sql','regex','jq','expr','lexer','parse','interp','njs','quickjs','hermes']),
 ('markup_data',['xml','json','yaml','html','css','toml','xslt','xpath','svg','proto','protobuf','cbor','msgpack','plist','unicode']),
 ('binary_exec',['elf','pe_','mach','dwarf','disas','bfd','binutils','objdump','capstone','disassemble','readelf','llvmfuzz','radare','bpf']),
 ('db',['sqlite','leveldb','rocksdb','db_','lmdb','duckdb']),
]
def ibucket(project, harness):
    p=(project or '').lower()
    if p in PROJECT_MAP: return PROJECT_MAP[p]
    h=(harness or '').lower()
    for name,kws in BUCKETS:
        if any(k in h for k in kws) or any(k in p for k in kws): return name
    return 'other'
def vbucket(crash):
    # NOTE: external ARVO crash_type uses a DIFFERENT vocabulary from the
    # features_1507 sanitizer strings ("Heap-buffer-overflow WRITE 8",
    # "UNKNOWN READ", "Bad-cast", "Null-dereference", ...). Keyword set is
    # extended to that vocabulary so the external buckets stay accurate and
    # comparable to the 1507 scheme.
    c=(crash or '').lower()
    if not c: return 'unknown'
    # ASan "UNKNOWN READ/WRITE" = access to an undetermined memory region; the
    # crash category is genuinely uncategorised -> 'unknown', not 'other'.
    if c.startswith('unknown'): return 'unknown'
    for name,kws in [('heap-overflow',['heap-buffer-overflow']),('uninit',['uninitialized']),
        ('segv',['segv','null-dereference','null-deref','wild']),
        ('stack-overflow',['stack-buffer-overflow','stack-overflow']),
        ('uaf',['use-after-free','use-after-poison','use-after-return','use-after-scope','heap-use-after']),
        ('global-overflow',['global-buffer-overflow']),
        ('double-free',['double-free','attempting free','invalid-free','bad-free','free on address']),
        ('ubsan',['runtime error','overflow','shift','divide','misaligned','out of bounds','index',
                  'bad-cast','object-size','vla-bound','function-pointer-type','non-positive','undefined']),
        ('leak',['leak']),('timeout-oom',['timeout','out-of-memory','oom']),('underflow',['underflow']),('container',['container'])]:
        if any(k in c for k in kws): return name
    return 'other'

# ---- recover the real harness name from a built env's out/ dir --------------
DATA_EXT_OK = False  # we treat any name with a '.' as a data/seed file
def harness_of(idnum):
    for half in ('vul','fix'):
        od=f"{OUT_ROOT}/{idnum}/{half}/out"
        if not os.path.isdir(od): continue
        files=os.listdir(od)
        if not files: continue
        # data/seed files carry extensions; the fuzzer binary does not.
        nodot=[f for f in files if '.' not in f]
        cand = nodot or files
        if len(cand)==1: return cand[0]
        # multiple no-dot binaries: prefer a fuzz-named one, else executable
        fz=[f for f in cand if 'fuzz' in f.lower()]
        if len(fz)==1: return fz[0]
        if fz: cand=fz
        ex=[f for f in cand if os.access(f"{od}/{f}", os.X_OK)]
        if len(ex)==1: return ex[0]
        return (ex or cand)[0]
    return None

# ---- enumerate BUILT envs (both vul+fix binaries present) -------------------
ids=[]
for d in sorted(os.listdir(OUT_ROOT), key=lambda x:(not x.isdigit(), int(x) if x.isdigit() else x)):
    if os.path.isfile(f"{OUT_ROOT}/{d}/vul/arvo") and os.path.isfile(f"{OUT_ROOT}/{d}/fix/arvo"):
        ids.append(d)

import hashlib
def poc_status(idnum):
    p=f"{OUT_ROOT}/{idnum}/poc"
    if not os.path.isfile(p): return (False, None, None)
    b=open(p,'rb').read()
    return (True, len(b), hashlib.sha256(b).hexdigest())

rows={}
for idnum in ids:
    tid=f"arvo:{idnum}"
    meta=arvo.get(tid, {})
    project=meta.get('project'); crash=meta.get('crash_type')
    h=harness_of(idnum)
    has_poc, poc_size, poc_sha = poc_status(idnum)
    rows[tid]=dict(idnum=idnum, project=project, harness=h, crash_type=crash,
                   lang=meta.get('language') or 'unknown',
                   vuln=vbucket(crash), inp=ibucket(project, h),
                   has_vul=True, has_fix=True,
                   has_poc=has_poc, poc_size=poc_size, poc_sha256=poc_sha,
                   env_dir=f"{OUT_ROOT}/{idnum}",
                   in_arvo_json=tid in arvo)

# ---- write per-task classification + features -------------------------------
json.dump(rows, open(f"{HERE}/features_external_built.json","w"), indent=0)

# ---- authoritative per-environment DEFINITION manifest (environments.jsonl) --
# One self-contained line per built external-ARVO env: what it is, how it
# crashes, what input modality, and the on-disk artifacts (vul/fix binaries +
# reference PoC) that make it runnable.
with open(f"{HERE}/environments.jsonl","w") as fh:
    for tid in sorted(rows, key=lambda t:int(t.split(':')[1])):
        r=rows[tid]
        rec=dict(
            task_id=tid, source="arvo", id_num=r['idnum'],
            project=r['project'], language=r['lang'],
            harness=r['harness'],
            crash_type=r['crash_type'],            # raw sanitizer string
            vuln_class=r['vuln'], input_class=r['inp'],
            artifacts=dict(
                vul_binary=f"{r['env_dir']}/vul/arvo",
                fix_binary=f"{r['env_dir']}/fix/arvo",
                reference_poc=(f"{r['env_dir']}/poc" if r['has_poc'] else None),
                poc_size=r['poc_size'], poc_sha256=r['poc_sha256'],
            ),
            grading="binary",                      # vul crashes / fix does not
        )
        fh.write(json.dumps(rec)+"\n")

with open(f"{HERE}/classification_external.tsv","w") as fh:
    fh.write("task_id\tproject\tharness\tcrash_type\tvuln\tinp\n")
    for tid in sorted(rows, key=lambda t:int(t.split(':')[1])):
        r=rows[tid]
        fh.write(f"{tid}\t{r['project']}\t{r['harness']}\t{r['crash_type']}\t{r['vuln']}\t{r['inp']}\n")

# ---- distribution report (same shape as report.md) --------------------------
buf=io.StringIO()
N=len(rows)
buf.write(f"# External-ARVO built environments — classification ({N} envs)\n\n")
buf.write("Same vuln/input buckets as `sample.py` (the 1507->100 scheme). These are the\n")
buf.write("ARVO tasks NOT in CyberGym-1507 that we have BUILT (vul+fix binaries present in\n")
buf.write("`arvo-external-server-data/arvo/<id>/`). Input-type uses the REAL harness name\n")
buf.write("recovered from each env's `out/` dir; vuln-type uses arvo.json crash_type.\n\n")
for d in ['vuln','inp']:
    c=collections.Counter(r[d] for r in rows.values())
    buf.write(f"## {d}\n\n| value | count | pct |\n|---|---|---|\n")
    for k,v in c.most_common():
        buf.write(f"| {k} | {v} | {100*v/N:.1f}% |\n")
    buf.write("\n")
# project top-30
buf.write("## top projects\n\n| project | count |\n|---|---|\n")
for k,v in collections.Counter(r['project'] for r in rows.values()).most_common(30):
    buf.write(f"| {k} | {v} |\n")
open(f"{HERE}/report_external.md","w").write(buf.getvalue())

# ---- console summary --------------------------------------------------------
print(f"built external-ARVO envs classified: {N}")
for d in ['vuln','inp']:
    c=collections.Counter(r[d] for r in rows.values())
    print(f"--- {d} ---")
    for k,v in c.most_common():
        print(f"   {k:14s} {v:5d}  {100*v/N:5.1f}%")
oth=[t for t,r in rows.items() if r['inp']=='other']
print(f"\ninp=other: {len(oth)} ({100*len(oth)/N:.1f}%)  sample projects:",
      collections.Counter(rows[t]['project'] for t in oth).most_common(15))
npoc=sum(1 for r in rows.values() if r['has_poc'])
print(f"reference PoC present: {npoc}/{N} ({100*npoc/N:.1f}%)  [extractor may still be running]")
