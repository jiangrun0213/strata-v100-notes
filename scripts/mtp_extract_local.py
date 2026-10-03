#!/usr/bin/env python3
import json, os, struct, hashlib, sys
INDEX  = "/home/jiangrun02123/strata-webroot/Qwen/Qwen3.8-Flash-Next/resolve/de4b8e4d43b917e7706784d8bb445c9af86a3540/model.safetensors.index.json"
SHARDS = "/home/jiangrun02123/models/flash-next-bf16-shards"
OUT    = "/home/jiangrun02123/Strata-data/mtp"
REPO   = "https://huggingface.co/Qwen/Qwen3.8-Flash-Next"
idx = json.load(open(INDEX))
wm = idx["weight_map"]
mtp = {k: v for k, v in wm.items() if k.startswith("mtp.")}
shards = sorted(set(mtp.values()))
print(f"MTP tensors: {len(mtp)}, shards: {len(shards)}")
def shard_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return 8 + n, json.loads(f.read(n))
tdir = os.path.join(OUT, "tensors")
os.makedirs(tdir, exist_ok=True)
rows, total = [], 0
for shard in shards:
    sp = os.path.join(SHARDS, shard)
    if not os.path.exists(sp):
        sys.exit(f"missing shard: {sp}")
    base, header = shard_header(sp)
    for name, meta in header.items():
        if name in mtp and mtp[name] == shard:
            a, b = meta["data_offsets"]
            rows.append(dict(name=name, shard=shard, dtype=meta["dtype"], shape=meta["shape"], start=base + a, end=base + b - 1, bytes=b - a))
            total += b - a
missing = sorted(set(mtp) - set(r["name"] for r in rows))
if missing:
    sys.exit(f"tensors in index but absent from shard headers: {missing}")
rows.sort(key=lambda r: r["name"])
with open(os.path.join(OUT, "mtp-inventory.json"), "w") as f:
    json.dump(dict(repo=REPO, total_bytes=total, tensors=rows), f, indent=1)
manifest = []
for i, r in enumerate(rows, 1):
    path = os.path.join(tdir, r["name"] + ".bin")
    sp = os.path.join(SHARDS, r["shard"])
    with open(sp, "rb") as src, open(path, "wb") as dst:
        src.seek(r["start"])
        remain = r["bytes"]
        while remain > 0:
            chunk = src.read(min(remain, 1 << 24))
            if not chunk:
                sys.exit(f"short read on {sp}")
            dst.write(chunk)
            remain -= len(chunk)
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    manifest.append(dict(r, file=os.path.relpath(path, OUT), sha256=digest))
    print(f"[{i}/{len(rows)}] {r['name']}  {r['bytes']/1e6:.1f} MB  sha256 {digest[:12]}")
with open(os.path.join(OUT, "mtp-manifest.json"), "w") as f:
    json.dump(manifest, f, indent=1)
print(f"DONE: {total/1e9:.2f} GB in {len(rows)} tensors -> {OUT}")
