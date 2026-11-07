#!/usr/bin/env python3
"""Cut the vision tower out of the DeepSeek-V4.1-Flash checkpoint into one safetensors file.

usage: vision_extract.py CKPT_DIR OUT.safetensors
Keeps vision.*, aligner.* and image_start / image_end / image_newline exactly as stored (byte copies,
no dtype change). Stdlib only."""
import hashlib, json, os, struct, sys

ckpt, out = sys.argv[1], sys.argv[2]
wm = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))["weight_map"]
want = sorted(n for n in wm if n.startswith(("vision.", "aligner.")) or n in ("image_start", "image_end", "image_newline"))
by_file = {}
for n in want:
    by_file.setdefault(wm[n], []).append(n)

tensors = {}  # name -> (dtype, shape, bytes)
for f, names in sorted(by_file.items()):
    with open(os.path.join(ckpt, f), "rb") as fp:
        hlen = struct.unpack("<Q", fp.read(8))[0]
        hdr = json.loads(fp.read(hlen))
        base = 8 + hlen
        for n in names:
            m = hdr[n]
            a, b = m["data_offsets"]
            fp.seek(base + a)
            tensors[n] = (m["dtype"], m["shape"], fp.read(b - a))

meta, off = {}, 0
for n in sorted(tensors):
    dt, shape, data = tensors[n]
    meta[n] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + len(data)]}
    off += len(data)
meta["__metadata__"] = {"format": "pt", "source": "DeepSeek-V4.1-Flash-1017 vision tower (vision_extract.py)"}
h = json.dumps(meta, separators=(",", ":")).encode()
h += b" " * (-len(h) % 8)
tmp = out + ".tmp"
with open(tmp, "wb") as fp:
    fp.write(struct.pack("<Q", len(h)))
    fp.write(h)
    for n in sorted(tensors):
        fp.write(tensors[n][2])
os.replace(tmp, out)
dts = sorted({v[0] for v in tensors.values()})
print(f"{len(tensors)} tensors, {off / 2**30:.3f} GiB, dtypes {dts} -> {out}")
sha = hashlib.sha256(open(out, "rb").read()).hexdigest()
print("sha256", sha)
