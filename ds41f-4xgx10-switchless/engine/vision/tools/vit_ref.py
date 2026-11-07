#!/usr/bin/env python3
"""The reference's vision tower on one image (ref/inference vision.py + image_processor.py), for vit_check.

Run in the SGLang image (torch, PIL, safetensors), with ref/inference on PYTHONPATH:
    vit_ref.py VISION.safetensors IMAGE OUT_PREFIX
Writes OUT_PREFIX.u8 (the letterboxed image, uint8 RGB, what the front-end sends the engine) and
OUT_PREFIX.f32 (the aligner's rows, float32 [n_llm_h * n_llm_w][5120]), and prints the grid. The
pixels are checked to reproduce the reference's patches exactly."""
import io, sys
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageOps
from safetensors.torch import load_file

import image_processor as ip
from vision import Aligner, ViT

weights, image_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
args = SimpleNamespace(vision_patch_size=14, vision_downsample_ratio=3, vision_max_n_token=1024,
                       vision_min_pixels=544 * 544, vision_max_wh_ratio=None, vision_dim=1024, vision_n_heads=16,
                       vision_inter_dim=2816, vision_n_layers=32, vision_rope_theta=10000.0, dim=5120)
record = {"url": image_path}
patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = ip.load_image(record, args)

# the letterboxed pixels, as load_image makes them (vision_max_wh_ratio is None: always ImageOps.pad)
with Image.open(io.BytesIO(ip.load_image_bytes(record))) as src:
    image = src.convert("RGB")
_, _, best_h, best_w = ip.plan_image_grid(image.width, image.height, args)
padded = ImageOps.pad(image, (best_w, best_h), color=(127, 127, 127))
px = np.asarray(padded, dtype=np.uint8)
p = args.vision_patch_size
x = torch.from_numpy(px.astype(np.float32)).permute(2, 0, 1) / 255
x = ((x - 0.5) / 0.5).to(torch.bfloat16)
again = x.reshape(3, n_vit_h, p, n_vit_w, p).permute(1, 3, 0, 2, 4).reshape(n_vit_h * n_vit_w, 3, p, p)
assert torch.equal(again, patches), "the letterboxed pixels do not reproduce the reference's patches"
px.tofile(out + ".u8")

torch.set_default_dtype(torch.bfloat16)
torch.set_default_device("cuda")
vit, aligner = ViT(args), Aligner(args)
sd = load_file(weights)
vit.load_state_dict({k[len("vision."):]: v for k, v in sd.items() if k.startswith("vision.")})
aligner.load_state_dict({k[len("aligner."):]: v for k, v in sd.items() if k.startswith("aligner.")})
with torch.inference_mode():
    emb = aligner(vit(patches.cuda(), n_vit_h, n_vit_w), n_vit_h, n_vit_w)
emb.float().cpu().numpy().tofile(out + ".f32")
print(f"{image_path}: {image.width}x{image.height} -> pixels {best_w}x{best_h}, patches {n_vit_h}x{n_vit_w}, "
      f"tokens {n_llm_h}x{n_llm_w} (span {ip.num_image_tokens(n_llm_h, n_llm_w)})")
print(f"GRID {n_vit_h} {n_vit_w} {n_llm_h} {n_llm_w}")
