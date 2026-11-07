#!/usr/bin/env python3
"""The reference tower in float32 (weights upcast) on the same patches: the noise floor for vit_check.
    vit_fp32.py VISION.safetensors IMAGE OUT.f32"""
import sys
from types import SimpleNamespace
import torch
from safetensors.torch import load_file
import image_processor as ip
from vision import Aligner, ViT
weights, image_path, out = sys.argv[1:4]
args = SimpleNamespace(vision_patch_size=14, vision_downsample_ratio=3, vision_max_n_token=1024,
                       vision_min_pixels=544 * 544, vision_max_wh_ratio=None, vision_dim=1024, vision_n_heads=16,
                       vision_inter_dim=2816, vision_n_layers=32, vision_rope_theta=10000.0, dim=5120)
patches, nh, nw, _, _ = ip.load_image({"url": image_path}, args)
torch.set_default_dtype(torch.float32)
torch.set_default_device("cuda")
torch.backends.cuda.matmul.allow_tf32 = False
vit, aligner = ViT(args), Aligner(args)
sd = {k: v.float() for k, v in load_file(weights).items()}
vit.load_state_dict({k[7:]: v for k, v in sd.items() if k.startswith("vision.")})
aligner.load_state_dict({k[8:]: v for k, v in sd.items() if k.startswith("aligner.")})
with torch.inference_mode():
    emb = aligner(vit(patches.float().cuda(), nh, nw), nh, nw)
emb.cpu().numpy().tofile(out)
print("wrote", out, tuple(emb.shape))
