# Vision

dsgb10 serves images as the reference does (ref/inference: `vision.py`, `image_processor.py`, `model.py`):
the front-end prepares each image's pixels, every rank runs DeepSeek's ViT and aligner on them, and the
prefill embeds the image's span from the aligner's rows. Nothing runs outside the engine and dsfront.

## What the model does with an image

- **Preprocessing** (`image_processor.py`): the picture becomes RGB, its patch grid is planned
  (`plan_image_grid`: at least 544² pixels, at most 1024 LLM tokens, multiples of 14), and it is
  letterboxed onto that grid with Pillow's bicubic resampling and grey 127 bars (`ImageOps.pad`).
- **The tower** (`vision.py`): a 32-layer ViT (dim 1024, 16 heads, 2D RoPE, full attention over the
  patches, SwiGLU MLP), then the aligner: a 3×3 unfold of the patch grid and two linears into 5120.
- **The span** (`image_processor.prepare_vl_inputs`): `IMAGE_START`, then per token row `n_llm_w`
  `IMAGE` slots and one `IMAGE_NEW_LINE`, then `IMAGE_END`. The slots take the aligner's rows in reading
  order; the three delimiters have learned embeddings (`image_start`, `image_newline`, `image_end`).
- **In the LLM** (`model.py`): those positions route with the gate's `bias_vl` in every MoE layer,
  take no part in an Engram n-gram (their compressed id is `DEAD`) and get no Engram contribution (the
  gate is shut). Attention, the indexer and everything else treat them as any position.

## How dsgb10 does it

**dsfront** (`frontend/src/image.rs`). Image blocks (`image_url`, `image`, the Responses API's
`input_image`) are accepted; encoding.rs already puts one `<｜deepseek_image｜>` per image in the prompt.
Each image is loaded (base64 data, a `data:` URL or an http(s) URL; no local paths), decoded (the
`image` crate), planned and letterboxed by a port of Pillow 12.3's `Resample.c` (the same coefficients,
fixed-point rounding and uint8 intermediate). Each placeholder id then becomes the image's span. Every
position of the span carries one negative id, `0x80000000 | 30 bits of a digest of the image's bytes`:
- the engine reads a negative id as an image position;
- two images never share a prefix in the session caches (dsfront's table matches by text, the engine's
  by ids: the engine's longest common prefix stops at the first image that differs).
The SUBMIT frame (`kFlagImages`) carries, after the prompt, each image's span start, grids and pixels
(`fe::ImageHeader`, docs/protocol.md).

**The tower** (`engine/vision/`). `vis::Vision` loads `vision.safetensors` (266 tensors, 0.9 GiB, cut from
the checkpoint by `tools/vision_extract.py`, byte copies) and runs the reference's arithmetic module by
module: bf16 tensors with fp32 accumulation, each Linear's bias added before its one rounding, RMSNorm and
RoPE in fp32, exact GELU. GEMMs are cuBLAS's (static); attention is a fused kernel (`k_flash`: wmma bf16,
online softmax in fp32, P rounded to bf16 for P V, nothing of size N² in memory). Every buffer and cuBLAS's
workspace are made at load, and one small image runs through it then, so nothing allocates later.
2.62 GiB per rank with the image-row buffer.

**Every rank** gets a request's images by `kCmdImages` (the pixels through `bcast_payload`), only those whose
span reaches the positions the request computes (`[q, n)`); each runs the tower into `img_emb`
(`model/image.cu`). No collective: every rank has every row its prefill rows need.

**The prefill** (`model/prefill.cu`):
- `vis_rows` turns the chunk's image ids into `<｜deepseek_image｜>` for the token embedding, then
  `k_vis_embed` overwrites this rank's image rows (all four mHC copies) with the aligner rows and the
  delimiter vectors;
- `ffn_p`'s router selects with `bias_vl` on image rows (`router_topk`'s second bias);
- `engram_combine` passes image rows through; `hash_positions` treats a negative id as dead;
- CED's replay recomputes the mask for its 128 rows from the history (`vis_mask`).
A chunk holding an image position always takes the prefill path, never the 8-row step.

**Checks in rank 0** (`serve.cpp check_images`): every image's span inside the prompt and in order, each
span one negative id throughout, no image position outside a span, at most `kMaxImageRows` (8192) aligner
rows a request. An engine without `--vision` refuses images.

## Measured (2026-11-07)

**The tower against the reference** (`vision/tools/vit_ref.py` in the SGLang image, `vit_check`), five
images (800×600 PNG, a 1188×365 screenshot, a 4000×3000 JPEG, 300×1200, a 100×80 RGBA):
- cosine of the aligner rows 0.9965–0.9998;
- against the reference run in fp32, dsgb10's rows are as close as the reference's own bf16 run
  (shapes 0.99918 vs 0.99928, tall 0.99637 vs 0.99704, tiny 0.99584 vs 0.99622): bf16 noise, not a fault;
- encode: 62 ms (35×44 patches), 122 ms (43×58), 1.07 s (81×108, the largest grid).

**The pixels** (`examples/imgprep.rs` against `vit_ref.py`'s): the four PNGs byte for byte; the JPEG
0.17 % of bytes off by at most 2 (the decoder, not the resampling).

**End to end** through the proxy: shapes, colours and text read correctly; a follow-up turn on the same
image reuses 333 of 343 prompt tokens; the same text with another image reuses none and answers about the
new image; two images in one request; streaming; refused inputs (bad bytes, a local path, audio) get 400.

**Text unchanged**: the bench ladder (8.9–9.5k tok/s prefill, decode 74.0 / 130.8 / 186.6), qeval 71/75
with no flips against the build without vision, T > 0 scan 0 of 35, needle at 124k.
