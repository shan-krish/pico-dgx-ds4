// The vision tower (ref/inference/vision.py, image_processor.py): DeepSeek's ViT over one image's
// patches, then the aligner's 3x3 downsample into the LLM's embedding space. Every rank runs it on the
// image's pixels (the front-end letterboxes them to the patch grid), so each holds every image row its
// prefill rows need without a collective.
//
// The arithmetic follows the reference module by module: bf16 tensors with fp32 accumulation, a
// Linear's bias added to its fp32 sums before the one rounding, RMSNorm in fp32 with fp32 weights,
// 2D RoPE in fp32, softmax in fp32 with P rounded to bf16 before P V, exact (erf) GELU. GEMMs are
// cuBLAS's (bf16 in, fp32 compute).
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <string>

namespace vis {

constexpr int kDim = 1024;       // vision_dim
constexpr int kHeads = 16;       // vision_n_heads
constexpr int kHeadDim = kDim / kHeads;
constexpr int kInter = 2816;     // vision_inter_dim
constexpr int kLayers = 32;      // vision_n_layers
constexpr int kPatch = 14;       // vision_patch_size
constexpr int kDown = 3;         // vision_downsample_ratio
constexpr int kLlmDim = 5120;    // the LLM's dim
constexpr int kMaxTokens = 1024; // vision_max_n_token: an image's whole span, delimiters included
constexpr int kPatchK = 3 * kPatch * kPatch;  // 588
constexpr float kRopeTheta = 10000.f;
constexpr float kNormEps = 1e-6f;
// The largest patch grid an image can have: its span n_llm_h * (n_llm_w + 1) + 2 <= kMaxTokens, and
// each LLM token covers at most 3 x 3 patches.
constexpr int kMaxPatches = kDown * kDown * (kMaxTokens - 2);

// One image's grids (image_processor.plan_image_grid): the ViT's patch grid and the aligner's token grid
// (ceil(n_vit / 3) each way). Its span in the prompt is IMAGE_START, n_llm_h rows of n_llm_w IMAGE and
// one IMAGE_NEW_LINE, then IMAGE_END.
struct Grid {
  int n_vit_h = 0, n_vit_w = 0, n_llm_h = 0, n_llm_w = 0;
  int span() const { return n_llm_h * (n_llm_w + 1) + 2; }
  int rows() const { return n_llm_h * n_llm_w; }  // aligner rows: the span's IMAGE slots in reading order
  size_t pixel_bytes() const { return (size_t)n_vit_h * kPatch * n_vit_w * kPatch * 3; }
  bool valid() const;  // positive, the token grid is the patch grid's ceil(/3), within kMaxTokens
};

class Vision {
 public:
  // Loads the tower from a safetensors file holding vision.*, aligner.* and image_start / image_end /
  // image_newline (bf16, as the checkpoint stores them) and makes every buffer and the cuBLAS workspace
  // now, then runs one small image through it, so that encode() allocates nothing.
  static std::unique_ptr<Vision> load(const std::string& path, cudaStream_t st);
  ~Vision();

  // pixels: the letterboxed image, uint8 RGB rows of n_vit_w * 14 pixels, n_vit_h * 14 of them (host
  // memory; copied through a pinned buffer). out: device bf16 [g.rows()][kLlmDim], the aligner's rows.
  void encode(const uint8_t* pixels, const Grid& g, __nv_bfloat16* out, cudaStream_t st);

  // The span delimiters' learned embeddings, device bf16 [kLlmDim].
  const __nv_bfloat16* image_start() const { return start_; }
  const __nv_bfloat16* image_end() const { return end_; }
  const __nv_bfloat16* image_newline() const { return newline_; }
  size_t device_bytes() const { return bytes_; }

  struct Impl;

 private:
  Vision() = default;
  std::unique_ptr<Impl> d_;
  const __nv_bfloat16 *start_ = nullptr, *end_ = nullptr, *newline_ = nullptr;
  size_t bytes_ = 0;
};

// The span position o (0 .. g.span() - 1) of an image: -2 IMAGE_START, -3 IMAGE_NEW_LINE, -4 IMAGE_END,
// else the aligner row it takes.
inline int span_slot(const Grid& g, int o) {
  if (o == 0) return -2;
  if (o == g.span() - 1) return -4;
  const int j = o - 1, r = j / (g.n_llm_w + 1), c = j % (g.n_llm_w + 1);
  return c == g.n_llm_w ? -3 : r * g.n_llm_w + c;
}

}  // namespace vis
