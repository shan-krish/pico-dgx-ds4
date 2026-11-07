// One image through the vision tower, compared with the reference's aligner output (tools/vit_ref.py).
//   vit_check WEIGHTS PIXELS.u8 n_vit_h n_vit_w [REF.f32] [OUT.f32]
// PIXELS: the letterboxed image, uint8 RGB, (n_vit_h * 14) x (n_vit_w * 14). REF: float32 [rows][5120].
// Prints the time per encode and, with REF, the cosine similarity and the largest differences.
#include <cuda_bf16.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "common/check.h"
#include "vision/vision.h"

static std::vector<char> slurp(const char* path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) die("cannot open %s", path);
  return std::vector<char>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

int main(int argc, char** argv) {
  if (argc < 5) die("usage: vit_check WEIGHTS PIXELS.u8 n_vit_h n_vit_w [REF.f32] [OUT.f32]");
  vis::Grid g;
  g.n_vit_h = std::atoi(argv[3]), g.n_vit_w = std::atoi(argv[4]);
  g.n_llm_h = (g.n_vit_h + 2) / 3, g.n_llm_w = (g.n_vit_w + 2) / 3;
  if (!g.valid()) die("bad grid");
  const std::vector<char> px = slurp(argv[2]);
  if (px.size() != g.pixel_bytes()) die("pixels: %zu bytes, want %zu", px.size(), g.pixel_bytes());
  cudaStream_t st;
  CUDA_CHECK(cudaStreamCreate(&st));
  auto t0 = std::chrono::steady_clock::now();
  auto v = vis::Vision::load(argv[1], st);
  auto t1 = std::chrono::steady_clock::now();
  std::printf("loaded in %.2f s, %.2f GiB on the device\n", std::chrono::duration<double>(t1 - t0).count(),
              v->device_bytes() / 1073741824.0);
  const size_t n = (size_t)g.rows() * vis::kLlmDim;
  __nv_bfloat16* out = nullptr;
  CUDA_CHECK(cudaMalloc(&out, n * 2));
  double best = 1e9;
  for (int i = 0; i < 4; ++i) {
    auto a = std::chrono::steady_clock::now();
    v->encode(reinterpret_cast<const uint8_t*>(px.data()), g, out, st);
    CUDA_CHECK(cudaStreamSynchronize(st));
    best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count());
  }
  std::printf("grid %dx%d patches -> %dx%d tokens (span %d): encode %.1f ms\n", g.n_vit_h, g.n_vit_w, g.n_llm_h,
              g.n_llm_w, g.span(), best * 1e3);
  std::vector<__nv_bfloat16> hb(n);
  CUDA_CHECK(cudaMemcpy(hb.data(), out, n * 2, cudaMemcpyDeviceToHost));
  std::vector<float> h(n);
  for (size_t i = 0; i < n; ++i) h[i] = __bfloat162float(hb[i]);
  if (argc > 6) std::ofstream(argv[6], std::ios::binary).write(reinterpret_cast<const char*>(h.data()), n * 4);
  if (argc > 5) {
    const std::vector<char> rb = slurp(argv[5]);
    if (rb.size() != n * 4) die("ref: %zu bytes, want %zu", rb.size(), n * 4);
    const float* r = reinterpret_cast<const float*>(rb.data());
    double dot = 0, na = 0, nb = 0, maxd = 0, sumd = 0, worst_row = 1;
    for (int row = 0; row < g.rows(); ++row) {
      double d = 0, a2 = 0, b2 = 0;
      for (int j = 0; j < vis::kLlmDim; ++j) {
        const size_t i = (size_t)row * vis::kLlmDim + j;
        d += (double)h[i] * r[i], a2 += (double)h[i] * h[i], b2 += (double)r[i] * r[i];
        maxd = std::max(maxd, (double)std::fabs(h[i] - r[i]));
        sumd += std::fabs(h[i] - r[i]);
      }
      dot += d, na += a2, nb += b2;
      worst_row = std::min(worst_row, d / std::sqrt(a2 * b2));
    }
    std::printf("vs reference: cosine %.6f (worst row %.6f), mean |diff| %.5f, max |diff| %.4f, ref rms %.4f\n",
                dot / std::sqrt(na * nb), worst_row, sumd / n, maxd, std::sqrt(nb / n));
  }
  return 0;
}
