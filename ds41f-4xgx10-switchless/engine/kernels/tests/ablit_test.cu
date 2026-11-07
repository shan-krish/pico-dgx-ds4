// Standalone check of the abliteration kernel (kernels/ablit.cu). Allocates a few MB and runs on one
// GPU, so it is safe beside a serving engine. Not part of the service build.
//
//   nvcc -O2 -std=c++17 -arch=sm_121a -Iengine/kernels ablit_test.cu engine/kernels/ablit.cu -o ablit_test
//   ./ablit_test
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "dsk.h"

static void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
    std::exit(1);
  }
}

int main() {
  constexpr int K = 5120, R = 8, NSLOTS = 4;
  const float lam = 3.3f;
  std::mt19937 rng(7);
  std::normal_distribution<float> nd(0.f, 0.5f);

  std::vector<float> r(K), y(R * K);
  for (auto& v : r) v = nd(rng);
  // unit-normalise r, as the artifact's is
  double n2 = 0;
  for (float v : r) n2 += (double)v * v;
  const float inv = (float)(1.0 / std::sqrt(n2));
  for (auto& v : r) v *= inv;
  for (auto& v : y) v = nd(rng);

  std::vector<int32_t> slot(R), slot_var(NSLOTS, 0);
  slot_var[0] = 1;  // slot 0 is the abliterated session
  slot_var[1] = 0;  // slot 1 stock
  for (int i = 0; i < R; i++) slot[i] = i % 2;  // rows alternate: ablit, stock, ablit, ...

  __nv_bfloat16 *dy = nullptr, *dref = nullptr;
  float *dr = nullptr;
  int32_t *dvar = nullptr, *dslot = nullptr;
  check(cudaMalloc(&dy, sizeof(__nv_bfloat16) * R * K), "malloc y");
  check(cudaMalloc(&dref, sizeof(__nv_bfloat16) * R * K), "malloc ref");
  check(cudaMalloc(&dr, sizeof(float) * K), "malloc r");
  check(cudaMalloc(&dvar, sizeof(int32_t) * NSLOTS), "malloc slot_var");
  check(cudaMalloc(&dslot, sizeof(int32_t) * R), "malloc slot");

  // bf16 copies of y (the reference works on what the bf16 tensor actually holds)
  std::vector<__nv_bfloat16> hy(R * K);
  for (int i = 0; i < R * K; i++) hy[i] = __float2bfloat16(y[i]);
  check(cudaMemcpy(dy, hy.data(), sizeof(__nv_bfloat16) * R * K, cudaMemcpyHostToDevice), "h2d y");
  check(cudaMemcpy(dref, hy.data(), sizeof(__nv_bfloat16) * R * K, cudaMemcpyHostToDevice), "h2d ref");
  check(cudaMemcpy(dr, r.data(), sizeof(float) * K, cudaMemcpyHostToDevice), "h2d r");
  check(cudaMemcpy(dvar, slot_var.data(), sizeof(int32_t) * NSLOTS, cudaMemcpyHostToDevice), "h2d slot_var");
  check(cudaMemcpy(dslot, slot.data(), sizeof(int32_t) * R, cudaMemcpyHostToDevice), "h2d slot");

  dsk::ablit_rows(dy, K, dr, lam, dvar, dslot, R, NSLOTS, 0, 0);
  check(cudaDeviceSynchronize(), "kernel");

  std::vector<__nv_bfloat16> got(R * K);
  check(cudaMemcpy(got.data(), dy, sizeof(__nv_bfloat16) * R * K, cudaMemcpyDeviceToHost), "d2h y");
  std::vector<__nv_bfloat16> before(R * K);
  check(cudaMemcpy(before.data(), dref, sizeof(__nv_bfloat16) * R * K, cudaMemcpyDeviceToHost), "d2h ref");

  // reference on the host, per active row, in fp32 from the bf16 inputs
  int bad = 0;
  double worst = 0;
  for (int row = 0; row < R; row++) {
    const bool active = slot_var[slot[row]] != 0;
    if (!active) {
      for (int i = 0; i < K; i++)
        if (got[row * K + i] != before[row * K + i]) {
          std::printf("row %d (stock) changed at %d\n", row, i);
          bad++;
          break;
        }
      continue;
    }
    std::vector<float> yf(K);
    for (int i = 0; i < K; i++) yf[i] = __bfloat162float(before[row * K + i]);
    dsk::ablit_row_reference(yf.data(), K, r.data(), lam);
    for (int i = 0; i < K; i++) {
      const float d = std::fabs(__bfloat162float(got[row * K + i]) - yf[i]);
      const float tol = 2.f * std::fabs(yf[i]) * 0.0039f + 0.004f;  // bf16 rounding of the kernel's output
      if (d > tol) {
        std::printf("row %d mismatch at %d: %.5f vs %.5f\n", row, i, __bfloat162float(got[row * K + i]), yf[i]);
        bad++;
        break;
      }
      worst = std::max(worst, (double)d);
    }
  }
  // uniform mode (prefill): all_var = 1 touches every row
  check(cudaMemcpy(dy, hy.data(), sizeof(__nv_bfloat16) * R * K, cudaMemcpyHostToDevice), "h2d y");
  dsk::ablit_rows(dy, K, dr, lam, dvar, nullptr, R, NSLOTS, 1, 0);
  check(cudaDeviceSynchronize(), "kernel uniform");
  check(cudaMemcpy(got.data(), dy, sizeof(__nv_bfloat16) * R * K, cudaMemcpyDeviceToHost), "d2h y2");
  for (int row = 0; row < R; row++) {
    std::vector<float> yf(K);
    for (int i = 0; i < K; i++) yf[i] = __bfloat162float(before[row * K + i]);
    dsk::ablit_row_reference(yf.data(), K, r.data(), lam);
    for (int i = 0; i < K; i++) {
      const float d = std::fabs(__bfloat162float(got[row * K + i]) - yf[i]);
      if (d > 2.f * std::fabs(yf[i]) * 0.0039f + 0.004f) {
        std::printf("uniform row %d mismatch at %d\n", row, i);
        bad++;
        break;
      }
    }
  }
  // all_var = 0 must be a bitwise no-op
  check(cudaMemcpy(dy, hy.data(), sizeof(__nv_bfloat16) * R * K, cudaMemcpyHostToDevice), "h2d y3");
  dsk::ablit_rows(dy, K, dr, lam, dvar, nullptr, R, NSLOTS, 0, 0);
  check(cudaDeviceSynchronize(), "kernel off");
  check(cudaMemcpy(got.data(), dy, sizeof(__nv_bfloat16) * R * K, cudaMemcpyDeviceToHost), "d2h y4");
  for (int i = 0; i < R * K; i++)
    if (got[i] != hy[i]) {
      std::printf("all_var 0 changed element %d\n", i);
      bad++;
      break;
    }

  std::printf("%s: rows %d, slots {0:ablit,1:stock}, max |err| %.6f, failures %d\n", bad ? "FAIL" : "PASS", R, worst, bad);
  return bad ? 1 : 0;
}