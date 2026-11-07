// RMSNorm (fp32 math, bf16 out, optionally fused with act_quant of the output) and the router's
// selection step (sqrt(softplus), bias for selection only, top-k, normalise, scale), as model.py.
#include "common/check.h"
#include "kernels/common.cuh"
#include "kernels/dsk.h"

namespace dsk {

namespace {

// One CTA per row (K/8 threads, at most 1024).
__global__ void __launch_bounds__(1024) k_rmsnorm(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
                                                  int D, float eps, __nv_bfloat16* __restrict__ y, uint8_t* __restrict__ q,
                                                  uint8_t* __restrict__ s) {
  __shared__ float red[264];
  const size_t row = blockIdx.x;
  pdl_launch_dependents();
  pdl_wait();
  rmsnorm_rows(x + row * D, D, w, 1, D, eps, red, y + row * D, D, q ? q + row * D : nullptr, D,
               s ? s + row * (D / 32) : nullptr, nullptr, nullptr);
}

// One warp per row.
__global__ void k_router_topk(const float* __restrict__ logits, const float* __restrict__ bias, int E, int M,
                              int* __restrict__ idx, float* __restrict__ wts, float route_scale,
                              const float* __restrict__ bias2, const uint8_t* __restrict__ use2) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  pdl_launch_dependents();
  pdl_wait();
  if (warp >= M) return;
  int id;
  float wv;
  router_row(logits + (size_t)warp * E, use2 && use2[warp] ? bias2 : bias, E, route_scale, id, wv);
  if (lane < kTopK) idx[warp * kTopK + lane] = id, wts[warp * kTopK + lane] = wv;
}

// wide64: batch-aware re-routing (night T3, common.cuh router_row_oea) for a whole step's rows in one CTA: the
// batch's core (every row's top k0), then each row's picks preferring the core among its top kp.
__global__ void __launch_bounds__(1024) k_router_topk_oea(const float* __restrict__ logits, const float* __restrict__ bias,
                                                          int E, int M, int* __restrict__ idx, float* __restrict__ wts,
                                                          float route_scale, int k0, int kp) {
  __shared__ uint32_t cmask[16];
  const int tid = threadIdx.x, lane = tid & 31;
  pdl_launch_dependents();
  pdl_wait();
  if (tid < 16) cmask[tid] = 0;
  __syncthreads();
  for (int r = tid >> 5; r < M; r += blockDim.x >> 5) {
    int id;
    float w;
    router_row(logits + (size_t)r * E, bias, E, route_scale, id, w);
    if (lane < k0) atomicOr(&cmask[id >> 5], 1u << (id & 31));
  }
  __syncthreads();
  for (int r = tid >> 5; r < M; r += blockDim.x >> 5) {
    int id;
    float w;
    router_row_oea(logits + (size_t)r * E, bias, E, route_scale, cmask, k0, kp, id, w);
    if (lane < kTopK) idx[r * kTopK + lane] = id, wts[r * kTopK + lane] = w;
  }
}

}  // namespace

void router_topk_oea(const float* logits, const float* bias, int M, int E, int* idx, float* wts, float route_scale, int k0,
                     int kp, cudaStream_t st) {
  if (E > 512 || k0 < 1 || k0 >= kTopK || kp < kTopK || kp > 16) die("router_topk_oea: E=%d k0=%d kp=%d", E, k0, kp);
  launch_pdl(k_router_topk_oea, dim3(1), dim3(1024), 0, st, true, logits, bias, E, M, idx, wts, route_scale, k0, kp);
}

void rmsnorm(const __nv_bfloat16* x, const __nv_bfloat16* w, int M, int D, float eps, __nv_bfloat16* y, uint8_t* q,
             uint8_t* s, cudaStream_t st) {
  if (D % 256 || D > 8192) die("rmsnorm: D=%d", D);
  launch_pdl(k_rmsnorm, dim3(M), dim3(D / 8), 0, st, true, x, w, D, eps, y, q, s);
}

void router_topk(const float* logits, const float* bias, int M, int E, int* idx, float* wts, float route_scale,
                 cudaStream_t st, const float* bias2, const uint8_t* use2) {
  if (E > 512 || !bias2 != !use2) die("router_topk: E=%d", E);
  launch_pdl(k_router_topk, dim3((M + 3) / 4), dim3(128), 0, st, true, logits, bias, E, M, idx, wts, route_scale, bias2,
             use2);
}

}  // namespace dsk
