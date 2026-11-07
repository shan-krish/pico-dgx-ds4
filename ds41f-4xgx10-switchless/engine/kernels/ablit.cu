// Rank-1 abliteration on the attention output projection's partials (the SGLang side of this feature:
// patches/sglang-dual-variant-abliteration.patch in the shan-krish/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4 fork).
//
// y <- y - lambda_L * m * (r . y) * r, applied to EVERY rank's partial sums of wo_b before the all-reduce.
// The map is linear, so projecting the partials and then summing is the same as summing and then
// projecting: the collective is untouched and each rank only touches what it already owns.
//
// r is the same vector on every rank (kDim floats), lambda is per layer. m is looked up per row through
// the row's session slot (StepIn::slot -> StepIn::slot_var): 0 = stock (return at once, bitwise no-op),
// 1 = abliterated. Keying on the slot rather than the row means the one-session paths, batch steps,
// verify steps and the draft pass all resolve the variant the same way, with no per-row bookkeeping.
#include <cuda_bf16.h>

#include "dsk.h"

namespace dsk {
namespace {

__global__ void k_ablit_rows(__nv_bfloat16* __restrict__ y, int K, const float* __restrict__ r,
                             float lam, const int* __restrict__ slot_var,
                             const int32_t* __restrict__ slot, int R, int nslots, int all_var) {
  const int row = blockIdx.x;
  if (row >= R) return;
  int v = all_var;
  if (slot) {
    const int s = slot[row];
    v = (s >= 0 && s < nslots) ? slot_var[s] : 0;
  }
  if (v == 0) return;
  __nv_bfloat16* p = y + (size_t)row * K;
  float acc = 0.f;
  for (int i = threadIdx.x; i < K; i += blockDim.x) acc += __bfloat162float(p[i]) * r[i];
  __shared__ float sdot[256];
  sdot[threadIdx.x] = acc;
  __syncthreads();
  for (int o = 128; o > 0; o >>= 1) {
    if (threadIdx.x < o) sdot[threadIdx.x] += sdot[threadIdx.x + o];
    __syncthreads();
  }
  __shared__ float c;
  if (threadIdx.x == 0) c = lam * sdot[0];
  __syncthreads();
  for (int i = threadIdx.x; i < K; i += blockDim.x)
    p[i] = __float2bfloat16(__bfloat162float(p[i]) - c * r[i]);
}

}  // namespace

void ablit_rows(__nv_bfloat16* y, int K, const float* r, float lam, const int* slot_var, const int32_t* slot, int R,
                int nslots, int all_var, cudaStream_t st) {
  if (R <= 0 || lam == 0.f || (slot == nullptr && all_var == 0)) return;
  k_ablit_rows<<<R, 256, 0, st>>>(y, K, r, lam, slot_var, slot, R, nslots, all_var);
}

// The self-test's reference: one row, projected on the host.
void ablit_row_reference(float* y, int K, const float* r, float lam) {
  float d = 0.f;
  for (int i = 0; i < K; i++) d += y[i] * r[i];
  for (int i = 0; i < K; i++) y[i] -= lam * d * r[i];
}
}  // namespace dsk