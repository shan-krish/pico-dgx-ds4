// The vision tower (vision.h): weights, buffers, kernels and the forward.
#include "vision/vision.h"

#include <cublas_v2.h>
#include <mma.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <vector>

#include "common/check.h"
#include "weights/safetensors.h"

namespace vis {

using bf16 = __nv_bfloat16;

#define CUBLAS_CHECK(x)                                                                  \
  do {                                                                                   \
    cublasStatus_t s_ = (x);                                                             \
    if (s_ != CUBLAS_STATUS_SUCCESS) die("%s:%d: %s: cuBLAS status %d", __FILE__, __LINE__, #x, (int)s_); \
  } while (0)

bool Grid::valid() const {
  return n_vit_h > 0 && n_vit_w > 0 && n_llm_h == (n_vit_h + kDown - 1) / kDown &&
         n_llm_w == (n_vit_w + kDown - 1) / kDown && span() <= kMaxTokens && n_vit_h * n_vit_w <= kMaxPatches;
}

namespace {

constexpr int kQChunk = 256;                        // attention: query rows per pass (every head at once)
constexpr size_t kWorkspace = 32u << 20;            // cuBLAS's
constexpr int kAlignK = kDim * kDown * kDown;       // 9216: the aligner's input

__device__ __forceinline__ float bf(bf16 v) { return __bfloat162float(v); }
__device__ __forceinline__ bf16 rb(float v) { return __float2bfloat16_rn(v); }

// pixels (uint8 HWC, W = nw * 14) -> patches [nh * nw][3][14][14], (v / 255 - 0.5) / 0.5 in fp32, then bf16
__global__ void k_patchify(const uint8_t* __restrict__ px, int nh, int nw, bf16* __restrict__ out) {
  const int p = blockIdx.x, ph = p / nw, pw = p % nw, W = nw * kPatch;
  for (int k = threadIdx.x; k < kPatchK; k += blockDim.x) {
    const int c = k / (kPatch * kPatch), y = (k / kPatch) % kPatch, x = k % kPatch;
    const float v = (float)px[((size_t)(ph * kPatch + y) * W + pw * kPatch + x) * 3 + c];
    out[(size_t)p * kPatchK + k] = rb(__fdiv_rn(__fdiv_rn(v, 255.f) - 0.5f, 0.5f));
  }
}

// y = bf16(acc + bias) (a Linear with bias); with res: res = bf16(res + y) instead (x + sublayer(x))
__global__ void k_bias(const float* __restrict__ acc, const bf16* __restrict__ bias, int n, int cols,
                       bf16* __restrict__ y, bf16* __restrict__ res) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)n * cols) return;
  const bf16 v = rb(acc[i] + bf(bias[i % cols]));
  if (res) res[i] = rb(bf(res[i]) + bf(v));
  else y[i] = v;
}

// res = bf16(res + y)
__global__ void k_add(const bf16* __restrict__ y, size_t n, bf16* __restrict__ res) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) res[i] = rb(bf(res[i]) + bf(y[i]));
}

// RMSNorm over kDim per row, fp32 math, fp32 weight: bf16(w * (x * rsqrt(mean(x^2) + eps)))
__global__ void __launch_bounds__(256) k_rmsnorm(const bf16* __restrict__ x, const float* __restrict__ w,
                                                 bf16* __restrict__ y) {
  __shared__ float red[8];
  const bf16* xr = x + (size_t)blockIdx.x * kDim;
  float v[kDim / 256], ss = 0.f;
#pragma unroll
  for (int i = 0; i < kDim / 256; ++i) v[i] = bf(xr[threadIdx.x + 256 * i]), ss += v[i] * v[i];
#pragma unroll
  for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
  __syncthreads();
  if (threadIdx.x < 32) {
    float t = threadIdx.x < 8 ? red[threadIdx.x] : 0.f;
#pragma unroll
    for (int o = 4; o; o >>= 1) t += __shfl_xor_sync(0xffffffff, t, o);
    if (threadIdx.x == 0) red[0] = t;
  }
  __syncthreads();
  const float r = rsqrtf(red[0] / kDim + kNormEps);
  bf16* yr = y + (size_t)blockIdx.x * kDim;
#pragma unroll
  for (int i = 0; i < kDim / 256; ++i) {
    const int j = threadIdx.x + 256 * i;
    yr[j] = rb(w[j] * (v[i] * r));
  }
}

// 2D RoPE on q and k of qkv [N][3 * kDim], in place: per head, x1 = dims 0..31, x2 = 32..63; angle d < 16
// is row * inv_freq[d], d >= 16 is column * inv_freq[d - 16] (get_vision_cos_sin)
__global__ void k_rope2d(bf16* __restrict__ qkv, int nw, const float* __restrict__ inv_freq) {
  const int p = blockIdx.x, hpos = p / nw, wpos = p % nw;
  for (int t = threadIdx.x; t < 2 * kHeads * 32; t += blockDim.x) {
    const int which = t / (kHeads * 32), h = (t / 32) % kHeads, d = t % 32;
    const float ang = (d < 16 ? (float)hpos : (float)wpos) * inv_freq[d & 15];
    const float c = cosf(ang), s = sinf(ang);
    bf16* base = qkv + (size_t)p * 3 * kDim + which * kDim + h * kHeadDim;
    const float x1 = bf(base[d]), x2 = bf(base[d + 32]);
    base[d] = rb(x1 * c - x2 * s);
    base[d + 32] = rb(x2 * c + x1 * s);
  }
}

// softmax over each row of S (fp32, n columns) -> P (bf16)
__global__ void __launch_bounds__(256) k_softmax(const float* __restrict__ S, int n, bf16* __restrict__ P) {
  __shared__ float red[8];
  const float* s = S + (size_t)blockIdx.x * n;
  bf16* p = P + (size_t)blockIdx.x * n;
  float m = -INFINITY;
  for (int j = threadIdx.x; j < n; j += 256) m = fmaxf(m, s[j]);
#pragma unroll
  for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m;
  __syncthreads();
  m = red[0];
#pragma unroll
  for (int i = 1; i < 8; ++i) m = fmaxf(m, red[i]);
  __syncthreads();
  float sum = 0.f;
  for (int j = threadIdx.x; j < n; j += 256) sum += expf(s[j] - m);
#pragma unroll
  for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = sum;
  __syncthreads();
  sum = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) sum += red[i];
  const float inv = 1.f / sum;
  for (int j = threadIdx.x; j < n; j += 256) p[j] = rb(expf(s[j] - m) * inv);
}

// act = bf16(bf16(silu(gate)) * up), gate | up = the halves of each row of mid [N][2 * kInter]
__global__ void k_silu_mul(const bf16* __restrict__ mid, int n, bf16* __restrict__ act) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)n * kInter) return;
  const size_t r = i / kInter, c = i % kInter;
  const float g = bf(mid[r * 2 * kInter + c]), u = bf(mid[r * 2 * kInter + kInter + c]);
  act[i] = rb(bf(rb(g / (1.f + expf(-g)))) * u);
}

// the aligner's input: x [nh * nw][kDim] as a [kDim][nh][nw] map, zero-padded to multiples of 3, unfolded by
// 3x3 blocks: out[l][c * 9 + ki * 3 + kj] = x[(3 lh + ki) * nw + 3 lw + kj][c], l = lh * Lw + lw (F.unfold)
__global__ void k_unfold(const bf16* __restrict__ x, int nh, int nw, bf16* __restrict__ out) {
  const int l = blockIdx.x, Lw = (nw + kDown - 1) / kDown, lh = l / Lw, lw = l % Lw;
  for (int k = threadIdx.x; k < kAlignK; k += blockDim.x) {
    const int c = k / 9, ki = (k % 9) / 3, kj = k % 3, y = lh * kDown + ki, xx = lw * kDown + kj;
    out[(size_t)l * kAlignK + k] = y < nh && xx < nw ? x[((size_t)y * nw + xx) * kDim + c] : rb(0.f);
  }
}

// y = bf16(gelu(bf16(acc + bias))), exact GELU
__global__ void k_bias_gelu(const float* __restrict__ acc, const bf16* __restrict__ bias, int n, int cols,
                            bf16* __restrict__ y) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)n * cols) return;
  const float t = bf(rb(acc[i] + bf(bias[i % cols])));
  y[i] = rb(0.5f * t * (1.f + erff(t * 0.70710678118654752440f)));
}

unsigned blocks(size_t n, int t = 256) { return (unsigned)((n + t - 1) / t); }

// Fused attention (flash): a CTA of 4 warps takes 64 queries of one head, 16 per warp, and walks the keys in
// tiles of 64. S = Q K^T and P V on the tensor cores (wmma bf16, fp32 accumulators); the softmax online in
// fp32 (running max and sum per row, fp32 sums of the unrounded p), P rounded to bf16 for P V, the output
// accumulated in fp32 and divided by the row's sum at the end. Nothing of size N^2 leaves the SM.
constexpr int kFM = 64, kFN = 64, kFWarps = 4, kLdH = kHeadDim + 8, kLdS = kFN + 4, kLdO = kHeadDim + 4;
constexpr size_t kFlashSmem = (size_t)3 * kFN * kLdH * 2                      // Q, K, V tiles
                              + (size_t)kFWarps * 16 * (kLdS + kLdO) * 4      // S (and P V's), O per warp
                              + (size_t)kFWarps * 16 * kLdH * 2               // P per warp
                              + (size_t)kFWarps * 16 * 3 * 4;                 // alpha, m, l per row

__global__ void __launch_bounds__(128) k_flash(const bf16* __restrict__ qkv, int N, float scale, bf16* __restrict__ o) {
  namespace w = nvcuda::wmma;
  extern __shared__ __align__(128) unsigned char sm[];
  bf16* Qs = reinterpret_cast<bf16*>(sm);
  bf16* Ks = Qs + kFM * kLdH;
  bf16* Vs = Ks + kFN * kLdH;
  float* Sw = reinterpret_cast<float*>(Vs + kFN * kLdH);
  float* Ow = Sw + kFWarps * 16 * kLdS;
  bf16* Pw = reinterpret_cast<bf16*>(Ow + kFWarps * 16 * kLdO);
  float* rowv = reinterpret_cast<float*>(Pw + kFWarps * 16 * kLdH);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, h = blockIdx.y, q0 = blockIdx.x * kFM;
  float* S = Sw + warp * 16 * kLdS;
  float* O = Ow + warp * 16 * kLdO;
  bf16* P = Pw + warp * 16 * kLdH;
  float* alpha = rowv + warp * 48;
  float* m = alpha + 16;
  float* l = alpha + 32;
  const uint4 zero4 = make_uint4(0, 0, 0, 0);
  // the CTA's queries
  for (int i = threadIdx.x; i < kFM * kHeadDim / 8; i += blockDim.x) {
    const int r = i / (kHeadDim / 8), c = (i % (kHeadDim / 8)) * 8;
    const int q = q0 + r;
    *reinterpret_cast<uint4*>(Qs + r * kLdH + c) =
        q < N ? *reinterpret_cast<const uint4*>(qkv + (size_t)q * 3 * kDim + h * kHeadDim + c) : zero4;
  }
  for (int i = lane; i < 16 * kHeadDim; i += 32) O[(i / kHeadDim) * kLdO + i % kHeadDim] = 0.f;
  if (lane < 16) m[lane] = -INFINITY, l[lane] = 0.f;
  __syncthreads();
  w::fragment<w::matrix_a, 16, 16, 16, bf16, w::row_major> qa[kHeadDim / 16];
#pragma unroll
  for (int kk = 0; kk < kHeadDim / 16; ++kk) w::load_matrix_sync(qa[kk], Qs + warp * 16 * kLdH + kk * 16, kLdH);
  const int r = lane >> 1, half = lane & 1;  // the softmax: two lanes per row, 32 columns each
  for (int k0 = 0; k0 < N; k0 += kFN) {
    __syncthreads();  // every warp is done with the last tiles
    for (int i = threadIdx.x; i < kFN * kHeadDim / 8; i += blockDim.x) {
      const int rr = i / (kHeadDim / 8), c = (i % (kHeadDim / 8)) * 8, k = k0 + rr;
      const bf16* src = qkv + (size_t)k * 3 * kDim + h * kHeadDim + c;
      *reinterpret_cast<uint4*>(Ks + rr * kLdH + c) = k < N ? *reinterpret_cast<const uint4*>(src + kDim) : zero4;
      *reinterpret_cast<uint4*>(Vs + rr * kLdH + c) = k < N ? *reinterpret_cast<const uint4*>(src + 2 * kDim) : zero4;
    }
    __syncthreads();
    // S = Q K^T for the warp's 16 rows and the tile's 64 keys
#pragma unroll
    for (int nb = 0; nb < kFN / 16; ++nb) {
      w::fragment<w::accumulator, 16, 16, 16, float> acc;
      w::fill_fragment(acc, 0.f);
#pragma unroll
      for (int kk = 0; kk < kHeadDim / 16; ++kk) {
        w::fragment<w::matrix_b, 16, 16, 16, bf16, w::col_major> kb;
        w::load_matrix_sync(kb, Ks + nb * 16 * kLdH + kk * 16, kLdH);
        w::mma_sync(acc, qa[kk], kb, acc);
      }
      w::store_matrix_sync(S + nb * 16, acc, kLdS, w::mem_row_major);
    }
    __syncwarp();
    {  // the online softmax of row r over the tile
      float s[32], mx = -INFINITY;
#pragma unroll
      for (int j = 0; j < 32; ++j) {
        const int c = half * 32 + j;
        s[j] = k0 + c < N ? S[r * kLdS + c] * scale : -INFINITY;
        mx = fmaxf(mx, s[j]);
      }
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 1));
      const float m_old = m[r], m_new = fmaxf(m_old, mx);
      float sum = 0.f;
#pragma unroll
      for (int j = 0; j < 32; ++j) {
        const float p = s[j] == -INFINITY ? 0.f : expf(s[j] - m_new);
        sum += p;
        P[r * kLdH + half * 32 + j] = rb(p);
      }
      sum += __shfl_xor_sync(0xffffffff, sum, 1);
      __syncwarp();
      if (half == 0) {
        const float a = m_old == -INFINITY ? 0.f : expf(m_old - m_new);
        alpha[r] = a, l[r] = l[r] * a + sum, m[r] = m_new;
      }
    }
    __syncwarp();
    // P V into S's buffer, then O = O * alpha + P V
#pragma unroll
    for (int nb = 0; nb < kHeadDim / 16; ++nb) {
      w::fragment<w::accumulator, 16, 16, 16, float> acc;
      w::fill_fragment(acc, 0.f);
#pragma unroll
      for (int kk = 0; kk < kFN / 16; ++kk) {
        w::fragment<w::matrix_a, 16, 16, 16, bf16, w::row_major> pa;
        w::fragment<w::matrix_b, 16, 16, 16, bf16, w::row_major> vb;
        w::load_matrix_sync(pa, P + kk * 16, kLdH);
        w::load_matrix_sync(vb, Vs + kk * 16 * kLdH + nb * 16, kLdH);
        w::mma_sync(acc, pa, vb, acc);
      }
      w::store_matrix_sync(S + nb * 16, acc, kLdS, w::mem_row_major);
    }
    __syncwarp();
    for (int i = lane; i < 16 * kHeadDim; i += 32) {
      const int rr = i / kHeadDim, c = i % kHeadDim;
      O[rr * kLdO + c] = O[rr * kLdO + c] * alpha[rr] + S[rr * kLdS + c];
    }
    __syncwarp();
  }
  for (int i = lane; i < 16 * kHeadDim; i += 32) {
    const int rr = i / kHeadDim, c = i % kHeadDim, q = q0 + warp * 16 + rr;
    if (q < N) o[(size_t)q * kDim + h * kHeadDim + c] = rb(O[rr * kLdO + c] / l[rr]);
  }
}

}  // namespace

struct Vision::Impl {
  cublasHandle_t cb = nullptr;
  void* ws = nullptr;
  // weights (device)
  struct Block {
    bf16 *wqkv, *bqkv, *wo, *bo, *w1, *w2;
    float *n1, *n2;
  } blk[kLayers];
  bf16 *pw, *pb, *a1w, *a1b, *a2w, *a2b;
  float *nf, *inv_freq;
  // buffers (device)
  uint8_t* px = nullptr;  // pixels
  uint8_t* px_h = nullptr;  // pinned staging
  bf16 *patches, *x, *h, *qkv, *o, *mid, *act, *al_in, *al_mid;
  float *acc, *S;
  bf16* P;
  std::vector<void*> allocs;

  template <class T>
  T* dalloc(size_t n, size_t& bytes) {
    void* p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, n * sizeof(T)));
    allocs.push_back(p);
    bytes += n * sizeof(T);
    return static_cast<T*>(p);
  }

  // y[n][out] = x[n][in] W[out][in]^T, fp32 sums: into acc (C32) or rounded into a bf16 y
  void linear(const bf16* x, int n, const bf16* W, int out, int in, void* y, bool f32, cudaStream_t st) {
    const float one = 1.f, zero = 0.f;
    CUBLAS_CHECK(cublasSetStream(cb, st));
    CUBLAS_CHECK(cublasGemmEx(cb, CUBLAS_OP_T, CUBLAS_OP_N, out, n, in, &one, W, CUDA_R_16BF, in, x, CUDA_R_16BF, in,
                              &zero, y, f32 ? CUDA_R_32F : CUDA_R_16BF, out, CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT));
  }

  // full bidirectional attention over the N patches, every head; qkv [N][3 kDim] (RoPE applied), o [N][kDim]:
  // the fused kernel, or (gemm_attn, a check) S and P materialised a chunk of queries at a time
  bool gemm_attn = false;
  void attention(int N, cudaStream_t st) {
    if (!gemm_attn) {
      k_flash<<<dim3((N + kFM - 1) / kFM, kHeads), kFWarps * 32, kFlashSmem, st>>>(qkv, N, 1.f / std::sqrt((float)kHeadDim), o);
      CUDA_CHECK(cudaGetLastError());
      return;
    }
    CUBLAS_CHECK(cublasSetStream(cb, st));
    const float scale = 1.f / std::sqrt((float)kHeadDim), one = 1.f, zero = 0.f;
    for (int q0 = 0; q0 < N; q0 += kQChunk) {
      const int nq = std::min(kQChunk, N - q0);
      // S_h [nq][N] = Q_h K_h^T * scale: column-major S_h^T [N][nq] = K_h^T(op T of [64][N]) x Q_h [64][nq]
      CUBLAS_CHECK(cublasGemmStridedBatchedEx(
          cb, CUBLAS_OP_T, CUBLAS_OP_N, N, nq, kHeadDim, &scale, qkv + kDim, CUDA_R_16BF, 3 * kDim, kHeadDim,
          qkv + (size_t)q0 * 3 * kDim, CUDA_R_16BF, 3 * kDim, kHeadDim, &zero, S, CUDA_R_32F, N, (long long)nq * N,
          kHeads, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
      k_softmax<<<nq * kHeads, 256, 0, st>>>(S, N, P);
      CUDA_CHECK(cudaGetLastError());
      // O_h [nq][64] = P_h V_h: column-major O_h^T [64][nq] = V_h^T [64][N] x P_h^T [N][nq]
      CUBLAS_CHECK(cublasGemmStridedBatchedEx(
          cb, CUBLAS_OP_N, CUBLAS_OP_N, kHeadDim, nq, N, &one, qkv + 2 * kDim, CUDA_R_16BF, 3 * kDim, kHeadDim, P,
          CUDA_R_16BF, N, (long long)nq * N, &zero, o + (size_t)q0 * kDim, CUDA_R_16BF, kDim, kHeadDim, kHeads,
          CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    }
  }
};

namespace {

// The file's tensors: name -> (offset, bytes, shape), bf16 only.
struct StFile {
  std::string path;
  struct T {
    uint64_t off = 0, bytes = 0;
    std::vector<int64_t> shape;
  };
  std::map<std::string, T> t;
  std::vector<char> data;

  static StFile open(const std::string& path) {
    StFile f;
    f.path = path;
    std::ifstream in(path, std::ios::binary);
    if (!in) die("vision: cannot open %s", path.c_str());
    in.seekg(0, std::ios::end);
    const size_t size = (size_t)in.tellg();
    in.seekg(0);
    f.data.resize(size);
    if (!in.read(f.data.data(), size)) die("vision: cannot read %s", path.c_str());
    uint64_t hl = 0;
    std::memcpy(&hl, f.data.data(), 8);
    if (8 + hl > size) die("vision: %s: bad header", path.c_str());
    const dsw::Json h = dsw::Json::parse(std::string(f.data.data() + 8, hl));
    for (const auto& [name, v] : h.obj) {
      if (name == "__metadata__") continue;
      const dsw::Json* dt = v.get("dtype");
      const dsw::Json* sh = v.get("shape");
      const dsw::Json* of = v.get("data_offsets");
      if (!dt || !sh || !of || dt->s != "BF16") die("vision: %s: %s is not a bf16 tensor", path.c_str(), name.c_str());
      T x;
      x.off = 8 + hl + (uint64_t)of->arr[0].i;
      x.bytes = (uint64_t)(of->arr[1].i - of->arr[0].i);
      for (const auto& d : sh->arr) x.shape.push_back(d.i);
      if (x.off + x.bytes > size) die("vision: %s: %s past the end", path.c_str(), name.c_str());
      f.t[name] = x;
    }
    return f;
  }
  const T& need(const std::string& name, std::vector<int64_t> shape) const {
    auto it = t.find(name);
    if (it == t.end()) die("vision: %s has no %s", path.c_str(), name.c_str());
    if (it->second.shape != shape) die("vision: %s: %s has another shape", path.c_str(), name.c_str());
    return it->second;
  }
};

}  // namespace

std::unique_ptr<Vision> Vision::load(const std::string& path, cudaStream_t st) {
  const StFile f = StFile::open(path);
  std::unique_ptr<Vision> v(new Vision);
  v->d_ = std::make_unique<Impl>();
  Impl& d = *v->d_;
  size_t& bytes = v->bytes_;
  auto up = [&](const std::string& name, std::vector<int64_t> shape) {
    const StFile::T& x = f.need(name, shape);
    bf16* p = d.dalloc<bf16>(x.bytes / 2, bytes);
    CUDA_CHECK(cudaMemcpy(p, f.data.data() + x.off, x.bytes, cudaMemcpyHostToDevice));
    return p;
  };
  auto up_f32 = [&](const std::string& name, int n) {  // a norm weight: bf16 in the file, fp32 as the module holds it
    const StFile::T& x = f.need(name, {n});
    std::vector<float> h(n);
    const uint16_t* s = reinterpret_cast<const uint16_t*>(f.data.data() + x.off);
    for (int i = 0; i < n; ++i) {
      const uint32_t b = (uint32_t)s[i] << 16;
      std::memcpy(&h[i], &b, 4);
    }
    float* p = d.dalloc<float>(n, bytes);
    CUDA_CHECK(cudaMemcpy(p, h.data(), n * 4, cudaMemcpyHostToDevice));
    return p;
  };
  d.pw = up("vision.patch_embed.proj.weight", {kDim, kPatchK});
  d.pb = up("vision.patch_embed.proj.bias", {kDim});
  for (int L = 0; L < kLayers; ++L) {
    const std::string p = "vision.blocks." + std::to_string(L) + ".";
    Impl::Block& b = d.blk[L];
    b.wqkv = up(p + "attn.wqkv.weight", {3 * kDim, kDim});
    b.bqkv = up(p + "attn.wqkv.bias", {3 * kDim});
    b.wo = up(p + "attn.wo.weight", {kDim, kDim});
    b.bo = up(p + "attn.wo.bias", {kDim});
    b.w1 = up(p + "mlp.w1.weight", {2 * kInter, kDim});
    b.w2 = up(p + "mlp.w2.weight", {kDim, kInter});
    b.n1 = up_f32(p + "norm1.weight", kDim);
    b.n2 = up_f32(p + "norm2.weight", kDim);
  }
  d.nf = up_f32("vision.norm.weight", kDim);
  d.a1w = up("aligner.w1.weight", {kLlmDim, kAlignK});
  d.a1b = up("aligner.w1.bias", {kLlmDim});
  d.a2w = up("aligner.w2.weight", {kLlmDim, kLlmDim});
  d.a2b = up("aligner.w2.bias", {kLlmDim});
  v->start_ = up("image_start", {kLlmDim});
  v->end_ = up("image_end", {kLlmDim});
  v->newline_ = up("image_newline", {kLlmDim});
  {  // inv_freq = 1 / theta^(2i / 32), i < 16 (rope_dim = head_dim / 2 = 32), fp32 as torch computes it
    float h[16];
    for (int i = 0; i < 16; ++i) h[i] = 1.f / std::pow(kRopeTheta, (float)(2 * i) / 32.f);
    d.inv_freq = d.dalloc<float>(16, bytes);
    CUDA_CHECK(cudaMemcpy(d.inv_freq, h, sizeof h, cudaMemcpyHostToDevice));
  }
  const size_t N = kMaxPatches, L = kMaxTokens;
  d.px = d.dalloc<uint8_t>(N * kPatch * kPatch * 3, bytes);
  CUDA_CHECK(cudaMallocHost(&d.px_h, N * kPatch * kPatch * 3));
  d.patches = d.dalloc<bf16>(N * kPatchK, bytes);
  d.x = d.dalloc<bf16>(N * kDim, bytes);
  d.h = d.dalloc<bf16>(N * kDim, bytes);
  d.qkv = d.dalloc<bf16>(N * 3 * kDim, bytes);
  d.o = d.dalloc<bf16>(N * kDim, bytes);
  d.mid = d.dalloc<bf16>(N * 2 * kInter, bytes);
  d.act = d.dalloc<bf16>(N * kInter, bytes);
  d.al_in = d.dalloc<bf16>(L * kAlignK, bytes);
  d.al_mid = d.dalloc<bf16>(L * kLlmDim, bytes);
  d.acc = d.dalloc<float>(std::max(N * 3 * kDim, L * kLlmDim), bytes);
  d.S = d.dalloc<float>((size_t)kHeads * kQChunk * N, bytes);
  d.P = d.dalloc<bf16>((size_t)kHeads * kQChunk * N, bytes);
  d.ws = d.dalloc<uint8_t>(kWorkspace, bytes);
  CUDA_CHECK(cudaFuncSetAttribute(k_flash, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)kFlashSmem));
  if (const char* e = std::getenv("DSGB10_VISION_GEMM_ATTN")) d.gemm_attn = e[0] == '1';
  CUBLAS_CHECK(cublasCreate(&d.cb));
  CUBLAS_CHECK(cublasSetWorkspace(d.cb, d.ws, kWorkspace));
  CUBLAS_CHECK(cublasSetMathMode(d.cb, CUBLAS_DEFAULT_MATH));
  {  // one small image through every path, so that nothing allocates later
    const Grid g{40, 40, 14, 14};
    std::vector<uint8_t> px(g.pixel_bytes(), 127);
    bf16* out = d.dalloc<bf16>((size_t)g.rows() * kLlmDim, bytes);
    v->encode(px.data(), g, out, st);
    CUDA_CHECK(cudaStreamSynchronize(st));
  }
  return v;
}

Vision::~Vision() {
  if (!d_) return;
  if (d_->cb) cublasDestroy(d_->cb);
  for (void* p : d_->allocs) cudaFree(p);
  if (d_->px_h) cudaFreeHost(d_->px_h);
}

void Vision::encode(const uint8_t* pixels, const Grid& g, bf16* out, cudaStream_t st) {
  if (!g.valid()) die("vision: bad grid %dx%d -> %dx%d", g.n_vit_h, g.n_vit_w, g.n_llm_h, g.n_llm_w);
  Impl& d = *d_;
  const int N = g.n_vit_h * g.n_vit_w, n = g.rows();
  const size_t pb = g.pixel_bytes();
  CUDA_CHECK(cudaStreamSynchronize(st));  // the pinned buffer's last copy is done
  std::memcpy(d.px_h, pixels, pb);
  CUDA_CHECK(cudaMemcpyAsync(d.px, d.px_h, pb, cudaMemcpyHostToDevice, st));
  k_patchify<<<N, 256, 0, st>>>(d.px, g.n_vit_h, g.n_vit_w, d.patches);
  CUDA_CHECK(cudaGetLastError());
  // patch_embed
  d.linear(d.patches, N, d.pw, kDim, kPatchK, d.acc, true, st);
  k_bias<<<blocks((size_t)N * kDim), 256, 0, st>>>(d.acc, d.pb, N, kDim, d.x, nullptr);
  for (int L = 0; L < kLayers; ++L) {
    const Impl::Block& b = d.blk[L];
    // x = x + wo(attn(norm1(x)))
    k_rmsnorm<<<N, 256, 0, st>>>(d.x, b.n1, d.h);
    d.linear(d.h, N, b.wqkv, 3 * kDim, kDim, d.acc, true, st);
    k_bias<<<blocks((size_t)N * 3 * kDim), 256, 0, st>>>(d.acc, b.bqkv, N, 3 * kDim, d.qkv, nullptr);
    k_rope2d<<<N, 256, 0, st>>>(d.qkv, g.n_vit_w, d.inv_freq);
    CUDA_CHECK(cudaGetLastError());
    d.attention(N, st);
    d.linear(d.o, N, b.wo, kDim, kDim, d.acc, true, st);
    k_bias<<<blocks((size_t)N * kDim), 256, 0, st>>>(d.acc, b.bo, N, kDim, nullptr, d.x);
    // x = x + w2(silu(gate) * up), w1 / w2 without bias
    k_rmsnorm<<<N, 256, 0, st>>>(d.x, b.n2, d.h);
    d.linear(d.h, N, b.w1, 2 * kInter, kDim, d.mid, false, st);
    k_silu_mul<<<blocks((size_t)N * kInter), 256, 0, st>>>(d.mid, N, d.act);
    d.linear(d.act, N, b.w2, kDim, kInter, d.o, false, st);
    k_add<<<blocks((size_t)N * kDim), 256, 0, st>>>(d.o, (size_t)N * kDim, d.x);
    CUDA_CHECK(cudaGetLastError());
  }
  k_rmsnorm<<<N, 256, 0, st>>>(d.x, d.nf, d.h);
  // aligner: 3x3 unfold, w1 + gelu, w2
  k_unfold<<<n, 256, 0, st>>>(d.h, g.n_vit_h, g.n_vit_w, d.al_in);
  d.linear(d.al_in, n, d.a1w, kLlmDim, kAlignK, d.acc, true, st);
  k_bias_gelu<<<blocks((size_t)n * kLlmDim), 256, 0, st>>>(d.acc, d.a1b, n, kLlmDim, d.al_mid);
  d.linear(d.al_mid, n, d.a2w, kLlmDim, kLlmDim, d.acc, true, st);
  k_bias<<<blocks((size_t)n * kLlmDim), 256, 0, st>>>(d.acc, d.a2b, n, kLlmDim, out, nullptr);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace vis
