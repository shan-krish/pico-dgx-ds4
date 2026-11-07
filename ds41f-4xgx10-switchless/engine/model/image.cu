// Images in the model (model.h set_images, docs/vision.md): the vision tower's rows for a request's image
// spans, and per prefill pass the rows that take them, the routing mask and Engram's.
//
// A position inside an image span carries a negative id (model.h image_id): the span's IMAGE_START,
// IMAGE rows with IMAGE_NEW_LINE after each, IMAGE_END, all one id. The reference (ref/inference model.py)
// embeds every such position as <｜deepseek_image｜> and then overwrites it: the delimiters with their learned
// vectors, the IMAGE slots with the aligner's rows in reading order. Those positions also route with the
// gate's bias_vl in every MoE layer, take no part in an Engram n-gram and get no Engram contribution.
#include "model/impl.h"
#include "model/prefill.h"

namespace model {

using namespace detail;

namespace {

// rows [0, n) of this rank's slice: src[i] >= 0 takes image row src[i]; -2 / -3 / -4 IMAGE_START /
// IMAGE_NEW_LINE / IMAGE_END; -1 keeps the token's embedding. Every one of the kHc copies of the stream.
__global__ void k_vis_embed(const int* __restrict__ src, const bf16* __restrict__ rows, const bf16* __restrict__ start,
                            const bf16* __restrict__ newline, const bf16* __restrict__ end, bf16* __restrict__ x) {
  const int r = blockIdx.x, s = src[r];
  if (s == -1) return;
  const bf16* v = s >= 0 ? rows + (size_t)s * kDim : s == -2 ? start : s == -3 ? newline : end;
  const uint4* in = reinterpret_cast<const uint4*>(v);
  uint4* out = reinterpret_cast<uint4*>(x + (size_t)r * kHcDim);
  for (int i = threadIdx.x; i < kDim / 8; i += blockDim.x) {
    const uint4 w = in[i];
#pragma unroll
    for (int c = 0; c < kHc; ++c) out[c * (kDim / 8) + i] = w;
  }
}

}  // namespace

void Model::Impl::vis_alloc() {
  if (vsrc_d) return;
  const int cap = std::max(opt.prefill_rows, kReplay);
  vsrc_d = dalloc<int>(cap);
  vmask_d = dalloc<uint8_t>(cap);
  vsrc_h = halloc<int32_t>(cap);
  vmask_h = halloc<uint8_t>(cap);
}

// The image (and its slot) covering position q, from the request's set; dies when none does: rank 0
// checked the prompt's spans against the images before it sent either.
int Model::Impl::vis_slot(int q) const {
  for (const ImgEnt& e : imgs)
    if (q >= e.pos && q < e.pos + e.g.span()) {
      const int s = vis::span_slot(e.g, q - e.pos);
      return s >= 0 ? e.row0 + s : s;
    }
  die("model: position %d is in an image span, but no image of the request covers it", q);
}

bool Model::Impl::vis_mask(int a, int M) {
  bool any = false;
  for (int i = 0; i < M; ++i) any |= image_id(hist[a + i]);
  if (!any) return false;
  vis_alloc();
  for (int i = 0; i < M; ++i) vmask_h[i] = image_id(hist[a + i]) ? 1 : 0;
  CUDA_CHECK(cudaMemcpyAsync(vmask_d, vmask_h, M, cudaMemcpyHostToDevice, st));
  return true;
}

bool Model::Impl::vis_rows(int a, int M, int32_t* tok) {
  if (!vis_mask(a, M)) return false;
  for (int i = 0; i < M; ++i) {
    if (!image_id(tok[i])) {
      vsrc_h[i] = -1;
      continue;
    }
    vsrc_h[i] = vis_slot(a + i);
    tok[i] = kImageTokenId;  // the token embedding the span's rows replace (the reference's input_ids)
  }
  CUDA_CHECK(cudaMemcpyAsync(vsrc_d, vsrc_h, (size_t)M * 4, cudaMemcpyHostToDevice, st));
  return true;
}

void Model::Impl::vis_embed(int r0, int n, bf16* x) {
  if (!vision) die("model: an image span, and no vision tower (--vision)");
  k_vis_embed<<<n, 128, 0, st>>>(vsrc_d + r0, img_emb, vision->image_start(), vision->image_newline(),
                                 vision->image_end(), x);
  CUDA_CHECK(cudaGetLastError());
}

void Model::load_vision(const std::string& path) {
  Impl& d = *d_;
  if (d.vision) die("model: the vision tower is loaded already");
  size_t free0 = 0, free1 = 0, total = 0;
  CUDA_CHECK(cudaMemGetInfo(&free0, &total));
  d.vision = vis::Vision::load(path, d.st);
  d.img_emb = dalloc<bf16>((size_t)kMaxImageRows * kDim);
  d.vis_alloc();
  CUDA_CHECK(cudaStreamSynchronize(d.st));
  CUDA_CHECK(cudaMemGetInfo(&free1, &total));
  std::printf("rank %d: vision tower from %s: %.2f GiB (%.2f GiB free)\n", d.rank, path.c_str(),
              (double)(free0 - free1) / 1073741824.0, free1 / 1073741824.0);
}

bool Model::has_vision() const { return d_->vision != nullptr; }

void Model::set_images(const std::vector<ImageIn>& images) {
  Impl& d = *d_;
  d.imgs.clear();
  if (images.empty()) return;
  if (!d.vision) die("model: images, and no vision tower (--vision)");
  d.drain();  // a chunk in flight may still read the last set's rows
  int row = 0;
  for (const ImageIn& im : images) {
    if (!im.g.valid() || im.pos < 0) die("model: bad image at %d", im.pos);
    if (row + im.g.rows() > kMaxImageRows) die("model: more than %d image rows in one request", kMaxImageRows);
    d.vision->encode(im.pixels, im.g, d.img_emb + (size_t)row * kDim, d.st);
    d.imgs.push_back({im.pos, im.g, row});
    row += im.g.rows();
  }
  CUDA_CHECK(cudaStreamSynchronize(d.st));
}

}  // namespace model
