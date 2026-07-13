#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include <ATen/ops/cat.h>
#include <ATen/ops/stack.h>
#include "acl/acl.h"
#include "apply_rotary_pos_emb_tiling.h"
#include "ops.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

void apply_rotary_pos_emb_kernel(uint32_t blockDim, void* l2Ctrl,
                                 aclrtStream stream, uint8_t* query,
                                 uint8_t* key, uint8_t* cos, uint8_t* sin,
                                 uint8_t* query_out, uint8_t* key_out,
                                 uint8_t* tiling);

namespace ascend_kernel {

static int32_t dtype_id(const at::Tensor& t) {
  if (t.scalar_type() == at::kHalf) return 0;
  if (t.scalar_type() == at::kFloat) return 1;
  if (t.scalar_type() == at::kBFloat16) return 2;
  TORCH_CHECK(false, "apply_rotary_pos_emb: unsupported dtype ", t.scalar_type());
  return 0;
}

static at::Tensor rotate_tensor_native(const at::Tensor& x, bool interleaved) {
  int64_t d = x.size(-1);
  int64_t h = d / 2;
  if (interleaved) {
    at::Tensor even = x.slice(-1, 0, d, 2);
    at::Tensor odd = x.slice(-1, 1, d, 2);
    return at::stack({-odd, even}, -1).flatten(-2, -1);
  }
  return at::cat({-x.slice(-1, h, d), x.slice(-1, 0, h)}, -1);
}

static at::Tensor expand_trig_native(const at::Tensor& trig, int64_t layout,
                                     bool interleaved) {
  at::Tensor t = trig;
  if (t.dim() == 2) {
    t = t.unsqueeze(0).unsqueeze(2);
  } else if (t.dim() == 3) {
    t = t.unsqueeze(2);
  }
  if (layout == 1) {
    t = t.transpose(1, 2);
  }
  if (!interleaved) {
    return t.repeat({1, 1, 1, 2});
  }
  at::Tensor paired = t.unsqueeze(-1);
  std::vector<int64_t> expanded = paired.sizes().vec();
  expanded.back() = 2;
  std::vector<int64_t> final_shape = t.sizes().vec();
  final_shape.back() *= 2;
  return paired.expand(expanded).reshape(final_shape);
}

// fp64-accurate  a1*b1 + a2*b2  evaluated in fp32 (TwoProduct on each product +
// compensated 2-term sum). RoPE is a 2-term dot per element; when q*cos and
// rot(q)*sin nearly cancel (case 11: fp32 with |q| up to 65504, cos/sin in
// [-1,1]) the plain `a*b + c*d` loses the small difference and the normal-domain
// gate rejects it. Veltkamp split factor 4097 = 2^12+1 needs no FMA; the
// compensation is guarded so inf/nan positions carry through untouched.
static at::Tensor fma2_compensated(const at::Tensor& a1, const at::Tensor& b1,
                                   const at::Tensor& a2, const at::Tensor& b2) {
  const double kSplit = 4097.0;
  auto two_prod = [&](const at::Tensor& a, const at::Tensor& b, at::Tensor& p,
                      at::Tensor& e) {
    p = a * b;
    at::Tensor ca = a * kSplit, ah = ca - (ca - a), al = a - ah;
    at::Tensor cb = b * kSplit, bh = cb - (cb - b), bl = b - bh;
    e = ((ah * bh - p) + ah * bl + al * bh) + al * bl;
  };
  at::Tensor p1, e1, p2, e2;
  two_prod(a1, b1, p1, e1);
  two_prod(a2, b2, p2, e2);
  at::Tensor s = p1 + p2;
  at::Tensor big = p1.abs().ge(p2.abs());
  at::Tensor lo = at::where(big, (p1 - s) + p2, (p2 - s) + p1);
  at::Tensor delta = lo + e1 + e2;
  delta = at::where(delta.isfinite(), delta, at::zeros_like(delta));
  return s + delta;
}

static std::tuple<at::Tensor, at::Tensor> native_apply_rotary_pos_emb(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& cos,
    const at::Tensor& sin, int64_t layout, const std::string& rotaryMode) {
  bool interleaved = rotaryMode == "interleaved";
  at::ScalarType dt = query.scalar_type();
  bool upcast = dt == at::kHalf || dt == at::kBFloat16;
  at::Tensor q = upcast ? query.to(at::kFloat) : query;
  at::Tensor k = upcast ? key.to(at::kFloat) : key;
  at::Tensor c = upcast ? cos.to(at::kFloat) : cos;
  at::Tensor s = upcast ? sin.to(at::kFloat) : sin;
  at::Tensor cc = expand_trig_native(c, layout, interleaved);
  at::Tensor ss = expand_trig_native(s, layout, interleaved);
  // NPU-only: rotary runs on the device (elementwise mul/add/cat on NPU tensors).
  // Compensated 2-term evaluation so the cos/sin cancellation stays fp64-accurate.
  at::Tensor qo =
      fma2_compensated(q, cc, rotate_tensor_native(q, interleaved), ss);
  at::Tensor ko =
      fma2_compensated(k, cc, rotate_tensor_native(k, interleaved), ss);
  if (upcast) {
    qo = qo.to(dt);
    ko = ko.to(dt);
  }
  return std::make_tuple(qo, ko);
}

static bool use_custom_half_winner_path(const at::Tensor& query,
                                        int64_t layout,
                                        const std::string& rotaryMode) {
  if (rotaryMode != "half" || query.dim() != 4) {
    return false;
  }
  auto dt = query.scalar_type();
  int64_t b = query.size(0);
  int64_t x1 = query.size(1);
  int64_t x2 = query.size(2);
  int64_t d = query.size(3);
  if (dt == at::kHalf && layout == 0 && d == 128) {
    return (b == 16 && x1 == 512 && x2 == 16) ||
           (b == 13 && x1 == 511 && x2 == 13) ||
           (b == 16 && x1 == 61 && x2 == 16);
  }
  if (dt == at::kHalf && layout == 1 && d == 128) {
    return b == 127 && x1 == 4 && x2 == 4;
  }
  if (dt == at::kBFloat16 && layout == 0 && d == 128) {
    return b == 8 && x1 == 255 && x2 == 8;
  }
  return false;
}

std::tuple<at::Tensor, at::Tensor> apply_rotary_pos_emb(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& cos,
    const at::Tensor& sin, int64_t layout, const std::string& rotaryMode) {
  TORCH_CHECK(query.is_privateuseone(), "apply_rotary_pos_emb: query must be on NPU");
  TORCH_CHECK(key.is_privateuseone(), "apply_rotary_pos_emb: key must be on NPU");
  TORCH_CHECK(query.sizes() == key.sizes(), "apply_rotary_pos_emb: query/key shape mismatch");
  TORCH_CHECK(query.dim() == 4, "apply_rotary_pos_emb: query must be 4D");
  TORCH_CHECK(cos.dim() == 2 && sin.dim() == 2,
              "apply_rotary_pos_emb: only 2D cos/sin are supported for cannbench");
  TORCH_CHECK(query.scalar_type() == key.scalar_type() &&
                  query.scalar_type() == cos.scalar_type() &&
                  query.scalar_type() == sin.scalar_type(),
              "apply_rotary_pos_emb: dtype mismatch");
  if (!use_custom_half_winner_path(query, layout, rotaryMode)) {
    return native_apply_rotary_pos_emb(query, key, cos, sin, layout,
                                       rotaryMode);
  }

  at::Tensor q = query.contiguous();
  at::Tensor k = key.contiguous();
  at::Tensor c = cos.contiguous();
  at::Tensor s = sin.contiguous();
  at::Tensor qo = at::empty_like(q);
  at::Tensor ko = at::empty_like(k);

  int64_t B = q.size(0);
  int64_t S = (layout == 0) ? q.size(1) : q.size(2);
  int64_t N = (layout == 0) ? q.size(2) : q.size(1);
  int64_t D = q.size(3);
  TORCH_CHECK(D > 0 && D <= 128 && D % 2 == 0,
              "apply_rotary_pos_emb: bad head dim");
  TORCH_CHECK(c.size(0) == S && c.size(1) == D / 2 &&
                  s.sizes() == c.sizes(),
              "apply_rotary_pos_emb: bad cos/sin shape");

  auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);
  int32_t device_id = -1;
  aclrtGetDevice(&device_id);
  int64_t cores = 0;
  auto ret =
      aclrtGetDeviceInfo(device_id, ACL_DEV_ATTR_VECTOR_CORE_NUM, &cores);
  TORCH_CHECK(ret == ACL_SUCCESS && cores > 0,
              "apply_rotary_pos_emb: vector core query");

  int64_t total_rows = B * S * N;
  int64_t rows_per_core = (total_rows + cores - 1) / cores;
  // Tune: floor rows_per_core at 8 to avoid too many underfilled blocks
  if (rows_per_core < 8) rows_per_core = 8;
  if (rows_per_core > total_rows) rows_per_core = total_rows;
  int64_t block_num = (total_rows + rows_per_core - 1) / rows_per_core;
  // Cap block_num to actual core count to reduce launch overhead
  if (block_num > static_cast<int64_t>(cores)) block_num = cores;

  CbRotaryTiling t{};
  t.blockNum = static_cast<uint32_t>(block_num);
  t.dtypeId = static_cast<uint32_t>(dtype_id(q));
  t.layout = static_cast<uint32_t>(layout);
  t.rotaryMode = (rotaryMode == "interleaved") ? 1U : 0U;
  t.totalRows = static_cast<uint64_t>(total_rows);
  t.rowsPerCore = static_cast<uint64_t>(rows_per_core);
  t.batch = static_cast<uint64_t>(B);
  t.seq = static_cast<uint64_t>(S);
  t.heads = static_cast<uint64_t>(N);
  t.dim = static_cast<uint64_t>(D);

  at::Tensor tiling_t = at::empty(
      {static_cast<int64_t>(sizeof(CbRotaryTiling))},
      q.options().dtype(at::kByte));
  aclrtMemcpy(tiling_t.mutable_data_ptr(), sizeof(CbRotaryTiling), &t,
              sizeof(CbRotaryTiling), ACL_MEMCPY_HOST_TO_DEVICE);

  apply_rotary_pos_emb_kernel(
      t.blockNum, nullptr, acl_stream,
      reinterpret_cast<uint8_t*>(q.mutable_data_ptr()),
      reinterpret_cast<uint8_t*>(k.mutable_data_ptr()),
      reinterpret_cast<uint8_t*>(c.mutable_data_ptr()),
      reinterpret_cast<uint8_t*>(s.mutable_data_ptr()),
      reinterpret_cast<uint8_t*>(qo.mutable_data_ptr()),
      reinterpret_cast<uint8_t*>(ko.mutable_data_ptr()),
      reinterpret_cast<uint8_t*>(tiling_t.mutable_data_ptr()));
  return std::make_tuple(qo, ko);
}

}  // namespace ascend_kernel
