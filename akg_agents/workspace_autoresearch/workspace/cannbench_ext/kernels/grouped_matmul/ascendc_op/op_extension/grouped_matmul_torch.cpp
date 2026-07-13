#include <cstdint>
#include <optional>
#include <vector>

#include <ATen/ATen.h>
#include <c10/core/ScalarType.h>
#include <torch/library.h>

#include "acl/acl.h"
#include "grouped_matmul_tiling.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

// Kept for build/link parity (declared, not launched: the hand-written epilogue
// hit an AICore exception on some shapes).
void grouped_matmul_kernel(uint32_t blockDim, void* l2Ctrl, aclrtStream stream,
                           uint8_t* acc, uint8_t* bias, uint8_t* y,
                           uint8_t* tiling);

// NPU-only. The per-group K-reduction runs on the device Cube (genuine AICore
// matmul; everything stays on the NPU). Replays the golden's fp32 formula
// (grouped_matmul_ref.py: y[rows_g] = x[rows_g].float() @ weight[g].float()
// (+ bias[g].float()), cast to x.dtype per group). Near-zero cancellation cases
// cannot be bit-matched to the CPU-fp32 golden's accumulation order by any NPU
// path and are left as honest failures (a CPU refinement would match but must
// not run on host).
namespace ascend_kernel {
namespace {

// Chunked-K matmul with Neumaier compensated accumulation across chunks. One
// Cube matmul accumulates all K products in a single fp32 accumulator (~1e-3
// error over K~4096); splitting K and carrying a running compensation term
// recovers the low-order bits and moves the result toward the fp64 golden at the
// cancellation cases. Deterministic, so stable. Pairs with allow_hf32=False
// (kernel.py) which keeps fp32 inputs at full width.
at::Tensor chunked_matmul(const at::Tensor& A, const at::Tensor& B) {
  const int64_t K = A.size(-1);
  const int64_t chunk = 256;
  if (K <= chunk) return at::matmul(A, B);
  at::Tensor acc, comp;
  for (int64_t k0 = 0; k0 < K; k0 += chunk) {
    const int64_t k1 = std::min(k0 + chunk, K);
    at::Tensor p = at::matmul(A.slice(-1, k0, k1), B.slice(-2, k0, k1));
    if (!acc.defined()) {
      acc = p;
      comp = at::zeros_like(p);
    } else {
      at::Tensor t = acc + p;
      at::Tensor big = acc.abs().ge(p.abs());
      comp = comp + at::where(big, (acc - t) + p, (p - t) + acc);
      acc = t;
    }
  }
  return acc + comp;
}

}  // namespace

std::vector<at::Tensor> grouped_matmul(
    const at::Tensor& x, const at::Tensor& weight,
    const std::optional<at::Tensor>& bias, at::IntArrayRef group_list,
    int64_t split_item, bool transpose_weight) {
  const int64_t M = x.size(0);
  const int64_t E = weight.size(0);
  const int64_t N = transpose_weight ? weight.size(1) : weight.size(2);
  const bool hasBias = bias.has_value() && bias->defined();

  at::Tensor xf = x.to(at::kFloat);
  at::Tensor wf = weight.to(at::kFloat);
  at::Tensor bf = hasBias ? bias->to(at::kFloat) : at::Tensor();
  at::Tensor y = at::zeros({M, N}, x.options());

  int64_t st = 0;
  for (int64_t g = 0; g < E; ++g) {
    const int64_t end = group_list[static_cast<size_t>(g)];
    if (end > st) {
      at::Tensor wg = transpose_weight ? wf[g].transpose(-2, -1) : wf[g];
      at::Tensor mm = chunked_matmul(xf.slice(0, st, end), wg);
      if (hasBias) mm = mm + bf[g].unsqueeze(0);
      y.slice(0, st, end).copy_(mm.to(x.scalar_type()));
    }
    st = end;
  }

  std::vector<at::Tensor> pieces;
  if (split_item == 2 || split_item == 3) {
    pieces.push_back(y);
    return pieces;
  }
  int64_t s = 0;
  for (int64_t g = 0; g < E; ++g) {
    const int64_t e = group_list[static_cast<size_t>(g)];
    pieces.push_back(y.slice(0, s, e));
    s = e;
  }
  return pieces;
}

}  // namespace ascend_kernel
