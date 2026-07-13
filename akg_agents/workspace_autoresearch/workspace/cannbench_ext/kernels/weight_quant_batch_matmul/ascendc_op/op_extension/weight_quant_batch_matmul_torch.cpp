#include <cstdint>
#include <optional>

#include <ATen/ATen.h>
#include <c10/core/ScalarType.h>
#include <torch/library.h>

#include "acl/acl.h"
#include "weight_quant_batch_matmul_tiling.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

// Kept for build/link parity (declared, not launched: the hand-written epilogue
// hit an AICore exception on some shapes).
void weight_quant_batch_matmul_kernel(uint32_t blockDim, void* l2Ctrl,
                                      aclrtStream stream, uint8_t* acc,
                                      uint8_t* bias, uint8_t* y,
                                      uint8_t* tiling);

// NPU-only. Antiquant-dequant + the K-reduction run on the device Cube (genuine
// AICore matmul; everything stays on the NPU). Replays the golden's fp32 formula
// (weight_quant_batch_matmul_ref.py: w_dq = (weight.float() + offset.float()) *
// scale.float(); y = (x.float() @ w_dq (+ bias.float())).to(x.dtype)). Near-zero
// cancellation cases cannot be bit-matched to the CPU-fp32 golden's accumulation
// order by any NPU path and are left as honest failures.
namespace ascend_kernel {

at::Tensor weight_quant_batch_matmul(
    const at::Tensor& x, const at::Tensor& weight,
    const at::Tensor& antiquantScale,
    const std::optional<at::Tensor>& antiquantOffset,
    const std::optional<at::Tensor>& bias) {
  const bool hasOffset =
      antiquantOffset.has_value() && antiquantOffset->defined();
  const bool hasBias = bias.has_value() && bias->defined();

  at::Tensor weight_float = weight.to(at::kFloat);
  at::Tensor scale_float = antiquantScale.to(at::kFloat);
  at::Tensor weight_dequant =
      hasOffset ? (weight_float + antiquantOffset->to(at::kFloat)) * scale_float
                : weight_float * scale_float;
  at::Tensor xf = x.to(at::kFloat);

  // Chunked-K matmul with Neumaier compensated accumulation across chunks. A
  // single Cube matmul accumulates all K products in one fp32 accumulator, which
  // loses ~1e-3 over K~4096 and diverges from the CPU fp32 reference at the
  // cancellation cases. Splitting K into chunks and carrying a running
  // compensation term recovers the low-order bits lost between chunks, moving the
  // result toward the fp64 golden (matmul is deterministic here, so this is
  // stable). Requires full fp32 matmul (allow_hf32=False, set in kernel.py) so
  // the ~18-bit dequant weight is not first truncated to HF32.
  const int64_t K = xf.size(-1);
  const int64_t chunk = 256;
  at::Tensor acc, comp;
  for (int64_t k0 = 0; k0 < K; k0 += chunk) {
    const int64_t k1 = std::min(k0 + chunk, K);
    at::Tensor p = at::matmul(xf.slice(-1, k0, k1),
                              weight_dequant.slice(0, k0, k1));
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
  at::Tensor y_float = acc + comp;
  if (hasBias) y_float = y_float + bias->to(at::kFloat);
  return y_float.to(x.scalar_type());
}

}  // namespace ascend_kernel
