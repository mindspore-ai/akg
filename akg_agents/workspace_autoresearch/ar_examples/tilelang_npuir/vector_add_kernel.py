import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401


tilelang.cache.clear_cache()


def vec_add_kernel(n, block_n, dtype="float32"):
    n_blocks = n // block_n

    @T.prim_func
    def main(
        a: T.Tensor((n), dtype),
        b: T.Tensor((n), dtype),
        c: T.Tensor((n), dtype),
        shape: T.int32,
    ):
        with T.Kernel(n_blocks, is_npu=True) as (cid, _):
            a_vec = T.alloc_ub((block_n), dtype)
            b_vec = T.alloc_ub((block_n), dtype)
            c_vec = T.alloc_ub((block_n), dtype)

            offset = cid * block_n
            remain = shape - offset
            tail_size = T.min(block_n, remain)
            T.copy(a[offset], a_vec, [tail_size])
            T.copy(b[offset], b_vec, [tail_size])
            T.npuir_add(a_vec, b_vec, c_vec)
            T.copy(c_vec, c[offset], [tail_size])

    return main


def vector_add_tilelang_npuir_torch(a: torch.Tensor,
                                    b: torch.Tensor) -> torch.Tensor:
    a = a.contiguous()
    b = b.contiguous()
    n = a.numel()
    block_n = 1024
    out = torch.empty_like(a)
    func = vec_add_kernel(n, block_n, "float32")
    compiled = tilelang.compile(func, target="npuir")
    compiled(a.reshape(-1), b.reshape(-1), out.reshape(-1), n)
    return out
