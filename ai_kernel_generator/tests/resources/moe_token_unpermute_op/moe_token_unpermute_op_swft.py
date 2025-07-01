from core import *
from api import *

BLOCK_DIM = 8


@sub_kernel(core_num=8)
def moe_token_unpermute_op_impl_npu(gm_permute_token, gm_sorted_idx, gm_probs, gm_output):
    block_idx = get_block_idx()
    token_num = 16
    top_k = 8
    hidden = 7168

    tokens_per_core = (token_num + BLOCK_DIM - 1) // BLOCK_DIM

    # Initialize local output buffer
    ub_idx = move_to_ub(gm_sorted_idx)
    prob_ub = move_to_ub(gm_probs)

    for i in range(tokens_per_core):
        start_token = block_idx * tokens_per_core + i

        tmp_s = Scalar("FP16", 0.0)
        local_out = vector_dup(tmp_s, [1, hidden], False)

        for k in range(top_k):
            # Load sorted_idx
            idx = move_to_scalar(ub_idx[k * token_num + start_token])

            # Load permute_token row
            dst_row = slice_to_ub(gm_permute_token, [idx, 0], slicesize=[1, hidden])

            # Load prob
            prob = move_to_scalar(prob_ub[start_token, k])

            # # Compute weighted row and accumulate
            weighted_row = vmuls(dst_row, prob)
            local_out = vadd(local_out, weighted_row)

        # Write back to GM
        insert_to_gm(gm_output, local_out, [start_token, 0], slicesize=[1, hidden])


def moe_token_unpermute_op_host_run():
    set_context("310P")

    # Define input tensors
    token_num = 16
    top_k = 8
    hidden = 7168

    gm_permute_token = Tensor("GM", "FP16", [token_num * top_k, hidden], format="ND", multi_core=False)
    gm_sorted_idx = Tensor("GM", "INT32", [token_num * top_k], format="ND", multi_core=False)
    gm_probs = Tensor("GM", "FP16", [token_num, top_k], format="ND", multi_core=False)
    gm_output = Tensor("GM", "FP16", [token_num, hidden], format="ND", multi_core=False)

    moe_token_unpermute_op_impl_npu(gm_permute_token, gm_sorted_idx, gm_probs, gm_output)
    compile_kernel("./moe_token_unpermute_op_kernel.cce")


if __name__ == '__main__':
    moe_token_unpermute_op_host_run()
