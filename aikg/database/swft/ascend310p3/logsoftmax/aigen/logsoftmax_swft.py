from core import *
from api import *

BATCH_SIZE = 16
DIM = 16384
BLOCK_DIM = 8
SAMPLES_PER_CORE = BATCH_SIZE // BLOCK_DIM

@sub_kernel(core_num=BLOCK_DIM)
def logsoftmax_impl_npu(gm_input, gm_output):
    core_idx = get_block_idx()
    start_batch = core_idx * SAMPLES_PER_CORE
    
    # Process each sample in pipeline
    for sample_idx in range(SAMPLES_PER_CORE):
        global_idx = start_batch + sample_idx
        
        # 1. Load input data
        ub_input = slice_to_ub(gm_input, [global_idx, 0], [1, DIM])
        
        # 2. Compute max value
        ub_max = vcmax(ub_input, reduce_axis=-1)
        
        # 3. Compute shifted = input - max_val
        scalar_max = move_to_scalar(ub_max)
        ub_shifted = vsubs(ub_input, scalar_max)
        
        # 4. Compute exp(shifted)
        ub_exp = vexp(ub_shifted)
        
        # 5. Compute sum of exp
        ub_sum_exp = vcadd(ub_exp, reduce_axis=-1)
        
        # 6. Compute log(sum_exp)
        ub_log_sum_exp = vln(ub_sum_exp)
        
        # 7. Compute log_softmax = shifted - log_sum_exp
        scalar_log_sum_exp = move_to_scalar(ub_log_sum_exp)
        ub_output = vsubs(ub_shifted, scalar_log_sum_exp)
        
        # 8. Write back result
        insert_to_gm(gm_output, ub_output, [global_idx, 0], [1, DIM])

def logsoftmax_host_run():
    set_context("310P")
    gm_input = Tensor("GM", "FP16", [BATCH_SIZE, DIM], format="ND", multi_core=False)
    gm_output = Tensor("GM", "FP16", [BATCH_SIZE, DIM], format="ND", multi_core=False)
    
    logsoftmax_impl_npu(gm_input, gm_output)

if __name__ == '__main__':
    logsoftmax_host_run()
    compile_kernel("./logsoftmax.cce")