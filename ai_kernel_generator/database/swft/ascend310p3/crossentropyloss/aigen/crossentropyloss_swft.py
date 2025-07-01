from core import *
from api import *

@sub_kernel(core_num=1)
def crossentropyloss_impl_npu(gm_input, gm_target, gm_output):
    # Hardcoded parameters from tiling
    BATCH_SIZE = 4096
    NUM_CLASSES = 10
    TILE_BATCH_SIZE = 64
    LOOP_COUNT = BATCH_SIZE // TILE_BATCH_SIZE
    
    # Only core 0 executes the computation
    if get_block_idx() != 0:
        return
    
    # Create accumulator
    zero = Scalar("FP32", 0.0)
    accumulator = vector_dup(zero, [1], False)
    
    # Main processing loop
    for i in range(LOOP_COUNT):
        start_idx = i * TILE_BATCH_SIZE
        end_idx = start_idx + TILE_BATCH_SIZE
        
        # Load input and target data
        ub_input = slice_to_ub(gm_input, [start_idx, 0], [TILE_BATCH_SIZE, NUM_CLASSES])
        ub_target = slice_to_ub(gm_target, [start_idx], [TILE_BATCH_SIZE])
        
        # Compute max values
        ub_max = vcmax(ub_input, reduce_axis=-1)
        
        # Subtract max values
        ub_shifted = vsub(ub_input, ub_max)
        
        # Compute exp
        ub_exp = vexp(ub_shifted)
        
        # Compute sum of exp
        ub_sum_exp = vcadd(ub_exp, reduce_axis=-1)
        
        # Compute log softmax
        ub_log_sum_exp = vln(ub_sum_exp)
        ub_log_softmax = vsub(ub_shifted, ub_log_sum_exp)
        
        # Select target values
        ub_selected = vector_dup(zero, [TILE_BATCH_SIZE], False)
        for j in range(TILE_BATCH_SIZE):
            # Get target index
            index_scalar = move_to_scalar(ub_target[j:j+1])
            index_tile = vector_dup(index_scalar, [1], False)
            
            # Gather corresponding log softmax value
            value_tile = slice_to_ub(ub_log_softmax, [j, index_scalar], [1, 1])
            
            # Store selected value
            insert_to_gm(ub_selected, value_tile, [j], [1])
        
        # Convert to FP32 and accumulate
        ub_selected_fp32 = vconv(ub_selected, "FP32")
        ub_sum_selected = vcadd(ub_selected_fp32, reduce_axis=-1)
        accumulator = vadd(accumulator, ub_sum_selected)
    
    # Compute final loss
    batch_size_scalar = Scalar("FP32", float(BATCH_SIZE))
    ub_batch_size = vector_dup(batch_size_scalar, [1], False)
    
    ub_loss = vdiv(accumulator, ub_batch_size)
    negative_one = Scalar("FP32", -1.0)
    ub_loss = vmuls(ub_loss, negative_one)
    
    # Convert to FP16 and store result
    ub_loss_fp16 = vconv(ub_loss, "FP16")
    gm_output.load(ub_loss_fp16)

def crossentropyloss_host_run():
    set_context("310P")
    BATCH_SIZE = 4096
    NUM_CLASSES = 10
    
    gm_input = Tensor("GM", "FP16", [BATCH_SIZE, NUM_CLASSES], "ND", False)
    gm_target = Tensor("GM", "INT64", [BATCH_SIZE], "ND", False)
    gm_output = Tensor("GM", "FP16", [1], "ND", False)
    
    crossentropyloss_impl_npu(gm_input, gm_target, gm_output)

if __name__ == '__main__':
    crossentropyloss_host_run()
    compile_kernel("./crossentropyloss.cce")