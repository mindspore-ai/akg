from core import *
from api import *

@sub_kernel(core_num=8)
def max_reduction_over_a_dimension_impl_npu(gm_input, gm_output):
    # Hardcoded parameters from tiling
    BATCH_SIZE = 16
    DIM1 = 256
    DIM2 = 256
    SAMPLES_PER_CORE = BATCH_SIZE // 8  # 8 cores, 2 batches per core
    
    # Get current core index
    block_idx = get_block_idx()
    
    # Process each batch assigned to this core
    for i in range(SAMPLES_PER_CORE):
        current_batch = block_idx * SAMPLES_PER_CORE + i
        

        # Initialize output with minimum FP16 value
        min_fp16 = Scalar("FP16", -65504.0)
        ub_output = vector_dup(min_fp16, [1, 1, DIM2], False)

        # Reduction loop over DIM1
        for j in range(DIM1):
            # Load input slice [current_batch, j, :] into UB
            ub_input = slice_to_ub(gm_input, [current_batch, j, 0], [1, 1, DIM2])
            
            # Compute max between current output and new input
            ub_output = vmax(ub_output, ub_input)
        
        # Store result back to GM
        insert_to_gm(gm_output, ub_output, [current_batch, 0], [1, DIM2])

def max_reduction_over_a_dimension_host_run():
    set_context("310P")
    
    # Define input tensor (shape [16, 256, 256])
    gm_input = Tensor("GM", "FP16", [16, 256, 256], "ND", False)
    
    # Define output tensor (shape [16, 256])
    gm_output = Tensor("GM", "FP16", [16, 256], "ND", False)
    
    # Call the NPU implementation
    max_reduction_over_a_dimension_impl_npu(gm_input, gm_output)

if __name__ == '__main__':
    max_reduction_over_a_dimension_host_run()
    compile_kernel("./max_reduction_over_a_dimension.cce")