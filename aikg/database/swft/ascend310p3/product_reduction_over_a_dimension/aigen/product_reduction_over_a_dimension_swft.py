from core import *
from api import *

@sub_kernel(core_num=8)
def product_reduction_over_a_dimension_impl_npu(gm_input, gm_output):
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
        init_fp32 = Scalar("FP32", 1.0)
        reduce_buf_fp32 = vector_dup(init_fp32, [1, 1, DIM2], False)

        # Reduction loop over DIM1
        for j in range(DIM1):
            # Load input slice [current_batch, j, :] into UB
            ub_input = slice_to_ub(gm_input, [current_batch, j, 0], [1, 1, DIM2])

            # Convert to FP32 for precision
            ub_fp32 = vconv(ub_input_reshaped, "FP32")
            
            # Accumulate sum
            reduce_buf_fp32 = vmul(reduce_buf_fp32, ub_fp32)
        
        # Convert back to FP16
        ub_output = vconv(reduce_buf_fp32, "FP16")
        
        # Store result back to GM
        insert_to_gm(gm_output, ub_output, [current_batch, 0], [1, DIM2])

def product_reduction_over_a_dimension_host_run():
    set_context("310P")
    
    # Define input tensor (shape [16, 256, 256])
    gm_input = Tensor("GM", "FP16", [16, 256, 256], "ND", False)
    
    # Define output tensor (shape [16, 256])
    gm_output = Tensor("GM", "FP16", [16, 256], "ND", False)
    
    # Call the NPU implementation
    product_reduction_over_a_dimension_impl_npu(gm_input, gm_output)

if __name__ == '__main__':
    product_reduction_over_a_dimension_host_run()
    compile_kernel("./product_reduction_over_a_dimension.cce")