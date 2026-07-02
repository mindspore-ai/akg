SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# all tasks in kernelbench level1
# python ${SCRIPT_DIR}/run_test_kernels_passk.py \
#   --level level1 \
#   --pass-k 1 \
#   --parallel-num 8 \
#   --task-type precision_only \
#   --devices 0 \


# attn tasks
python ${SCRIPT_DIR}/run_test_kernels_passk.py \
  --attention flash_attn,page_attn,radix_attn \
  --workflow mathir_multi_kernel_gen_workflow \
  --pass-k 1 \
  --parallel-num 4 \
  --devices 0 \
  --task-type precision_only \

  