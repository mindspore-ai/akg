SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /usr/local/Ascend/ascend-toolkit/set_env.sh  # 910b必须
export TRITON_ALL_BLOCKS_PARALLEL=1  # 允许超大一维逻辑grid由Triton-Ascend自动映射到物理核心循环

# attn tasks
python ${SCRIPT_DIR}/run_test_kernels_passk.py \
  --attention flash_attn,page_attn,radix_attn \
  --workflow mathir_multi_kernel_gen_workflow \
  --pass-k 1 \
  --parallel-num 4 \
  --devices 0,1,2,3 \
  --task-type precision_only \
  --framework torch \
  --dsl triton_ascend \
  --backend ascend \
  --arch ascend910b3 \
  --config-path "${SCRIPT_DIR}/../../python/akg_agents/op/config/triton_ascend_mathir_config.yaml"

# all tasks in kernelbench level1
# python "${SCRIPT_DIR}/run_test_kernels_passk.py" \
#   --level level1 \
#   --pass-k 1 \
#   --parallel-num 8 \
#   --workflow mathir_multi_kernel_gen_workflow \
#   --task-type precision_only \
#   --devices 0,1,2,3,4,5,6,7 \
#   --framework torch \
#   --dsl triton_ascend \
#   --backend ascend \
#   --arch ascend910b3 \
#   --config-path "${SCRIPT_DIR}/../../python/akg_agents/op/config/triton_ascend_mathir_config.yaml"


