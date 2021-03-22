# how many multi-processing to build
export BUILD_PARALLEL_NUM=4

# set the default gpu devices, plz never change it
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# set the real devices you want to use
export USE_GPU_DEVICES=0,1,2,3

export RUNTIME_MODE=gpu

export PROFILING_MODE=true

# ascend config
export DEVICE_ID=0
export DEVICE_TOTAL_NUM=8
