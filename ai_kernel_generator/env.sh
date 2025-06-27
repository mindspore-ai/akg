export PYTHONPATH=$(pwd)/python/ai_kernel_generator:${PYTHONPATH}

# 设置黄区GPU环境无代理
export no_proxy=10.90.55.107,${no_proxy}