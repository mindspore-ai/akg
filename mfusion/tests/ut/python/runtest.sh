#!/bin/bash

set -e

BASEPATH=$(cd "$(dirname $0)"; pwd)

export PYTHONPATH="${BASEPATH}:${PYTHONPATH}"

# 切换到 GCC 11.30（优先级高于 GCC 7.30）
export PATH=/usr/local/gcc/gcc1130/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/gcc/gcc1130/lib64:$LD_LIBRARY_PATH

# 可选：设置 CC/CXX 环境变量
export CC=/usr/local/gcc/gcc1130/bin/gcc
export CXX=/usr/local/gcc/gcc1130/bin/g++

# 验证 libstdc++ 版本是否满足要求
echo "Checking libstdc++ version:"
strings /usr/local/gcc/gcc1130/lib64/libstdc++.so.6 | grep GLIBCXX_3.4.29
if [ $? -eq 0 ]; then
    echo "GLIBCXX_3.4.29 found, good to go!"
else
    echo "Warning: GLIBCXX_3.4.29 not found in GCC 11.30 libstdc++"
fi

# 运行测试
pytest -v "${BASEPATH}"