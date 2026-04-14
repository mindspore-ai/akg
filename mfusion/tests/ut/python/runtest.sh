#!/bin/bash

set -e

BASEPATH=$(cd "$(dirname $0)"; pwd)

export PYTHONPATH="${BASEPATH}:${PYTHONPATH}"

# 切换到 GCC 11.30（优先级高于 GCC 7.30）
export PATH=/usr/local/gcc/gcc1130/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/gcc/gcc1130/lib64:$LD_LIBRARY_PATH

# 设置 CC/CXX 环境变量
export CC=/usr/local/gcc/gcc1130/bin/gcc
export CXX=/usr/local/gcc/gcc1130/bin/g++

# 运行测试
pytest -v "${BASEPATH}"