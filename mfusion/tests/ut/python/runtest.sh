#!/bin/bash

set -e

BASEPATH=$(cd "$(dirname $0)"; pwd)

export PYTHONPATH="${BASEPATH}:${PYTHONPATH}"

# 增加：打印 Python 和 pytest 路径
which python3
python3 --version
which pytest

# 增加：在导入 mfusion 前检查环境
python3 -c "import sys; print(sys.path)" 

# 增加：尝试单独导入 mfusion 并捕获段错误
python3 -c "import mfusion" || echo "Import mfusion failed with exit code $?"

# 运行原测试
pytest -v "${BASEPATH}"