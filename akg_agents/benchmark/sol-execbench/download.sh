#!/bin/bash
set -e

echo "开始下载 SOL-ExecBench 数据集..."

# 获取项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
THIRDPARTY_DIR="${PROJECT_ROOT}/thirdparty"

mkdir -p "${THIRDPARTY_DIR}"
cd "${THIRDPARTY_DIR}"

# 如果目录不存在，则克隆 GitHub 仓库
if [ ! -d "sol-execbench" ]; then
    echo "克隆 nvidia/sol-execbench 仓库到 thirdparty/sol-execbench..."
    git clone https://github.com/nvidia/sol-execbench sol-execbench
else
    echo "仓库 sol-execbench 已存在，跳过克隆。"
fi

cd sol-execbench

# 安装可能缺失的依赖
echo "安装依赖 datasets 和 huggingface_hub..."
pip install datasets huggingface_hub -q

# 运行下载脚本
echo "运行下载脚本从 HuggingFace 拉取数据..."
python scripts/download_solexecbench.py

echo "下载完成！数据存放在 thirdparty/sol-execbench/data/benchmark/"
