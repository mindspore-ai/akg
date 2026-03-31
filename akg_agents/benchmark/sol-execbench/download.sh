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
# 暂时关闭 set -e 以便捕获错误
set +e
python scripts/download_solexecbench.py
DOWNLOAD_STATUS=$?
set -e

if [ $DOWNLOAD_STATUS -ne 0 ]; then
    echo "HuggingFace 下载超时或失败，尝试从 GitCode 备用仓库下载数据集..."
    cd "${THIRDPARTY_DIR}"
    if [ ! -d "SOL-ExecBench-dataset" ]; then
        git clone https://gitcode.com/yiyanzhi_akane1/SOL-ExecBench.git SOL-ExecBench-dataset
    else
        echo "备用仓库 SOL-ExecBench-dataset 已存在，跳过克隆。"
    fi
    cd sol-execbench
    
    # 直接魔改下载脚本，将数据源指向本地备用仓库
    # 使用 Python 脚本替换，以确保在 macOS, Linux 和 Windows (Git Bash) 下都能稳定运行
    python -c "
import sys
path = 'scripts/download_solexecbench.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()
with open(path, 'w', encoding='utf-8') as f:
    f.write(content.replace('\"nvidia/SOL-ExecBench\"', '\"../SOL-ExecBench-dataset\"'))
"
    
    echo "运行下载脚本从本地备用仓库拉取数据..."
    python scripts/download_solexecbench.py
fi

echo "下载完成！数据存放在 thirdparty/sol-execbench/data/benchmark/"
