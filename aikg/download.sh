#!/bin/bash
set -e

# 解析命令行参数
WITH_LOCAL_MODEL=false
for arg in "$@"; do
  if [ "$arg" = "--with_local_model" ]; then
    WITH_LOCAL_MODEL=true
  fi
done

# 模型目标路径
MODEL_DIR="$HOME/.aikg/text2vec-large-chinese"

function check_python_and_deps() {
  if ! command -v python3 &> /dev/null; then
    echo "错误：未找到 python3，请安装 Python 3.10/3.11/3.12"
    exit 1
  fi

  # 检查是否已安装 huggingface_hub，若无则尝试安装（不强制，避免权限问题）
  if ! python3 -c "import huggingface_hub" &> /dev/null; then
    echo "检测到未安装 huggingface_hub，正在尝试安装..."
    if ! pip3 install --user huggingface_hub; then
      echo "错误：无法安装 huggingface_hub，请手动运行：pip3 install --user huggingface_hub"
      exit 1
    fi
  fi
}

function download_text2vec_large_chinese_lib() {
  mkdir -p "$HOME/.aikg"

  if [ -d "$MODEL_DIR" ]; then
    echo "模型目录已存在：$MODEL_DIR，跳过下载。如需重新下载，请先删除该目录。"
    return 0
  fi

  echo "正在下载 text2vec-large-chinese 模型..."
  
  # 使用 Python 脚本下载
  python3 -c "
import os, sys
from huggingface_hub import snapshot_download
try:
    snapshot_download(
        repo_id='GanymedeNil/text2vec-large-chinese',
        local_dir='$MODEL_DIR',
        local_dir_use_symlinks=False,
        resume_download=True,
        max_workers=4
    )
    print('✅ 模型下载成功！路径：$MODEL_DIR')
except Exception as e:
    print(f'❌ 下载失败: {e}', file=sys.stderr)
    sys.exit(1)
  "
}

# 主逻辑
if $WITH_LOCAL_MODEL; then
  check_python_and_deps
  download_text2vec_large_chinese_lib
fi