#!/bin/bash
set -e

# 解析命令行参数
WITH_LOCAL_MODEL=false
for arg in "$@"; do
  if [ "$arg" = "--with_local_model" ]; then
    WITH_LOCAL_MODEL=true
  fi
done

# 检查git命令是否存在
if ! command -v git &> /dev/null
then
    echo "错误：git未安装，请先安装git"
    exit 1
fi

function download_text2vec_large_chinese_lib() {
  if [ ! -d "thirdparty/text2vec-large-chinese" ]; then
    echo "正在克隆text2vec-large-chinese仓库..."
    if git clone https://hf-mirror.com/GanymedeNil/text2vec-large-chinese thirdparty/text2vec-large-chinese; then
      echo "text2vec-large-chinese库下载成功到thirdparty/text2vec-large-chinese目录"
    else
      echo "错误：仓库克隆失败，请检查网络连接或仓库地址"
      exit 1
    fi
  fi
}


if $WITH_LOCAL_MODEL; then
    download_text2vec_large_chinese_lib
fi
