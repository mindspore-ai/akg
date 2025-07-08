#!/bin/bash

# 设置错误时退出
set -e

# 清理旧的构建文件
echo "清理旧的构建文件..."
rm -rf build/ dist/ *.egg-info/

# 创建输出目录
echo "创建输出目录..."
mkdir -p output

# 构建 wheel 包
echo "开始构建 wheel 包..."
python setup.py bdist_wheel --dist-dir output

# 显示构建结果
echo "构建完成！"
echo "wheel 包位置:"
ls -l output/

# 可选：显示安装命令提示
echo -e "\n您可以使用以下命令安装包："
echo "pip install output/ai_kernel_generator-*-py3-none-any.whl" 