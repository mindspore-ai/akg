#!/bin/bash
set -e

# 确保目录存在
mkdir -p python/ai_kernel_generator/resources

# 生成 Logo
echo "Generating Logo to python/ai_kernel_generator/resources/logo.ans..."
FORCE_COLOR=1 npx oh-my-logo "AKG CLI" mint --filled > python/ai_kernel_generator/resources/logo.ans

echo "Done."
