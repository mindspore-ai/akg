#!/bin/bash
set -e

# 确保目录存在
mkdir -p python/akg_agents/resources

# 生成 Logo
echo "Generating Logo to python/akg_agents/resources/logo.ans..."
FORCE_COLOR=1 npx oh-my-logo "AKG CLI" mint --filled > python/akg_agents/resources/logo.ans

echo "Done."
