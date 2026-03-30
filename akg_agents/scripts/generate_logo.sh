#!/bin/bash
set -e

# 确保目录存在
mkdir -p python/akg_agents/op/resources

# 生成 Logo
echo "Generating Logo using build_logo.js..."
node scripts/build_logo.js

echo "Done."
