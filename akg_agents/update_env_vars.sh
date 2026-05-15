#!/bin/bash
# Copyright 2025 Huawei Technologies Co., Ltd
#
# 环境变量更新脚本
# 将 AKG_AGENTS_* 环境变量转换为 AKG_AGENTS_*
#
# 使用方法：
#   source update_env_vars.sh

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}环境变量迁移助手${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

echo -e "${YELLOW}当前 AKG_AGENTS_* 环境变量：${NC}"
env | grep "^AKG_AGENTS_" || echo "  (未找到 AKG_AGENTS_* 环境变量)"
echo ""

echo -e "${YELLOW}建议的新环境变量：${NC}"
echo ""

# 遍历所有 AKG_AGENTS_ 开头的环境变量
for var in $(env | grep "^AKG_AGENTS_" | cut -d= -f1); do
    value="${!var}"
    new_var="${var/AKG_AGENTS_/AKG_AGENTS_}"
    
    echo "# 原: export $var=\"$value\""
    echo "export $new_var=\"$value\""
    echo ""
    
    # 实际设置新的环境变量
    export "$new_var=$value"
done

echo -e "${GREEN}新环境变量已设置！${NC}"
echo ""

# 更新 env.sh 文件
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ENV_FILE="$SCRIPT_DIR/env.sh"

if [ -f "$ENV_FILE" ]; then
    echo -e "${YELLOW}正在更新 env.sh 文件...${NC}"
    
    # 备份原文件
    cp "$ENV_FILE" "$ENV_FILE.backup"
    echo -e "  ${GREEN}✓ 已备份到 env.sh.backup${NC}"
    
    # 更新内容
    sed -i.tmp 's|export PYTHONPATH=$(pwd)/python:|export PYTHONPATH=$(pwd)/python:|g' "$ENV_FILE"
    rm -f "$ENV_FILE.tmp"
    
    echo -e "  ${GREEN}✓ env.sh 已更新${NC}"
    echo ""
fi

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}提示：${NC}"
echo "1. 新的环境变量已在当前 shell 中生效"
echo "2. 请将以上 export 语句添加到你的 ~/.bashrc 或 ~/.zshrc"
echo "3. 或者更新你的环境变量管理工具配置"
echo ""
echo -e "${YELLOW}常用环境变量映射：${NC}"
echo "  AKG_AGENTS_BASE_URL          -> AKG_AGENTS_BASE_URL"
echo "  AKG_AGENTS_MODEL_NAME        -> AKG_AGENTS_MODEL_NAME"
echo "  AKG_AGENTS_API_KEY           -> AKG_AGENTS_API_KEY"
echo "  AKG_AGENTS_MODEL_ENABLE_THINK -> AKG_AGENTS_MODEL_ENABLE_THINK"
echo "  AKG_AGENTS_EMBEDDING_*       -> AKG_AGENTS_EMBEDDING_*"
echo "  AKG_AGENTS_LOG_LEVEL         -> AKG_AGENTS_LOG_LEVEL"
echo ""
