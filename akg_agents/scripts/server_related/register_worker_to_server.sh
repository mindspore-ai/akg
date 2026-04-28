#!/bin/bash
# 向 AIKG Server 注册 Worker Service
# 用法: ./scripts/server_related/register_worker_to_server.sh [server_url] [worker_url] [backend] [arch] [capacity]
# ./scripts/server_related/register_worker_to_server.sh http://localhost:8000 http://localhost:9001 cuda a100 1
# ./scripts/server_related/register_worker_to_server.sh http://localhost:8000 http://localhost:9001 ascend ascend910b4 1
#
# ========================================
# IPv4/IPv6 配置说明:
# ========================================
# 通过环境变量或参数来指定 URL:
#   - 参数方式: ./register_worker_to_server.sh <server_url> <worker_url> ...
#   - 环境变量: AKG_AGENTS_SERVER_URL, AKG_AGENTS_WORKER_URL
#
# IPv4 示例:
#   ./register_worker_to_server.sh http://192.168.1.100:8000 http://192.168.1.101:9001 cuda a100 1
#
# IPv6 示例 (注意 IPv6 地址需要用方括号包围):
#   ./register_worker_to_server.sh http://[2001:db8::1]:8000 http://[2001:db8::2]:9001 cuda a100 1
#   或者:
#   export AKG_AGENTS_SERVER_URL=http://[::1]:8000
#   export AKG_AGENTS_WORKER_URL=http://[::1]:9001
#   ./register_worker_to_server.sh "" "" cuda a100 1
# ========================================

set -e

SERVER_URL=${1:-${AKG_AGENTS_SERVER_URL:-http://localhost:8000}}
WORKER_URL=${2:-${AKG_AGENTS_WORKER_URL:-http://localhost:9001}}
BACKEND=${3:-cuda}
ARCH=${4:-a100}
CAPACITY=${5:-1}

echo "=========================================="
echo "注册 Worker 到 AIKG Server"
echo "=========================================="
echo "Server URL: $SERVER_URL"
echo "Worker URL: $WORKER_URL"
echo "Backend: $BACKEND"
echo "Arch: $ARCH"
echo "Capacity: $CAPACITY"
echo "=========================================="

# 注册 Worker
curl -X POST "$SERVER_URL/api/v1/workers/register" \
    -H "Content-Type: application/json" \
    -d "{
        \"url\": \"$WORKER_URL\",
        \"backend\": \"$BACKEND\",
        \"arch\": \"$ARCH\",
        \"capacity\": $CAPACITY,
        \"tags\": []
    }"

echo ""
echo "Worker 注册命令执行完成！"

# 验证注册
echo ""
echo "📋 当前已注册的 Workers:"
curl -s "$SERVER_URL/api/v1/workers/status" | python -m json.tool
