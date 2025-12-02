#!/bin/bash
# 向 AIKG Server 注册 Worker Service
# 用法: ./scripts/server_related/register_worker_to_server.sh [server_url] [worker_url] [backend] [arch] [capacity]
# ./scripts/server_related/register_worker_to_server.sh http://localhost:8000 http://localhost:9001 cuda a100 1
#./scripts/server_related/register_worker_to_server.sh http://localhost:8000 http://localhost:9001 ascend ascend910b4 1
set -e

SERVER_URL=${1:-http://localhost:8000}
WORKER_URL=${2:-${AIKG_WORKER_URL:-http://localhost:9001}}
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

