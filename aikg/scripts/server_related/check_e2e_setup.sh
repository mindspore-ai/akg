#!/bin/bash
# 检查 Client-Server-Worker 端到端环境是否就绪

SERVER_URL=${1:-http://localhost:8000}
WORKER_URL=${2:-${AIKG_WORKER_URL:-http://localhost:9001}}

echo "=========================================="
echo "检查 Client-Server-Worker 环境"
echo "=========================================="
echo "Server URL: $SERVER_URL"
echo "Worker URL: $WORKER_URL"
echo "=========================================="

# 检查 Server
echo ""
echo "1. 检查 AIKG Server..."
if curl -s -f "$SERVER_URL/docs" > /dev/null 2>&1; then
    echo "   ✅ Server 正在运行"
else
    echo "   ❌ Server 未运行或无法访问"
    echo "   💡 启动: ./scripts/server_related/start_server.sh"
    exit 1
fi

# 检查 Worker Service
echo ""
echo "2. 检查 Worker Service..."
if curl -s -f "$WORKER_URL/api/v1/status" > /dev/null 2>&1; then
    echo "   ✅ Worker Service 正在运行"
    WORKER_STATUS=$(curl -s "$WORKER_URL/api/v1/status")
    echo "   📋 Worker 状态: $WORKER_STATUS"
else
    echo "   ❌ Worker Service 未运行或无法访问"
    echo "   💡 启动: ./scripts/server_related/start_worker_service.sh"
    exit 1
fi

# 检查 Worker 注册
echo ""
echo "3. 检查 Worker 注册状态..."
WORKERS=$(curl -s "$SERVER_URL/api/v1/workers/status")
if echo "$WORKERS" | grep -q '"backend"'; then
    echo "   ✅ Worker 已注册到 Server"
    echo "   📋 已注册的 Workers:"
    echo "$WORKERS" | python -m json.tool 2>/dev/null || echo "$WORKERS"
else
    echo "   ⚠️  Worker 未注册到 Server"
    echo "   💡 注册: ./scripts/server_related/register_worker_to_server.sh $SERVER_URL $WORKER_URL cuda a100 1"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ 环境检查通过！可以运行 Client 测试了"
echo "=========================================="

