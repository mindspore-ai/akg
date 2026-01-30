#!/bin/bash
# 启动 AIKG Server 和 本地 Worker，并自动注册
# 用法: ./scripts/server_related/start_server_with_local_worker.sh [server_port] [worker_port] [backend] [arch] [devices]
# 示例: ./scripts/server_related/start_server_with_local_worker.sh 8000 9001 ascend ascend910b4 0,1,2,3
# 示例: ./scripts/server_related/start_server_with_local_worker.sh 8000 9001 cuda a100 0,1,2,3
#
# ========================================
# IPv4/IPv6 配置说明:
# ========================================
# 通过环境变量来控制监听地址和 URL 格式:
#
# 监听地址环境变量:
#   - AKG_AGENTS_SERVER_HOST: Server 监听地址，默认 0.0.0.0
#   - AKG_AGENTS_WORKER_HOST: Worker 监听地址，默认 0.0.0.0
#
# URL 环境变量 (用于服务发现和注册):
#   - AKG_AGENTS_SERVER_URL: Server 的访问地址，默认 http://localhost:$SERVER_PORT
#   - AKG_AGENTS_WORKER_URL: Worker 的访问地址，默认 http://localhost:$WORKER_PORT
#
# IPv4 示例:
#   export AKG_AGENTS_SERVER_HOST=0.0.0.0
#   export AKG_AGENTS_SERVER_URL=http://192.168.1.100:8000
#
# IPv6 示例 (注意 IPv6 地址需要用方括号包围):
#   export AKG_AGENTS_SERVER_HOST=::
#   export AKG_AGENTS_SERVER_URL=http://[2001:db8::1]:8000
#   export AKG_AGENTS_WORKER_HOST=::
#   export AKG_AGENTS_WORKER_URL=http://[2001:db8::1]:9001
#
# 双栈模式:
#   使用 :: 作为 host 可以同时监听 IPv4 和 IPv6
# ========================================

set -e

# 参数处理
SERVER_PORT=${1:-${AKG_AGENTS_SERVER_PORT:-8000}}
WORKER_PORT=${2:-${AKG_AGENTS_WORKER_PORT:-9001}}
BACKEND=${3:-cuda}
ARCH=${4:-a100}
DEVICES=${5:-0}

# 从环境变量获取 host，默认为 0.0.0.0
SERVER_HOST=${AKG_AGENTS_SERVER_HOST:-0.0.0.0}
WORKER_HOST=${AKG_AGENTS_WORKER_HOST:-0.0.0.0}

# URL 配置 - 支持从环境变量覆盖，以支持 IPv6 或自定义地址
# 默认使用 localhost，IPv6 场景需要通过环境变量设置如 http://[::1]:8000
SERVER_URL=${AKG_AGENTS_SERVER_URL:-http://localhost:$SERVER_PORT}
WORKER_URL=${AKG_AGENTS_WORKER_URL:-http://localhost:$WORKER_PORT}

# 计算 capacity (device 数量)
IFS=',' read -ra DEVICE_ARRAY <<< "$DEVICES"
CAPACITY=${#DEVICE_ARRAY[@]}

echo "=========================================="
echo "启动 AIKG Server 和 Local Worker (全自动)"
echo "=========================================="
echo "Server Host: $SERVER_HOST"
echo "Server Port: $SERVER_PORT"
echo "Server URL: $SERVER_URL"
echo "Worker Host: $WORKER_HOST"
echo "Worker Port: $WORKER_PORT"
echo "Worker URL: $WORKER_URL"
echo "Backend: $BACKEND"
echo "Arch: $ARCH"
echo "Devices: $DEVICES (Capacity: $CAPACITY)"
echo "=========================================="

cd "$(dirname "$0")/../.."
source env.sh

# 定义清理函数
cleanup() {
    echo ""
    echo "🛑 正在停止服务..."
    if [ -n "$WORKER_PID" ]; then
        echo "Killing Worker (PID: $WORKER_PID)..."
        kill $WORKER_PID 2>/dev/null || true
    fi
    if [ -n "$SERVER_PID" ]; then
        echo "Killing Server (PID: $SERVER_PID)..."
        kill $SERVER_PID 2>/dev/null || true
    fi
    exit 0
}

trap cleanup SIGINT SIGTERM

# 1. 启动 Server
echo "🚀 Starting Server on $SERVER_HOST:$SERVER_PORT..."
# 设置环境变量供 Python 代码使用
export AKG_AGENTS_SERVER_HOST=$SERVER_HOST
export AKG_AGENTS_SERVER_PORT=$SERVER_PORT

uvicorn akg_agents.server.app:app --host "$SERVER_HOST" --port $SERVER_PORT > server.log 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# 等待 Server 启动
echo "Waiting for Server to be ready..."
MAX_RETRIES=30
count=0
while ! curl -s "$SERVER_URL/docs" > /dev/null; do
    sleep 1
    count=$((count+1))
    if [ $count -ge $MAX_RETRIES ]; then
        echo "Error: Server failed to start in $MAX_RETRIES seconds."
        echo "--- Server Log ---"
        tail -n 20 server.log
        echo "------------------"
        cleanup
    fi
    echo -n "."
done
echo " Server is UP!"

# 2. 启动 Worker
echo "🚀 Starting Worker on $WORKER_HOST:$WORKER_PORT..."
export WORKER_BACKEND=$BACKEND
export WORKER_ARCH=$ARCH
export WORKER_DEVICES=$DEVICES
export WORKER_PORT=$WORKER_PORT
export WORKER_HOST=$WORKER_HOST

# 使用 python -m 启动 Worker
python -m akg_agents.worker.server > worker.log 2>&1 &
WORKER_PID=$!
echo "Worker PID: $WORKER_PID"

# 等待 Worker 启动
echo "Waiting for Worker to be ready..."
count=0
while ! curl -s "$WORKER_URL/api/v1/status" > /dev/null; do
    sleep 1
    count=$((count+1))
    if [ $count -ge $MAX_RETRIES ]; then
        echo "Error: Worker failed to start in $MAX_RETRIES seconds."
        echo "--- Worker Log ---"
        tail -n 20 worker.log
        echo "------------------"
        cleanup
    fi
    echo -n "."
done
echo " Worker is UP!"

# 3. 注册 Worker
echo "🔗 Registering Worker to Server..."
response=$(curl -s -X POST "$SERVER_URL/api/v1/workers/register" \
    -H "Content-Type: application/json" \
    -d "{
        \"url\": \"$WORKER_URL\",
        \"backend\": \"$BACKEND\",
        \"arch\": \"$ARCH\",
        \"capacity\": $CAPACITY,
        \"tags\": [\"local\"]
    }")

echo "Registration response: $response"

echo "=========================================="
echo "✅ 全部服务启动完成!"
echo "Server: $SERVER_URL"
echo "Worker: $WORKER_URL"
echo "按 Ctrl+C 停止所有服务"
echo "=========================================="

# 等待
wait
