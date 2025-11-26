#!/bin/bash
# 启动 AIKG Server 并自动注册 Worker Service
# 用法: ./scripts/server_related/start_server_with_worker.sh [server_port] [worker_url] [backend] [arch]

set -e

SERVER_PORT=${1:-8000}
WORKER_URL=${2:-${AIKG_WORKER_URL:-http://localhost:9001}}
BACKEND=${3:-cuda}
ARCH=${4:-a100}
CAPACITY=${5:-1}

echo "=========================================="
echo "启动 AIKG Server 并注册 Worker"
echo "=========================================="
echo "Server Port: $SERVER_PORT"
echo "Worker URL: $WORKER_URL"
echo "Backend: $BACKEND"
echo "Arch: $ARCH"
echo "Capacity: $CAPACITY"
echo "=========================================="

cd "$(dirname "$0")/../.."
source env.sh

# 等待一下，确保 Server 启动
echo ""
echo "⚠️  注意: 此脚本会启动 Server，但需要手动注册 Worker"
echo "   或者使用另一个终端运行:"
echo "   ./scripts/server_related/register_worker_to_server.sh http://localhost:$SERVER_PORT $WORKER_URL $BACKEND $ARCH $CAPACITY"
echo ""

# 启动 Server
echo "Starting AIKG Server on port $SERVER_PORT..."
python -m ai_kernel_generator.server.app

