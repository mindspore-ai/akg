#!/bin/bash
# 启动 AIKG Server
# 用法: ./scripts/server_related/start_server.sh [port]

set -e

PORT=${1:-8000}

echo "=========================================="
echo "启动 AIKG Server"
echo "=========================================="
echo "Port: $PORT"
echo "=========================================="

cd "$(dirname "$0")/../.."
source env.sh

echo "Starting AIKG Server on port $PORT..."
python -m ai_kernel_generator.server.app

# 或者使用 uvicorn 直接启动（更多控制选项）
# uvicorn ai_kernel_generator.server.app:app --host 0.0.0.0 --port $PORT

