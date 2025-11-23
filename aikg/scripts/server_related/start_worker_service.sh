#!/bin/bash
# 启动 AIKG Worker Service
# 用法: ./scripts/server_related/start_worker_service.sh [backend] [arch] [devices] [port]
# devices: 逗号分隔的设备ID列表，例如 "0,1,2,3,4,5" 或单个设备 "0"

set -e

# 默认配置
# devices 参数应为逗号分隔的设备ID列表，如 "0,1,2,3,4,5"，而不是设备数量
BACKEND=${1:-cuda}
ARCH=${2:-a100}
DEVICES=${3:-0}
PORT=${4:-9001}

echo "=========================================="
echo "启动 AIKG Worker Service"
echo "=========================================="
echo "Backend: $BACKEND"
echo "Arch: $ARCH"
echo "Devices: $DEVICES"
echo "Port: $PORT"
echo "=========================================="

# 设置环境变量
export WORKER_BACKEND=$BACKEND
export WORKER_ARCH=$ARCH
export WORKER_DEVICES=$DEVICES
export WORKER_PORT=$PORT

# 启动服务
cd "$(dirname "$0")/../.."
source env.sh

echo "Starting Worker Service on port $PORT..."
python -m ai_kernel_generator.worker.server

# 或者使用 uvicorn 直接启动（更多控制选项）
# uvicorn ai_kernel_generator.worker.server:app --host 0.0.0.0 --port $PORT

