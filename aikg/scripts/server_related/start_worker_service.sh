#!/bin/bash
# 启动 AIKG Worker Service
# 用法: ./scripts/server_related/start_worker_service.sh [backend] [arch] [devices] [port]
# devices: 逗号分隔的设备ID列表，例如 "0,1,2,3,4,5" 或单个设备 "0"
# ./scripts/server_related/start_worker_service.sh cuda a100 0,1,2,3 9001
# ./scripts/server_related/start_worker_service.sh ascend ascend910b4 0,1,2,3 9001
#
# ========================================
# IPv4/IPv6 配置说明:
# ========================================
# 通过环境变量 AIKG_WORKER_HOST 来控制监听地址:
#   - IPv4 监听所有接口: export AIKG_WORKER_HOST=0.0.0.0 (默认)
#   - IPv6 监听所有接口: export AIKG_WORKER_HOST=::
#   - IPv4 本地回环:    export AIKG_WORKER_HOST=127.0.0.1
#   - IPv6 本地回环:    export AIKG_WORKER_HOST=::1
#   - 指定 IPv4 地址:   export AIKG_WORKER_HOST=192.168.1.100
#   - 指定 IPv6 地址:   export AIKG_WORKER_HOST=2001:db8::1
#
# 注意: 使用 :: 可以同时监听 IPv4 和 IPv6 (dual-stack)，但需要操作系统支持
# ========================================

set -e

# 默认配置
# devices 参数应为逗号分隔的设备ID列表，如 "0,1,2,3,4,5"，而不是设备数量
BACKEND=${1:-cuda}
ARCH=${2:-a100}
DEVICES=${3:-0}
PORT=${4:-${AIKG_WORKER_PORT:-9001}}
# 从环境变量获取 host，默认为 0.0.0.0 (IPv4 全接口)
HOST=${AIKG_WORKER_HOST:-0.0.0.0}

echo "=========================================="
echo "启动 AIKG Worker Service"
echo "=========================================="
echo "Host: $HOST"
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
export WORKER_HOST=$HOST

# 启动服务
cd "$(dirname "$0")/../.."
source env.sh

echo "Starting Worker Service on $HOST:$PORT..."
python -m ai_kernel_generator.worker.server

# 或者使用 uvicorn 直接启动（更多控制选项）
# uvicorn ai_kernel_generator.worker.server:app --host "$HOST" --port $PORT
