#!/bin/bash
# 启动 AIKG Server
# 用法: ./scripts/server_related/start_server.sh [port]
# ./scripts/server_related/start_server.sh 8000
#
# ========================================
# IPv4/IPv6 配置说明:
# ========================================
# 通过环境变量 AIKG_SERVER_HOST 来控制监听地址:
#   - IPv4 监听所有接口: export AIKG_SERVER_HOST=0.0.0.0 (默认)
#   - IPv6 监听所有接口: export AIKG_SERVER_HOST=::
#   - IPv4 本地回环:    export AIKG_SERVER_HOST=127.0.0.1
#   - IPv6 本地回环:    export AIKG_SERVER_HOST=::1
#   - 指定 IPv4 地址:   export AIKG_SERVER_HOST=192.168.1.100
#   - 指定 IPv6 地址:   export AIKG_SERVER_HOST=2001:db8::1
#
# 注意: 使用 :: 可以同时监听 IPv4 和 IPv6 (dual-stack)，但需要操作系统支持
# ========================================

set -e

# 从环境变量获取 host，默认为 0.0.0.0 (IPv4 全接口)
HOST=${AIKG_SERVER_HOST:-0.0.0.0}
PORT=${1:-${AIKG_SERVER_PORT:-8000}}

echo "=========================================="
echo "启动 AIKG Server"
echo "=========================================="
echo "Host: $HOST"
echo "Port: $PORT"
echo "=========================================="

cd "$(dirname "$0")/../.."
source env.sh

echo "Starting AIKG Server on $HOST:$PORT..."
# python -m ai_kernel_generator.server.app

# 使用 uvicorn 直接启动以支持自定义端口
# 设置环境变量供 Python 代码使用
export AIKG_SERVER_HOST=$HOST
export AIKG_SERVER_PORT=$PORT

uvicorn ai_kernel_generator.server.app:app --host "$HOST" --port "$PORT"
