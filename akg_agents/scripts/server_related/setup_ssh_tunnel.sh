#!/bin/bash
# 建立 SSH 隧道连接到远程 Worker Service
# 用法: ./scripts/server_related/setup_ssh_tunnel.sh [local_port] [remote_port] [ssh_host] [ssh_port] [ssh_user]
# 或者通过环境变量设置: SSH_HOST, SSH_PORT, SSH_USER
#
# ========================================
# IPv4/IPv6 配置说明:
# ========================================
# SSH 本身支持 IPv4 和 IPv6，此脚本通过以下方式支持：
#
# SSH_HOST 可以是:
#   - IPv4 地址: 192.168.1.100
#   - IPv6 地址: 2001:db8::1
#   - 主机名:    server.example.com
#
# SSH 隧道绑定地址 (LOCAL_BIND_ADDR 环境变量):
#   - IPv4 本地: 127.0.0.1 (默认)
#   - IPv6 本地: ::1
#   - 所有接口: 0.0.0.0 或 ::
#
# 隧道远程端 (REMOTE_BIND_ADDR 环境变量):
#   - 默认 localhost
#   - IPv4: 127.0.0.1
#   - IPv6: ::1
#
# IPv6 示例:
#   export SSH_HOST=2001:db8::1
#   export LOCAL_BIND_ADDR=::1
#   export REMOTE_BIND_ADDR=::1
#   ./setup_ssh_tunnel.sh 9001 9001
#
# 建立隧道后访问方式:
#   - IPv4: http://127.0.0.1:$LOCAL_PORT
#   - IPv6: http://[::1]:$LOCAL_PORT
# ========================================

LOCAL_PORT=${1:-9001}
REMOTE_PORT=${2:-9001}
SSH_HOST=${3:-${SSH_HOST}}
SSH_PORT=${4:-${SSH_PORT:-22}}
SSH_USER=${5:-${SSH_USER:-${USER}}}

# 本地和远程绑定地址，支持 IPv6
LOCAL_BIND_ADDR=${LOCAL_BIND_ADDR:-localhost}
REMOTE_BIND_ADDR=${REMOTE_BIND_ADDR:-localhost}

# 检查必需的参数
if [ -z "$SSH_HOST" ]; then
    echo "错误: 未指定 SSH 主机地址"
    echo "用法: $0 [local_port] [remote_port] [ssh_host] [ssh_port] [ssh_user]"
    echo "或者设置环境变量: SSH_HOST, SSH_PORT, SSH_USER"
    echo ""
    echo "IPv6 示例:"
    echo "  export SSH_HOST=2001:db8::1"
    echo "  export LOCAL_BIND_ADDR=::1"
    echo "  $0 9001 9001"
    exit 1
fi

echo "=========================================="
echo "建立 SSH 隧道"
echo "=========================================="
echo "本地绑定: $LOCAL_BIND_ADDR:$LOCAL_PORT"
echo "远程绑定: $REMOTE_BIND_ADDR:$REMOTE_PORT"
echo "SSH 地址: $SSH_USER@$SSH_HOST:$SSH_PORT"
echo "=========================================="
echo ""
echo "⚠️  提示:"
echo "1. 此脚本会建立 SSH 隧道（前台运行）"
echo "2. 隧道建立后，可以通过以下地址访问远程 Worker Service:"
echo "   - IPv4: http://127.0.0.1:$LOCAL_PORT"
echo "   - IPv6: http://[::1]:$LOCAL_PORT"
echo "3. 按 Ctrl+C 可以关闭隧道"
echo "4. 需要输入 SSH 密码或使用 SSH 密钥认证"
echo ""
echo "正在建立隧道..."
echo ""

# 建立 SSH 隧道（前台运行，方便查看状态）
# -N: 不执行远程命令
# -L: 本地端口转发
# -o ServerAliveInterval=60: 保持连接活跃
ssh -N -L ${LOCAL_BIND_ADDR}:${LOCAL_PORT}:${REMOTE_BIND_ADDR}:${REMOTE_PORT} \
    -p ${SSH_PORT} \
    ${SSH_USER}@${SSH_HOST} \
    -o ServerAliveInterval=60 \
    -o ServerAliveCountMax=3

echo ""
echo "SSH 隧道已关闭"
