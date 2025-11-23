#!/bin/bash
# 建立 SSH 隧道连接到远程 Worker Service
# 用法: ./scripts/server_related/setup_ssh_tunnel.sh [local_port] [remote_port] [ssh_host] [ssh_port] [ssh_user]
# 或者通过环境变量设置: SSH_HOST, SSH_PORT, SSH_USER

LOCAL_PORT=${1:-9001}
REMOTE_PORT=${2:-9001}
SSH_HOST=${3:-${SSH_HOST}}
SSH_PORT=${4:-${SSH_PORT:-22}}
SSH_USER=${5:-${SSH_USER:-${USER}}}

# 检查必需的参数
if [ -z "$SSH_HOST" ]; then
    echo "错误: 未指定 SSH 主机地址"
    echo "用法: $0 [local_port] [remote_port] [ssh_host] [ssh_port] [ssh_user]"
    echo "或者设置环境变量: SSH_HOST, SSH_PORT, SSH_USER"
    exit 1
fi

echo "=========================================="
echo "建立 SSH 隧道"
echo "=========================================="
echo "本地端口: $LOCAL_PORT"
echo "远程端口: $REMOTE_PORT"
echo "SSH 地址: $SSH_USER@$SSH_HOST:$SSH_PORT"
echo "=========================================="
echo ""
echo "⚠️  提示:"
echo "1. 此脚本会建立 SSH 隧道（前台运行）"
echo "2. 隧道建立后，可以通过 http://localhost:$LOCAL_PORT 访问远程 Worker Service"
echo "3. 按 Ctrl+C 可以关闭隧道"
echo "4. 需要输入 SSH 密码或使用 SSH 密钥认证"
echo ""
echo "正在建立隧道..."
echo ""

# 建立 SSH 隧道（前台运行，方便查看状态）
# -N: 不执行远程命令
# -L: 本地端口转发
# -o ServerAliveInterval=60: 保持连接活跃
ssh -N -L ${LOCAL_PORT}:localhost:${REMOTE_PORT} \
    -p ${SSH_PORT} \
    ${SSH_USER}@${SSH_HOST} \
    -o ServerAliveInterval=60 \
    -o ServerAliveCountMax=3

echo ""
echo "SSH 隧道已关闭"

