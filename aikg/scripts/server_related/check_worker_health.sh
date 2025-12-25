#!/bin/bash
# 检查 Worker Service 健康状态
#
# ========================================
# IPv4/IPv6 配置说明:
# ========================================
# 通过环境变量或参数来指定 URL:
#   - 参数方式: ./check_worker_health.sh <worker_url>
#   - 环境变量: AIKG_WORKER_URL
#
# IPv4 示例:
#   ./check_worker_health.sh http://192.168.1.100:9001
#
# IPv6 示例 (注意 IPv6 地址需要用方括号包围):
#   ./check_worker_health.sh http://[2001:db8::1]:9001
#   或者:
#   export AIKG_WORKER_URL=http://[::1]:9001
#   ./check_worker_health.sh
# ========================================

WORKER_URL=${1:-${AIKG_WORKER_URL:-http://localhost:9001}}

echo "Checking Worker Service at: $WORKER_URL"
echo "=========================================="

# 检查状态
response=$(curl -s -w "\n%{http_code}" "$WORKER_URL/api/v1/status" 2>/dev/null)
http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | head -n-1)

if [ "$http_code" = "200" ]; then
    echo "✅ Worker Service is healthy"
    echo "Response: $body"
    exit 0
else
    echo "❌ Worker Service is not responding (HTTP $http_code)"
    echo "Response: $body"
    exit 1
fi
