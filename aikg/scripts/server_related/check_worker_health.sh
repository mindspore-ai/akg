#!/bin/bash
# 检查 Worker Service 健康状态

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

