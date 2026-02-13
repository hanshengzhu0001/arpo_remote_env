#!/bin/bash
# Quick verification script: test cluster -> remote env server connectivity
# Usage: ./scripts/verify_remote_env_connection.sh [SERVER_URL]
# Default: http://100.48.93.208:15001

SERVER_URL="${1:-http://100.48.93.208:15001}"
SERVER_IP=$(echo "$SERVER_URL" | sed -E 's|https?://([^:/]+).*|\1|')
SERVER_PORT=$(echo "$SERVER_URL" | sed -E 's|https?://[^:]+:([0-9]+).*|\1|' || echo "15001")

echo "=== Verifying connection to remote env server ==="
echo "Server: $SERVER_URL"
echo ""

echo "1. Testing basic connectivity (TCP port $SERVER_PORT)..."
if timeout 3 bash -c "cat < /dev/null > /dev/tcp/$SERVER_IP/$SERVER_PORT" 2>/dev/null; then
    echo "   ✓ Port $SERVER_PORT is reachable"
else
    echo "   ✗ Port $SERVER_PORT is NOT reachable (check security group/firewall)"
    exit 1
fi

echo ""
echo "2. Testing /health endpoint..."
HEALTH_RESPONSE=$(curl -s -w "\n%{http_code}" "$SERVER_URL/health" 2>&1)
HTTP_CODE=$(echo "$HEALTH_RESPONSE" | tail -n1)
BODY=$(echo "$HEALTH_RESPONSE" | head -n-1)

if [ "$HTTP_CODE" = "200" ]; then
    echo "   ✓ Health check passed (HTTP $HTTP_CODE)"
    echo "   Response: $BODY"
else
    echo "   ✗ Health check failed (HTTP $HTTP_CODE)"
    echo "   Response: $BODY"
    exit 1
fi

echo ""
echo "3. Testing /env/evaluate endpoint (may return 0.0 if no reset)..."
EVAL_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$SERVER_URL/env/evaluate" \
    -H "Content-Type: application/json" \
    -d '{}' \
    --max-time 10 2>&1)
EVAL_HTTP_CODE=$(echo "$EVAL_RESPONSE" | tail -n1)
EVAL_BODY=$(echo "$EVAL_RESPONSE" | head -n-1)

if [ "$EVAL_HTTP_CODE" = "200" ] || [ "$EVAL_HTTP_CODE" = "503" ]; then
    echo "   ✓ Evaluate endpoint responded (HTTP $EVAL_HTTP_CODE)"
    echo "   Response: $EVAL_BODY"
    if [ "$EVAL_HTTP_CODE" = "503" ]; then
        echo "   Note: 503 means env not ready (no reset yet) - this is expected"
    fi
else
    echo "   ✗ Evaluate endpoint failed (HTTP $EVAL_HTTP_CODE)"
    echo "   Response: $EVAL_BODY"
    exit 1
fi

echo ""
echo "=== All checks passed! Connection is ready. ==="
