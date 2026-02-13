#!/usr/bin/env bash
# Start the remote env server with AWS provider.
# On EC2: auto-detects AWS_SUBNET_ID and AWS_SECURITY_GROUP_ID from this instance if not set.
# Usage: from repo root, run:  ./scripts/start_remote_env_aws.sh
# Or:    bash scripts/start_remote_env_aws.sh

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Load .env if present (don't overwrite existing env)
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

export AWS_REGION="${AWS_REGION:-us-east-1}"

# On EC2, auto-detect subnet and security group from instance metadata if not set
if [ -z "$AWS_SUBNET_ID" ] || [ -z "$AWS_SECURITY_GROUP_ID" ]; then
  METADATA_URL="http://169.254.169.254/latest/meta-data"
  # IMDSv2 requires a token (many accounts use this by default)
  TOKEN=$(curl -s -f -m 2 -X PUT "$METADATA_URL/../api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 60" 2>/dev/null || true)
  if [ -n "$TOKEN" ]; then
    META() { curl -s -f -m 2 -H "X-aws-ec2-metadata-token: $TOKEN" "$METADATA_URL/$1"; }
  else
    META() { curl -s -f -m 2 "$METADATA_URL/$1"; }
  fi
  if META "instance-id" >/dev/null 2>&1; then
    MAC=$(META "network/interfaces/macs/" | head -1 | tr -d '\n\r' | sed 's|/$||')
    if [ -n "$MAC" ]; then
      [ -z "$AWS_SUBNET_ID" ] && AWS_SUBNET_ID=$(META "network/interfaces/macs/${MAC}/subnet-id")
      if [ -z "$AWS_SECURITY_GROUP_ID" ]; then
        AWS_SECURITY_GROUP_ID=$(META "network/interfaces/macs/${MAC}/security-group-ids" | head -1 | tr -d '\n\r')
      fi
    fi
    export AWS_SUBNET_ID
    export AWS_SECURITY_GROUP_ID
    if [ -n "$AWS_SUBNET_ID" ] && [ -n "$AWS_SECURITY_GROUP_ID" ]; then
      echo "Auto-detected on EC2: AWS_SUBNET_ID=$AWS_SUBNET_ID AWS_SECURITY_GROUP_ID=$AWS_SECURITY_GROUP_ID"
    fi
  fi
fi

if [ -z "$AWS_SUBNET_ID" ] || [ -z "$AWS_SECURITY_GROUP_ID" ]; then
  echo "Error: Set AWS_SUBNET_ID and AWS_SECURITY_GROUP_ID (in .env or env), or run this script on an EC2 instance in the target VPC."
  exit 1
fi

echo "Starting remote env server (AWS provider) on 0.0.0.0:15001 ..."
exec env PROVIDER=aws python -m uvicorn scripts.remote_env_server:app --host 0.0.0.0 --port 15001
