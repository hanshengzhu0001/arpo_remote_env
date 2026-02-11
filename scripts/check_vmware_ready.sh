#!/usr/bin/env bash
# Check if VMware Fusion is ready for arpo_remote_env (vmrun -T fusion).
# Run this from the project root or anywhere; it uses the system vmrun.

set -e
VMRUN="${VMRUN:-/Applications/VMware Fusion.app/Contents/Library/vmrun}"

echo "=== VMware readiness check ==="
echo ""

# 1. Fusion app present
if [[ ! -x "$VMRUN" ]]; then
  echo "FAIL: vmrun not found at $VMRUN"
  echo "      Install VMware Fusion from https://www.vmware.com/products/fusion.html"
  exit 1
fi
echo "OK: VMware Fusion vmrun found"

# 2. vmrun list (requires Fusion GUI to have been opened once so helpers load)
if ! "$VMRUN" -T fusion list 2>/dev/null; then
  echo ""
  echo "FAIL: vmrun list failed. This usually means:"
  echo "  1. Open VMware Fusion once from the Applications folder (or: open -a \"VMware Fusion\")."
  echo "  2. Approve any System Extension in System Settings → Privacy & Security if prompted."
  echo "  3. Then run this script again."
  exit 1
fi

echo ""
echo "=== VMware is ready for arpo_remote_env ==="
