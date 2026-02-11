#!/usr/bin/env bash
# Run apt update && apt upgrade inside the Ubuntu VM (after it's booted from disk).
# Uses vmrun runScriptInGuest. Set UBUNTU_AUTOINSTALL_PASSWORD if you changed it from default.
#
# Usage: ./scripts/run_guest_apt_update.sh
# (VM must be powered on and booted into the installed Ubuntu.)

set -e
VM_NAME="${VM_NAME:-Ubuntu-ARPO-Env}"
VMS_DIR="${VMS_DIR:-$HOME/Virtual Machines.localized}"
VMX_FILE="$VMS_DIR/${VM_NAME}.vmwarevm/${VM_NAME}.vmx"
VMRUN="${VMRUN:-/Applications/VMware Fusion.app/Contents/Library/vmrun}"
PASSWORD="${UBUNTU_AUTOINSTALL_PASSWORD:-ChangeMe123!}"

if [[ ! -f "$VMX_FILE" ]]; then
  echo "ERROR: VM not found at $VMX_FILE"
  exit 1
fi
if [[ ! -x "$VMRUN" ]]; then
  echo "ERROR: vmrun not found at $VMRUN"
  exit 1
fi

# Escape single quotes in password for use inside single-quoted script
PASS_ESC=$(echo "$PASSWORD" | sed "s/'/'\\\\''/g")
SCRIPT="echo '$PASS_ESC' | sudo -S env DEBIAN_FRONTEND=noninteractive apt-get update && echo '$PASS_ESC' | sudo -S env DEBIAN_FRONTEND=noninteractive apt-get upgrade -y"

echo "=== Run apt update && upgrade in guest ==="
echo "  VM: $VMX_FILE"
echo "  Waiting for guest IP (VM must be booted)..."
"$VMRUN" -T fusion getGuestIPAddress "$VMX_FILE" -wait
echo "  Running apt update && upgrade in guest..."
"$VMRUN" -T fusion runScriptInGuest "$VMX_FILE" -gu ubuntu -gp "$PASSWORD" /bin/bash "$SCRIPT"
echo "  Done."
echo ""
