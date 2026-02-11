#!/usr/bin/env bash
# Remove current VM installation and switch to Ubuntu Server (text installer, no GUI).
# Picks ISO by Mac chip: Apple Silicon -> arm64, Intel -> amd64.
# Steps: power off VM, remove virtual disk, point CD at Server ISO, create new disk.
#
# Run on your Mac. Requires: VMware Fusion, vmrun in PATH or default location.

set -e
VM_NAME="${VM_NAME:-Ubuntu-ARPO-Env}"
VMS_DIR="${VMS_DIR:-$HOME/Virtual Machines.localized}"
VM_BUNDLE="$VMS_DIR/${VM_NAME}.vmwarevm"
VMX_FILE="$VM_BUNDLE/${VM_NAME}.vmx"
DISK_FILE="$VM_BUNDLE/${VM_NAME}.vmdk"
DISK_SIZE_GB="${DISK_SIZE_GB:-40}"
VMRUN="${VMRUN:-/Applications/VMware Fusion.app/Contents/Library/vmrun}"
FUSION_VDISKMANAGER="/Applications/VMware Fusion.app/Contents/Library/vmware-vdiskmanager"
ISO_DIR="${ISO_DIR:-$HOME/Downloads}"

# Pick Server ISO by Mac architecture
UBUNTU_VER="24.04.3"
case "$(uname -m)" in
  arm64)
    # Apple Silicon (M1/M2/M3)
    ISO_NAME="${ISO_NAME:-ubuntu-${UBUNTU_VER}-live-server-arm64.iso}"
    ISO_URL="${ISO_URL:-https://cdimage.ubuntu.com/releases/24.04/release/${ISO_NAME}}"
    ;;
  x86_64)
    # Intel Mac
    ISO_NAME="${ISO_NAME:-ubuntu-${UBUNTU_VER}-live-server-amd64.iso}"
    ISO_URL="${ISO_URL:-https://releases.ubuntu.com/24.04/${ISO_NAME}}"
    ;;
  *)
    echo "ERROR: Unsupported arch $(uname -m). Set ISO_URL and ISO_NAME manually."
    exit 1
    ;;
esac
ISO_PATH="$ISO_DIR/$ISO_NAME"

echo "=== Switch VM to Server installer (text, no GUI) ==="
echo "  VM: $VM_BUNDLE"
echo "  Arch: $(uname -m) -> $ISO_NAME"
echo ""

if [[ ! -f "$VMX_FILE" ]]; then
  echo "ERROR: VM not found at $VMX_FILE"
  echo "  Set VM_NAME / VMS_DIR if your VM is elsewhere."
  exit 1
fi

# 1) Power off VM if running
if [[ -x "$VMRUN" ]]; then
  if "$VMRUN" -T fusion list 2>/dev/null | grep -q "$VMX_FILE"; then
    echo "Powering off VM..."
    "$VMRUN" -T fusion stop "$VMX_FILE" hard || true
    sleep 2
  else
    echo "VM not running."
  fi
else
  echo "Warning: vmrun not found. Please power off the VM manually in Fusion."
fi

# 2) Remove current virtual disk (removes current installation)
if [[ -f "$DISK_FILE" ]]; then
  echo "Removing current virtual disk (current installation)..."
  rm -f "$DISK_FILE"
  echo "  Deleted $DISK_FILE"
else
  echo "No existing virtual disk found."
fi

# 3) Download Server ISO if missing
if [[ ! -f "$ISO_PATH" ]]; then
  echo "Downloading Ubuntu Server $(uname -m) ISO..."
  mkdir -p "$ISO_DIR"
  curl -L -o "$ISO_PATH" "$ISO_URL"
  echo "  Downloaded $ISO_PATH"
else
  echo "Using existing Server ISO: $ISO_PATH"
fi

# 4) Point VM CD at Server ISO (update sata0:1.fileName in .vmx)
if grep -q 'sata0:1.fileName' "$VMX_FILE"; then
  ESCAPED_ISO=$(echo "$ISO_PATH" | sed 's/[\/&]/\\&/g')
  sed -i '' "s|sata0:1.fileName = .*|sata0:1.fileName = \"$ESCAPED_ISO\"|" "$VMX_FILE"
  echo "Updated CD to Server ISO in $VMX_FILE"
fi

# 5) Create new empty virtual disk
echo "Creating new virtual disk ${DISK_SIZE_GB}GB..."
if [[ -x "$FUSION_VDISKMANAGER" ]]; then
  "$FUSION_VDISKMANAGER" -c -s "${DISK_SIZE_GB}GB" -a lsilogic -t 0 "$DISK_FILE" || true
fi
if [[ ! -f "$DISK_FILE" ]]; then
  echo "Warning: Could not create disk with vdiskmanager (Fusion may need to be running)."
  echo "  In Fusion: Settings -> Add Device -> New Hard Disk (e.g. ${DISK_SIZE_GB}GB), then power on and boot from CD."
  if ! grep -q 'sata0:0.present = "TRUE"' "$VMX_FILE"; then
    echo "  VMX currently has no disk; add a disk in Fusion and point sata0:0 at it."
  fi
else
  echo "  Created $DISK_FILE"
  if ! grep -q 'sata0:0.present = "TRUE"' "$VMX_FILE" || ! grep -q "sata0:0.fileName = \"${VM_NAME}.vmdk\"" "$VMX_FILE"; then
    sed -i '' 's/^sata0:0.present = "FALSE"/sata0:0.present = "TRUE"/' "$VMX_FILE"
    sed -i '' "s|^sata0:0.fileName = \"\"|sata0:0.fileName = \"${VM_NAME}.vmdk\"|" "$VMX_FILE"
  fi
fi

echo ""
echo "=== Done ==="
echo "Next steps:"
echo "  1. Open VMware Fusion: File -> Open -> $VMX_FILE"
echo "  2. Power On the VM; it will boot from the Ubuntu Server ISO."
echo "  3. Use the text installer (keyboard only, no GUI/mouse needed)."
echo "  4. When asked, choose to use the whole disk for a clean install."
echo "  5. After install, install Open VM Tools and set up shared folder (see REMOTE_ENV_TUNNEL_SETUP.md)."
echo ""
