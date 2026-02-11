#!/usr/bin/env bash
# Create an Ubuntu VM in VMware Fusion with:
# - Nested virtualization enabled (vhv.enable = "TRUE")
# - Shared folder pointing to ARPO_replicate on the Mac
# - Boot from Ubuntu Server ISO (text installer). Apple Silicon -> arm64, Intel -> amd64.
#
# Run on your Mac. Requires: VMware Fusion, network for ISO download.
# After this script: open the VM in Fusion, boot from the ISO, install Ubuntu Server.

set -e
VM_NAME="${VM_NAME:-Ubuntu-ARPO-Env}"
VMS_DIR="${VMS_DIR:-$HOME/Virtual Machines.localized}"
VM_BUNDLE="$VMS_DIR/${VM_NAME}.vmwarevm"
VMX_FILE="$VM_BUNDLE/${VM_NAME}.vmx"
DISK_FILE="$VM_BUNDLE/${VM_NAME}.vmdk"
DISK_SIZE_GB="${DISK_SIZE_GB:-40}"
ISO_DIR="${ISO_DIR:-$HOME/Downloads}"
UBUNTU_VER="24.04.3"

# Pick Server ISO and guestOS by Mac architecture
case "$(uname -m)" in
  arm64)
    # Apple Silicon (M1/M2/M3)
    ISO_NAME="${ISO_NAME:-ubuntu-${UBUNTU_VER}-live-server-arm64.iso}"
    ISO_URL="${ISO_URL:-https://cdimage.ubuntu.com/releases/24.04/release/${ISO_NAME}}"
    GUEST_OS="${GUEST_OS:-arm-ubuntu-64}"
    ;;
  x86_64)
    # Intel Mac
    ISO_NAME="${ISO_NAME:-ubuntu-${UBUNTU_VER}-live-server-amd64.iso}"
    ISO_URL="${ISO_URL:-https://releases.ubuntu.com/24.04/${ISO_NAME}}"
    GUEST_OS="${GUEST_OS:-ubuntu-64}"
    ;;
  *)
    echo "ERROR: Unsupported arch $(uname -m). Set ISO_URL, ISO_NAME, GUEST_OS manually."
    exit 1
    ;;
esac
ISO_PATH="$ISO_DIR/$ISO_NAME"

# Shared folder: repo on Mac so the VM can access it
REPO_HOST_PATH="${REPO_HOST_PATH:-/Users/hanszhu/Desktop/ARPO_replicate}"
FUSION_VDISKMANAGER="/Applications/VMware Fusion.app/Contents/Library/vmware-vdiskmanager"

echo "=== Create Ubuntu VM for ARPO (Fusion) ==="
echo "  VM: $VM_BUNDLE"
echo "  Arch: $(uname -m) -> $ISO_NAME"
echo "  Shared folder: $REPO_HOST_PATH"
echo ""

# 1) Download Ubuntu Server ISO if missing
if [[ ! -f "$ISO_PATH" ]]; then
  echo "Downloading Ubuntu Server $(uname -m) ISO..."
  mkdir -p "$ISO_DIR"
  curl -L -o "$ISO_PATH" "$ISO_URL"
  echo "Downloaded $ISO_PATH"
else
  echo "Using existing ISO: $ISO_PATH"
fi

# 2) Create VM bundle directory
mkdir -p "$VM_BUNDLE"
cd "$VM_BUNDLE"

# 3) Create virtual disk (Fusion must be running for vdiskmanager; try anyway)
if [[ ! -f "$DISK_FILE" ]]; then
  echo "Creating virtual disk ${DISK_SIZE_GB}GB..."
  if [[ -x "$FUSION_VDISKMANAGER" ]]; then
    "$FUSION_VDISKMANAGER" -c -s "${DISK_SIZE_GB}GB" -a lsilogic -t 0 "$DISK_FILE" || true
  fi
  if [[ ! -f "$DISK_FILE" ]]; then
    echo "Warning: Could not create disk with vdiskmanager (Fusion may need to be running)."
    echo "You can create the disk from Fusion: Settings -> Add Disk, or create a new VM with the wizard and we'll only add settings."
  fi
else
  echo "Virtual disk already exists."
fi

# 4) Write .vmx with nested virtualization and shared folder
#    Use SATA for CD (ISO) and SCSI for disk; Fusion 13 ARM accepts this.
echo "Writing $VMX_FILE ..."
cat > "$VMX_FILE" << VMX
.encoding = "UTF-8"
config.version = "8"
virtualHW.version = "20"
guestOS = "$GUEST_OS"
firmware = "efi"
memsize = "4096"
numvcpus = "2"

vhv.enable = "TRUE"

sata0.present = "TRUE"
sata0:0.present = "TRUE"
sata0:0.fileName = "${VM_NAME}.vmdk"
sata0:1.present = "TRUE"
sata0:1.fileName = "$ISO_PATH"
sata0:1.deviceType = "cdrom-image"

ethernet0.present = "TRUE"
ethernet0.connectionType = "nat"
ethernet0.virtualDev = "vmxnet3"

usb.present = "TRUE"
usb_xhci.present = "TRUE"
ehci.present = "FALSE"

sharedFolder0.present = "TRUE"
sharedFolder0.enabled = "TRUE"
sharedFolder0.readAccess = "TRUE"
sharedFolder0.writeAccess = "TRUE"
sharedFolder0.folderName = "ARPO_replicate"
sharedFolder0.hostPath = "$REPO_HOST_PATH"

displayName = "$VM_NAME"
VMX

# If disk wasn't created, user must add one in Fusion (Settings -> Add Device -> New Hard Disk) and remove sata0:0 from .vmx or create disk manually.
if [[ ! -f "$DISK_FILE" ]]; then
  # Comment out disk so VM can at least start and boot from CD; user adds disk in Fusion
  sed -i '' 's/^sata0:0.present = "TRUE"/sata0:0.present = "FALSE"/' "$VMX_FILE"
  sed -i '' 's|^sata0:0.fileName = ".*"|sata0:0.fileName = ""|' "$VMX_FILE"
  echo "No virtual disk found. In Fusion: Settings -> Add Device -> New Hard Disk (e.g. 40GB), then edit .vmx to point sata0:0 at the new .vmdk."
fi

echo ""
echo "=== VM created ==="
echo "  VMX: $VMX_FILE"
echo "  Nested virtualization: vhv.enable = TRUE"
echo "  Shared folder: $REPO_HOST_PATH -> /mnt/hgfs/ARPO_replicate (after VMware Tools in guest)"
echo ""
echo "Next steps:"
echo "  1. Open VMware Fusion."
echo "  2. File -> Open -> select: $VMX_FILE"
echo "  3. Power On the VM; it will boot from the Ubuntu Server ISO (text installer, keyboard only)."
echo "  4. Complete the Ubuntu Server installation (install Open VM Tools when prompted for optional packages)."
echo "  5. After first boot, in the VM run: sudo mount -t fuse.vmhgfs-fuse .host:/ARPO_replicate /mnt -o allow_other  (or use /mnt/hgfs if available)."
echo ""

# Optional: open Fusion with this VM
read -p "Open this VM in VMware Fusion now? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[yY]$ ]]; then
  open "$VMX_FILE"
fi
