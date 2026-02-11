#!/usr/bin/env bash
# Attach the autoinstall seed ISO to the Ubuntu VM as a second CD, and disable 3D graphics.
# Run after build_ubuntu_autoinstall_seed.sh. VM must exist (create_ubuntu_vm_fusion.sh or switch_vm_to_server_install.sh).
#
# Requires: seed ISO at ~/ubuntu-seed.iso (or SEED_ISO env).

set -e
VM_NAME="${VM_NAME:-Ubuntu-ARPO-Env}"
VMS_DIR="${VMS_DIR:-$HOME/Virtual Machines.localized}"
VMX_FILE="$VMS_DIR/${VM_NAME}.vmwarevm/${VM_NAME}.vmx"
SEED_ISO="${SEED_ISO:-$HOME/ubuntu-seed.iso}"

if [[ ! -f "$SEED_ISO" ]]; then
  echo "ERROR: Seed ISO not found at $SEED_ISO"
  echo "  Run first: ./scripts/build_ubuntu_autoinstall_seed.sh"
  exit 1
fi
if [[ ! -f "$VMX_FILE" ]]; then
  echo "ERROR: VM not found at $VMX_FILE"
  exit 1
fi

echo "=== Attach autoinstall seed to VM ==="
echo "  VM:   $VMX_FILE"
echo "  Seed: $SEED_ISO"
echo ""

# Add second SATA CD (sata0:2) for seed ISO if not present
if ! grep -q 'sata0:2.present' "$VMX_FILE"; then
  echo "Adding second CD (seed ISO) to VMX..."
  cat >> "$VMX_FILE" << VMXSEED

sata0:2.present = "TRUE"
sata0:2.fileName = "$SEED_ISO"
sata0:2.deviceType = "cdrom-image"
VMXSEED
  echo "  Added sata0:2 -> $SEED_ISO"
else
  # Update existing sata0:2 path
  if grep -q 'sata0:2.fileName' "$VMX_FILE"; then
    ESCAPED_SEED=$(echo "$SEED_ISO" | sed 's/[\/&]/\\&/g')
    sed -i '' "s|sata0:2.fileName = .*|sata0:2.fileName = \"$ESCAPED_SEED\"|" "$VMX_FILE"
    echo "  Updated sata0:2 -> $SEED_ISO"
  fi
fi

# Disable 3D acceleration (avoids installer issues; subiquity prefers no 3D)
if ! grep -q 'mks.enableGL' "$VMX_FILE"; then
  echo "Disabling 3D graphics in VMX..."
  echo 'mks.enableGL = "FALSE"' >> "$VMX_FILE"
  echo "  Set mks.enableGL = FALSE"
else
  sed -i '' 's/mks.enableGL = .*/mks.enableGL = "FALSE"/' "$VMX_FILE"
  echo "  Updated mks.enableGL = FALSE"
fi

echo ""
echo "=== Done ==="
echo ""
echo "--- GRUB: trigger autoinstall ---"
echo "1. Power On the VM (Fusion)."
echo "2. At GRUB menu: highlight 'Try or Install Ubuntu Server', press e."
echo "3. Find the line starting with 'linux' (or 'linuxefi')."
echo "4. Append before the '---':   autoinstall ds=nocloud"
echo "   So it contains:   ... autoinstall ds=nocloud ---"
echo "5. Boot: Ctrl+X (or F10 / fn+F10)."
echo ""
echo "Install runs unattended. After reboot:"
echo "  - In Fusion: disconnect both CD images (installer + seed) so VM boots from disk."
echo "  - Log in: user ubuntu, password (the one you set in build_ubuntu_autoinstall_seed.sh)."
echo "  - Then: sudo apt update && sudo apt upgrade -y"
echo ""
