#!/usr/bin/env bash
# Build Ubuntu autoinstall (cloud-init) seed ISO for fully unattended Server install.
# Output: ~/ubuntu-seed.iso (label CIDATA). Use with Ubuntu Server ISO and GRUB: autoinstall ds=nocloud
#
# Set password via env (optional):
#   UBUNTU_AUTOINSTALL_PASSWORD='MySecurePass' ./scripts/build_ubuntu_autoinstall_seed.sh
# Default password: ChangeMe123!

set -e
SEED_DIR="${SEED_DIR:-$HOME/ubuntu-seed}"
SEED_ISO="${SEED_ISO:-$HOME/ubuntu-seed.iso}"
PASSWORD="${UBUNTU_AUTOINSTALL_PASSWORD:-ChangeMe123!}"
HOSTNAME="${UBUNTU_AUTOINSTALL_HOSTNAME:-ubuntu-vm}"
USERNAME="${UBUNTU_AUTOINSTALL_USERNAME:-ubuntu}"
TIMEZONE="${UBUNTU_AUTOINSTALL_TIMEZONE:-America/New_York}"

echo "=== Build Ubuntu autoinstall seed ISO ==="
echo "  Seed dir: $SEED_DIR"
echo "  Output:   $SEED_ISO"
echo "  Hostname: $HOSTNAME  User: $USERNAME  Timezone: $TIMEZONE"
echo ""

# 1) Generate SHA-512 password hash (openssl passwd -6 = SHA-512)
PASSHASH=$(openssl passwd -6 "$PASSWORD")
mkdir -p "$SEED_DIR"
cd "$SEED_DIR"

# 2) user-data
cat > user-data << EOF
#cloud-config
autoinstall:
  version: 1
  locale: en_US.UTF-8
  keyboard:
    layout: us
  timezone: $TIMEZONE

  identity:
    hostname: $HOSTNAME
    username: $USERNAME
    password: "$PASSHASH"

  ssh:
    install-server: true
    allow-pw: true

  storage:
    layout:
      name: direct

  package_update: true
  package_upgrade: true
  packages:
    - open-vm-tools

  # Run apt update/upgrade during install so you don't have to after first boot
  late-commands:
    - curtin in-target --target=/target -- env DEBIAN_FRONTEND=noninteractive apt-get update
    - curtin in-target --target=/target -- env DEBIAN_FRONTEND=noninteractive apt-get upgrade -y
EOF
echo "Wrote $SEED_DIR/user-data"

# 3) meta-data
cat > meta-data << EOF
instance-id: iid-ubuntu-autoinstall-01
local-hostname: $HOSTNAME
EOF
echo "Wrote $SEED_DIR/meta-data"

# 4) Build seed ISO with volume label CIDATA (subiquity looks for this)
echo "Building seed ISO..."
rm -f "$SEED_ISO" "${SEED_ISO}.cdr"
hdiutil makehybrid -o "$SEED_ISO" "$SEED_DIR" -iso -joliet -default-volume-name CIDATA

# 5) macOS sometimes outputs .cdr
if [[ -f "${SEED_ISO}.cdr" ]]; then
  mv "${SEED_ISO}.cdr" "$SEED_ISO"
  echo "Renamed ${SEED_ISO}.cdr -> $SEED_ISO"
fi

echo ""
echo "=== Done ==="
echo "  Seed ISO: $SEED_ISO"
echo ""
echo "Next: attach this ISO and the Ubuntu Server ISO to your VM, then at GRUB append: autoinstall ds=nocloud"
echo "  Run: ./scripts/attach_autoinstall_to_vm.sh   (to add seed ISO to VM and print GRUB steps)"
echo ""
