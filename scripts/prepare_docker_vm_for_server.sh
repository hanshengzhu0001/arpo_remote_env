#!/usr/bin/env bash
# Option 2: Pre-download the VM on this machine (Mac or any host with space), then copy
# only Ubuntu.qcow2 to the server. Server needs ~20GB for the image, not 30GB for zip+extract.
#
# Usage:
#   PREPARE_VM_SERVER=ubuntu@34.227.191.7 PREPARE_VM_KEY=path/to/key.pem ./scripts/prepare_docker_vm_for_server.sh
# Or set PREPARE_VM_* and run from repo root.

set -e
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
VMS_DIR="${VMS_DIR:-$REPO_ROOT/docker_vm_data}"
ZIP_URL="${ZIP_URL:-https://huggingface.co/datasets/xlangai/ubuntu_osworld/resolve/main/Ubuntu.qcow2.zip}"
ZIP_NAME="${ZIP_NAME:-Ubuntu.qcow2.zip}"
SSH_TARGET="${PREPARE_VM_SERVER:-}"
SSH_KEY="${PREPARE_VM_KEY:-}"

echo "=== Pre-download VM and copy to server ==="
echo "  Local dir: $VMS_DIR"
echo "  Server:    ${SSH_TARGET:-（not set, will only download+unzip）}"
echo ""

mkdir -p "$VMS_DIR"
cd "$VMS_DIR"

# 1) Download (curl -C - resumes if partial)
echo "Downloading (or resuming) $ZIP_URL ..."
curl -L -C - -o "$ZIP_NAME" "$ZIP_URL"

# 2) Unzip
if [[ ! -f Ubuntu.qcow2 ]]; then
  echo "Unzipping $ZIP_NAME ..."
  unzip -o "$ZIP_NAME"
else
  echo "Ubuntu.qcow2 already present, skipping unzip."
fi

# 3) Copy to server (rsync resumes if interrupted)
if [[ -n "$SSH_TARGET" ]]; then
  REMOTE_DIR="${PREPARE_VM_REMOTE_DIR:-arpo_remote_env/docker_vm_data}"
  echo "Copying Ubuntu.qcow2 to $SSH_TARGET:~/$REMOTE_DIR/ (resumes if partial) ..."
  ssh ${SSH_KEY:+-i "$SSH_KEY"} "$SSH_TARGET" "mkdir -p ~/$REMOTE_DIR"
  rsync -avz --progress --partial ${SSH_KEY:+-e "ssh -i $SSH_KEY"} "$VMS_DIR/Ubuntu.qcow2" "$SSH_TARGET:~/$REMOTE_DIR/"
  echo "Done. On server run (no need for SKIP_DOCKER_VM_DOWNLOAD; image is present):"
  echo "  cd ~/arpo_remote_env && python -m uvicorn scripts.remote_env_server:app --host 0.0.0.0 --port 15001"
else
  echo "Set PREPARE_VM_SERVER (e.g. ubuntu@34.227.191.7) and optionally PREPARE_VM_KEY to copy to server."
  echo "Ubuntu.qcow2 is at: $VMS_DIR/Ubuntu.qcow2"
fi
echo ""
