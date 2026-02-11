#!/usr/bin/env bash
# Run this script inside the Ubuntu VM (Fusion guest with nested virtualization enabled).
# It installs Docker, clones the repo with submodules, creates a venv, and installs deps.
# After this, run the env server and the reverse tunnel (see REMOTE_ENV_TUNNEL_SETUP.md).

set -e
REPO_URL="${REPO_URL:-https://github.com/gowathena/arpo_replica.git}"
REPO_DIR="${REPO_DIR:-$HOME/arpo_replica}"

echo "=== Ubuntu VM setup for remote env (Docker + repo + venv) ==="
echo "  REPO_URL=$REPO_URL"
echo "  REPO_DIR=$REPO_DIR"
echo ""

# --- Docker ---
if ! command -v docker &>/dev/null; then
  echo "Installing Docker..."
  sudo apt update
  sudo apt install -y ca-certificates curl
  sudo install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  sudo chmod a+r /etc/apt/keyrings/docker.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
  sudo apt update
  sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  sudo usermod -aG docker "$USER"
  echo "Docker installed. You may need to run: newgrp docker (or log out and back in)."
else
  echo "Docker already installed."
fi

# --- Repo ---
if [[ ! -d "$REPO_DIR/.git" ]]; then
  echo "Cloning repo (with submodules)..."
  git clone --recurse-submodules "$REPO_URL" "$REPO_DIR"
  cd "$REPO_DIR"
  git submodule update --init --recursive
else
  echo "Repo already present at $REPO_DIR"
  cd "$REPO_DIR"
  git submodule update --init --recursive
fi

# --- Venv + deps ---
if [[ ! -d "$REPO_DIR/.venv" ]]; then
  echo "Creating venv and installing dependencies..."
  python3 -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
  cd OSWorld && pip install -r requirements.txt && pip install -e . && cd ..
  pip install uvicorn
  echo "Venv and deps installed."
else
  echo "Venv already exists at $REPO_DIR/.venv"
fi

echo ""
echo "=== Next steps (run these in the VM) ==="
echo "1. Start env server (keep running):"
echo "   cd $REPO_DIR && source .venv/bin/activate"
echo "   PROVIDER=docker python -m uvicorn scripts.remote_env_server:app --host 127.0.0.1 --port 18082"
echo ""
echo "2. In another terminal, start reverse tunnel:"
echo "   ssh -R 15001:127.0.0.1:18082 kevinzyz@deepx-a100-40g-2 -N"
echo ""
echo "3. On the cluster: python -m verl.trainer.main config=configs/smoke_remote_env_tunnel.yaml"
echo ""
