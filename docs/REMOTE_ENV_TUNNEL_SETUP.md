# Remote Env Tunnel Setup (Fusion → Ubuntu VM → Cluster)

This guide walks through running ARPO training on the GPU cluster while the OSWorld environment runs inside an **Ubuntu VM** in VMware Fusion, with Docker (and `/dev/kvm`) inside that VM. The cluster reaches the env server via an SSH reverse tunnel.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Your Mac (VMware Fusion)                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │  Ubuntu VM (nested virtualization enabled)                           │ │
│  │  • Docker + OSWorld containers (/dev/kvm in guest)                    │ │
│  │  • Env server: http://127.0.0.1:18082  (PROVIDER=docker)            │ │
│  │  • Reverse tunnel: -R 15001:127.0.0.1:18082 → cluster                 │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ssh -R 15001:127.0.0.1:18082 kevinzyz@deepx-a100-40g-2 -N
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  GPU Cluster (e.g. deepx-a100-40g-2)                                    │
│  • Training sees env at http://127.0.0.1:15001 (tunnel to VM’s 18082)   │
│  • Run: config=configs/smoke_remote_env_tunnel.yaml                      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Part 1: VMware Fusion – Ubuntu VM with Nested Virtualization

### 1.1 Create Ubuntu VM

1. Open **VMware Fusion**.
2. **File → New** (or **+**).
3. Choose **Install from disc or image** and select an **Ubuntu Server** or **Ubuntu Desktop** ISO (e.g. 22.04 LTS).
4. Finish the wizard and install Ubuntu (defaults are fine).
5. Install **VMware Tools** / **Open VM Tools** inside the VM if prompted:
   ```bash
   sudo apt update && sudo apt install -y open-vm-tools open-vm-tools-desktop
   ```

### 1.2 Enable “Hypervisor applications” (nested virtualization)

1. In Fusion, select the **Ubuntu VM** → **Settings** (or right‑click → Settings).
2. Go to **Processors & Memory**.
3. Enable **“Run with hypervisor application in this virtual machine”** (or equivalent “Hypervisor applications” / nested virtualization option).
4. Allocate at least **4 GB RAM** and **2+ cores**.
5. Click **Apply** / **OK**.

This allows the VM to run Docker with KVM so OSWorld containers see `/dev/kvm`.

---

## Part 2: Inside the Ubuntu VM – Docker, Repo, Env Server

Run all of the following **inside the Ubuntu VM** (SSH or Fusion console).

### 2.1 Install Docker, clone repo, venv (optional one-shot script)

From the repo root on your Mac, you can copy the setup script into the VM and run it there:

```bash
# On the VM (after copying or cloning the repo):
chmod +x scripts/setup_ubuntu_vm_for_remote_env.sh
./scripts/setup_ubuntu_vm_for_remote_env.sh
```

Or do the steps manually below.

### 2.2 Install Docker (if not using the script)

```bash
sudo apt update
sudo apt install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER
newgrp docker   # or log out and back in
docker run hello-world
```

### 2.3 Clone repo (with submodules)

Use the repo that contains the **remote env server** (e.g. `scripts/remote_env_server.py`). Example:

```bash
cd ~
git clone --recurse-submodules https://github.com/gowathena/arpo_replica.git
cd arpo_replica
git submodule update --init --recursive
```

If your repo or branch names differ, adjust the URL and directory name (e.g. `arpo_remote_env`). The repo must contain the remote env server (e.g. `scripts/remote_env_server.py`).

### 2.4 Create venv and install dependencies

```bash
cd ~/arpo_replica   # or your repo path
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cd OSWorld && pip install -r requirements.txt && pip install -e . && cd ..
pip install uvicorn   # for the env server
```

### 2.5 Run the env server with PROVIDER=docker

Start the server so it uses Docker inside the VM (OSWorld containers will see `/dev/kvm`):

```bash
cd ~/arpo_replica
source .venv/bin/activate
PROVIDER=docker python -m uvicorn scripts.remote_env_server:app --host 127.0.0.1 --port 18082
```

Leave this terminal running. You should see something like:

- `Startup: PROVIDER=docker, ...`
- `Uvicorn running on http://127.0.0.1:18082`

---

## Part 3: Reverse tunnel (inside the same Ubuntu VM)

In a **second terminal** in the Ubuntu VM, start the reverse tunnel so the cluster can reach the env server:

```bash
ssh -R 15001:127.0.0.1:18082 kevinzyz@deepx-a100-40g-2 -N
```

- Replace `kevinzyz` and `deepx-a100-40g-2` with your cluster user and host (or use the `deepx` host from your `~/.ssh/config` if it points to the same machine).
- `-R 15001:127.0.0.1:18082`: on the cluster, port **15001** will forward to the VM’s **18082**.
- `-N`: no remote command (just keep the tunnel open).

Keep this session open. On the cluster, the env server will be available at **http://127.0.0.1:15001**.

---

## Part 4: On the cluster – run training

On the **GPU cluster** (e.g. SSH to `deepx` or `deepx-a100-40g-2`):

1. Ensure the repo is cloned and the environment is set up (as in [REMOTE_GPU_SETUP.md](../REMOTE_GPU_SETUP.md)).
2. Ensure the **reverse tunnel is running** from the VM (Part 3).
3. Start Ray with a resource so the single env worker can be scheduled on the head node (where the tunnel is):
   ```bash
   ray start --head --port=2468 --resources='{"docker:127.0.0.1": 1}'
   ```
4. Run training with the smoke remote-env tunnel config:
   ```bash
   cd ~/arpo_replica   # or your clone path
   conda activate arpo
   python -m verl.trainer.main config=configs/smoke_remote_env_tunnel.yaml
   ```

The config sets `env.remote_env_url: "http://127.0.0.1:15001"`. The current trainer schedules env workers by `docker:IP` and each worker uses a local `DesktopEnv`. To use the tunnel, the trainer/worker code must be extended to use `config.env.remote_env_url` when set (e.g. an HTTP client to the remote env server instead of starting a local Docker env). Until then, you can run with the same config and Ray resource for a single worker on the head node and implement the remote-env client in `EnvWorker` to call `http://127.0.0.1:15001`.

---

## Quick reference

| Where        | What |
|-------------|------|
| **Fusion**  | Ubuntu VM; Settings → Processors & Memory → enable “Hypervisor applications”. |
| **Ubuntu VM** | Install Docker, clone repo + submodules, venv, install deps, run `PROVIDER=docker python -m uvicorn scripts.remote_env_server:app --host 127.0.0.1 --port 18082`. |
| **Ubuntu VM** (2nd terminal) | `ssh -R 15001:127.0.0.1:18082 kevinzyz@deepx-a100-40g-2 -N`. |
| **Cluster**  | `python -m verl.trainer.main config=configs/smoke_remote_env_tunnel.yaml`. |

---

## Troubleshooting

- **Tunnel “Connection refused” on cluster**: Ensure the env server is running in the VM on port 18082 and the tunnel command is running.
- **Docker in VM**: After enabling nested virtualization, reboot the VM if Docker still doesn’t see KVM.
- **Cluster host**: If you use an SSH alias (e.g. `deepx`), run:  
  `ssh -R 15001:127.0.0.1:18082 deepx -N`  
  (replace `deepx` with your alias).
