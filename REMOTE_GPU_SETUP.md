# Remote GPU Cluster Setup via GCP Bastion

Complete guide for running ARPO training on remote GPU cluster using VSCode Remote SSH with ProxyJump.

---

## Network Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Connection Flow                           │
│                                                              │
│  Your Mac (VSCode)                                          │
│  └─> SSH                                                    │
│      └─> GCP VM Bastion (34.9.43.1 whitelisted)           │
│          └─> ProxyJump                                      │
│              └─> GPU Cluster (172.174.34.71)               │
│                  └─> Run ARPO Training                      │
│                                                              │
│  Firewall: Cluster only accepts 34.9.43.1                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Step 1: Local Prerequisites (Mac)

```bash
# Install tools
brew install python@3.13
brew install google-cloud-sdk

# Authenticate to GCP
gcloud init
gcloud auth login
gcloud config set project YOUR_PROJECT_ID  # Your GCP project

# Verify
gcloud config list
```

---

## Step 2: Create/Verify GCP Bastion VM

### Check Existing VM

```bash
# List your VMs
gcloud compute instances list

# Check the VM that appears in your ssh config
# Example: instance-20260128-042206.us-central1-f.gen-lang-client-0387779402
```

### Verify External IP

```bash
# Get the VM's external IP
gcloud compute instances describe INSTANCE_NAME \
    --zone=us-central1-f \
    --format='get(networkInterfaces[0].accessConfigs[0].natIP)'

# Should show: 34.9.43.1 (or the whitelisted IP)
```

### If IP Doesn't Match (Optional)

```bash
# Reserve static IP
gcloud compute addresses create bastion-ip --region=us-central1

# Get the IP
gcloud compute addresses describe bastion-ip --region=us-central1

# Assign to VM
gcloud compute instances delete-access-config INSTANCE_NAME \
    --access-config-name="external-nat" --zone=us-central1-f

gcloud compute instances add-access-config INSTANCE_NAME \
    --access-config-name="external-nat" \
    --address=34.9.43.1 \
    --zone=us-central1-f
```

---

## Step 3: Generate GCP SSH Config

```bash
# This auto-generates ~/.ssh/config entry for the GCP VM
gcloud compute config-ssh

# Output:
# Updated [~/.ssh/config].
# You should now be able to SSH into your GCP VM...
```

**Check ~/.ssh/config**:
```bash
cat ~/.ssh/config | grep -A 5 "instance-"
```

You should see:
```
Host instance-20260128-042206.us-central1-f.gen-lang-client-0387779402
  HostName 34.9.43.1
  IdentityFile ~/.ssh/google_compute_engine
  IdentitiesOnly yes
```

### Test Connection

```bash
ssh instance-20260128-042206.us-central1-f.gen-lang-client-0387779402

# Should connect to GCP VM
# Exit with: exit
```

---

## Step 4: Add GPU Cluster with ProxyJump

Edit `~/.ssh/config` and add:

```bash
# GPU Cluster (accessed via GCP bastion)
Host deepx
  HostName 172.174.34.71
  User kevinzyz
  IdentityFile ~/.ssh/kevinzyz_id_ed25519
  IdentitiesOnly yes
  ProxyJump instance-20260128-042206.us-central1-f.gen-lang-client-0387779402
  ServerAliveInterval 60
  ServerAliveCountMax 10
```

**Important**: Replace with your actual:
- Instance name (from `gcloud compute config-ssh` output)
- GPU cluster IP (172.174.34.71 or your actual IP)
- Username (kevinzyz or yours)
- Private key path

### Test ProxyJump Connection

```bash
ssh deepx

# Should:
# 1. SSH to GCP VM (34.9.43.1)
# 2. ProxyJump to GPU cluster (172.174.34.71)
# 3. Land on GPU server

# Verify you're on the right machine:
hostname
nvidia-smi  # Should show GPUs
```

---

## Step 5: VSCode Remote SSH

### Install Extension

1. Open VSCode
2. Extensions → Search "Remote - SSH"
3. Install: **Remote - SSH** by Microsoft

### Connect to GPU Cluster

1. **Cmd+Shift+P** → **Remote-SSH: Connect to Host...**
2. Select: **deepx** (your GPU cluster)
3. VSCode will:
   - SSH through GCP bastion (transparent)
   - Connect to GPU cluster
   - Open remote workspace

### Open ARPO Repository

1. **File → Open Folder**
2. Navigate to: `/home/kevinzyz/arpo_replica` (or wherever you cloned)
3. VSCode is now editing remotely!

---

## Step 6: Clone ARPO on GPU Cluster

**On the GPU server** (via SSH or VSCode terminal):

```bash
# Clone repository
git clone https://github.com/gowathena/arpo_replica.git
cd arpo_replica
git checkout arpo-cpu-replicate
git submodule update --init --recursive

# Create environment
conda create -n arpo python=3.10 -y
conda activate arpo

# Install dependencies
pip install -r requirements.txt
cd OSWorld && pip install -r requirements.txt && pip install -e . && cd ..

# Apply patches (Docker for cluster)
cp osworld_patches/run_uitars.py OSWorld/
cp osworld_patches/uitars_agent.py OSWorld/mm_agents/

# Update for Docker (not VMware)
sed -i 's/vmware/docker/g' OSWorld/run_uitars.py
```

---

## Step 7: Setup Docker on GPU Cluster

```bash
# Check if Docker is installed
docker --version

# If not installed:
sudo apt-get update
sudo apt-get install docker.io

# Add your user to docker group (no sudo needed)
sudo usermod -aG docker $USER
newgrp docker

# Verify
docker ps

# Pull OSWorld image
docker pull happysixd/osworld-docker:latest
```

---

## Step 8: Setup wandb

```bash
# On GPU cluster
conda activate arpo
pip install wandb

# Login
wandb login
# Paste API key when prompted
```

---

## Step 9: Run Training from VSCode

**In VSCode connected to deepx**:

1. Open terminal (Ctrl+`)
2. Navigate to repo:
   ```bash
   cd ~/arpo_replica
   conda activate arpo
   ```

3. **Start training**:
   ```bash
   # Set wandb key
   export WANDB_API_KEY="your-key"
   
   # Run smoke test
   python -m verl.trainer.main config=configs/smoke_test.yaml
   ```

4. **Monitor** in VSCode:
   - Terminal shows live output
   - Can edit code locally, runs remotely
   - wandb dashboard: https://wandb.ai/hanszhu05/arpo-smoke-test

---

## Complete ~/.ssh/config Example

```bash
# GCP Bastion (auto-generated by gcloud compute config-ssh)
Host instance-20260128-042206.us-central1-f.gen-lang-client-0387779402
  HostName 34.9.43.1
  IdentityFile ~/.ssh/google_compute_engine
  IdentitiesOnly yes
  User YOUR_GCP_USERNAME

# GPU Cluster (via ProxyJump)
Host deepx
  HostName 172.174.34.71
  User kevinzyz
  IdentityFile ~/.ssh/kevinzyz_id_ed25519
  IdentitiesOnly yes
  ProxyJump instance-20260128-042206.us-central1-f.gen-lang-client-0387779402
  ServerAliveInterval 60
  ServerAliveCountMax 10
  
  # Forward ports (optional - for Jupyter, tensorboard, etc.)
  LocalForward 8888 localhost:8888
  LocalForward 6006 localhost:6006
```

---

## Troubleshooting

### "Permission denied (publickey)"

**On GCP VM**:
```bash
# Generate key pair
ssh-keygen -t ed25519 -f ~/.ssh/kevinzyz_id_ed25519

# Copy public key to GPU cluster
ssh-copy-id -i ~/.ssh/kevinzyz_id_ed25519.pub kevinzyz@172.174.34.71
```

**On your Mac**:
```bash
# Copy private key from GCP VM to Mac
scp instance-...:~/.ssh/kevinzyz_id_ed25519 ~/.ssh/
chmod 600 ~/.ssh/kevinzyz_id_ed25519
```

### "Connection timeout"

- Verify GCP VM IP is whitelisted: `34.9.43.1`
- Check cluster firewall allows that IP
- Test: `ssh instance-...` then `ssh 172.174.34.71` manually

### "VSCode can't connect"

- Test SSH works first: `ssh deepx`
- Check VSCode Remote SSH settings
- Try: **Cmd+Shift+P** → **Remote-SSH: Kill VS Code Server on Host**

---

## Verification Checklist

- [ ] `gcloud` installed and authenticated
- [ ] GCP VM exists with external IP `34.9.43.1`
- [ ] `gcloud compute config-ssh` generated bastion entry
- [ ] Can SSH to GCP VM: `ssh instance-...`
- [ ] Added `deepx` host with ProxyJump
- [ ] Can SSH to cluster: `ssh deepx`
- [ ] `nvidia-smi` works on deepx
- [ ] Docker works on deepx: `docker ps`
- [ ] VSCode Remote SSH can connect to deepx
- [ ] Repository cloned on deepx
- [ ] Conda environment created on deepx

---

## Quick Test

```bash
# From your Mac
ssh deepx 'hostname && nvidia-smi && docker ps'

# Should show:
# - GPU cluster hostname
# - GPU info
# - Docker containers (if any)
```

If this works, you're ready to run ARPO training remotely via VSCode! 🚀

---

## Training Workflow

1. **Edit locally**: VSCode on Mac, connected to `deepx`
2. **Run remotely**: Terminal in VSCode executes on GPU cluster
3. **Monitor**: wandb dashboard + terminal output
4. **Results**: Saved on GPU cluster, sync with `scp` or mount

**All through one VSCode window!**
