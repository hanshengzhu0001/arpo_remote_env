# Remote env server – runbook

## SSH: use `ssh` not `sh`

To connect to the server, use **ssh** (not sh). Use the current EC2 IP (e.g. **100.48.93.208** if your env’s IP changed):

```bash
ssh -i /Users/hanszhu/Desktop/ARPO_replicate/osworld-key.pem ubuntu@100.48.93.208
```

If you type `sh -i ...` the shell will try to run your `.pem` file as a script and you’ll see errors like `-----BEGIN: command not found`.

---

## If the cluster sees "Connection refused" to the remote env (port 15001)

The training cluster talks to the remote env server at `http://<EC2_IP>:15001`. If you see:

- `Connection refused` or `Failed to establish a new connection: [Errno 111] Connection refused`

then the cluster cannot reach the server. Do the following.

### 1. Ensure the server process is running on the EC2 box

On the **EC2 instance** (SSH in first):

```bash
# Check if something is listening on 15001
sudo lsof -i :15001
# or
ss -tlnp | grep 15001
```

If nothing is listening, start the server and keep it running (e.g. in a `tmux` or `screen` session so it survives SSH disconnect):

```bash
cd ~/arpo_remote_env
source arpo_env/bin/activate

# Run in foreground (use tmux/screen so it stays up after you disconnect)
# Use 3600 (1 hr) or 7200 (2 hr) when VM runs without KVM so boot can finish
DOCKER_VM_READY_TIMEOUT=3600 python -m uvicorn scripts.remote_env_server:app --host 0.0.0.0 --port 15001
```

Or in background (log in `~/arpo_remote_env/uvicorn.log`):

```bash
cd ~/arpo_remote_env && source arpo_env/bin/activate
nohup env DOCKER_VM_READY_TIMEOUT=3600 python -m uvicorn scripts.remote_env_server:app --host 0.0.0.0 --port 15001 >> uvicorn.log 2>&1 &
```

Then verify from the same machine:

```bash
curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:15001/docs
# should print 200
```

### 2. Open port 15001 in the AWS Security Group

The EC2 instance’s **Security Group** must allow **inbound TCP port 15001** from the cluster.

- In AWS Console: EC2 → Instances → select the instance → Security tab → Security group → Edit inbound rules.
- Add rule: Type **Custom TCP**, Port **15001**, Source either:
  - The cluster’s IP or CIDR (e.g. `10.100.4.0/24` if workers are in that range), or
  - **0.0.0.0/0** for testing (anywhere).

If the cluster is in the same VPC and uses private IPs, the source should be the cluster subnet (e.g. `10.100.4.0/24`). If the cluster is on the internet, use its outbound IP or 0.0.0.0/0.

### 3. Use the correct IP in training config

Training must use the **IP that the cluster can reach**:

- If the cluster reaches EC2 over the **public internet**, use the instance’s **public IP** (e.g. 100.48.93.208 if that is the public IP).
- If the cluster is in the **same VPC** and you use **private IPs**, set the env URL to the instance’s **private IP** and ensure the security group allows 15001 from the cluster subnet.

---

## If you see "VM failed to become ready within timeout period" or 503 on /env/evaluate

The VM inside Docker on EC2 runs under **QEMU software emulation** (no KVM). Boot can take **20–40+ minutes** on a small instance. The server considers the VM ready only when **both** the screenshot endpoint and Chrome CDP (port 9222) respond.

### 1. Use a long timeout when starting the server (recommended: 1–2 hours)

On the **Ubuntu server**, stop the server (Ctrl+C), then start with **3600** (1 hour) or **7200** (2 hours). Defaults (1200s / 1800s) are often too short without KVM:

```bash
cd ~/arpo_remote_env
source arpo_env/bin/activate

# 1 hour – recommended without KVM
DOCKER_VM_READY_TIMEOUT=3600 python -m uvicorn scripts.remote_env_server:app --host 0.0.0.0 --port 15001

# Or 2 hours if the VM is very slow
DOCKER_VM_READY_TIMEOUT=7200 python -m uvicorn scripts.remote_env_server:app --host 0.0.0.0 --port 15001
```

Server logs will show: `Waiting for VM to be ready... Xs / Ys (screenshot=..., chrome_cdp=...)`. Both must become `True` before the first `/env/reset` succeeds; then `/env/evaluate` will stop returning 503.

### 2. Ensure only one reset runs at a time

The server has **one** env. If the training client sends many **concurrent** resets, each can start a new container and the previous one is torn down, so the VM never finishes booting. Prefer:

- **One** RemoteEnvWorker (or one client) talking to this server, or
- Client-side backoff when you get 503/500 so the same env isn’t reset again immediately.

### 3. Pre-warm the env before starting training (optional but helpful)

Before launching the training run:

1. Start uvicorn with `DOCKER_VM_READY_TIMEOUT=3600` (or 7200).
2. Trigger **one** `/env/reset` (e.g. run a minimal script that POSTs to `http://<EC2_IP>:15001/env/reset` or start training with a single step).
3. Wait until the server logs show the VM is ready (no timeout, or first evaluate returns 200/503 with “env not fully started” then later 200).
4. Then run the full training so the first `/env/evaluate` hits an already-ready env.

### 4. Check that the container and VM are booting

In another terminal on the same server:

```bash
docker ps
docker logs -f <container_id>
```

Confirm the container stays up and the VM is progressing (UEFI → GRUB → kernel). The server waits for both **screenshot** (port 5000) and **Chrome CDP** (port 9222) before marking ready.

### 5. Optional: larger instance

A larger EC2 instance (more vCPUs) can make QEMU boot faster.

---

---

## Using native EC2 instead of Docker+QEMU (no software emulation)

To avoid QEMU and use **native** t3 instances, run the remote env server with the **AWS provider**. The server launches a **separate** EC2 instance (OSWorld AMI) as the desktop; that instance boots in ~1–2 min.

### 1. Run uvicorn on an EC2 in your target VPC

Run the server on an EC2 instance (e.g. your t3) in the VPC where you want desktop instances. **Same VPC** is required so the server can reach the desktop’s private IP (ports 5000, 9222, 8006, 8080).

### 2. AWS credentials

The server must be able to call EC2. Either:

- **IAM role** (recommended): Attach an IAM role to the EC2 instance that runs uvicorn with permissions: `ec2:RunInstances`, `ec2:DescribeInstances`, `ec2:StartInstances`, `ec2:StopInstances`, and (if you use the registry) `ec2:CreateTags`, `ec2:DescribeTags`. No `aws configure` needed.
- Or run **`aws configure`** on that host and set Access Key / Secret for a user with the same permissions.

If credentials are missing or wrong, the server returns 503 with a clear message on first use.

### 3. Subnet and security group (same VPC)

- **Automatic**: If the server runs **on EC2** in the target VPC, the code **auto-detects** this instance’s subnet and security group and uses them for the desktop instance. No env vars needed for networking if your inbounds (5000, 9222, 8006, 8080) are already allowed for that security group.
With the **xlang-ai/OSWorld** submodule, **`AWS_SUBNET_ID` and `AWS_SECURITY_GROUP_ID` are required** (no auto-detect). Set them to the subnet and security group where the server runs so desktop instances launch in the same VPC.

The desktop instance’s security group must allow the ports listed in `OSWorld/desktop_env/providers/aws/AWS_GUIDELINE.md` (SSH 22, 5000, 5910, 8006, 8080, 8081, 9222, etc.).

### 4. OSWorld AMI (image for the desktop instance)

With the **xlang-ai/OSWorld** submodule, the desktop is launched from the **official OSWorld Ubuntu AMI** in `IMAGE_ID_MAP` (e.g. `us-east-1` → `ami-0d23263edb96951d8`). No override needed if that AMI is available in your account and region. See `OSWorld/desktop_env/providers/aws/AWS_GUIDELINE.md` for security group ports and full AWS setup.

### 5. Start the server

**Required** for AWS provider (xlang-ai/OSWorld): `AWS_REGION`, `AWS_SUBNET_ID`, `AWS_SECURITY_GROUP_ID`.

**Option A – On EC2 (recommended):** use the start script; it auto-detects subnet and security group from the instance:

```bash
cd ~/arpo_remote_env && source arpo_env/bin/activate
./scripts/start_remote_env_aws.sh
```

**Option B – Set by hand or via .env:** copy `.env.example` to `.env`, fill in your subnet and security group IDs, then:

```bash
cd ~/arpo_remote_env && source arpo_env/bin/activate
export AWS_REGION=us-east-1
export AWS_SUBNET_ID=subnet-xxxxxxxxxxxxxxxxx
export AWS_SECURITY_GROUP_ID=sg-xxxxxxxxxxxxxxxxx
PROVIDER=aws python -m uvicorn scripts.remote_env_server:app --host 0.0.0.0 --port 15001
```

Optional: `DEFAULT_TTL_MINUTES`, `ENABLE_TTL`, `AWS_SCHEDULER_ROLE_ARN` (see `OSWorld/desktop_env/providers/aws/config.py`). Default instance type in upstream is `t3.xlarge`.

No `DOCKER_VM_READY_TIMEOUT` needed; the desktop is a real EC2 and boots in ~1–2 min.

### 6. Summary

| | Docker provider | AWS provider |
|---|------------------|--------------|
| Desktop | QEMU + Ubuntu VM inside container | Native EC2 (OSWorld AMI) |
| Boot time | 20–40+ min (no KVM) | ~1–2 min |
| Set | `PROVIDER=docker` (default), `DOCKER_VM_READY_TIMEOUT=3600` | `PROVIDER=aws`, **`AWS_REGION`**, **`AWS_SUBNET_ID`**, **`AWS_SECURITY_GROUP_ID`** (required) |
| Needs | Docker, .qcow2 image | AWS credentials (or IAM role), OSWorld AMI; VPC subnet + SG with 5000/9222/8006/8080/5910 etc. (see AWS_GUIDELINE.md) |

---

**Git (if you see divergent branches):**  
On the server, to pull and rebase:  
`git pull --rebase origin arpo-cpu-replicate`  
(or merge: `git pull --no-rebase`).
