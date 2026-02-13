# Summary: Updates After New AWS Configuration

Updates made to use the **xlang-ai/OSWorld** submodule and run the remote env server on EC2 with the AWS provider.

---

## 1. OSWorld submodule

- **Switched** from `FanbinLu/OSWorld` to **`xlang-ai/OSWorld`** (upstream).
- Submodule now tracks upstream `main` (commit `e695a10`).
- **Official AMI** in code: `us-east-1` → **`ami-0d23263edb96951d8`** (`osworld_client_image_30G_0719`); confirmed visible in your account.

## 2. AWS provider requirements

- **Required env vars** (no auto-detect in upstream code): `AWS_REGION`, `AWS_SUBNET_ID`, `AWS_SECURITY_GROUP_ID`.
- **Optional:** `.env` in server directory; `DEFAULT_TTL_MINUTES`, `ENABLE_TTL`, `AWS_SCHEDULER_ROLE_ARN` (see `OSWorld/desktop_env/providers/aws/config.py`).
- **Instance type** in upstream: `t3.xlarge` (default).
- **Security group:** ports per `OSWorld/desktop_env/providers/aws/AWS_GUIDELINE.md` (22, 5000, 5910, 8006, 8080, 8081, 9222).

## 3. Config and scripts added

- **`.env.example`** – Template for `AWS_REGION`, `AWS_SUBNET_ID`, `AWS_SECURITY_GROUP_ID`.
- **`.env`** – Contains `AWS_REGION=us-east-1` only; subnet/SG left unset so the start script can auto-detect on EC2.
- **`scripts/start_remote_env_aws.sh`** – Start script that:
  - Loads `.env` if present.
  - Defaults `AWS_REGION=us-east-1`.
  - **On EC2:** auto-detects `AWS_SUBNET_ID` and `AWS_SECURITY_GROUP_ID` from instance metadata (IMDSv2 and IMDSv1).
  - Runs `PROVIDER=aws python -m uvicorn scripts.remote_env_server:app --host 0.0.0.0 --port 15001`.

## 4. Runbook changes

- **`docs/REMOTE_ENV_SERVER_RUNBOOK.md`** updated for xlang-ai/OSWorld:
  - **Option A:** On EC2, run `./scripts/start_remote_env_aws.sh` (auto-detects subnet/SG).
  - **Option B:** Set vars in `.env` or env and run uvicorn manually.
  - Documented required env vars and pointer to `AWS_GUIDELINE.md` for ports.

## 5. Deployment

- **Rsync** from Mac to EC2 (`ubuntu@100.48.93.208:~/arpo_remote_env/`) with exclusions (e.g. `.git`, `docker_vm_data`, `evaluation_examples`).
- `.env` and `scripts/start_remote_env_aws.sh` deployed and verified on server.

## 6. Current run flow

1. **EC2 server:** `cd ~/arpo_remote_env && source arpo_env/bin/activate && ./scripts/start_remote_env_aws.sh`
2. Script auto-detects **subnet** and **security group** from instance metadata and starts the server on **15001**.
3. **Training** uses a config with `env.remote_env_url: "http://<EC2_IP>:15001"` (e.g. `http://100.48.93.208:15001`).
4. **Security group** for the EC2 instance must allow **inbound TCP 15001** from the training client.

---

## Removed / superseded

- **REMOTE_ENV_TUNNEL_SETUP.md** – Tunnel (Fusion VM → cluster) flow; superseded by runbook for direct EC2 + AWS provider.
- **START_HERE.md** – Mac/VMware-centric quick start; README + runbook are the main entry points.
- **FILES.md** – Outdated file list; README and runbook cover current layout and usage.
