# Remote OSWorld env server (2-machine setup)

**Architecture:**
- **Resource (GPUs, model, rollout):** current cluster.
- **Executions (reset / step / evaluate):** remote env server on Mac, using **VMware** (default on macOS).

The cluster runs ARPO training and sends env requests (reset, step, evaluate) to the Mac over HTTP. On the Mac, the server uses VMware Fusion by default (no Docker/KVM required).

## 1. On Mac (env server – VMware)

Install VMware Fusion and ensure `vmrun` is on PATH. The server defaults to VMware on macOS.

```bash
cd ~/arpo_remote_env   # or your clone
source arpo_env/bin/activate
python -m uvicorn scripts.remote_env_server:app --host 127.0.0.1 --port 18082
```

Server listens on `127.0.0.1:18082`. Keep this terminal open. You should see `Startup: PROVIDER=vmware`.

## 2. On cluster (training – resource)

**Option A – Cluster can reach your Mac (same VPN/network)**  
- Set `env.remote_server_url` in `configs/smoke_remote_env.yaml` to your Mac URL, e.g. `http://YOUR_MAC_IP:18082`.  
- From cluster: `curl http://YOUR_MAC_IP:18082/health` then run:
  `python -m verl.trainer.main config=configs/smoke_remote_env.yaml`

**Option B – Cluster cannot reach your Mac (SSH reverse tunnel)**  
1. On Mac (env server already running on 18082): open a second terminal and run:
   `ssh -R 15001:127.0.0.1:18082 kevinzyz@<CLUSTER_HOST>`  
   Leave this SSH session open.  
2. On cluster: `curl http://127.0.0.1:15001/health` then:
   `python -m verl.trainer.main config=configs/smoke_remote_env_tunnel.yaml`

Tunnel maps cluster `localhost:15001` → Mac `localhost:18082`. Training config `smoke_remote_env_tunnel.yaml` already uses `http://127.0.0.1:15001`.

## API (for debugging)

- `POST /env/reset`  Body: `{"task_config": {...}}`  → `{env_idx, obs_messages, is_done, format_reward}`
- `POST /env/step`   Body: `{"prediction": "..."}`   → same shape
- `POST /env/evaluate` Body: `{}` → float
- `POST /env/history_messages` Body: `{}` → `{history_messages: [...]}`
- `GET /health` → `{status: "ok", provider: "vmware", ...}`

Images in JSON use a `b64` field (raw base64); the cluster adds the `data:image/jpeg;base64,` prefix when building messages.
