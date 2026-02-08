# Remote OSWorld env server (2-machine setup)

Run the **env** (OSWorld + VM) on your Mac (or AWS CPU) and **training** on the GPU cluster. The cluster talks to the env server over HTTP.

## 1. On Mac (env server)

```bash
cd /path/to/hansenzuishuai
# Install deps if needed: pip install fastapi uvicorn requests (and OSWorld/desktop_env deps)
python scripts/remote_env_server.py
```

Server listens on `0.0.0.0:5001`. Keep this terminal open.

## 2. On cluster (training)

**Option A – Cluster can reach your Mac (same VPN/network)**  
- Set `env.remote_server_url` in `configs/smoke_remote_env.yaml` to your Mac IP, e.g. `http://10.103.75.204:5001`.
- From cluster: `curl http://YOUR_MAC_IP:5001/health` then run `python -m verl.trainer.main config=configs/smoke_remote_env.yaml`

**Option B – Cluster cannot reach your Mac (SSH reverse tunnel)**  
1. On your Mac (env server already running): `ssh -R 5001:localhost:5001 kevinzyz@<CLUSTER_IP_or_HOSTNAME>` — leave this SSH session open.  
2. On cluster: `curl http://127.0.0.1:5001/health` then `python -m verl.trainer.main config=configs/smoke_remote_env_tunnel.yaml`  
The tunnel uses outbound SSH from the Mac; the cluster connects to `127.0.0.1:5001` on the cluster node.

## API (for debugging)

- `POST /env/reset`  Body: `{"task_config": {...}}`  → `{env_idx, obs_messages, is_done, format_reward}`
- `POST /env/step`   Body: `{"prediction": "..."}`   → same shape
- `POST /env/evaluate` Body: `{}` → float
- `POST /env/history_messages` Body: `{}` → `{history_messages: [...]}`
- `GET /health` → `{status: "ok"}`

Images in JSON use a `b64` field (raw base64); the cluster adds the `data:image/jpeg;base64,` prefix when building messages.
