# Remote OSWorld env server (2-machine setup)

Run the **env** (OSWorld + VM) on your Mac (or AWS CPU) and **training** on the GPU cluster. The cluster talks to the env server over HTTP.

## 1. On Mac (env server)

```bash
cd /path/to/hansenzuishuai
# Install deps if needed: pip install fastapi uvicorn requests (and OSWorld/desktop_env deps)
python scripts/remote_env_server.py
```

Server listens on `0.0.0.0:5000`. Ensure the VM image and Docker are set up (same as local Docker provider); the server uses one `DesktopEnv` (Docker + Ubuntu VM).

## 2. On cluster (training)

- Copy `configs/smoke_remote_env.yaml` and set `env.remote_server_url` to your Mac’s URL, e.g. `http://YOUR_MAC_IP:5000`.
- Cluster must be able to reach that URL (same VPN, or port forward, or Mac’s LAN IP if cluster is on same network).
- Use `num_envs: 1` (one server = one env).
- Run training as usual with the modified config.

## API (for debugging)

- `POST /env/reset`  Body: `{"task_config": {...}}`  → `{env_idx, obs_messages, is_done, format_reward}`
- `POST /env/step`   Body: `{"prediction": "..."}`   → same shape
- `POST /env/evaluate` Body: `{}` → float
- `POST /env/history_messages` Body: `{}` → `{history_messages: [...]}`
- `GET /health` → `{status: "ok"}`

Images in JSON use a `b64` field (raw base64); the cluster adds the `data:image/jpeg;base64,` prefix when building messages.
