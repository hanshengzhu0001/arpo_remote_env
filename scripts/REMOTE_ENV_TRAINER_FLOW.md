# Remote env: trainer flow and server logs

## Training-side checklist (AWS EC2 remote server)

- **Config key:** Use **`env.remote_server_url`** (not `remote_env_url`). Example: `remote_server_url: "http://100.48.93.208:15001"`.
- **Config file:** e.g. `configs/smoke_remote_env.yaml`; run with `python -m verl.trainer.main config=configs/smoke_remote_env.yaml`.
- **Connectivity:** EC2 security group must allow **inbound TCP 15001** from the training cluster. Verify from the cluster: `curl http://<EC2_IP>:15001/health` or `./scripts/verify_remote_env_connection.sh`.
- **Server:** On EC2, start with `./scripts/start_remote_env_aws.sh` (or run uvicorn with `PROVIDER=aws` and required AWS env vars). See `docs/REMOTE_ENV_SERVER_RUNBOOK.md`. The server loads `.env` at startup so `OPENAI_API_KEY` (and AWS vars) are available for tasks; the start script in [arpo_remote_env](https://github.com/hanshengzhu0001/arpo_remote_env) also sources `.env` before starting.

---

## Trainer order (reset → step → evaluate)

The trainer **does** perform reset, then steps, then evaluate. Exact flow in `verl/trainer/ray_trainer.py`:

1. **Reset**  
   - `start_reset_envs(batch_dict)` (line ~843) builds Ray futures: `worker.reset.remote(task_config)` per env.  
   - `reset_outputs = ray.get(reset_envs_object)` (line ~869) waits for all resets.  
   - So every episode starts with a **reset** on each env.

2. **Step loop**  
   - `for step_idx in range(self.config.env.max_steps)` (line ~875):  
     - `prepare_vllm_inputs_full(env_outputs)` → build VLM batch from current obs.  
     - `generate_sequences` → model predicts actions.  
     - `worker.step.remote(action_text)` → env step; `env_outputs = ray.get(futures)`.  
   - When a step returns `is_done=True` for an env, that env’s **evaluate** is fired (async):  
     - `eval_results_objects[cur_env_idx] = self.env_workers[cur_env_idx].evaluate.remote()` (line ~946).  
   - Loop breaks when `is_all_done` (all envs reported done).

3. **Evaluate**  
   - After the step loop: `eval_results = ray.get(eval_results_objects)` (line ~968).  
   - So **evaluate is only called for envs that have reported `is_done=True`** at least once (typically after one or more steps). If reset fails and we break with `batch_skipped`, we never call evaluate and use `eval_results = [0.0] * len(task_configs)`.

So the order is: **reset → step (0..max_steps) → evaluate** (only for envs that are done). Evaluate is not called before any step; it is called after the env has been stepped at least until `is_done` is True.

---

## What to look for in remote env server logs

Run the server with stdout visible (e.g. `python scripts/remote_env_server.py` or via your process manager). During a full training run you should see:

| Event | Log line(s) |
|-------|-------------|
| **Reset requested** | (No single “reset received” line; first evidence is success/failure below.) |
| **Reset OK** | `Reset OK: VM screenshot obtained (... bytes), returning obs_messages with image. Instruction: ...` |
| **Reset failed** | `Env reset exception: ...` or `Reset: screenshot is None (VM/container not ready...). Returning obs_messages=None.` |
| **Step** | `step_parse: ... actions=[...] format_reward=... pred_preview=...` (once per step) |
| **Evaluate requested** | `POST /env/evaluate received (instruction=..., step_counter=...)` |
| **Evaluate OK** | `Evaluation completed: score=..., instruction=..., step_counter=...` |
| **Evaluate skipped (503)** | `Evaluation skipped: env not fully started (no setup_controller)...` or `...env has no setup_controller (reset failed)...` |
| **Evaluate error (0.0)** | `Evaluation error: ...` and traceback |

If you see **no** `POST /env/evaluate received` during a run, the cluster is not reaching `/env/evaluate` (e.g. connection/timeout or trainer never got `is_done=True`). If you see that line but then `Evaluation skipped` (503) or `Evaluation error`, the server is returning 503 or 0.0 and the client will record eval 0.

---

## Quick checks

- **Cluster → server:** From a cluster node: `curl -s http://54.89.232.89:15001/health` and `python scripts/test_evaluate_endpoint.py`.  
- **Server logs:** Tail the process that runs `remote_env_server.py` while training; confirm `Reset OK`, `step_parse`, and `POST /env/evaluate received` / `Evaluation completed` (or the 503/error lines above).
