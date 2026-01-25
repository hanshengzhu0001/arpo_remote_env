# ARPO Training with Colab GPU Server

## Overview

Train UI-TARS-2B using Colab GPU for inference + Mac for OSWorld environments.

## Architecture

```
Colab GPU                          Mac
┌────────────────────┐           ┌──────────────────┐
│ UI-TARS-2B Server  │◄──────────│ VERL Trainer     │
│ Flask + ngrok      │───────────►│ 4-8 OSWorld VMs  │
│ ~10-30 sec/step    │           │ Experience Replay│
└────────────────────┘           └──────────────────┘
  T4 or A100 GPU                   128 tasks
```

## Setup (One-Time)

### 1. Start Colab GPU Server

Use `notebooks/GPU_Server_for_OSWorld.ipynb` but change model to UI-TARS-2B:

**In Cell 3**, change:
```python
MODEL = "ByteDance-Seed/UI-TARS-2B-SFT"  # Instead of 7B
```

Run cells 1-5 on Colab, get ngrok URL.

### 2. Update Mac Configuration

```bash
cd ~/Desktop/ARPO_replicate

# Update agent
COLAB_URL="https://your-ngrok-url.ngrok-free.dev"
sed -i '' "s|http://localhost:9000/v1|${COLAB_URL}/v1|g" OSWorld/mm_agents/uitars_agent.py
```

### 3. Update Training Config

Edit `configs/config_uitars_2b_mac.yaml`:

```yaml
data:
  train_files: test_data/osworld_examples/train_all_128.json  # 128 tasks

worker:
  rollout:
    use_external_server: true
    server_url: "https://your-ngrok-url.ngrok-free.dev/v1"

env:
  num_envs: 4  # 4 VMware VMs (adjust based on resources)

trainer:
  total_episodes: 10  # 10 epochs
```

## Running Training

```bash
cd ~/Desktop/ARPO_replicate

# Make sure Colab server is running!
curl https://your-ngrok-url.ngrok-free.dev/health

# Start training
bash scripts/train_uitars_2b_arpo.sh
```

## Expected Performance

| Metric | Value |
|--------|-------|
| **Tasks** | 128 |
| **Environments** | 4 VMs |
| **Rollouts/task** | 2 |
| **Epochs** | 10 |
| **Total steps** | ~10,240 |
| **Inference/step** | 10-30 sec (GPU) |
| **Time/epoch** | ~10-20 hours |
| **Total time** | **~100-200 hours** |

## Monitoring

```bash
# Watch training logs
tail -f logs/training.log

# Check results
ls -la results_training_2b/

# Check checkpoints
ls -la checkpoints_training_2b/
```

## Tips

1. **Colab free tier**: Disconnects after ~12 hours
   - Save training state before disconnect
   - Resume with checkpoint

2. **Multiple VMs**: Adjust `num_envs` based on Mac resources
   - 16GB RAM: 2 VMs
   - 32GB RAM: 4-8 VMs

3. **Monitor GPU**: Keep Colab tab open, check for errors

4. **Network**: Stable internet needed for Colab ↔ Mac

## Troubleshooting

### "ngrok session expired"
- Free ngrok: 2-hour limit
- Re-run Colab Cell 5, update Mac config with new URL

### "Out of memory" on Mac
- Reduce `num_envs` to 2
- Close other applications

### Colab disconnects
- Upgrade to Colab Pro ($10/month) for longer sessions
- Or checkpoint frequently and resume

---

**For full setup**: See `arpo_training_notebook.ipynb` cells 30-38
