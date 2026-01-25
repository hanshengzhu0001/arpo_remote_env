# Weights & Biases Integration

Track your ARPO training with wandb for comprehensive logging and visualization.

## Setup (One-Time)

### 1. Install wandb

```bash
conda activate arpo
pip install wandb
```

### 2. Login to wandb

```bash
wandb login
```

When prompted, paste your API key from: https://wandb.ai/authorize

**Or set environment variable**:
```bash
export WANDB_API_KEY="your-api-key-here"
```

### 3. Configure Your Entity

Edit `configs/wandb_config.yaml`:

```yaml
wandb:
  enabled: true
  entity: "YOUR-WANDB-ENTITY"  # ⬅️ UPDATE THIS (e.g., "athenazh-university-of-pennsylvania")
  project: "arpo-uitars-training"
  name: "uitars-2b-128tasks-run1"
```

---

## What Gets Logged

### Per Epoch:
- **average_reward**: Mean reward across all tasks
- **success_rate**: Percentage of tasks completed (score >= 0.9)
- **tasks_completed**: Number of tasks evaluated
- **replay_buffer_size**: Number of cached successful trajectories
- **policy_loss**: Policy gradient loss
- **learning_rate**: Current learning rate

### Per Task:
- **task_reward**: Individual task score (0.0 or 1.0)
- **num_steps**: Steps taken to complete task
- **inference_time**: Total time spent on model inference

### System:
- **gpu_memory**: GPU memory usage (if on Colab)
- **training_time**: Elapsed time per epoch
- **checkpoint_saved**: When checkpoints are saved

---

## Using wandb in Training

### Option 1: Use wandb-Enabled Script

```bash
cd ~/Desktop/ARPO_replicate

# Run training with wandb logging
python scripts/run_training_with_wandb.py \
    --wandb-config configs/wandb_config.yaml \
    --training-config configs/config_uitars_2b_mac.yaml \
    --epochs 10 \
    --result-dir results_training_128
```

### Option 2: Manual Integration

Add to your training code:

```python
import wandb

# Initialize
wandb.init(
    entity="your-entity",
    project="arpo-uitars-training",
    config={
        "model": "UI-TARS-2B",
        "tasks": 128,
        "epochs": 10,
    }
)

# Log metrics
wandb.log({
    "epoch": epoch,
    "average_reward": avg_reward,
    "success_rate": success_rate,
})

# Finish
wandb.finish()
```

---

## Viewing Results

After training starts, view your run at:
```
https://wandb.ai/YOUR-ENTITY/arpo-uitars-training
```

### Dashboards Show:
- 📈 Reward curves over epochs
- 📊 Success rate progression
- 🎯 Per-task performance
- ⏱️ Training time metrics
- 💾 Model checkpoints
- 📝 Hyperparameters and config

---

## Example wandb Dashboard

After training, you'll see:

**Summary**:
- Run: uitars-2b-128tasks-run1
- Status: ✅ Finished
- Duration: 5 days, 3 hours
- Final success rate: 68.5%

**Metrics**:
- Average reward: 0.123 → 0.685 (5.6x improvement!)
- Success rate: 12% → 68%
- Replay buffer: 0 → 87 tasks cached

**Artifacts**:
- Checkpoints: epoch_2, epoch_4, epoch_6, epoch_8, epoch_10
- Config: Full training configuration saved

---

## Disabling wandb

If you don't want wandb logging:

Edit `configs/wandb_config.yaml`:
```yaml
wandb:
  enabled: false  # ⬅️ Set to false
```

Training will continue without wandb.

---

## Tips

1. **Tag runs**: Use tags like "baseline", "with-replay", "ablation" to organize experiments
2. **Compare runs**: wandb makes it easy to compare different configurations
3. **Share results**: Get shareable links for your runs
4. **Resume training**: wandb tracks checkpoints for resuming interrupted runs

---

## Troubleshooting

### "wandb.errors.UsageError: api_key not configured"
→ Run `wandb login` first

### "Project not found"
→ Check entity name in config (must match your wandb account/team)

### Logs not appearing
→ Check `wandb.log()` is called inside training loop
→ Verify network connection (wandb needs internet)

---

**Ready to track your ARPO training!** 🎯
