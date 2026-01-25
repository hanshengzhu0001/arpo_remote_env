# Full VERL ARPO Training on Colab A100

Complete setup for running full ARPO training with VERL framework on Colab A100.

## Overview

Everything runs on Colab A100:
```
Colab A100
┌────────────────────────────────┐
│ ✅ VERL Trainer               │
│ ✅ UI-TARS-2B Model (trainable)│
│ ✅ 2 Docker OSWorld containers │
│ ✅ Experience Replay Buffer    │
│ ✅ Policy Updates (GRPO)       │
└────────────────────────────────┘
```

No Mac needed - pure Colab solution!

---

## Setup Steps

### 1. Open Colab

Go to: https://colab.research.google.com

### 2. Upload Notebook

Upload `notebooks/ARPO_Training_VERL_Colab.ipynb`

### 3. Select Runtime

- Runtime → Change runtime type
- Hardware accelerator: **GPU**
- GPU type: **A100** (important!)
- Click Save

### 4. Run Cells in Order

**Cell 1**: Check GPU (verify A100)  
**Cell 2**: Clone repository  
**Cell 3**: Install dependencies (~5 min)  
**Cell 4**: Setup Docker for OSWorld (~10 min)  
**Cell 5**: Start Ray cluster  
**Cell 6**: Configure wandb (enter API key)  
**Cell 7**: Update OSWorld for Docker  
**Cell 8**: Create training config  
**Cell 9**: Run VERL training ⏰ (~20-40 hours)

---

## Key Differences from Mac Setup

| Aspect | Mac Setup | Colab Setup |
|--------|-----------|-------------|
| **Model** | Frozen on Colab server | Trainable on Colab |
| **OSWorld** | VMware on Mac | Docker on Colab |
| **Training** | Mac orchestration | Full VERL on Colab |
| **Updates** | No weight updates | ✅ Full policy optimization |
| **Replay** | Not available | ✅ Experience replay |

---

## Training Configuration

```yaml
Model: UI-TARS-2B (2B params, trainable)
Tasks: 128 (all 10 domains)
Environments: 2 Docker containers
Rollouts: 4 per task
Epochs: 1
Max steps: 16

Algorithm: GRPO + Experience Replay
Learning rate: 1e-6
Batch size: 8
```

---

## Expected Performance

| Metric | Value |
|--------|-------|
| **Setup time** | ~15 minutes |
| **Training time** | ~20-40 hours |
| **GPU memory** | ~30-35 GB |
| **Total tasks** | 128 × 4 rollouts = 512 rollouts |
| **Policy updates** | Every 8 tasks |

---

## Monitoring

**In Colab**:
- Watch Cell 9 output for training logs
- See reward progression
- Monitor GPU usage: `!nvidia-smi`

**On wandb**:
- Dashboard: https://wandb.ai/hanszhu05/arpo-uitars-training
- Real-time metrics
- Reward curves
- Success rate progression

---

## Important Notes

### Colab Limitations:
1. **Session timeout**: Free tier disconnects after ~12 hours
   - **Solution**: Use Colab Pro ($10/month) or pause/resume
   
2. **Idle timeout**: Disconnects if no output for ~90 minutes
   - **Solution**: Training produces output continuously
   
3. **Daily limits**: Free tier has GPU usage limits
   - **Solution**: Colab Pro for extended training

### Docker on Colab:
- OSWorld Docker containers work out-of-box
- No manual VM download needed
- Lighter weight than VMware

### Checkpoints:
- Saved every 20-30 tasks
- Download checkpoints before session ends
- Can resume training from checkpoint

---

## After Training

Download results:

```python
# In a new cell after training completes
from google.colab import files

# Download checkpoints
!zip -r checkpoints.zip checkpoints/
files.download('checkpoints.zip')

# Download results
!zip -r results.zip results/
files.download('results.zip')
```

---

## Advantages of This Approach

✅ **Complete ARPO**: Full experience replay + policy optimization  
✅ **Self-contained**: Everything on Colab  
✅ **True training**: Model weights actually update  
✅ **Simpler**: No Mac-Colab coordination  
✅ **Reproducible**: Anyone with Colab can replicate  

---

## Estimated Costs

- **Free tier**: Not practical (session limits)
- **Colab Pro**: $10/month
  - A100: ~$1/hour
  - 30 hours = ~$30 total
  - **Total: $40** for complete training

**Alternative**: Use RunPod/Lambda Labs directly (~$20-30 total)

---

**Ready to train!** 🚀

See `notebooks/ARPO_Training_VERL_Colab.ipynb`
