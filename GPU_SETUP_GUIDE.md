# GPU Setup Guide - Complete Workflow

## Overview

This guide explains how to use **Colab A100 GPU** for inference while keeping your **tested Mac OSWorld setup**.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Complete Setup                          │
│                                                          │
│  Colab A100 (VSCode)          Mac (Local)              │
│  ┌──────────────────┐         ┌──────────────────┐     │
│  │ GPU Server       │         │ OSWorld VM       │     │
│  │ UI-TARS 7B       │◄────────│ Ubuntu (VMware)  │     │
│  │ Flask :9000      │         │                  │     │
│  │ + ngrok tunnel   │────────►│ Screenshots      │     │
│  │                  │         │ Actions          │     │
│  │ ~2-5 sec/step    │         │ Task execution   │     │
│  └──────────────────┘         └──────────────────┘     │
└─────────────────────────────────────────────────────────┘
```

**Why this works**:
- ✅ Your Mac OSWorld setup is already tested and working
- ✅ Only the model changes (localhost → Colab URL)
- ✅ No need to setup OSWorld on Colab (complex)
- ✅ GPU inference (fast) + Mac VM (familiar)

---

## Step-by-Step Instructions

### Part 1: Start GPU Server on Colab

#### Option A: VS Code + Colab (Your Setup)

1. **Open in VS Code**:
   ```bash
   code notebooks/GPU_Server_for_OSWorld.ipynb
   ```

2. **Connect to Colab**:
   - Click kernel selector (top right)
   - Select "Existing Jupyter Server"
   - Choose "Google Colab"
   - Select **A100 GPU** runtime

3. **Run cells 1-4**:
   - Cell 1: Install dependencies
   - Cell 2: Load model (~2 minutes)
   - Cell 3: Create server
   - Cell 4: Start with ngrok → **GET PUBLIC URL**

4. **Copy the URL** from cell 4 output:
   ```
   📍 Public URL: https://xxxx-xx-xx-xxx-xxx.ngrok.io
   ```

#### Option B: Colab Web Interface

1. Go to: https://colab.research.google.com
2. Upload `notebooks/GPU_Server_for_OSWorld.ipynb`
3. Runtime → Change runtime → A100 GPU
4. Run cells 1-4
5. Copy public URL

---

### Part 2: Configure Mac OSWorld

1. **Update agent to use Colab URL**:
   ```bash
   cd /Users/hanszhu/Desktop/ARPO_replicate
   
   # Edit OSWorld/mm_agents/uitars_agent.py
   # Find line (~562):
   #   base_url="http://localhost:9000/v1",
   # Change to:
   #   base_url="https://YOUR-NGROK-URL.ngrok.io/v1",
   ```

2. **Or use sed** to update automatically:
   ```bash
   NGROK_URL="https://xxxx.ngrok.io"  # Your URL from Colab
   
   sed -i '' "s|http://localhost:9000/v1|${NGROK_URL}/v1|g" OSWorld/mm_agents/uitars_agent.py
   
   echo "✅ Updated to use Colab GPU server"
   ```

---

### Part 3: Run OSWorld Evaluation

```bash
cd /Users/hanszhu/Desktop/ARPO_replicate/OSWorld

# Test on 5 original tasks
python run_uitars.py \
    --headless \
    --observation_type screenshot \
    --max_steps 15 \
    --model arpo-uitars-7b \
    --temperature 0.6 \
    --max_tokens 256 \
    --test_all_meta_path ../test_data/osworld_examples/test_10tasks.json \
    --result_dir ../results/gpu_eval_original/

# Test on 5 noisy tasks
python run_uitars.py \
    --headless \
    --observation_type screenshot \
    --max_steps 15 \
    --model arpo-uitars-7b \
    --temperature 0.6 \
    --max_tokens 256 \
    --test_all_meta_path ../test_data/osworld_examples/test_10tasks_noisy.json \
    --result_dir ../results/gpu_eval_noisy/
```

---

## Expected Results

### Timing:
- **Per step**: 2-5 seconds (vs 60 min on CPU!)
- **Per task**: ~30-75 seconds (15 steps)
- **5 tasks**: ~2-6 minutes
- **10 tasks total**: ~5-12 minutes

### Success Indicators:

**On Colab** (watch Cell 4 output):
```
Generated in 3.45s
Generated in 2.89s
Generated in 4.12s
...
```

**On Mac** (OSWorld terminal):
```
Prediction: Thought: ... Action: LEFT_CLICK(...)
[INFO] Step 1: LEFT_CLICK(...)
[INFO] Got screenshot successfully
...
[INFO] Reward: 1.00  ← Task succeeded!
```

---

## Troubleshooting

### Issue: "Connection refused" on Mac
**Cause**: ngrok URL not updated or incorrect  
**Fix**: Double-check URL in `uitars_agent.py`, include `/v1` suffix

### Issue: "Timeout" even with GPU
**Cause**: Network latency (Colab ↔ Mac)  
**Fix**: Increase timeout in `uitars_agent.py` to 60s (should be plenty)

### Issue: "ngrok session expired"
**Cause**: Free ngrok has 2-hour limit  
**Fix**: Re-run cell 4 to get new URL, update Mac config

### Issue: GPU out of memory
**Cause**: 4-bit might not be enough  
**Fix**: Restart Colab runtime, try T4 GPU (slower but works)

---

## Performance Comparison

| Setup | Device | Model | Time/Step | 10 Tasks |
|-------|--------|-------|-----------|----------|
| **Before** | Mac CPU | UI-TARS-2B | 60 min | ~600 min (10 hrs) |
| **After** | Colab A100 | ARPO 7B | 3 sec | ~10 min (5-12 min) |

**Speedup**: **1,200x faster!** (60 min → 3 sec)

---

## File Paths (After Reorganization)

- **GPU Server Notebook**: `notebooks/GPU_Server_for_OSWorld.ipynb` ⭐
- **Agent Config**: `OSWorld/mm_agents/uitars_agent.py`
- **Test Data**: `test_data/osworld_examples/`
- **Results**: `results/gpu_eval_original/`, `results/gpu_eval_noisy/`

---

## Success Criteria

After running on GPU, you should have:

- [ ] Colab server running (cell 4 active)
- [ ] ngrok public URL copied
- [ ] Mac agent updated with URL
- [ ] 10 tasks completed (5 original + 5 noisy)
- [ ] Average ~3-5 sec per step
- [ ] Results saved to `results/gpu_eval*/`

If all pass → **GPU setup works perfectly!** 🎉

---

##Summary

**This setup**:
- ✅ Uses your TESTED Mac OSWorld VM (no changes needed!)
- ✅ Just swaps model server (local CPU → Colab GPU)
- ✅ Same code, same pipeline, 1200x faster
- ✅ Ready to run right now!

**Next**: Open `notebooks/GPU_Server_for_OSWorld.ipynb` in VSCode + Colab! 🚀
