# Quick Start - GPU Testing (Colab A100)

## 🎯 Goal

Test ARPO UITARS 7B model on 10 OSWorld tasks using GPU.

---

## 🚀 Method 1: Colab Notebook (Recommended)

### Step 1: Open in Colab
1. Go to: https://github.com/gowathena/arpo_replica
2. Open `notebooks/ARPO_OSWorld_Evaluation.ipynb`
3. Click "Open in Colab" badge

### Step 2: Select GPU Runtime
- Runtime → Change runtime type → **A100 GPU**
- Or use **T4 GPU** (free tier)

### Step 3: Run All Cells
- Runtime → Run all
- Wait ~5-10 minutes for completion

### Expected Output:
```
✅ GPU: Tesla A100-SXM4-40GB
✅ Model loaded!
💾 GPU Memory Used: 4.82 GB

📊 Average inference time: 3.45s
📊 Original tasks avg: 3.42s
📊 Noisy tasks avg: 3.48s
```

---

## 🖥️ Method 2: VS Code + Colab GPU

### Step 1: Install Colab Extension
```bash
# In VS Code
code --install-extension ms-toolsai.jupyter
```

### Step 2: Connect to Colab
1. Open `notebooks/ARPO_OSWorld_Evaluation.ipynb` in VS Code
2. Click kernel selector → **Select Another Kernel**
3. Choose **Existing Jupyter Server**
4. Enter Colab URL (get from colab.research.google.com)

### Step 3: Run Cells
- Select A100 GPU runtime in Colab
- Run cells in VS Code

---

## 📊 What Gets Tested

### Model:
- **Name**: Fanbin/ARPO_UITARS1.5_7B
- **Size**: 7B parameters (4-bit quantized → ~4.8GB VRAM)
- **Performance**: 83.9% on 128 OSWorld tasks (from paper)

### Tasks:
- **5 Original**: Standard OSWorld Chrome tasks
- **5 Noisy**: Same tasks with distractor browsing history
- **Total**: 10 tasks

### Metrics:
- Inference time per step
- GPU memory usage
- Action prediction quality

---

## 🎓 Comparison: CPU vs GPU

| Metric | Mac CPU (2B) | Colab GPU (7B) |
|--------|--------------|----------------|
| **Model** | UI-TARS-2B | ARPO UITARS 7B |
| **Inference** | ~60 min/step | ~2-5 sec/step |
| **Speed** | 1x | **720x faster** |
| **Training (8 tasks)** | 400 hours | 5-10 hours |
| **Practical?** | ❌ No | ✅ Yes |

---

##Tips for Colab:

1. **Free tier (T4)**: Works but slower (~5-10 sec/step)
2. **Colab Pro (A100)**: Recommended (~2-3 sec/step)
3. **Session limits**: Colab disconnects after ~12 hours
4. **Save checkpoints**: Download results before session ends

---

## 🐛 Troubleshooting

### "CUDA out of memory"
- Use T4 instead of A100
- Or restart runtime: Runtime → Restart runtime

### "Model not found"
- Model is public, no HF token needed
- Check internet connection

### "Inference too slow"
- Verify GPU is selected (not CPU)
- Check: `torch.cuda.is_available()` returns `True`

---

## ✅ Success Criteria

After running the notebook, you should see:
- [ ] Model loads in ~1-2 minutes
- [ ] GPU memory usage ~4-5 GB
- [ ] Inference time ~2-5 seconds per step
- [ ] 10 tasks complete successfully
- [ ] Results saved to dataframe

---

**Ready to test on Colab A100!** 🚀

See `notebooks/ARPO_OSWorld_Evaluation.ipynb`
