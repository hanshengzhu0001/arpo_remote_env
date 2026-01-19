# Running GPU Server on Colab Web

Since VSCode ↔ Colab has connection issues, use the Colab web interface for the GPU server.

## Step-by-Step

### 1. Open Colab in Browser

Go to: https://colab.research.google.com

### 2. Upload Notebook

**Option A: From GitHub (Recommended)**
1. Click "File" → "Open notebook"
2. Select "GitHub" tab
3. Paste: `https://github.com/gowathena/arpo_replica/blob/arpo-cpu-replicate/notebooks/GPU_Server_for_OSWorld.ipynb`
4. Click the search icon
5. Open the notebook

**Option B: Upload from Mac**
1. Click "File" → "Upload notebook"
2. Select: `/Users/hanszhu/Desktop/ARPO_replicate/notebooks/GPU_Server_for_OSWorld.ipynb`

### 3. Select GPU Runtime

1. Click "Runtime" → "Change runtime type"
2. Hardware accelerator: **GPU**
3. GPU type: **A100** (or T4 if A100 unavailable)
4. Click "Save"

### 4. Run Cells

Run cells **1-5** in order:
1. Install dependencies (~30 seconds)
2. Configure ngrok (~1 second)
3. Load model (~2 minutes) ⏰
4. Create Flask server (~1 second)
5. Start server + ngrok ✅

### 5. Copy URL

From Cell 5 output, copy the URL:
```
GPU_SERVER_URL = "https://xxxx-xxxx.ngrok-free.dev"
```

### 6. Keep Tab Open

- **Don't close the Colab tab!** (Server will stop)
- Minimize it and leave it running
- Can check logs anytime

---

## On Your Mac (VSCode)

### Open Evaluation Notebook

```bash
cd /Users/hanszhu/Desktop/ARPO_replicate
code notebooks/ARPO_OSWorld_Evaluation.ipynb
```

### Select Kernel

1. Click kernel selector (top right)
2. Choose: **Python 3.10.16 ('arpo')**

### Paste URL

In Cell 4, update:
```python
GPU_SERVER_URL = "https://YOUR-URL-FROM-COLAB.ngrok-free.dev"
```

### Run All Cells

Click "Run All" - evaluation will run on your Mac using Colab GPU for inference!

---

## Monitoring

### Colab Tab
- Watch Cell 5 output for inference logs:
  ```
  [15:30:42] Generated in 3.45s
  [15:30:48] Generated in 2.89s
  ...
  ```

### Mac Terminal
- Watch OSWorld output:
  ```bash
  tail -f logs/test_osworld_uitars.log
  ```

---

## Troubleshooting

### "ngrok session expired"
- Free ngrok sessions last 2 hours
- Re-run Cell 5 to get new URL
- Update Mac evaluation notebook with new URL

### "502 Bad Gateway"
- Server died on Colab
- Check Colab tab for errors
- Re-run Cells 4-5

### "Connection timeout"
- Colab went to sleep
- Click in Colab tab to wake it up
- May need to re-run Cell 5

---

## Expected Timeline

| Task | Time |
|------|------|
| Setup Colab | 3 minutes |
| Load model | 2 minutes |
| 10 tasks on Mac | ~10-15 minutes |
| **Total** | **~15-20 minutes** |

---

## Benefits of This Approach

✅ **Stable**: Colab web is more reliable than VSCode connection  
✅ **Simple**: Just keep one browser tab open  
✅ **Fast**: A100 GPU inference (2-5 sec/step)  
✅ **Clean**: Mac runs OSWorld, Colab runs model  

**Ready to start!** 🚀
