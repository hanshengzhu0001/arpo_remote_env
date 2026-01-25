# Setup Guide for New Users

If you're getting "vmrun: command not found", follow these steps:

## Step 1: Install VMware Fusion (Required!)

**⚠️ IMPORTANT**: The Ubuntu VM (~38GB) is NOT included in the repository. Each user must download it separately!

### A. Download VMware Fusion
- **Free for personal use**: https://www.vmware.com/products/fusion.html
- **Broadcom download**: https://support.broadcom.com/group/ecx/productdownloads?subfamily=VMware+Fusion

### B. Install VMware Fusion
1. Download the `.dmg` file (~600MB)
2. Open the `.dmg` and drag VMware Fusion to Applications folder
3. Launch VMware Fusion once
4. Follow the setup wizard (accept license, etc.)
5. Close VMware Fusion (we'll use command-line)

---

## Step 2: Add vmrun to PATH

Open Terminal and run:

```bash
# Add VMware Fusion to your PATH
echo 'export PATH="/Applications/VMware Fusion.app/Contents/Library:$PATH"' >> ~/.zshrc

# Reload your shell configuration
source ~/.zshrc

# Or if you use bash instead of zsh:
echo 'export PATH="/Applications/VMware Fusion.app/Contents/Library:$PATH"' >> ~/.bash_profile
source ~/.bash_profile
```

---

## Step 3: Verify vmrun Works

```bash
vmrun -T fusion list
```

**Expected output**:
```
Total running VMs: 0
```

If you see this, vmrun is working! ✅

---

## Step 4: Clone and Setup Repository

```bash
# Clone repository
git clone https://github.com/gowathena/arpo_replica.git
cd arpo_replica
git checkout arpo-cpu-replicate

# Initialize OSWorld submodule
git submodule update --init --recursive

# Apply our modifications
cp osworld_patches/uitars_agent.py OSWorld/mm_agents/
cp osworld_patches/run_multienv_uitars.py OSWorld/
cp osworld_patches/run_uitars.py OSWorld/

# Create conda environment
conda create -n arpo python=3.10 -y
conda activate arpo

# Install dependencies
pip install -r requirements.txt
cd OSWorld && pip install -r requirements.txt && pip install -e . && cd ..
```

**Note**: Ubuntu VM is NOT included in the repository! It will auto-download (~38GB, ~10 minutes) on first run.

---

## Step 5: Update Colab Server URL

Edit `OSWorld/mm_agents/uitars_agent.py` line 562:

```python
# Change this line:
base_url="https://miller-unshapeable-melany.ngrok-free.dev/v1"

# To your actual Colab ngrok URL from GPU server notebook Cell 5
base_url="https://YOUR-NGROK-URL.ngrok-free.dev/v1"
```

---

## Step 6: First Run (Downloads VM Automatically)

**⚠️ First run will download Ubuntu VM (~38GB, takes ~10 minutes)**

```bash
# Start a simple test to trigger VM download
cd OSWorld
python run_uitars.py \
    --headless \
    --observation_type screenshot \
    --max_steps 3 \
    --test_config_base_dir ../test_data/osworld_examples \
    --test_all_meta_path ../test_data/osworld_examples/test_chrome_10.json \
    --result_dir ../results/test/
```

You'll see:
```
Downloading the virtual machine image... (~15GB download)
Unzipping... (~5 minutes)
Starting VM...
VM is ready!
```

VM downloads to: `OSWorld/vmware_vm_data/Ubuntu0/` (38GB total)

## Step 7: Run Full Evaluation

After VM is downloaded, open `notebooks/ARPO_OSWorld_Evaluation.ipynb`:

1. Select `arpo` kernel
2. Update Cell 4 with your Colab ngrok URL
3. Run cells in order

---

## Common Issues

### "vmrun: command not found"
→ VMware Fusion not installed or not in PATH. Do Steps 1-3 above.

### "No module named 'desktop_env'"
→ OSWorld not installed. Run:
```bash
cd OSWorld && pip install -e . && cd ..
```

### "Left tasks: chrome: 0"
→ Old results exist. The notebook auto-clears them, but you can manually clear:
```bash
rm -rf results/gpu_eval_chrome_10
```

### "Connection error" to Colab
→ GPU server not running or wrong URL. Check:
1. Colab Cell 5 is still running
2. Copy the correct ngrok URL
3. Update in both places (uitars_agent.py and evaluation notebook)

---

## Expected Timeline

- **VMware setup**: 10 minutes (first time)
- **Environment setup**: 5 minutes
- **VM download**: Automatic on first run (~10 min for 15GB)
- **10 Chrome tasks**: ~45 minutes
- **20 tasks (original + noisy)**: ~1.5 hours

---

## Need Help?

See:
- `README.md` - Main setup guide
- `TROUBLESHOOTING.md` - Common issues
- `GPU_SETUP_GUIDE.md` - Colab GPU setup
- `OSWORLD_MODIFICATIONS.md` - What we changed in OSWorld

**Good luck!** 🚀
