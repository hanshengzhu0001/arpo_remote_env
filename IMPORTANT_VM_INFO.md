# ⚠️ IMPORTANT: VM Is NOT Included in Repository

## What Gets Cloned from GitHub

When you clone this repository, you get:
- ✅ Code, scripts, notebooks
- ✅ Documentation
- ✅ Test data (JSON files)
- ✅ Configuration files
- ❌ **NOT the Ubuntu VM** (~38GB)

## What You Must Download Separately

### 1. VMware Fusion
- **Size**: ~600MB installer
- **Download**: https://www.vmware.com/products/fusion.html
- **Required**: YES - must install on every Mac

### 2. Ubuntu VM
- **Size**: ~38GB (15GB download + 23GB extracted)
- **Source**: Auto-downloads from HuggingFace
- **Download**: Automatic on first run
- **Location**: `OSWorld/vmware_vm_data/Ubuntu0/`

---

## First Run: Automatic VM Download

When you first run any OSWorld evaluation, it will:

```
[INFO] No free virtual machine available. Generating a new one...
[INFO] Downloading the virtual machine image...
Progress: ████████████ 15GB/15GB

[INFO] Unzipping... (this takes ~5 minutes)
[INFO] Files extracted to: ./vmware_vm_data/Ubuntu0

[INFO] Starting VM...
[INFO] Virtual machine started.
[INFO] VM is ready!
```

**Time**: ~10-15 minutes (depends on internet speed)

**Only happens once!** Subsequent runs use the existing VM.

---

## Why VM is Not in Repository

1. **Size**: 38GB is too large for GitHub
2. **License**: VM image has separate licensing
3. **Updates**: HuggingFace hosts the latest version
4. **Storage**: Would make repository huge

---

## What This Means for Your Friend

Your friend must:

1. ✅ **Install VMware Fusion** (600MB, 5 minutes)
2. ✅ **Add vmrun to PATH** (1 command)
3. ✅ **Clone repository** (code only, fast)
4. ✅ **Run first evaluation** (triggers 38GB VM download, 10-15 minutes)

**Cannot skip steps 1-2!** VMware Fusion must be installed.

**Cannot share VM files!** Each user downloads separately from HuggingFace.

---

## Quick Checklist for New Users

Before running any evaluation:

- [ ] VMware Fusion installed and launched once
- [ ] `vmrun` command works: `vmrun -T fusion list`
- [ ] Repository cloned: `git clone ...`
- [ ] Conda environment created: `conda create -n arpo python=3.10`
- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] OSWorld patches applied: `cp osworld_patches/*.py OSWorld/...`
- [ ] Ready for first run (will download VM automatically)

---

See `SETUP_FOR_NEW_USERS.md` for complete step-by-step instructions!
