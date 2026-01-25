# Fix: "VM Cannot Be Found" Error

## Problem

Error message:
```
Cannot open VM: .../OSWorld/vmware_vm_data/Ubuntu0/Ubuntu0.vmx, The virtual machine cannot be found
```

This happens when:
- VM download was interrupted
- VM files deleted but registry not cleared
- VM registry is corrupted

---

## Solution: Clear Registry and Re-download

```bash
cd ~/Documents/ARPO/arpo_replica/OSWorld

# Step 1: Clear VM registry
rm -f .vmware_vms .vmware_lck

# Step 2: Remove partial VM files (if any)
rm -rf vmware_vm_data

# Step 3: Let OSWorld re-download VM
python run_uitars.py \
    --headless \
    --observation_type screenshot \
    --max_steps 3 \
    --test_config_base_dir ../test_data/osworld_examples \
    --test_all_meta_path ../test_data/osworld_examples/test_chrome_10.json \
    --result_dir ../results/test/
```

This will:
1. Download fresh Ubuntu VM (~15GB, ~5-10 min)
2. Extract to `vmware_vm_data/Ubuntu0/` (~38GB total)
3. Start VM and create snapshot
4. Run test successfully

---

## What Happened

The `.vmware_vms` file tracks allocated VMs:
```
Ubuntu0:/path/to/Ubuntu0.vmx:0
```

But the actual VM files at that path were missing (incomplete download or moved).

**Solution**: Delete the registry so OSWorld starts fresh!

---

## Quick Fix Commands

```bash
cd ~/Documents/ARPO/arpo_replica/OSWorld

# One-liner to fix
rm -f .vmware_vms .vmware_lck && rm -rf vmware_vm_data && echo "✅ Cleared - ready for fresh download"

# Then run any evaluation to trigger download
```

---

## Prevent This Issue

**Don't interrupt** the first run when you see:
```
Downloading the virtual machine image...
Unzipping... (this takes ~5 minutes)
```

Let it complete! If interrupted, you'll need to clear and re-download.
