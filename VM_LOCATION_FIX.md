# VM Location Fix

## Issue

VM created in wrong location: `arpo_replica/vmware_vm_data/` instead of `arpo_replica/OSWorld/vmware_vm_data/`

## Why This Happens

OSWorld scripts run from the `OSWorld/` directory, so:
- `VMS_DIR = "./vmware_vm_data"` 
- Resolves to: `OSWorld/vmware_vm_data/` ✅

If you run from repo root, it creates VM in wrong place.

---

## Solution: Move VM to Correct Location

```bash
cd ~/Documents/ARPO/arpo_replica

# Check if VM is in wrong location
if [ -d vmware_vm_data/Ubuntu0 ]; then
    echo "VM found in wrong location - moving..."
    
    # Create correct directory
    mkdir -p OSWorld/vmware_vm_data
    
    # Move VM
    mv vmware_vm_data/Ubuntu0 OSWorld/vmware_vm_data/
    
    # Remove old directory
    rmdir vmware_vm_data 2>/dev/null || rm -rf vmware_vm_data
    
    echo "✅ VM moved to OSWorld/vmware_vm_data/Ubuntu0/"
    echo "✅ You can now run evaluations!"
else
    echo "VM not found in wrong location - check OSWorld/vmware_vm_data/"
fi
```

---

## Verify Correct Location

```bash
cd ~/Documents/ARPO/arpo_replica

# Should see the VM here:
ls -la OSWorld/vmware_vm_data/Ubuntu0/Ubuntu0.vmx

# This should work:
vmrun -T fusion list
# Or:
cd OSWorld
python -c "from desktop_env.providers.vmware.manager import VMwareVMManager; m = VMwareVMManager(); print('VM path:', m.get_vm_path('Ubuntu'))"
```

---

## Correct Directory Structure

```
arpo_replica/
├── OSWorld/                           ← Scripts run from here
│   ├── run_uitars.py
│   ├── vmware_vm_data/               ← VM should be here! ✅
│   │   └── Ubuntu0/
│   │       └── Ubuntu0.vmx
│   └── ...
├── test_data/
├── notebooks/
└── ...
```

**NOT**:
```
arpo_replica/
├── vmware_vm_data/                   ← Wrong! ❌
│   └── Ubuntu0/
├── OSWorld/
└── ...
```

---

## If VM Download Failed

If the VM partially downloaded or is corrupted:

```bash
# Remove and re-download
rm -rf OSWorld/vmware_vm_data

# Run any evaluation - VM will auto-download
cd OSWorld
python run_uitars.py \
    --headless \
    --observation_type screenshot \
    --max_steps 3 \
    --test_config_base_dir ../test_data/osworld_examples \
    --test_all_meta_path ../test_data/osworld_examples/test_chrome_10.json \
    --result_dir ../results/test/
```

VM will download to correct location: `OSWorld/vmware_vm_data/Ubuntu0/`

---

**Quick Fix**: Move `vmware_vm_data` folder into `OSWorld/` directory!
