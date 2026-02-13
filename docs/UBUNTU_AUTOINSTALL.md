# Ubuntu Server Unattended Install (Autoinstall)

Fully unattended Ubuntu Server install in VMware Fusion using cloud-init autoinstall and a seed ISO.

## 1. Create the autoinstall seed ISO (Mac)

From the repo (or set `SEED_DIR` / `SEED_ISO` if you prefer):

```bash
cd /path/to/ARPO_replicate

# Default password: ChangeMe123!  (override with env)
export UBUNTU_AUTOINSTALL_PASSWORD='YourSecurePassword'   # optional
./scripts/build_ubuntu_autoinstall_seed.sh
```

This creates:

- `~/ubuntu-seed/` with `user-data` and `meta-data`
- `~/ubuntu-seed.iso` (volume label **CIDATA**)

Optional env vars: `UBUNTU_AUTOINSTALL_HOSTNAME`, `UBUNTU_AUTOINSTALL_USERNAME`, `UBUNTU_AUTOINSTALL_TIMEZONE`.

## 2. Attach seed ISO to the VM and disable 3D

Ensure the VM exists and uses the **Ubuntu Server** installer ISO (e.g. run `./scripts/switch_vm_to_server_install.sh` first). Then:

```bash
./scripts/attach_autoinstall_to_vm.sh
```

This script:

- Adds a **second CD** in the VM pointing to `~/ubuntu-seed.iso`
- Sets **mks.enableGL = "FALSE"** (3D graphics off) in the .vmx

You can also in Fusion: **Settings → Display** → uncheck **Accelerate 3D graphics**.

## 3. Boot and trigger autoinstall from GRUB

1. **Power On** the VM in Fusion.
2. At the **GRUB** menu, highlight **Try or Install Ubuntu Server** and press **e**.
3. Find the line that starts with `linux` (or `linuxefi`).
4. Append **before** the `---` at the end of that line: **`autoinstall ds=nocloud`**  
   Example:  
   `linux /casper/vmlinuz ---` → `linux /casper/vmlinuz autoinstall ds=nocloud ---`
5. Boot with **Ctrl+X** (or **F10** / **fn+F10**).

The installer runs unattended and skips the language and interactive screens.

## 4. After install

1. VM reboots. In Fusion, **disconnect both CD images** (Server installer ISO and seed ISO) so the VM boots from the virtual disk.
2. Log in:
   - **User:** `ubuntu`
   - **Password:** the one you set (e.g. `ChangeMe123!` or `UBUNTU_AUTOINSTALL_PASSWORD`).
3. Update:

   ```bash
   sudo apt update && sudo apt upgrade -y
   ```

4. Set up shared folder and env server as in [REMOTE_ENV_SERVER_RUNBOOK.md](REMOTE_ENV_SERVER_RUNBOOK.md).

## Summary

| Step | Command / action |
|------|-------------------|
| Build seed ISO | `./scripts/build_ubuntu_autoinstall_seed.sh` |
| Attach seed + disable 3D | `./scripts/attach_autoinstall_to_vm.sh` |
| Boot VM | Power On in Fusion |
| GRUB | **e** → add `autoinstall ds=nocloud` before `---` → **Ctrl+X** |
| After reboot | Eject both CDs, log in as `ubuntu`, `sudo apt update && sudo apt upgrade -y` |
