#!/usr/bin/env bash
# Flash Raspberry Pi OS Lite + Sahayak onto the USB SD card.
# Run in Terminal.app (Cursor cannot write raw disks on macOS):
#   bash "/Users/jl-mayank/sih 2026/tools/flash_sd_mac.sh"
set -euo pipefail

IMG="/tmp/sahayak-pi/raspios-lite.img"
REPO_TGZ="/tmp/sahayak-pi-repo.tgz"
DISK="disk2"
RDISK="/dev/rdisk2"

if [[ ! -f "$IMG" ]]; then
  echo "Missing $IMG — image was not downloaded."
  exit 1
fi

INFO="$(diskutil info "$DISK")"
echo "$INFO" | grep -q 'Protocol:[[:space:]]*USB' || { echo "REFUSE: $DISK is not USB"; exit 1; }
echo "$INFO" | grep -q 'Removable Media:[[:space:]]*Removable' || { echo "REFUSE: $DISK is not removable"; exit 1; }
echo "$INFO" | grep 'Disk Size'
echo
if [[ "${1:-}" != "--yes" ]]; then
  read -r -p "This WIPES the SD ($DISK, 31.5 GB). Type YES to continue: " ok
  [[ "$ok" == "YES" ]] || { echo "aborted"; exit 1; }
fi

echo "Unmounting..."
diskutil unmountDisk "$DISK"

echo "Writing OS (2–5 min). Enter your Mac password if asked."
sudo dd if="$IMG" of="$RDISK" bs=4m
sync

echo "Waiting for boot partition..."
for _ in {1..20}; do
  diskutil mountDisk "$DISK" >/dev/null 2>&1 || true
  if [[ -d /Volumes/bootfs ]]; then
    BOOT=/Volumes/bootfs
    break
  fi
  if [[ -d /Volumes/boot ]]; then
    BOOT=/Volumes/boot
    break
  fi
  sleep 1
done
if [[ -z "${BOOT:-}" ]]; then
  echo "Boot partition did not mount. SD is imaged; SSH enable may need a rerun."
  exit 1
fi

echo "Customizing $BOOT ..."
# SSH + user pi / password sahayak
: > "$BOOT/ssh"
printf 'pi:%s\n' '$6$FhXNeCA2RddZtse/$WdRBIfJC8W3vpJtA8tqaUfkHrBtBG5KbjIzkpKmxSZb4HH.SAGzJZY32sIgXUHQN6jk0353OeVw2Ws1xwYsfi.' > "$BOOT/userconf.txt"

# USB-C ethernet gadget so the Mac can SSH without Wi-Fi
if [[ -f "$BOOT/config.txt" ]] && ! grep -q 'dtoverlay=dwc2' "$BOOT/config.txt"; then
  printf '\n[all]\ndtoverlay=dwc2,dr_mode=peripheral\n' >> "$BOOT/config.txt"
fi
if [[ -f "$BOOT/cmdline.txt" ]] && ! grep -q 'g_ether' "$BOOT/cmdline.txt"; then
  python3 - <<'PY'
from pathlib import Path
p = Path("/Volumes/bootfs/cmdline.txt") if Path("/Volumes/bootfs/cmdline.txt").exists() else Path("/Volumes/boot/cmdline.txt")
t = p.read_text().strip()
if "modules-load=dwc2,g_ether" not in t:
    t = t.replace("rootwait", "rootwait modules-load=dwc2,g_ether", 1)
    p.write_text(t + "\n")
PY
fi

if [[ -f "$REPO_TGZ" ]]; then
  cp "$REPO_TGZ" "$BOOT/sahayak.tgz"
fi
cat > "$BOOT/SAHAYAK.txt" <<'EOF'
Sahayak Pi 4B
User: pi
Password: sahayak
SSH is on.

After boot (wait ~2 min):
  ssh pi@raspberrypi.local
  # or, USB-C to Mac: a new Ethernet gadget appears; then
  ssh pi@raspberrypi.local

Unpack the project:
  sudo mkdir -p /home/pi/sahayak
  sudo tar -xzf /boot/firmware/sahayak.tgz -C /home/pi/sahayak
  sudo chown -R pi:pi /home/pi/sahayak
EOF

sync
echo
echo "DONE. Eject the card:"
echo "  diskutil eject $DISK"
echo "Put it in the Pi 4B, power with 5V/3A, wait 2 minutes, then:"
echo "  ssh pi@raspberrypi.local"
echo "Password: sahayak"
