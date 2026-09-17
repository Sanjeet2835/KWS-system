# Pi 4B first boot (blank SD + 3.5" panel)

Do this on the Mac. I cannot image the card from here. After SSH works, I do the rest.

## 1. Image the SD (Mac)

1. Install [Raspberry Pi Imager](https://www.raspberrypi.com/software/).
2. Insert the new SD (USB reader).
3. Imager:
   - Device: **Raspberry Pi 4**
   - OS: **Raspberry Pi OS (64-bit)** — the full desktop one, not Lite (we need the 3.5" UI).
   - Storage: the new card.
4. Click the gear / **Edit settings** before Write:
   - Hostname: `sahayak`
   - Username: `pi`
   - Password: pick one and remember it (I will need it for SSH).
   - Wi-Fi: **same network as this Mac**, SSID + password, country **IN**
   - Enable **SSH** → Use password authentication
5. Write. Eject the card.

## 2. Hardware (first boot — no mic)

- SD into the Pi 4B (contacts toward the board).
- **Official 5 V / 3 A USB-C power** (or a known-good 3 A supply). Phone chargers brown out.
- **3.5" display:**
  - **HDMI panel:** HDMI cable to the Pi HDMI port closest to USB-C. Power the panel if it has a separate USB.
  - **GPIO hat** (sits on all 40 pins): seat it straight on the header. The screen may stay white/black until I install the overlay over SSH. That is OK.
- **Do not** plug the INMP441 or amp onto the Pi tonight. The hat covers the header; I2S also fights GPIO18 on many 3.5" clones. Mic stays for the S3 after resolder.
- Ethernet into the router is optional but makes SSH easier than Wi-Fi.

## 3. Power on and wait ~2 minutes

First boot resizes the filesystem. Green LED should flicker.

## 4. Tell me this (then I take over)

From another phone/laptop on the same Wi-Fi, or the Pi screen if it already shows a desktop:

```
hostname -I
```

Or from the Mac:

```
ping -c 2 sahayak.local
```

Then send me:

- the **IP** (e.g. `192.168.1.42`) or that `sahayak.local` ping works
- whether the 3.5" is **HDMI** or a **GPIO hat**

Do **not** paste the SSH password in chat. I will `ssh pi@<ip>`; type the password in the terminal prompt when it asks.

I will:

```
ssh pi@<ip>
./tools/push_to_pi.sh pi@<ip>
```

and install the Sahayak server + 480×320 UI on that screen.

## 5. ESP32 (already flashed)

Leave the S3 unplugged until tomorrow. After resolder:

| Device | Pad | GPIO |
|---|---|---|
| INMP441 | SCK WS SD | 4 5 6 |
| INMP441 | VDD GND L/R | 3V3 GND GND |
| OLED | SDA SCL | 8 9 |
| OLED | VCC GND | 3V3 or 5V, GND |

Speak: serial should show `SPEECH` not `listen L=0`. OLED if alive: top bars + `OLED ok`.
