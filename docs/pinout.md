# Hardware Wiring — Pi 4B Node

Target: Raspberry Pi 4B, INMP441 I2S mic, MAX98357A I2S amp, 4 ohm speaker,
3.5 inch SPI touch display.

**Read the [GPIO conflict warning](#gpio-conflict-warning-read-before-wiring) before you
connect anything.** There is a known collision between the 3.5 inch display and the I2S
bus that will silently break mic capture.

---

## GPIO conflict warning — read before wiring

The Pi's I2S peripheral is hard-wired to **GPIO18 / 19 / 20 / 21** on the 40-pin header.
It cannot be remapped; the alternate PCM mapping (GPIO28-31) is not exposed on the 40-pin
header. So I2S must have those four pins.

Two problems follow:

**1. Many 3.5 inch SPI displays use GPIO18 for backlight control.**
This includes several Waveshare "RPi LCD (A)/(C)" variants and the common MPI3501 clones.
GPIO18 is `PCM_CLK`. If the display holds it, the mic gets no bit clock and capture
produces silence or noise with no obvious error.

Check your display's schematic or driver overlay for GPIO18 before wiring. Resolutions,
in order of preference:

- Configure the display overlay with backlight disabled or on a different pin, and tie the
  backlight permanently on
- Physically lift/mask pin 12 on the display's header connector
- Drop the SPI display and demo on HDMI instead — costs the "standalone device" look but
  keeps the audio path intact
- Keep the SPI display and move the mic to **USB audio**. Only `node/capture.py` changes;
  the feature spec, model, and protocol are untouched. This is a legitimate fallback, not
  a defeat

**2. The display physically covers the 40-pin header.**
These panels seat on all 40 pins, leaving nowhere to attach the mic and amp. You need a
**stacking / pass-through GPIO header extender** (2x20 female-to-male riser) so the
display sits above and the pins remain reachable. Order one; it is the cheapest part in
the build and the whole thing is un-demoable without it.

---

## INMP441 microphone (I2S input)

| Mic pin | Function | BCM | Physical pin |
|---|---|---|---|
| VDD | 3.3 V power | — | 1 |
| GND | Ground | — | 6 |
| SCK | Bit clock | GPIO18 (PCM_CLK) | 12 |
| WS | Word select / LR clock | GPIO19 (PCM_FS) | 35 |
| SD | Serial data out | GPIO20 (PCM_DIN) | 38 |
| L/R | Channel select | tie to GND | 39 |

`L/R` tied to GND puts the mic on the **left** channel. Remember this: capture is stereo
and the right channel will be empty. Extracting the wrong channel is the most common
cause of a "dead mic" that is actually wired correctly.

The INMP441 emits **24-bit samples in 32-bit slots**, MSB-aligned. Capture as `S32_LE`
and shift right by 16 to get int16. Do not assume the low bits are meaningful.

Power on 3.3 V, not 5 V. The INMP441 is a 3.3 V part.

## MAX98357A amplifier (I2S output)

Shares the clock lines with the mic and adds one data pin.

| Amp pin | Function | BCM | Physical pin |
|---|---|---|---|
| VIN | 5 V power | — | 2 or 4 |
| GND | Ground | — | 9 |
| BCLK | Bit clock | GPIO18 (PCM_CLK) | 12 |
| LRC | Word select | GPIO19 (PCM_FS) | 35 |
| DIN | Serial data in | GPIO21 (PCM_DOUT) | 40 |
| GAIN | Gain select | leave floating for 9 dB | — |
| SD | Shutdown / channel | leave floating for stereo-average | — |

Speaker connects to the amp's `+` and `-` screw terminals. 4 ohm is fine; the MAX98357A
drives 4-8 ohm.

Sharing BCLK and LRC between mic and amp is correct and intended — that is how a
full-duplex I2S bus works. The `googlevoicehat-soundcard` overlay exposes both directions.

**Chime fallback:** if simultaneous capture and playback proves fiddly, use the Pi's
3.5 mm jack for the wake chime and keep the amp for Phase G. The chime is a UX detail, not
a scored metric — do not lose days to it.

## 3.5 inch SPI touch display

Typical mapping for XPT2046-based panels. **Verify against your specific panel;** these
vary between clones.

| Function | BCM | Physical pin |
|---|---|---|
| SPI MOSI | GPIO10 | 19 |
| SPI MISO | GPIO9 | 21 |
| SPI SCLK | GPIO11 | 23 |
| LCD CS | GPIO8 (CE0) | 24 |
| Touch CS | GPIO7 (CE1) | 26 |
| LCD DC/RS | GPIO25 | 22 |
| LCD RESET | GPIO27 | 13 |
| Touch IRQ | GPIO17 | 11 |
| Backlight | **GPIO18 — CONFLICT** | 12 |

---

## Pin budget summary

Pins claimed after everything is connected:

- I2S audio: 18, 19, 20, 21
- Display SPI + control: 7, 8, 9, 10, 11, 17, 25, 27
- Power/ground: pins 1, 2/4, 6, 9, 39

No conflicts remain **provided** the display's GPIO18 backlight is resolved.

## Pre-power checklist

Run through this before first boot with the mic attached.

- [ ] Stacking header fitted; display seated, mic and amp pins reachable
- [ ] Display's GPIO18 usage checked and resolved
- [ ] INMP441 VDD on **3.3 V** (pin 1), not 5 V
- [ ] MAX98357A VIN on **5 V** (pin 2 or 4)
- [ ] Mic `L/R` tied to GND
- [ ] No wire bridging 3.3 V and 5 V rails
- [ ] Speaker on amp terminals, not on GPIO
- [ ] Continuity check GND between Pi, mic, and amp

Next: [i2s-setup.md](i2s-setup.md) for the software configuration.
