# Pi 4B I2S Setup — Phase A

Goal of Phase A: **a clean speech waveform captured from the INMP441 at 16 kHz mono
int16.** Nothing more. No model, no streaming, no display.

Wire the hardware first per [pinout.md](pinout.md), including the GPIO18 conflict check.

---

## 1. System dependencies

Package list carried over from the team's earlier `raspi-aibot/setup_pi.sh`, which was
already validated on Pi hardware.

```bash
sudo apt-get update
sudo apt-get install -y \
  python3-pyaudio portaudio19-dev \
  libsdl2-dev ffmpeg flac \
  python3-numpy python3-scipy python3-matplotlib \
  alsa-utils
```

`espeak` from the old script is not needed — we do not synthesize speech on the node.

## 2. Enable I2S

Edit the firmware config. On Raspberry Pi OS **Bookworm** the path is
`/boot/firmware/config.txt`; on Bullseye and earlier it is `/boot/config.txt`.

```bash
sudo nano /boot/firmware/config.txt
```

Add:

```ini
# I2S for INMP441 mic + MAX98357A amp
dtparam=i2s=on
dtoverlay=googlevoicehat-soundcard
```

The `googlevoicehat-soundcard` overlay is the pragmatic choice: it presents a single ALSA
card with **both capture and playback** over a shared I2S bus, which is exactly the
mic-plus-amp topology we have. It is locked to **48 kHz, S32_LE**.

If your display overlay also lives in this file, make sure it does not claim GPIO18.

Reboot:

```bash
sudo reboot
```

## 3. Verify the card exists

```bash
arecord -l
cat /proc/asound/cards
dmesg | grep -i -E "i2s|voicehat|snd"
```

Expect a card named `sndrpigooglevoi`. If it is absent, the overlay did not load — the
config edit or the reboot did not take. Fix that before touching wiring.

## 4. First capture

```bash
arecord -D plughw:CARD=sndrpigooglevoi,DEV=0 \
  -c 2 -f S32_LE -r 48000 -d 5 \
  -V stereo /tmp/mic_test.wav
```

The `-V stereo` flag draws live level meters for both channels. This is the single most
useful diagnostic in Phase A: speak, and watch which channel moves.

With `L/R` tied to GND, only the **left** meter should respond. A dead left channel and a
live right channel means `L/R` is floating or tied high.

Play it back to confirm:

```bash
aplay /tmp/mic_test.wav
```

## 5. Convert to the pipeline format

The captured file is 48 kHz stereo S32_LE. The pipeline wants **16 kHz mono int16**.

Two steps, both of which belong in `node/capture.py`:

1. **Channel extract** — take the left channel only
2. **Bit depth** — the INMP441's 24 bits sit MSB-aligned in a 32-bit slot, so `>> 16`
   yields int16. A high-pass filter around 60-80 Hz is worth adding; these mics carry a
   noticeable DC offset and low-frequency rumble
3. **Decimate 48000 to 16000** — exact factor of 3, via `scipy.signal.resample_poly(x, 1, 3)`,
   which applies the anti-alias filter for you

**Why decimate rather than ask ALSA for 16 kHz.** `plughw` will happily resample, but the
quality and filter are opaque and can differ between machines. We control the decimation
so that the 16 kHz int16 stream entering the feature extractor is reproducible. This
matters for Phase G: the ESP32 configures its I2S peripheral natively at 16 kHz and skips
decimation entirely, and we need both paths to feed the feature spec identical data.

## 6. Gate A validation

Capture roughly 5 seconds containing the word "Sahayak" spoken twice, then check:

- [ ] Waveform shows clear speech envelope against a quiet noise floor
- [ ] **No clipping** — peak magnitude comfortably below full scale. If it clips, back off
      from the mic; the INMP441 has no gain control
- [ ] Noise floor is low and flat with no 50 Hz hum
- [ ] DC offset removed by the high-pass
- [ ] Duration is accurate — 5.0 s of audio, not 4.2 s or 6.1 s. A duration error means
      the sample rate is not what you think it is
- [ ] Spectrogram shows speech energy concentrated below 4 kHz, with formant structure
      visible

Plot both waveform and spectrogram. A spectrogram catches problems a waveform hides:
aliasing from bad decimation, a stuck bit, or wrong channel ordering.

**Gate A passes when a human looking at the plots agrees it is clean speech at a verified
16 kHz.** Do not proceed to Phase B on a marginal capture — every downstream accuracy
problem will be blamed on the model.

---

## Troubleshooting

**Total silence, both channels.** Overlay not loaded, or GPIO18 held by the display.
Check `dmesg`, then disconnect the display entirely and retest. This isolates the conflict
faster than any amount of inspection.

**Silence on left, signal on right.** `L/R` pin is not tied to GND.

**Loud static or full-scale noise.** Usually SD (data) miswired, or VDD on 5 V instead of
3.3 V. Power down and recheck before assuming a dead part.

**Very quiet but present.** Likely the shift is wrong — try `>> 14` or inspect raw 32-bit
values to find where the 24 significant bits actually sit. Print a few raw samples in hex;
the alignment becomes obvious.

**`arecord` reports device busy.** Something else holds the card — often PulseAudio or
PipeWire. `systemctl --user stop pipewire pipewire-pulse` for the duration of testing.

**Playback works, capture does not.** The amp and mic share clocks, so playback succeeding
proves the bus is alive and narrows the fault to the mic's SD line or `L/R`.
