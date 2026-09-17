# Sahayak — the one file to read

This is the single document for the project. Read it in order. Other files under
`docs/` are wiring checklists, not a second explanation.

You do **not** need to already know machine learning, I2S, or Raspberry Pi internals.
Every term is defined the first time it appears.

---

## 0. What we are building, in one paragraph

A box that sits quietly and listens. When you say a **keyword** (default: *Sahayak*),
it notices **on the device itself**, then instantly sends the audio *after* that word
to a speech-to-text program on the same Raspberry Pi 4B (or later, from a smaller
chip to the Pi). The screen shows “listening / awake”, the confidence, the transcript,
and the delay. Nothing goes to Google, Alexa, or any cloud for the wake-up step.

That is SIH problem statement **26172**, implemented honestly.

---

## 1. The problem statement (source of truth)

The official PS sets two numbers for **continuous listening until the keyword**:

- **RAM below 256 KB** (the KWS listen path: model arena + buffers)
- **CPU under 10%** while waiting for the keyword

These are binding. The ESP32-S3 is the device that can meet them honestly. The Pi 4B
is the ASR server and the fallback demo; do not claim the whole Pi is 256 KB.

**What the PS actually says:**

1. Build an ultra-lightweight, highly accurate **keyword spotting (KWS)** model.
2. It must run **locally** on a **low-power** device.
3. After the keyword, **stream the following audio** to a **remote ASR** server.
4. Streaming must be **instant**, with **minimal data** and **low latency**.
5. **Open-source only.** No commercial voice-activation SDK.
6. Allowed tools: TensorFlow Lite for Microcontrollers, PyTorch Mobile, or similar.
7. **No pretrained global keywords** like “Hey Google” or “Alexa”. Train a **custom** word.

**How they score you:**

| Metric | Exact meaning |
|---|---|
| Efficiency | How big the model is in RAM/Flash, and how much CPU it uses while only listening |
| Accuracy | How often it catches the real word, and how rarely it wakes on other speech |
| Latency | Time from the **end of the keyword** until the ASR **receives** the audio stream |

Latency is not “when we detected it”. The detector needs a little audio *after* the
word before it is sure. That extra wait is part of the number. We measure from the
true end of the spoken word.

We host ASR on the Pi over the local network, not on the public internet. The PS
says “cloud ASR”. We disclose this: the demo works without internet; the delay we
report is the node-to-server hop, which is the hop we control.

---

## 2. Words you will see everywhere

**Keyword / wake word.** A short word that means “start listening for a command”.
Ours is configurable. Product name is **Sahayak**. The spoken wake word is
`marvin` (Google Speech Commands v2, trained by us). Changing the word is a
**retrain**, not a rewrite of the app. See [section 10](#10-how-to-change-the-keyword).

**Keyword spotting (KWS).** A tiny classifier that answers one question: “did that
last second of audio contain *our* word?” It is not full speech recognition. It
does not understand sentences.

**ASR (Automatic Speech Recognition).** The bigger program that turns speech into
text. We use **Vosk** (open source). It only runs *after* the keyword.

**TinyML.** Machine learning small enough for microcontrollers (chips with
hundreds of KB of RAM, not gigabytes).

**Feature.** A compact number computed from raw audio (see MFCC below). The model
never sees raw waveforms. It sees a small grid of features.

**INT8 quantization.** Storing the model’s numbers as 8-bit integers instead of
32-bit floats. About 4× smaller, much faster on small chips, a little less
accurate. We still train in float, then convert.

**Node.** The always-on listener. **ESP32-S3** firmware in `firmware/` is the scored
device. The Pi 4B Python node is the safety-net / baseline.

**Server.** The ASR + screen. Always stays on the Pi 4B. The node talks to it
through a socket (a network pipe), even when both run on the same board.

**I2S.** A digital audio wire protocol. Clock + left/right tick + data. Our mic
and amp both speak I2S.

**GPIO.** Numbered pins on the Pi. “BCM 18” means the chip’s pin name 18, which
is physical pin 12 on the header.

**Energy gate / VAD-lite.** If the last chunk of audio is almost silence, skip
the neural net. This is how idle CPU stays low.

**False accept / false reject.** Waking when you did not say the word / not
waking when you did. Accuracy is both of these, not just “it works when I say it”.

---

## 3. Why we are not copying the old Chotu project

There is an earlier repo, `raspi-aibot`. It felt like a voice assistant. The
wake-up was:

1. Send microphone audio to **Google’s cloud** speech-to-text.
2. Search the returned sentence for the letters `chotu`.

That fails the PS on three fronts: detection is not local, it uses a commercial
cloud SDK, and there is no trained custom model — so there is no model size, no
idle CPU, and no real latency story.

We keep only three things from it: the Pi audio package list, the idea of playing
a short sound on wake, and the lesson of what *not* to do.

That repo also committed a live Gemini API key. Revoke it. Unrelated to SIH, but
do it.

---

## 4. The big idea (architecture)

```
 microphone
     |
     v
 [node]  always on, tiny model, no network until wake
     |  on wake: send 20 ms audio frames over a socket
     v
 [server]  Vosk turns speech into text, screen shows it
```

Today both boxes are processes on the **same Pi 4B**. They still use a socket, on
purpose. If you get an ESP32 tomorrow, you move only the node. The server does
not change. You point it at a new IP.

That is the most important design decision in the repo.

What the node does, every 20 milliseconds:

1. Read 320 samples of audio (20 ms at 16,000 samples per second).
2. Keep the last 500 ms in a **ring buffer** (a fixed circular tape; old samples
   fall off the back). This is **pre-roll** — so the first syllable of the
   command after the keyword is not lost.
3. If the chunk is quiet, do nothing else. (**Energy gate.**)
4. If it has speech energy, compute **MFCCs** and ask the **detector**.
5. If the detector says “keyword”: play a short chime, connect to the server,
   send the pre-roll, then keep sending live audio until silence.

The detector has two backends behind one interface:

- `StubDetector` — fires on sustained speech energy. Used to prove the pipes
  work *before* a model exists.
- `TFLiteDetector` — the real INT8 model. Drop in a `.tflite` file; no other
  code changes.

---

## 5. Machine learning, from sound to a yes/no

### 5.1 Why not feed raw audio into a neural net?

A second of 16 kHz audio is 16,000 numbers. A small chip cannot run a model on
that every 20 ms. Also, raw samples are very sensitive to loudness and mic
colour. We first turn sound into a **spectrogram-like** picture that is small
and stable.

### 5.2 From waveform to MFCC (the feature)

Think of a piano-roll of the last ~1 second:

1. Cut the audio into overlapping windows: **30 ms** long, hop **20 ms**.
2. For each window, measure how much energy sits in each of **40** frequency
   bands spaced the way human hearing is spaced (**mel scale**: we hear pitch
   logarithmically, not linearly).
3. Take a logarithm (loudness is also logarithmic).
4. Compress those 40 numbers into **10** coefficients with a DCT. Those 10
   numbers are the **MFCCs** (Mel-Frequency Cepstral Coefficients). They are
   the standard compact fingerprint of a short speech sound.
5. Stack **49** windows. You now have a grid of shape **49 × 10**. That is
   the model input. About one second of speech, ~490 numbers instead of 16,000.

This recipe is fixed in `shared/feature_spec.py`. The Pi and a future ESP32
must compute the **same** numbers for the same audio, or the model will
silently fail. That file is the contract.

### 5.3 The model family: DS-CNN-S (predefined architecture)

We do **not** invent a new network. We use **DS-CNN-S**, a published tiny
keyword-spotting architecture from the “Hello Edge” / ARM research line, the
same family used in TensorFlow’s Speech Commands examples.

**CNN** = convolutional neural net: it slides small filters over the 49×10
grid and learns local patterns (“this looks like the ‘sa’ of Sahayak”).

**Depthwise separable (DS)** = a cheaper CNN. Instead of one expensive 3D
filter, it does a cheap per-channel blur and then a 1×1 mix. Same idea as
MobileNet. That is why it fits on a microcontroller.

**S** = the small variant. There are M and L versions. We start at S.

**Three output classes:**

- `keyword` — our current word (default sahayak)
- `unknown` — other speech
- `silence` — background

The network itself does not contain the letters “Sahayak”. Those letters live
in the **training data** and in a config string. That is why we can change the
word later without redesigning the model.

### 5.4 “Predefined model” vs the PS rule

The PS forbids models already trained to hear **“Hey Google” / “Alexa”**.
It does **not** forbid using a known **architecture**, or starting from
weights trained on the open **Speech Commands** dataset (yes/no/up/down/…),
which is a research set, not a commercial assistant.

What we do:

- Architecture is predefined: DS-CNN-S.
- Optional warm start: Speech Commands weights, then we **replace the last
  layer** and train on *our* word.
- The shipped detector is always a model **we** trained for the configured
  keyword. We never ship a “Hey Google” engine.

To try a different word: change `KEYWORD` in `shared/config.py`, ingest or
generate data, retrain, flash. The node, server, and protocol do not change.

### 5.5 How training works (high level)

Spoken keyword is `marvin`. Product name is Sahayak. Data is Google Speech
Commands v2 (CC-BY-4.0): thousands of real 1-second clips, official test list
held out so validation speakers never appear in train. We train DS-CNN-S
ourselves (INT8 TFLite). No Porcupine, WakeNet, Alexa, or ASR string-match.

Live INMP441 clips mix in later, after the mic is resoldered. They are not
required to train.

### 5.6 After the model: smoothing, so it does not flicker

Neural nets flicker. One frame 0.91, next 0.40, next 0.88. We:

- **Average** the last few “keyword” scores (posterior smoothing).
- Require the average to stay high for several frames (**debounce**).
- After a wake, refuse to wake again for ~1.5 s.
- **Mute** the detector while the chime plays, so the speaker cannot trigger
  itself.

---

## 6. Hardware (Pi 4B safety-net build)

You already have: Pi 4B, 6-pin I2S mic (INMP441), 4 ohm speaker, MAX98357A amp,
3.5 inch SPI touch display.

### 6.1 Microphone — INMP441 (digital I2S)

This is not a USB mic and not a 3.5 mm mic. It sends **numbers** over three
wires, not an analog voltage.

| Mic pin | Meaning | Pi |
|---|---|---|
| VDD | Power | **3.3 V only** (physical pin 1) |
| GND | Ground | GND |
| SCK | Bit clock | GPIO18 / pin 12 |
| WS | Left/right clock | GPIO19 / pin 35 |
| SD | Audio data out | GPIO20 / pin 38 |
| L/R | Which ear | Tie to GND = **left** channel |

The chip outputs 24-bit samples sitting in 32-bit slots. Software captures
`S32_LE`, keeps the left channel, shifts down to 16-bit, then resamples
48 kHz → 16 kHz (the I2S overlay on the Pi likes 48 kHz; our model wants 16 kHz).

**Why 16 kHz?** Speech energy lives mostly below 8 kHz. Nyquist says you need
2× that, so 16 kHz is the TinyML default. Higher rates waste RAM and CPU for
no keyword-spotting gain.

### 6.2 Speaker — 4 ohm, through the MAX98357A amp

The Pi cannot drive a speaker from a GPIO. The amp is an I2S **output** chip.
It shares the same clocks as the mic and takes one extra data pin (GPIO21).
The speaker screws onto the amp, not onto the Pi.

If I2S playback fights the mic, use the Pi’s 3.5 mm jack for the wake chime
only. The chime is for the demo. It is not a scored metric.

### 6.3 Display — 3.5 inch SPI touch

It plugs onto the 40-pin header. Two traps:

1. **It covers every pin.** You need a **stacking header** (2×20 riser) or you
   cannot attach the mic and amp.
2. **Many clones use GPIO18 for backlight.** GPIO18 is also the I2S clock. If
   both claim it, the mic is silently dead. Check your panel. Fixes: move or
   disable backlight on 18, lift pin 12 on the display connector, demo on HDMI,
   or fall back to a USB mic (only `node/capture.py` changes).

### 6.4 Software on the Pi for I2S

In `/boot/firmware/config.txt` (Bookworm) add:

```
dtparam=i2s=on
dtoverlay=googlevoicehat-soundcard
```

Reboot. `arecord -l` should show `sndrpigooglevoi`. First test:

```
arecord -D plughw:CARD=sndrpigooglevoi,DEV=0 -c 2 -f S32_LE -r 48000 -d 5 -V stereo /tmp/t.wav
```

`-V stereo` draws live meters. Left should move. If right moves instead, `L/R`
is not tied to ground.

Full pin tables: `docs/pinout.md`. Command-by-command bring-up: `docs/i2s-setup.md`.

---

## 7. Software map (what each folder is)

```
GUIDE.md                 <-- you are here
shared/                  rules both node and server obey
  config.py              keyword, ports, thresholds (change the word here)
  feature_spec.py        MFCC recipe (do not drift this)
  protocol.py            bytes on the socket
node/                    the always-on listener
  capture.py             mic / file / USB
  features.py            MFCC (golden reference for a future ESP32)
  energy.py              silence skip
  detector.py            StubDetector + TFLiteDetector
  stream.py              send frames to the server
  chime.py               short beep
  main.py                the loop
server/                  stays on the Pi 4B forever
  asr_server.py          socket + Vosk (or a local mock if Vosk is missing)
  ui.py                  480×320 “instrument” screen
  latency_harness.py     keyword-end → first byte received
training/                run on a laptop or the Pi, not in the live loop
  model.py               DS-CNN-S definition
  generate_dataset.py    TTS + augmentation
  train.py               train, quantize, export models/<keyword>.tflite
firmware/                ESP32-S3 node (scored device)
  include/pins.h         INMP441 on D4/D5/D6
  src/main.cpp           I2S + RMS speech/silence probe
tools/
  plot_waveform.py       Phase A: is the mic clean?
  profile.py             RAM / CPU numbers for the deck
  push_to_pi.sh          copy this folder over SSH tomorrow
  s3_monitor.py          serial log from the S3
```

Node and server are **separate programs**. Start the server first, then the node.

---

## 8. The wire protocol (what “stream to ASR” means)

When the node wakes it opens a TCP connection to `ASR_HOST:ASR_PORT`
(default `127.0.0.1:8765`).

```
header (little-endian):
  4 bytes  magic     "KWS1"
  1 byte   version   1
  2 bytes  preroll_ms
  4 bytes  sample_rate (16000)
then: many frames
  2 bytes  nbytes
  nbytes   int16 little-endian PCM  (320 samples = 640 bytes for 20 ms)
end:
  2 bytes  0
```

Why raw PCM, not MP3/Opus? Compression costs CPU and adds delay. 16 kHz mono
int16 is 32 kB/s — tiny on a LAN, and an ESP32 can emit it without a codec.

Pre-roll is sent first so the ASR hears the start of the command, not the
middle.

---

## 9. How you will move this project onto the Pi (tomorrow)

You said: one SD card, then enable SSH, then copy files. Do this.

**On the Pi, first boot (with a keyboard/HDMI once):**

1. Raspberry Pi Imager → Raspberry Pi OS (32-bit Bookworm is fine; 64-bit also
   works). In Imager, open the gear icon and **enable SSH**, set username
   `pi` and a password, optionally set Wi-Fi.
2. Boot. Find the IP: `hostname -I` on the Pi, or look in your router.
3. From then on you work from the Mac.

**On the Mac, every time you want to send code:**

```
./tools/push_to_pi.sh pi@192.168.x.x
```

That is `rsync`. It copies this folder to `~/sahayak` on the Pi and skips
`venv`, `__pycache__`, and big `data/` clips.

Then SSH in:

```
ssh pi@192.168.x.x
cd ~/sahayak
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# first time only, for real ASR:
# wget the small Vosk model (see README) and unpack into models/vosk
python -m server.asr_server &
python -m node.main
```

Until the SD card exists, you can run the same two commands **on this Mac**.
`node/main.py --source file --wav path.wav` uses a recording instead of I2S.
The mock ASR kicks in if Vosk is not installed, so the pipes can be tested
with no extra downloads.

---

## 10. How to change the keyword

1. Open `shared/config.py` and set `KEYWORD = "yourword"` (lowercase, one word
   works best). Prefer a Speech Commands v2 word (`marvin`, `sheila`, `happy`)
   so you get thousands of real speakers: `python -m training.ingest_speech_commands`.
2. Or synthesise: `python -m training.generate_dataset --keyword yourword`
3. `python -m training.train --keyword yourword`
4. Flash the node. It embeds `models/yourword.int8.tflite`.

The product name on the UI stays Sahayak. Only the detector word changes.

Avoid `yes` / `no` / `go` / `stop` — too short, high false-accept.

---

## 11. ESP32-S3 node (the scored device)

The board on USB is an **ESP32-S3-DevKitC** (not Nano), 16 MB flash, 8 MB PSRAM.
MAC: `b4:3a:45:a4:3e:08`. Firmware lives in `firmware/`.

**INMP441 wiring (3.3 V only) — use the GPIO numbers printed on the DevKit:**

| Mic silk | DevKit pin |
|---|---|
| VDD / VCC | 3V3 |
| GND | GND |
| L/R | GND |
| SCK / BCLK | **4** |
| WS / LRCK | **5** |
| SD / DOUT | **6** |

HW-104 analog amp stays on the **Pi headphone jack**, not on these I2S pins.

**What runs on it:** I2S capture, energy gate, KWS1 streaming and telemetry on
core 1 (Arduino's `loopTask`); the 70 Hz highpass, sparse-filterbank MFCC and the
INT8 DS-CNN-S (TensorFlow Lite Micro with ESP-NN kernels) on core 0. Inference
takes tens of milliseconds, so it cannot share a core with a 20 ms audio hop.

**Serial console** (`REC`/`STOP` are what `tools/record_s3.py` drives):

| Command | Effect |
|---|---|
| `MIC` | toggle 4 Hz `MIC rms=… peak=… dc=…` lines |
| `REC` / `STOP` | start/stop streaming base64 audio hops for dataset recording |
| `STATUS` | model bytes, arena high-water mark, wake count, last inference µs |
| `BENCH` | 20 forced inferences; prints steady-state MFCC / invoke µs and speech CPU |

Flash and check the mic. After `pio run -t upload`, macOS often leaves GPIO0 low
and the chip sits in download mode — boot it with `python -m tools.s3_monitor --reset`
rather than `pio device monitor`:

```
cd firmware && ~/.platformio/penv/bin/pio run -t upload
cd .. && python -m tools.s3_monitor --reset --seconds 8
python -m tools.mic_check
python -m tools.bench
```

`mic_check` prints a PASS/FAIL verdict. If RMS stays flat the mic is not wired,
L/R is not on GND, or VDD is on 5 V. Nothing measured downstream means anything
until this passes.

**Rebuild the model and everything generated from it:**

```
python -m training.ingest_speech_commands   # Speech Commands v2 → data/marvin/
python -m training.train                     # trains, scores, exports
python -m tools.mfcc_parity                  # C frontend == Python frontend
cd firmware && pio run -t upload
```

`training.train` calls `training.export_firmware`, which regenerates
`firmware/include/mfcc_tables.h` and `firmware/include/kws_model.h`. Both are
generated files — never hand-edit them, and rebuild the firmware after training
or the device will run features the model was not trained on.

What still stays on the Pi: Vosk, 3.5" swipe UI, latency harness. Same protocol later.

---

## 12. Decision log (why we chose each thing)

| Decision | Why |
|---|---|
| Pi 4B as the first complete demo | Hardware you have now; touch screen; can host ASR |
| Node and server as two processes | ESP32 becomes an IP change, not a rewrite |
| DS-CNN-S, not a custom net | Published, TinyML-proven, keyword lives in data not topology |
| Keyword in `config.py` | Retrain to swap the word; PS wants a custom word, not a frozen one |
| Speech Commands only as a backbone / unknown class | Legal under the PS; not “Hey Google” | 
| Stub detector first | Never debug “pipes” and “model” at the same time |
| Vosk, not Whisper, on the server | Streaming, small, open source; Whisper is batch and heavy |
| LAN ASR, not public cloud | Demo without internet; disclose vs the word “cloud” |
| 16 kHz mono int16 on the wire | Minimum useful speech; ESP32-friendly; no codec |
| Energy gate before the net | Idle CPU is a scored metric |
| Report *subsystem* RAM on the Pi | A Linux SBC cannot honestly claim a 256 KB whole-system footprint |
| No underclocking the Pi to “look small” | It raises CPU % and does not shrink RAM; judges will see through it |
| Synthetic TTS + augmentation | No time for a 100-speaker record week |
| Hold out real team voices | Only honest accuracy number |
| I2S INMP441, not USB, as the default | Same mic you will put on an ESP32; USB is the fallback |
| pygame 480×320 UI | Matches the 3.5" panel; no browser on the Pi |
| One GUIDE, not a pile of essays | You asked for one file |

---

## 13. What “done” looks like (gates)

- **A** — Clean 16 kHz waveform from the mic. No clipping. (`tools/plot_waveform.py`)
- **B** — Say something, see text on the screen, get a latency number. Stub is OK.
- **C** — Dataset exists: keyword, unknowns, lookalikes, noise.
- **D** — Real `.tflite` in `models/`, accuracy on *your* voices is acceptable.
- **E** — TV / Hindi / lookalikes do not wake it constantly.
- **F** — Footprint, idle CPU, latency, TPR/FAR written down for the deck.
  Each has a tool; `python -m tools.results_report` collects them into
  `docs/results.md` and lists whatever is still missing:

  | Number | Command |
  |---|---|
  | false accepts/hour + idle CPU | `python -m tools.far_test --minutes 60 --note "<material>"` |
  | true positive rate | `python -m tools.tpr_test --speaker <name> --per-distance 10` |
  | keyword-end → ASR first byte | `python -m server.latency_harness --wav <clip> --trials 12` |
  | threshold / debounce knee | `python -m tools.tune_threshold` |

- **G** — (optional) ESP32 node, same protocol, comparison table.

You can demo B to a faculty member before D is finished. That is why the stub
exists.

---

## 14. Honest limits (say these out loud)

- The Pi 4B is **not** a low-power MCU. We present it as the reference
  implementation. Efficiency marks are fully earned only after Phase G.
- TTS voices are not Indian-accent diverse enough. Fine-tune on real voices
  as soon as you can spare an hour with four people.
- “Near-zero false activations” is an operational target (< 1/hour of
  non-keyword audio). It is not a promise the first model will hit.
- If the display steals GPIO18, the mic will look broken. Check that before
  blaming software.

---

## 15. Glossary (short)

| Term | One line |
|---|---|
| Sample | One audio number. 16,000 of them = one second at 16 kHz |
| int16 | Whole numbers from −32768 to 32767. Standard CD-like PCM, 16-bit |
| Ring buffer | Fixed array; write index wraps around. No malloc in the hot path |
| Tensor arena | The RAM slab a TFLite Micro model is allowed to use |
| Posterior | The model’s probability for a class, after softmax |
| Overlay | A Raspberry Pi firmware snippet that turns on a hardware block (I2S) |
| ALSA | Linux sound system. `arecord` / `aplay` talk to it |
| Socket | Two programs sending bytes. TCP in our case |
| Quantization | Float weights → int8 weights |
| Transfer learning | Start from a trained net, train the last layer on new classes |

When something in the code contradicts this file, fix the code or update this
file in the same change. Do not leave a third story elsewhere.
