# Anuvani — technical specification (as implemented)

This file is a code-true dump of the listen-node stack for the spoken wake word **marvin**. Every number below is taken from the repository as it exists now: `shared/feature_spec.py`, `node/features.py`, `training/`, `firmware/`, `models/`, and `data/marvin/`. Nothing here is estimated.

There is **no `.tlf` format** in this project. The model path is:

| Artifact | Path | Role | Size |
| --- | --- | --- | --- |
| Keras (float32, train-time) | `models/marvin.keras` | DS-CNN-S with BatchNorm; saved after training | 424,055 bytes |
| TensorFlow Lite float32 | `models/marvin.float.tflite` | Folded-BN reference graph | **92,548 bytes** |
| TensorFlow Lite INT8 | `models/marvin.int8.tflite` | What actually runs on the ESP32-S3 | **43,152 bytes** |
| TFLM C array | `firmware/include/kws_model.h` | Same 43,152 INT8 bytes as `KWS_MODEL[43152]`, `KWS_MODEL_LEN 43152` | 43,152 model bytes |

TFL = TensorFlow Lite (`.tflite`). TFLM = TensorFlow Lite for Microcontrollers, which loads the C array in place. “TLF” is not a file this repo produces.

The network input is **49 frames × 10 MFCC coefficients = 490 values**, shape `(49, 10, 1)`. If “4910” appears in notes, that is a miscount of 49×10.

---

## 1. Model summary — before and after INT8 quantization

Architecture: **DS-CNN-S** (`training/model.py`), Hello Edge / ARM TinyML keyword net. Three softmax classes, in this order:

0. `keyword` (marvin)
1. `unknown`
2. `silence`

### 1.1 Keras graph (float32, what is trained)

Input: `(None, 49, 10, 1)`.

| Layer | Output shape | Params |
| --- | --- | --- |
| InputLayer `mfcc` | `(None, 49, 10, 1)` | 0 |
| Conv2D 64, kernel `(10, 4)`, stride `(2, 2)`, padding SAME, **no bias** | `(None, 25, 5, 64)` | 2,560 |
| BatchNormalization | `(None, 25, 5, 64)` | 256 |
| ReLU | | 0 |
| 4× block: ZeroPadding2D 1 px → DepthwiseConv2D 3×3 VALID, no bias → BN → ReLU → Conv2D 64 1×1 SAME, no bias → BN → ReLU | stays `(None, 25, 5, 64)` | 576 + 256 + 4,096 + 256 per block |
| GlobalAveragePooling2D | `(None, 64)` | 0 |
| Dense `logits` (3) | `(None, 3)` | 195 |
| Softmax `probs` | `(None, 3)` | 0 |

Parameter count of the Keras model (`build_dscnn_s().count_params()`):

| | Count | Bytes at float32 |
| --- | --- | --- |
| **Total** | **23,747** | 92.76 KB |
| Trainable | 22,595 | 88.26 KB |
| Non-trainable (BN moving stats) | 1,152 | 4.50 KB |

Export **folds every BN into the preceding conv** (`training/export_firmware.py` `_fold_conv_bn`) so the on-device graph has no BatchNorm ops. TFLM resolver registers exactly seven operators: `CONV_2D`, `DEPTHWISE_CONV_2D`, `PAD`, `RELU`, `MEAN`, `FULLY_CONNECTED`, `SOFTMAX`.

### 1.2 Float TFLite vs INT8 TFLite vs TFLM

| | Float32 TFLite | Full INT8 TFLite / TFLM |
| --- | --- | --- |
| File | `models/marvin.float.tflite` | `models/marvin.int8.tflite` = `KWS_MODEL[]` |
| Bytes | 92,548 | **43,152** (~2.1× smaller) |
| Input | `float32`, shape `[1, 49, 10, 1]`, no quant | `int8`, same shape |
| Output | `float32`, shape `[1, 3]` | `int8`, same shape |
| Input scale / zero-point | none | **scale = 0.6878264546394348**, **zero_point = 84** |
| Output scale / zero-point | none | **scale = 0.00390625** (= 1/256), **zero_point = −128** |
| Tensor count (interpreter) | 42 | 38 |

On-device dequant of the three class scores:

```
p = (q - out_zp) * out_scale
  = (q + 128) * 0.00390625
```

which maps int8 `q ∈ [-128, 127]` onto approximately `[0, 0.996]`. Class order in the output tensor is `[p_keyword, p_unknown, p_silence]`.

INT8 vs float parity on 200 representative clips (`models/marvin.int8_parity.json`):

| | |
| --- | --- |
| Argmax agreement | 0.995 (199 / 200) |
| Mean \|Δ p_keyword\| | 0.00612 |
| Max \|Δ p_keyword\| | 0.15769 |
| Calibrated on | `models/marvin.rep.npy` |

Quantization-aware training was **not** used. See §5.

---

## 2. Audio processing pipeline (every parameter)

Single source of truth: `shared/feature_spec.py`. Python (`node/features.py`) and C (`firmware/src/mfcc.cpp`) are required to match frame-for-frame. Tables in `firmware/include/mfcc_tables.h` are generated from the Python window, mel filterbank, DCT, and highpass SOS.

### 2.1 Time / frequency constants

| Parameter | Value | How it is derived |
| --- | --- | --- |
| Sample rate | **16,000 Hz** | `SPEC.sample_rate` |
| Frame / window length | **30 ms = 480 samples** | `window_ms=30` |
| Frame stride / hop | **20 ms = 320 samples** | `hop_ms=20` (hop **is** the stride) |
| Number of frames | **49** | `n_frames` |
| Clip length | **15,840 samples = 0.990 s** | `480 + 48 × 320` |
| FFT size | **512** | 480-sample window zero-padded to radix-2 |
| FFT bins (rfft) | **257** | `n_fft/2 + 1` |
| Mel filters | **40** | `n_mels` |
| MFCC coefficients | **10** | `n_mfcc` (includes C0) |
| Feature tensor | **(49, 10, 1)** | 490 floats |
| Mel fmin | **20 Hz** | |
| Mel fmax | **4,000 Hz** | well under Nyquist 8 kHz |
| Pre-emphasis | **0.97** | `x[0]` unchanged; `x[n] ← x[n] − 0.97·x[n−1]` for n≥1 |
| Window | **periodic Hann**, `scipy.signal.windows.hann(480, sym=False)` | table `MFCC_HANN[480]` |
| Log-mel floor | **1e-10** | `log(max(mel, 1e-10))` |
| DCT | orthonormal type-II: `√(2/n_mels)`, then C0 row `× √0.5` | `node/features.py` `_dct_matrix` |
| Highpass | 2nd-order Butterworth, **70 Hz**, SOS | Python `scipy.signal.sosfilt`; C biquad from the same taps |
| Highpass taps (exported) | B0 = 9.8075006225e-01, B1 = −1.9615001245e+00, B2 = 9.8075006225e-01, A1 = −1.9611295301e+00, A2 = 9.6187071893e-01 | `mfcc_tables.h` |
| Pi capture rate (legacy overlay) | 48,000 Hz, then exact 3:1 decimate | S3 path is native 16 kHz; no decimation |
| Int16 ↔ float | divide / multiply by **32768.0** on the node (`kInv16`); Python WAV load uses the same |

Mel scale: `mel = 2595 · log10(1 + f/700)`, 42 points from 20 Hz to 4000 Hz, triangular filters on FFT bins.

There is **no per-coefficient mean/variance normalization** and **no CMVN**. The only amplitude scaling before MFCC is:

1. hop DC removal + `MIC_GAIN = 12` (firmware capture),
2. `clip_match_train_level` / `match_train_level`: if clip RMS ≥ **0.008**, boost toward **0.10 FS**, max gain **10×**, never attenuate.

### 2.2 How the 49×10 (490) MFCC input is produced

Clip `x[0 … 15839]` at 16 kHz, already highpassed **from rest** (biquad state starts at zero; Python does this inside `load_wav_mono_16k`, C does it inside `mfcc_from_clip`).

For frame `i = 0 … 48`:

1. Take 480 samples starting at `i × 320`.
2. Pre-emphasize inside the window (sample 0 unchanged).
3. Multiply by the 480-point periodic Hann.
4. Zero-pad to 512.
5. Forward real FFT (numpy `rfft` in Python; in-place radix-2 FFT with numpy-matching sign `exp(−2πj kn/N)` in C).
6. Power spectrum `|X[k]|²` (C only computes bins below `MFCC_MAX_BIN`, the last bin any mel filter touches).
7. 40 triangular mel filters → 40 energies.
8. `log(max(e, 1e-10))`.
9. 10-point DCT-II → one row of 10 MFCCs.

Stack 49 rows, add a channel axis → `(49, 10, 1)`. Flattened length **490** (`MFCC_FEAT_LEN`). That tensor is the model input.

Firmware overlap trick: after the first 480 highpassed samples, each later frame keeps the last **160** samples (`WIN − HOP`) and highpasses only the new 320, so the biquad stays continuous and matches Python’s full-clip `sosfilt`.

---

## 3. Dataset composition

Source of the bulk of the data: **Google Speech Commands v0.02**, license **CC-BY-4.0**, ingested by `python -m training.ingest_speech_commands`.

Official **`testing_list.txt`** is the validation set (speaker-disjoint). Official **`validation_list.txt` is excluded from train** so that list stays unused, but it is **not** a third split. **There is no separate test set** in this repo.

### 3.1 On-disk Speech Commands ingest (`data/marvin/split.json`)

| Split | Keyword (marvin) | Unknown (other words) | Silence (`_background_noise_` slices) |
| --- | --- | --- | --- |
| Train | **1,905** | **5,000** (capped) | **639** |
| Val (`testing_list.txt`) | **195** | **10,810** | **159** |
| **On disk in `keyword/` / `unknown/` / `silence/`** | **2,100** | **15,810** | **798** |

- Keyword is the GSC word `marvin`. Short commands `yes/no/go/stop/on/off/up/down` are never used as the wake word; they remain unknowns.
- Train unknowns: random sample of 5,000 clips from the other words that are in neither official list.
- Silence: `_background_noise_` WAV files sliced at hop `CLIP_SAMPLES/2 = 7,920`, cap 2,000, then 20% of the kept slices go to val.

### 3.2 Real INMP441 recordings (`tools/record_s3.py`)

Recorded through the **ESP32-S3’s own INMP441**, USB serial `REC` mode, default clip **1,400 ms**, 16 kHz int16. Named `<label>_<speaker>_<idx>.wav`.

| Folder | WAV count | Speakers / conditions |
| --- | --- | --- |
| `keyword_real/` | **260** | `mayank` 180 (close-mic), `mayank1m` 80 (same speaker at ~1 m) |
| `unknown_real/` | **129** | `mayank` 129 |
| `silence_real/` | **40** | `mayank` 40 (room tone through the same mic) |
| `noise_real/` | **2** long files | `mayank` 1, `anon` 1 (TV / room; sliced into training unknowns) |
| `lookalike_real/` | **0** | not present |

These real files are **not** listed in `split.json` `val_files`, so they all go to **train**.

### 3.3 What the last training run actually saw (`models/marvin.metrics.json`)

`collect()` then:

- 2 extra **waveform-augmented copies of every train keyword** (so each train keyword clip becomes 3),
- up to **400** random `CLIP_SAMPLES` slices from `noise_real/` labelled **unknown**.

| Split | n | Keyword | Unknown | Silence |
| --- | --- | --- | --- | --- |
| **Train** | **12,703** | **6,495** | **5,529** | **679** |
| **Val** | **11,164** | **195** | **10,810** | **159** |
| Test | — | — | — | — |

Train keyword 6,495 = `(1,905 GSC + 260 real) × 3`.  
Train unknown 5,529 = `5,000 GSC + 129 real + 400 noise slices`.  
Train silence 679 = `639 GSC + 40 real`.

`n_real_speakers` recorded in metrics: **2,238** (GSC speaker hashes plus local names). Val is speaker-disjoint from train via `testing_list.txt`.

Recording conditions, in one sentence: GSC is crowd-sourced 16 kHz close-mic 1-second-ish words; local extras are this INMP441, this room, speaker `mayank`, close and ~1 m, plus two long background recordings.

---

## 4. Complete training pipeline

Entry: `python -m training.train` (defaults below).

### 4.1 Load / features

1. Read every WAV under the folders in `SOURCES`.
2. `load_wav_mono_16k` → float32 16 kHz + 70 Hz highpass.
3. Truncate or zero-pad to 15,840 samples.
4. `match_train_level` (RMS boost to 0.10 if ≥ 0.008).
5. `features_from_clip` → `(49, 10, 1)` float32.
6. Split: if `split.json` exists, membership in `val_files`; else hold out one speaker.

### 4.2 Waveform augmentation (train keyword only)

`augment_waveform` with `p_noise=0.9`, applied **twice** per train keyword clip:

| Op | Spec |
| --- | --- |
| Gain | uniform **0.08 … 1.15** (always) |
| Time roll | p = **0.7**, ±**1,600** samples |
| Time stretch | p = **0.35**, factor **0.88 … 1.12**, scipy resample, then crop/pad back to 15,840 |
| Additive noise | p = **0.9**, SNR **0 … 18 dB**, from `noise_real/` if present else pink-ish synthetic (+ 50 Hz hum with p=0.3) |
| Clip | `[−1, 1]` |

Val is never waveform-augmented.

### 4.3 Per-batch SpecAugment

`AugmentedBatches` + `tf.data` generator, regenerated every epoch.

Default (all classes, p=0.8):

- 2 time masks, width 0…8 frames, filled with the clip mean
- 1 frequency mask, width 0…2 coefficients, start ≥ 1 (C0 usually spared)

Keyword rows in the batch then get a second pass (p=1.0): 3 time masks max 10, 2 freq masks max 3.

Keyword indices are **concatenated once more** into the epoch order (2× oversample in the generator). `steps_per_epoch = len(x_train) // batch`.

### 4.4 Optimization

| Knob | Default |
| --- | --- |
| Optimizer | **Adam**, lr **2×10⁻³** (Keras defaults β1=0.9, β2=0.999) |
| Loss | **sparse_categorical_crossentropy** |
| Batch size | **32** |
| Epochs | **50** (may stop earlier) |
| Metrics | accuracy + custom **KeywordRecall** (TP / (TP+FN) on class 0) |
| Scheduler | `ReduceLROnPlateau(monitor=val_loss, factor=0.5, patience=6, min_lr=1e-5)` |
| Early stop / checkpoint | `BestWakeCallback`: ignore until `val_accuracy ≥ 0.90`, then keep weights with best `val_keyword_recall`; patience **12**; restore those weights on stop |
| Class weights | inverse frequency vs the majority class, then keyword **× 1.35** |

For the last run that produced `marvin.metrics.json`:

```
class_weight[keyword]  = 1.35          # 6495/6495 × 1.35
class_weight[unknown]  ≈ 1.175         # 6495/5529
class_weight[silence]  ≈ 9.566         # 6495/679
```

There is **no Keras `ModelCheckpoint` of every epoch**. The only saved weights are the restored best-recall snapshot written once to `models/marvin.keras`. Then `export_all` writes float/INT8 TFLite, MFCC tables, and `kws_model.h`.

### 4.5 Evaluation metrics (last run)

Threshold stored in `marvin.metrics.json` is **0.5** (the `WAKE_SCORE_THRESHOLD` at train time). Firmware now uses **0.52** (see §8). These numbers are **per-clip argmax / p_keyword**, not the 3-hit debounce used on the node.

| Split | n | Accuracy | Confusion (true\pred kw, unk, sil) | @thr 0.5 TPR | @thr 0.5 FAR |
| --- | --- | --- | --- | --- | --- |
| Train | 12,703 | 0.8307 | kw 4405/1843/247; unk 0/5469/60; sil 0/0/679 | 0.6778 | 0.0 |
| Val | 11,164 | **0.9920** | kw 152/42/1; unk 1/10764/45; sil 0/0/159 | **0.7795** | **0.0001** (1 FA / 10,969 negatives) |

Train accuracy is pulled down by the two augmented keyword copies; val is the speaker-disjoint GSC test list and is the number to quote.

---

## 5. Float32 → INT8 quantization

**Quantization-aware training is not used.** `training/qat.py` exits immediately: `tensorflow-model-optimization` 0.8 rejects Keras 3 Functional models. The flashed net is **post-training full-integer quantization**.

### 5.1 Converter settings (`export_tflite_fused`)

1. Fold BN into conv/depthwise/pointwise weights and biases.
2. Wrap as a `tf.function` with input `float32 [1, 49, 10, 1]`.
3. Convert **float** TFLite (no extra flags) → `marvin.float.tflite`.
4. Convert again with:

```
optimizations = [tf.lite.Optimize.DEFAULT]
target_spec.supported_ops = [TFLITE_BUILTINS_INT8]
inference_input_type = int8
inference_output_type = int8
representative_dataset = every row of marvin.rep.npy
```

### 5.2 Representative dataset

`save_representative` writes **300** MFCC tensors, `float32`, shape `(300, 49, 10, 1)`, file `models/marvin.rep.npy` (588,128 bytes). Preference order: train clips marked `real`, then fill from all train. Seed 0.

If that file is missing, export falls back to `N(0, 0.4)` noise and prints a warning. The current INT8 model **was** calibrated on `marvin.rep.npy`.

Calibration is **not adaptive on the device**. Scales and zero-points are frozen in the `.tflite` tensors. At boot, `kws_begin()` copies:

```
g_in_scale  = 0.6878264546394348
g_in_zp     = 84
g_out_scale = 0.00390625
g_out_zp    = -128
```

### 5.3 Quantize / dequantize on the S3

```
q = round(v / in_scale) + in_zp
q = clamp(q, -128, 127)     # int8 input
p = (out - out_zp) * out_scale
```

`v` is a float MFCC. This is the same affine map TFLite used during conversion.

---

## 6. ESP32 firmware architecture

Board: **ESP32-S3-DevKitC-1**, Arduino + ESP-IDF I2S, PlatformIO env `s3node`. CPU frequency is whatever the Arduino core boots (logged as `cpu=%u MHz`). PSRAM is enabled in the build flags but the **TFLM arena is internal SRAM**, 16-byte aligned, **64 KB** (`KWS_ARENA_BYTES`). Model weights stay in flash as `KWS_MODEL[]`.

### 6.1 Tasks, cores, stacks

Arduino `loopTask` is pinned to **core 1** (`ARDUINO_RUNNING_CORE`).

| Task | Core | Stack | Priority | Work |
| --- | --- | --- | --- | --- |
| `loopTask` (`loop()`) | **1** | **16,384** bytes (`ARDUINO_LOOP_STACK_SIZE`) | Arduino default | I2S hop, energy gate, ring, KWS1, TEL1, serial, OLED |
| `kws` FreeRTOS task | **0** (`KWS_TASK_CORE = 1 - ARDUINO_RUNNING_CORE`) | **8,192** | **2** | MFCC + TFLM `Invoke` |

Signalling: binary semaphore `g_infer_go`. Core 1 copies the ring into `g_infer_clip[15840]`, sets `g_infer_busy`, gives the semaphore. Core 0 runs `kws_infer_clip`, writes `g_infer_out`, sets `g_infer_fresh`, clears busy. One in-flight clip; **no queue**. If inference is still running, `infer_request()` returns false and that hop is skipped.

Idle listen path (energy gate closed): capture + RMS + `memmove` of the ring only. Target **< 10% of a 20 ms hop** of CPU, **< 256 KB RAM** for the listen node. Speech-state CPU (BENCH) is computed as `(mfcc + invoke) / (INFER_EVERY_HOPS × 20 ms)` with `INFER_EVERY_HOPS = 5`; the **live** path infers **as soon as the previous clip finishes**, not strictly every 5 hops.

### 6.2 Buffers and DMA

| Buffer | Size | Notes |
| --- | --- | --- |
| `g_ring` | 15,840 × int16 = 31,680 B | Sliding 0.99 s; newest hop at the end |
| `g_infer_clip` | same | Snapshot for core 0 |
| `g_i2s_zeros` / `g_i2s_rx` | 640 × int32 = 2,560 B each | One hop of stereo 32-bit |
| `g_hop_raw` | 320 × int32 | Chosen 24-bit samples |
| TFLM arena | 64 KB SRAM | High-water printed at boot (~56 KB in `docs/results.md`) |
| MFCC scratch | `g_win[480]`, `g_re/im[512]`, power, log-mel | static, not stack |

I2S DMA: `dma_buf_count = 4`, `dma_buf_len = 256`, interrupt `ESP_INTR_FLAG_LEVEL1`. Dummy TX descriptor auto-clear. The listen-loop CPU figure **subtracts** time blocked in `i2s_read`/`i2s_write`.

### 6.3 Boot and loop cadence

`setup()`: clear ring → I2S → OLED → `kws_begin()` → one warm-up inference on silence → create `kws` task.

`loop()`: poll serial → capture 20 ms hop → ring push → (optional REC/MIC) → Wi-Fi → energy gate → stream or request inference → consume `g_infer_fresh` → TEL1/OLED once per second → if the hop took < 18 ms, `delayMicroseconds` the remainder so the 20 ms grid stays aligned.

### 6.4 Wake state machine (firmware)

States implied by `state` string and flags:

```
listen  --energy 3 hops-->  speech  --3 consecutive KWS hits-->  wake (KWS1 open)
   ^                         |                                      |
   |                    energy drops /                              |
   |                    refractory                                  |
   +----- stream end (3.5 s quiet or 10 s timeout) -----------------+
```

- `listen`: CNN off if not `speechy`.
- `speech`: ring full, not in refractory, energy gate open → `infer_request` every hop that is not busy. `kws_reset()` on speech **onset** so a buffer of zeros cannot pull the smoother down.
- `wake` / `g_streaming`: hop PCM goes to KWS1; detector not used for a new wake until hang-up.
- After hang-up: `kws_reset()`, ring zeroed, energy hits zeroed, **refractory 1,800 ms**.

OLED I2C: SDA **GPIO 8**, SCL **GPIO 9**, SSD1306 128×64 addr `0x3C`, optional (firmware continues if missing).

Wi-Fi: STA, `setSleep(true)`, ASR host from `wifi_secrets.h` with `.local` mDNS fallback. Credentials are **not** repeated in this document.

---

## 7. INMP441 I2S configuration

Header GPIO numbers (`firmware/include/pins.h`):

| INMP441 pin | ESP32-S3 GPIO |
| --- | --- |
| VDD | 3V3 (3.3 V only) |
| GND | GND |
| L/R | **GND** (left channel) |
| SCK / BCLK | **15** |
| WS / LRCK | **16** |
| SD / DOUT | **17** |
| (dummy I2S DOUT, required because the driver is RX+TX) | **11** |

`init_i2s()` (`firmware/src/main.cpp`):

| Field | Value |
| --- | --- |
| Port | `I2S_NUM_0` |
| Mode | `MASTER \| RX \| TX` |
| Sample rate (WS) | **16,000 Hz** |
| Slot width | **32-bit** (`I2S_BITS_PER_SAMPLE_32BIT`, `I2S_BITS_PER_CHAN_32BIT`) |
| Channel format | `I2S_CHANNEL_FMT_RIGHT_LEFT` (stereo) |
| Channels | 2 (`I2S_TDM_ACTIVE_CH0 \| CH1`) |
| Communication | `I2S_COMM_FORMAT_STAND_I2S` (Philips I2S) |
| Left-align | **true** |
| APLL | **false** (PLL clock) |
| MCLK multiple | **256** → MCLK = 16,000 × 256 = **4.096 MHz** if generated; INMP441 does not need MCLK |
| BCLK | 16,000 × 32 bits × 2 channels = **1.024 MHz** (64 SCK pulses per WS, as INMP441 requires) |
| DMA | 4 buffers × 256 frames |
| TX | dummy zeros written every hop so the RX DMA stays clocked |

INMP441 puts **24-bit PCM left-justified in a 32-bit slot**.

### 7.1 Sample conversion before MFCC (every hop)

1. `i2s_write` 640 int32 zeros (320 stereo frames).
2. `i2s_read` 640 int32.
3. For each stereo pair: arithmetic `>> 8` to 24-bit, take the slot with **larger absolute value** (left if L/R=GND, but this survives packing/pin mistakes).
4. Subtract the hop **mean** (DC block).
5. Multiply by **`MIC_GAIN = 12`**, clamp to 24-bit range ±2²³.
6. Pack to int16 with `>> 8`.
7. RMS for the energy gate is computed on the 24-bit-after-gain samples as `sqrt(mean((s/8388608)²))` — **not** on the int16 values.
8. 320 int16 samples are shifted into `g_ring`.
9. Before inference only: copy ring, then `clip_match_train_level` (RMS 0.008 floor, target 0.10, max 10×).
10. `mfcc_from_clip`: `int16 / 32768` → 70 Hz highpass from rest → 49×10 MFCC.

---

## 8. Energy gate

The gate does **not learn**. The threshold is a compile-time constant. It is evaluated **every 20 ms hop**.

| Parameter | Value |
| --- | --- |
| Signal | hop RMS after DC-block and ×12, on the 24-bit / 8,388,608 scale |
| Threshold | **`ENERGY_RMS_THRESHOLD = 0.003`** |
| Window | **one hop = 320 samples = 20 ms** (no longer smoother) |
| Consecutive | **`ENERGY_CONSECUTIVE = 3`** |

Algorithm (`loop()`):

```
if rms > 0.003:  hits = min(3, hits + 1)
else:            hits = max(0, hits - 1)
speechy = (hits >= 3)
```

So the CNN turns on after **3 consecutive hops above 0.003** (60 ms of energy) and turns off on the **first hop** that drops `hits` below 3. Idle hops never call `infer_request`.

---

## 9. Wake decision logic

All knobs live in `firmware/include/pins.h` and are mirrored in `shared/config.py` for the Python tools. **These are the deployed values.** (The last `marvin.metrics.json` still records smoothing window 2 / debounce 2 / thr 0.5 from that training run.)

### 9.1 Per-inference score

After TFLM softmax dequant:

- Push `p_keyword` into a circular buffer of length **`SMOOTH_WINDOW = 3`**.
- `score = mean` of the last `n` values (`n` grows from 1 to 3). On speech onset and after a stream, `kws_reset()` clears this buffer.

### 9.2 Hit / miss

A frame is a **hit** iff **both**:

1. `score >= WAKE_SCORE_THRESHOLD` (**0.52**)
2. `p_keyword >= p_unknown + KWS_MARGIN` (**margin 0.08**)

Otherwise `hits = 0` (consecutive counter resets).

### 9.3 Debounce

`awake = (hits >= DEBOUNCE_HITS)` with **`DEBOUNCE_HITS = 3`**.

Because inference runs ~every 60–70 ms while speechy (MFCC ~30 ms + invoke ~36 ms, no queue), three consecutive hits is roughly **0.2 s** of sustained keyword posterior, not three 20 ms hops.

### 9.4 False-trigger prevention

| Mechanism | What it does |
| --- | --- |
| Energy gate | CNN off in silence / hiss |
| Margin 0.08 vs unknown | TV/command spikes that light `unknown` as well as `keyword` do not count |
| 3-hit debounce | single-frame spikes cannot wake |
| Speech-onset smoother reset | leading silence zeros cannot dilute the average |
| Refractory **1,800 ms** after hang-up (or failed TCP connect) | no immediate re-wake |
| Keyword audio never leaves the chip | KWS1 opens **after** `awake` |

### 9.5 Wake output and hang-up

On `r.awake` and not already streaming and out of refractory:

1. `WAKE t=… score=… pkw=… pun=… psil=…` on USB serial.
2. OLED mood = wake.
3. TCP connect to ASR **port 8765**, `TCP_NODELAY`.
4. KWS1 header + **500 ms preroll** (last **8,000** samples of the ring = 25 hops).
5. Every subsequent hop is streamed until:
   - hop RMS (same energy-gate scale) stays ≤ **`STREAM_VOICE_RMS = 0.028`** for **`SILENCE_END_MS = 3,500`**, or
   - **`STREAM_TIMEOUT_MS = 10,000`** since stream start.
6. Send end-of-stream, close TCP, `kws_reset()`, zero the ring, start refractory.

A hop with RMS > 0.028 refreshes `g_last_voice_ms`. Room noise after ×12 sits ~0.01–0.02 FS, so 0.028 is above hiss and below speech.

---

## 10. KWS1 TCP protocol

Defined in `shared/protocol.py`; written by `stream_begin` / `stream_hop` / `stream_end`. Little-endian throughout. **No timestamp field. No sequence number. No message-type enum.** Framing is length-prefixed PCM after a one-time header.

Transport: **TCP**, node → Raspberry Pi, **port 8765**. Magic ASCII `KWS1`.

### 10.1 Sequence

```
ESP32                          Pi (asr_server.handle_client)
  |  TCP connect :8765              |
  |  (reachability probe: connect + close, no header — Pi treats as probe)
  |                                 |
  |  11-byte header                 |
  |  N frames: uint16 nbytes + PCM  |  drop first preroll_ms of audio from ASR
  |  ... live hops ...              |  feed remaining PCM to Vosk
  |  uint16 0  (end)                |
  |  close                          |  transcriber.finish()
```

### 10.2 Header (11 bytes)

`struct.Struct("<4sBHI")` = `HEADER_BYTES = 11`.

| Offset | Type | Field | Deployed value |
| --- | --- | --- | --- |
| 0 | 4s | magic | `KWS1` (`0x4B 0x57 0x53 0x31`) |
| 4 | uint8 | version | **1** |
| 5 | uint16 LE | preroll_ms | **500** |
| 7 | uint32 LE | sample_rate | **16000** |

Reject if magic ≠ `KWS1` or version ≠ 1.

### 10.3 Frames

Each audio frame:

| Field | Type | Meaning |
| --- | --- | --- |
| nbytes | uint16 LE | payload length in bytes |
| payload | nbytes bytes | **int16 little-endian PCM**, mono, 16 kHz |

Live hops: `nbytes = 640` (320 samples × 2 bytes). Preroll: 25 such frames (8,000 samples = 500 ms).

**Termination:** `nbytes = 0`, no payload. That is the only end marker. There is no checksum.

**Sample format:** signed 16-bit PCM, native endian of the node which is little-endian, already after DC-block, ×12, and `>>8` pack — **not** re-highpassed for the stream. The Pi does **not** run MFCC on this audio; it is ASR input only.

**Frame type:** there is one type (PCM). Preroll vs live is distinguished only by order (preroll is sent immediately after the header, before the next captured hops). The server skips `preroll_ms × sample_rate × 2` bytes so Vosk does not transcribe the wake word itself.

### 10.4 TEL1 (not part of KWS1)

UDP **port 8766**, ASCII line once per second, prefix `TEL1`, fields `state cpu idle heap rssi rms kw mfcc_us inv_us arena wakes lat_ms`. Used by the Pi UI and by `tools/bench.py`. Independent of the TCP audio stream.

---

## 11. File map (quick)

| Path | What it is |
| --- | --- |
| `models/marvin.keras` | Float Keras checkpoint |
| `models/marvin.float.tflite` | Float TFLite, 92,548 B |
| `models/marvin.int8.tflite` | INT8 TFLite, 43,152 B |
| `firmware/include/kws_model.h` | Same INT8 bytes for TFLM |
| `models/marvin.rep.npy` | 300 MFCCs for PTQ calibration |
| `models/marvin.int8_parity.json` | INT8 vs float agreement |
| `models/marvin.metrics.json` | Last train/val report |
| `data/marvin/split.json` | GSC ingest counts + val file list |
| `shared/feature_spec.py` | MFCC recipe |
| `shared/protocol.py` | KWS1 header/frame packing |
| `firmware/include/pins.h` | I2S pins, energy, wake, I2S-side knobs |
| `firmware/src/main.cpp` | Listen loop, I2S, KWS1, energy gate |
| `firmware/src/kws.cpp` | Quantize, TFLM, smoother, debounce |
| `firmware/src/mfcc.cpp` | Highpass + MFCC |

---

## 12. What this document does not claim

- No `.tlf` file exists.
- Input length is **490**, not 4910.
- Quantization-aware training is **not** in the training path that produced the flashed model.
- KWS1 has **no** timestamps or sequence numbers.
- The energy gate **does not adapt**; 0.003 is fixed.
- Idle RAM/CPU quotas apply to the **ESP32-S3 listen path**, not to the Raspberry Pi ASR/UI.
- ASR after wake is **offline Vosk on the Pi** (LAN), not a public cloud service.
