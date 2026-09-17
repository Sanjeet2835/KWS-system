# Source of Truth — SIH Problem Statement 26172

This file is the **authoritative reference** for every design decision in this repo.
When a design choice is questioned, it is resolved against the verbatim text below,
not against anything written elsewhere in these docs.

Rule: **do not paraphrase the PS into a requirement.** If a constraint is not in the
verbatim text, it is a team-chosen target and must be labelled as such (see
[Derived targets](#derived-targets-chosen-by-us-not-stated-by-the-ps)).

---

## Verbatim text

> **Description**
>
> Build an ultra-lightweight, highly accurate keyword spotting (KWS) model that runs
> locally on a low-power device. Upon detecting the keyword, the system must instantly
> and efficiently stream the subsequent audio to a remote Automated Speech Recognition
> (ASR) server with minimal data overhead and latency.
>
> **Key Metrics for Evaluation**
>
> - Efficiency: Model size (RAM/Flash footprint) and CPU usage during idle listening.
> - Accuracy: High true-positive rate for the keyword with near-zero false activations.
> - Latency: The time delta between the keyword ending and the cloud ASR receiving the audio stream.
>
> **Software & Framework Restrictions**
>
> - Open-Source Only: The use of proprietary, closed-source, or commercial voice-activation SDKs is strictly prohibited.
> - Allowed Frameworks: Teams must build their keyword spotting (KWS) pipelines using open-source machine learning and TinyML frameworks. Recommended tools include TensorFlow Lite for Microcontrollers, PyTorch Mobile or similar.
> - No Pre-Trained Global Keywords: Teams cannot use models pre-trained on generic smart-assistant keywords like 'Hey Google' or 'Alexa'. They need to train on a custom key word.

---

## What the PS actually mandates

Read literally, there are seven binding requirements and **zero numeric thresholds**.

| # | Requirement (binding) | Where satisfied |
|---|---|---|
| R1 | KWS model runs **locally** on a low-power device | `node/` runs the INT8 model on-device; no audio leaves the node before a detection |
| R2 | Model is **ultra-lightweight** and **highly accurate** | DS-CNN-S, full INT8 quantization |
| R3 | On detection, **stream subsequent audio** to a **remote** ASR server | `node/` opens a socket to `server/asr_server.py` on wake |
| R4 | Streaming is **instant**, with **minimal data overhead** | 16 kHz mono int16, 20 ms frames, no re-encode; connection opens on wake |
| R5 | **Open-source only**; no proprietary/commercial voice-activation SDK | TFLite/TFLM, Vosk, Piper. See [prohibited list](#explicitly-prohibited-in-this-repo) |
| R6 | Built on open-source ML / TinyML frameworks | TensorFlow Lite for Microcontrollers |
| R7 | **Custom keyword**, not a pretrained global one | "Sahayak", trained by us from synthetic + augmented data |

And three scored metrics:

| Metric | Exact PS wording | What we must be able to report |
|---|---|---|
| Efficiency | "Model size (RAM/Flash footprint) and CPU usage during idle listening" | Two numbers: footprint, and idle-listening CPU |
| Accuracy | "High true-positive rate ... with near-zero false activations" | TPR, and false activations per unit time |
| Latency | "time delta between the keyword ending and the cloud ASR receiving the audio stream" | Keyword-end timestamp to ASR-receipt timestamp, one clock |

### Note on the latency definition

The PS defines latency from **keyword ending**, not from detection. These differ: the
detector needs a trailing window after the keyword before it can fire. That gap is part
of the measured number and cannot be hidden. `server/latency_harness.py` must timestamp
the true end of the keyword audio, not the moment the detector fires.

The PS says "cloud ASR". Our ASR server is remote-over-network but LAN-hosted. This is a
deliberate, disclosed deviation: it keeps the demo functional without internet, and the
measured delta is network-path-honest for the node-to-server hop. State it openly rather
than implying internet transit.

---

## Official efficiency limits (from the official PS)

Copied from SIH 2026 Expected Solution on [sih.gov.in/sih2026PS](https://sih.gov.in/sih2026PS)
(PS **SIH26172**, ISRO):

> The edge software application must run smoothly within an environment restricted
> to less than 256KB of RAM and consume under 10% CPU utilization **while idling
> in continuous listening mode**.

Those two numbers apply **only to idle listen**. After the keyword, the PS wants
the node to stream audio to **cloud / remote ASR** — that hop is allowed to use
more CPU. Do not hold the stream path to 10%.

| Limit | Value | When it applies |
|---|---|---|
| KWS RAM | **< 256 KB** | Listen / infer path on the node (arena + buffers). Not Linux. Not PSRAM. |
| Idle CPU | **< 10%** | Quiet continuous listening, before the keyword. Not speech. Not streaming. |

The device that can satisfy these literally is the **ESP32-S3**. The Pi 4B is the ASR
server and the internal-round demo; on the Pi we still report **subsystem** RAM only.

## Derived targets (ours, not in the PS)

| Target | Value | Why |
|---|---|---|
| False activations | < 1 per hour of non-keyword audio | Operationalizes "near-zero" |
| Keyword-end to ASR receipt | < 500 ms median | Perceptually "instant" |

---

## Explicitly prohibited in this repo

Direct consequences of R5 and R7. Any of these in the detection path is disqualifying.

- Cloud STT used for wake detection — `speech_recognition.recognize_google`, Azure Speech, AWS Transcribe
- Commercial wake-word SDKs — Picovoice Porcupine (non-free tiers), Snowboy, Sensory TrulyHandsfree, Alexa/Google wake engines
- Any pretrained model whose keyword is a generic assistant phrase
- Detecting the keyword by **string-matching a transcript** from any ASR. This is the
  anti-pattern: it means audio left the device before detection, there is no model to
  size, and no idle-CPU story. See the prior-art note in the plan.

Permitted, and what we use: TensorFlow / TFLite / TFLM (Apache-2.0), Vosk (Apache-2.0),
Piper TTS (MIT), Speech Commands v2 (CC-BY-4.0), MUSAN / ESC-50 for noise.

---

## The one disclosed caveat

The Pi 4B build cannot honestly report a KB-scale whole-system footprint; it is a Linux
SBC. On the Pi we report **KWS subsystem** footprint (`arena_used_bytes()` plus
preallocated buffers) and **pinned-core** idle CPU, and we say plainly that the host is a
reference implementation. Phase G moves the node to an ESP32-S3, where the efficiency
metric becomes fully and literally satisfiable.

Overclaiming here is the single fastest way to lose credibility in Q&A. Do not do it.
