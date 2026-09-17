#pragma once

/*
 * ESP32-S3-DevKitC (not Nano). Use the GPIO numbers printed on the header.
 *
 * INMP441 (3.3 V only). Moved off 4/5/6 after those pins stayed silent:
 *   VDD -> 3V3
 *   GND -> GND
 *   L/R -> GND
 *   SCK / BCLK -> GPIO 15
 *   WS  / LRCK -> GPIO 16
 *   SD  / DOUT -> GPIO 17
 *
 * SSD1306 128×64 OLED: SDA GPIO 8, SCL GPIO 9 (optional; firmware degrades)
 * HW-104 stays on the Pi jack.
 */

#define PIN_I2S_SCK 15
#define PIN_I2S_WS 16
#define PIN_I2S_SD 17

#define PIN_OLED_SDA 8
#define PIN_OLED_SCL 9

#define I2S_PORT I2S_NUM_0
#define SAMPLE_RATE 16000
#define HOP_SAMPLES 320
#define CLIP_SAMPLES 15840
#define PREROLL_SAMPLES 8000
#define ENERGY_RMS_THRESHOLD 0.003f
#define ENERGY_CONSECUTIVE 3
#define INFER_EVERY_HOPS 5
#define REFRACTORY_MS 1800
// Hang up after this much quiet. Room noise sits ~0.01–0.02 FS after gain 12,
// so STREAM_VOICE_RMS must be higher or last_voice never expires (12 s timeout).
#define SILENCE_END_MS 3500
#define STREAM_VOICE_RMS 0.028f
#define STREAM_TIMEOUT_MS 10000

/*
 * 0.52 / 3 / 3 plus a keyword-vs-unknown margin: marvin still crosses for ~0.5 s,
 * one-frame TV/command spikes do not. Do not go back to 0.55 — that missed 1 m.
 */
#define WAKE_SCORE_THRESHOLD 0.52f
#define SMOOTH_WINDOW 3
#define DEBOUNCE_HITS 3
#define KWS_MARGIN 0.08f

// After DC-block. 12x: 1 m speech lands near the close-mic MFCCs the net saw.
#define MIC_GAIN 12

// TFLM scratch. The model's tensors need ~50 KB and the interpreter's own
// bookkeeping sits in here too. Boot prints the high-water mark; trim to it.
#define KWS_ARENA_BYTES (64 * 1024)
