#include "mfcc.h"

#include <math.h>
#include <string.h>

#include "mfcc_tables.h"

// Everything in this file must match node/features.py frame for frame. The
// tables in mfcc_tables.h are generated from that Python, so the only thing
// that can drift is the arithmetic below.

static const float kInv16 = 1.0f / 32768.0f;

// Scratch. Static, not stack: the inference task runs with a small stack and
// these are far too big to live on it.
static float g_win[MFCC_WIN];       // highpassed audio for the current frame
static float g_re[MFCC_NFFT];
static float g_im[MFCC_NFFT];
static float g_power[MFCC_MAX_BIN];
static float g_logmel[MFCC_N_MELS];

// --- 2nd-order Butterworth highpass, taps from scipy via export_firmware ---

struct Biquad {
  float x1, x2, y1, y2;
};

static void hp_reset(Biquad *s) {
  s->x1 = s->x2 = s->y1 = s->y2 = 0.0f;
}

static inline float hp_step(Biquad *s, float x) {
  const float y = HP_B0 * x + HP_B1 * s->x1 + HP_B2 * s->x2 - HP_A1 * s->y1 - HP_A2 * s->y2;
  s->x2 = s->x1;
  s->x1 = x;
  s->y2 = s->y1;
  s->y1 = y;
  return y;
}

// --- in-place radix-2 FFT, forward sign, natural-order input ---

static void fft_forward(float *re, float *im) {
  for (int i = 0; i < MFCC_NFFT; ++i) {
    const int j = FFT_BITREV[i];
    if (j > i) {
      float t = re[i];
      re[i] = re[j];
      re[j] = t;
      t = im[i];
      im[i] = im[j];
      im[j] = t;
    }
  }
  for (int len = 2; len <= MFCC_NFFT; len <<= 1) {
    const int half = len >> 1;
    const int step = MFCC_NFFT / len;
    for (int base = 0; base < MFCC_NFFT; base += len) {
      int tw = 0;
      for (int k = 0; k < half; ++k, tw += step) {
        const float wr = FFT_TWIDDLE_RE[tw];
        const float wi = FFT_TWIDDLE_IM[tw];
        const int a = base + k;
        const int b = a + half;
        const float tr = re[b] * wr - im[b] * wi;
        const float ti = re[b] * wi + im[b] * wr;
        re[b] = re[a] - tr;
        im[b] = im[a] - ti;
        re[a] += tr;
        im[a] += ti;
      }
    }
  }
}

static void frame_to_mfcc(const float *win, float *row) {
  // Pre-emphasis inside the window, exactly as mfcc_frame() does: sample 0 is
  // left alone, the rest get x[n] - 0.97*x[n-1]. Written into the FFT buffer so
  // `win` stays clean for the next frame, which overlaps this one by 160.
  g_re[0] = win[0] * MFCC_HANN[0];
  g_im[0] = 0.0f;
  for (int n = 1; n < MFCC_WIN; ++n) {
    g_re[n] = (win[n] - MFCC_PREEMPH * win[n - 1]) * MFCC_HANN[n];
    g_im[n] = 0.0f;
  }
  // Zero-pad 480 -> 512 so the transform is radix-2.
  memset(g_re + MFCC_WIN, 0, (MFCC_NFFT - MFCC_WIN) * sizeof(float));
  memset(g_im + MFCC_WIN, 0, (MFCC_NFFT - MFCC_WIN) * sizeof(float));

  fft_forward(g_re, g_im);

  // Only bins below MFCC_MAX_BIN feed a mel filter (fmax_hz is well under
  // Nyquist), so squaring the rest would be wasted work.
  for (int k = 0; k < MFCC_MAX_BIN; ++k) {
    g_power[k] = g_re[k] * g_re[k] + g_im[k] * g_im[k];
  }

  // Sparse filterbank: each mel band is a short triangular span, stored as
  // (start, len) into a packed weight array. See _sparse_fbank() in
  // training/export_firmware.py.
  const float *w = MFCC_FBANK_W;
  for (int m = 0; m < MFCC_N_MELS; ++m) {
    const int start = MFCC_FBANK_START[m];
    const int len = MFCC_FBANK_LEN[m];
    const float *p = g_power + start;
    float acc = 0.0f;
    for (int k = 0; k < len; ++k) {
      acc += w[k] * p[k];
    }
    w += len;
    g_logmel[m] = logf(acc < 1e-10f ? 1e-10f : acc);
  }

  for (int c = 0; c < MFCC_N_MFCC; ++c) {
    const float *dct = MFCC_DCT + c * MFCC_N_MELS;
    float acc = 0.0f;
    for (int m = 0; m < MFCC_N_MELS; ++m) {
      acc += dct[m] * g_logmel[m];
    }
    row[c] = acc;
  }
}

void mfcc_from_clip(const int16_t *clip, float *out) {
  Biquad hp;
  hp_reset(&hp);

  // The highpass is a streaming filter but Python runs it per clip from rest,
  // so we do the same and reuse the filtered tail across overlapping frames.
  for (int n = 0; n < MFCC_WIN; ++n) {
    g_win[n] = hp_step(&hp, (float)clip[n] * kInv16);
  }
  frame_to_mfcc(g_win, out);

  const int keep = MFCC_WIN - MFCC_HOP;  // 160 samples shared with the next frame
  for (int f = 1; f < MFCC_N_FRAMES; ++f) {
    memmove(g_win, g_win + MFCC_HOP, keep * sizeof(float));
    const int16_t *src = clip + MFCC_WIN + (f - 1) * MFCC_HOP;
    for (int n = 0; n < MFCC_HOP; ++n) {
      g_win[keep + n] = hp_step(&hp, (float)src[n] * kInv16);
    }
    frame_to_mfcc(g_win, out + f * MFCC_N_MFCC);
  }
}
