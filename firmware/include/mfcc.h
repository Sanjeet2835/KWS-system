#pragma once

#include <stdint.h>

// Feature dimensions, generated alongside the tables from shared/feature_spec.py.
#include "mfcc_shape.h"

// Fill out[MFCC_FEAT_LEN] from CLIP_SAMPLES int16 samples. Applies the same
// highpass, pre-emphasis, window, FFT and Log-Mel steps as node/features.py; see
// tools/mfcc_parity.py for the test that keeps the two in agreement.
void mfcc_from_clip(const int16_t *clip, float *out);
