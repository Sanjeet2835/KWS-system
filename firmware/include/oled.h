#pragma once

#include <stdint.h>

// SSD1306 128×64 on PIN_OLED_SDA / PIN_OLED_SCL. Safe if the panel is missing:
// oled_begin() returns false and every later call is a no-op.

bool oled_begin();
bool oled_present();

// mood: 0 sleep, 1 speech, 2 awake. Redraws only when mood/score/cpu change
// or when force is true. Keep this off the 20 ms hop except at 1 Hz.
void oled_show(int mood, float kw_score, float cpu_pct, bool force);
