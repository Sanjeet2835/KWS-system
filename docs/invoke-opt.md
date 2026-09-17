# Invoke-time optimization (timeboxed)

Steady-state TFLM invoke was **36.1 ms** on the S3 (`results/bench.json`). The hop
budget is 100 ms (`INFER_EVERY_HOPS=5` × 20 ms), so this is not a demo blocker.
The target for Q&A is **< 15 ms**.

## What we changed

- DS-CNN-S depthwise layers are now `ZeroPadding2D(1) + DepthwiseConv2D VALID`
  instead of `padding="same"`. Same geometry; the converted graph is PAD +
  DEPTHWISE_CONV, which ESP-NN's S3 SIMD kernels can take. SAME-padded
  depthwise on the odd 25×5 map was the suspected reference-kernel fallback.
- Firmware already registers `AddPad()` (`firmware/src/kws.cpp`).
- Profile build: `pio run -e s3profile` (`-DKWS_PROFILE=1`) logs per-op
  timings via TFLM `MicroProfiler` on each `BENCH` / invoke. Off in the demo
  build so the timer reads do not inflate idle CPU.

## Fallback

Keep the 36 ms SAME-padding model (`models/marvin.int8.tflite` from the first
Speech Commands train) if the PAD+VALID retrain is slower or disagrees with
float. Do not flash a slower graph.

## After the retrain (2026-08-31)

Flashed PAD+VALID INT8. `tools.bench` invoke **36.150 ms** vs previous **36.132 ms**. No win.
Idle CPU **9.3%** mean / **9.5%** max. Keep the graph (same hop budget); do not claim sub-15 ms.
Profile env: `pio run -e s3profile` (not default — `default_envs = s3node`).
