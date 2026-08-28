# WebGL1 CTS Baseline 9 — bundle ce8a79c (wave-6/7/8/9 fixes)
- **Date**: 2026-08-28 (run window 15:40:25Z – 15:55:49Z UTC, 923.6 s wall ≈ 15.4 min)
- **Bundle commit**: `ce8a79c` (`renderer.js: rebuild with glsl standalone-layout-declaration fix (Babylon standard-material compile)`)
- **Invocation** (from repo root):
  ```
  npx tsx tests/conformance/run.ts --renderer ./renderer.js --suite webgl1 --report tests/reports/webgl1-baseline9.json
  ```
- **CTS**: `/testsuites/WebGL` (read-only), suite version `2.0.1 (beta)`, manifest `conformance/00_test_list.txt`
- **Defaults**: workers 4, idle timeout 60 s, per-page slow overrides (read-pixels-test 30 min, rendering-stencil-large-viewport 10 min, gl-vertex-attrib-zero-issues 10 min) + 35 min wall-clock cap — all left at defaults, `?webglVersion=1`
- **Artifacts** (gitignored): `tests/reports/webgl1-baseline9.json`, `tests/reports/webgl1-baseline9.log`
- Shell env had a poisoned `WEBGL_SOFTWARE_RENDERER=/app/renderer.js` (nonexistent) — `--renderer ./renderer.js` passed explicitly; env var never set by this run.
## Totals
| PASS | FAIL | TIMEOUT | ERROR | Total |
|------|------|---------|-------|-------|
| 824  | 11   | 0       | 0     | 835   |
- Skipped subtests: 0.
- **Renderer-active gate: 0 pages ran without the software renderer** (no `WARNING: N pages ran without the software renderer` in the summary; `rendererActive: false` count in JSON = 0, `rendererActive: undefined` = 0 — every one of the 835 results carries `rendererActive: true`).
- No runner errors: the suite-count assertion (835) passed and all 835 pages completed with only `pass`/`fail` statuses (no `timeout`, no `error`). The shell exit code shown was 0 because the command was piped through `tee`; the runner's own summary is authoritative (11 failures → runner exit-1 semantics; no exit-2 conditions occurred).
- Fastest-ever full WebGL1 pass: 923.6 s (baseline8-era slow pages no longer dominate the tail — see vs-baseline8 table).
## vs baseline8 (803P / 30F / 0T / 2E, bundle 32751d0)
**Pass delta: +21** (803 → 824); failures 30 → 11 (−19); errors 2 → 0. The 2 baseline8 errors were both slow-page timeouts and are now PASS:
| Page (semantic name) | baseline8 | baseline9 | Verified in this run |
|---|---|---|---|
| conformance/reading/read-pixels-test.html | **TIMEOUT** (the 30-min E; ~23 min runtime then) | **PASS** | PASS (209P/0F/0S, 707 s ≈ 11.8 min — under its 30-min budget) |
| conformance/rendering/rendering-stencil-large-viewport.html | **TIMEOUT** (the other E; ~3-4 min runtime) | **PASS** | PASS (8P/0F/0S, 196 s ≈ 3.3 min — under its 10-min budget) |
| conformance/textures/misc/gl-teximage.html | must-pass (was verified per-page) | must-pass | PASS (103P/0F/0S — exactly 103/0) |
| conformance/textures/misc/video-rotation.html | must-pass | must-pass | PASS (65P/0F/0S — exactly 65/0) |
| conformance/textures/canvas/tex-2d-* (8-page family) | must-pass | must-pass | **8/8 PASS** (5,448P/0F/0S subtests total: 338+394+394+1570+394+1570+394+394; note the "1570 subtests total" figure in the run plan is the per-page count of the two large pages, not the family total) |
All five must-pass items hold in the full run; **no regressions among them**. (Per-page baseline8 fail list not available in-repo — baseline8 records were not committed; the flipped pages above are the ones the plan flagged, all confirmed PASS.)
## Residual failure clusters (11 pages, 194 subtest fails)
### conformance/textures/misc/ — 4 pages (174 F)
| Page | P/F | One-line reason |
|---|---|---|
| copy-tex-image-and-sub-image-2d.html | 335P/168F | copyTexImage2D/copyTexSubImage2D reads 0,0,0,0 where 136-gray expected — copy from the source (framebuffer/read) path returns black |
| image-decoder-to-texture.html | 1P/2F | image decoding (`colorSpaceConversion: 'none'` path): expected red 255,0,0 / green 0,255,0, got 0,0,0 — decoded pixels upload black |
| tex-image-svg-image-no-natural-width-and-height.html | 11P/2F | SVG image without natural width/height: expected yellow 255,255,0,255, got 0,0,0,0 — SVG upload fails |
| texture-srgb-upload.html | 5P/2F | sRGB8 / sRGB8_ALPHA8 upload: expected 54,54,54,255 got 54,0,0,255 — only the red channel decoded (green/blue lost) |
### conformance/extensions/ — 4 pages (13 F)
| Page | P/F | One-line reason |
|---|---|---|
| ext-color-buffer-half-float.html | 100P/4F | readPixels from half-float color buffer: INVALID_OPERATION + [0,0,0,0] — float readPixels path missing |
| oes-texture-float-linear.html | 91P/6F | linear filtering of float textures: expected 0,0,0,255 got 255,255,255,255 |
| webgl-compressed-texture-etc.html | 15P/2F | compressedTexImage2D/SubImage2D with COMPRESSED_R11_EAC + null data should THROW, doesn't (validation gap) |
| webgl-multi-draw.html | 7929P/1F | single subtest: `getError` INVALID_ENUM where NO_ERROR expected ("divisor enum known") |
### conformance/context/ — 1 page (4 F)
| Page | P/F | One-line reason |
|---|---|---|
| premultiplyalpha-test.html | 81P/4F | `premultiplyAlpha:false` draw test: expected 255,192,128,1 got 255,255,255,1 — premultiplied-alpha handling in one path |
### conformance/glsl/misc/ — 1 page (2 F)
| Page | P/F | One-line reason |
|---|---|---|
| shader-with-double-underscore.html | 1P/2F | fragment shader with double-underscore identifier fails to compile (expected success) — GLSL compiler rejects `__` identifiers |
### conformance/offscreencanvas/ — 1 page (1 F)
| Page | P/F | One-line reason |
|---|---|---|
| context-attribute-preserve-drawing-buffer.html | 2P/1F | `preserveDrawingBuffer:false`: buffer NOT cleared after present — expected 0,0,0,0 got 255,0,255,255 after 2000 ms |
## Unexpected / investigated
- **0 timeouts, 0 errors** — nothing to rerun for timeout/error causes. The two baseline8 timeouts are now PASS (see table above).
- **All 11 failing pages rerun singly** (same bundle + harness shim injection, 1 page at a time) to capture per-subtest failure messages and console/pageerror output — every page reproduced with **identical P/F counts**, i.e. all 11 failures are deterministic, none flaky. Root causes listed above come from those reruns.
- **Harness note (not fixed, per run scope):** the JSON report's `messages` array is always empty for subtest failures — the in-page shim collects `window.__softglCts.messages` (first 20, ≤500 chars) but they are never shipped into the result. Per-page root causes had to be extracted by rerunning the failing pages and reading the shim state directly. Worth wiring `state.messages` into the snapshot/results in a future harness change.
- No page ran without the software renderer; bundle injection verified active on all 835 pages.
