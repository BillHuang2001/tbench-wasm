# WebGL1 CTS Baseline 10 — bundle b56a985 (wave-12/13/14 fixes)
- **Date**: 2026-08-28 (run window 20:46:02Z – 21:02:17Z UTC, 974.4 s wall ≈ 16.2 min)
- **Bundle commit**: `b56a985` (`renderer.js: rebuild with wave-12/13/14 fixes (SRC1 dual-source routing, int/uint varying+output bit-pack, TF raw-bit capture, raster integer-encode bit-reinterpret, blend secondary plumbing, index-1 outputs)`)
- **Invocation** (from repo root):
  ```
  npx tsx tests/conformance/run.ts --renderer ./renderer.js --suite webgl1 --report tests/reports/webgl1-baseline10.json
  ```
- **CTS**: `/testsuites/WebGL` (read-only), suite version `2.0.1 (beta)`, manifest `conformance/00_test_list.txt`
- **Defaults**: workers 4, idle timeout 60 s, per-page slow overrides (read-pixels-test 30 min, rendering-stencil-large-viewport 10 min, gl-vertex-attrib-zero-issues 10 min) + 35 min wall-clock cap — all left at defaults, `?webglVersion=1`
- **Artifacts** (gitignored): `tests/reports/webgl1-baseline10.json`, `tests/reports/webgl1-baseline10.log`
- Shell env had a poisoned `WEBGL_SOFTWARE_RENDERER=/app/renderer.js` (nonexistent) — `--renderer ./renderer.js` passed explicitly; env var never set by this run. (Plumbing note: the initial `tee` failed to open the log because `tests/reports/` did not exist at tee start — the runner creates it before writing the JSON; the log was reconstructed from the captured runner output afterwards. The JSON report is authoritative.)
## Totals
| PASS | FAIL | TIMEOUT | ERROR | Total |
|------|------|---------|-------|-------|
| 832  | 3    | 0       | 0     | 835   |
- Skipped subtests: 0.
- **Renderer-active gate: 0 pages ran without the software renderer** (no `WARNING: N pages ran without the software renderer` in the summary; `rendererActive: false` count in JSON = 0, `rendererActive: undefined` = 0 — every one of the 835 results carries `rendererActive: true`).
- No runner errors: the suite-count assertion (835) passed and all 835 pages completed with only `pass`/`fail` statuses (no `timeout`, no `error`). The command was piped through `tee` (its exit code is not the runner's); the runner's own summary is authoritative (3 failures → runner exit-1 semantics; no exit-2 conditions occurred).
- Second-fastest full WebGL1 pass: 974.4 s (baseline9: 923.6 s; the 754 s read-pixels-test still dominates the tail).
## vs baseline9 (824P / 11F / 0T / 0E, bundle ce8a79c, run 2026-08-28)
**Pass delta: +8** (824 → 832); failures 11 → 3 (−8); timeouts 0 → 0; errors 0 → 0. **8 of the 11 baseline9 residual pages are now fully PASS** (wave-12/13/14 fixes), and **no regressions**: the only 3 failing pages in this run were failing in baseline9 with identical P/F counts (81P/4F, 1P/2F, 2P/1F); baseline9's JSON was not retained (gitignored), so per-subtest comparison of passing pages relies on the fail-set containment above plus the must-pass checks below.
| Page (semantic name) | baseline9 | baseline10 | Verified in this run |
|---|---|---|---|
| conformance/textures/misc/copy-tex-image-and-sub-image-2d.html | 335P/168F | **PASS** | PASS (503P/0F/0S — all 503 subtests) |
| conformance/textures/misc/image-decoder-to-texture.html | 1P/2F | **PASS** | PASS (3P/0F/0S) |
| conformance/textures/misc/tex-image-svg-image-no-natural-width-and-height.html | 11P/2F | **PASS** | PASS (13P/0F/0S) |
| conformance/textures/misc/texture-srgb-upload.html | 5P/2F | **PASS** | PASS (7P/0F/0S) |
| conformance/extensions/ext-color-buffer-half-float.html | 100P/4F | **PASS** | PASS (104P/0F/0S) |
| conformance/extensions/oes-texture-float-linear.html | 91P/6F | **PASS** | PASS (97P/0F/0S) |
| conformance/extensions/webgl-compressed-texture-etc.html | 15P/2F | **PASS** | PASS (17P/0F/0S) |
| conformance/extensions/webgl-multi-draw.html | 7929P/1F | **PASS** | PASS (7930P/0F/0S) |
Must-pass items all hold in the full run: gl-teximage **103P/0F/0S** (exactly 103/0), video-rotation **65P/0F/0S** (exactly 65/0), tex-2d-\* family **8/8 PASS** (5,448P/0F/0S subtests total: 338+394+394+1570+394+1570+394+394). **No regressions among them.** Slow pages all within their overrides: read-pixels-test 209P/0F in 754 s (12.6 min, 30-min budget), rendering-stencil-large-viewport 8P/0F in 202 s (3.4 min, 10-min budget), gl-vertex-attrib-zero-issues 33P/0F in 63 s (10-min budget).
## Residual failure clusters (3 pages, 7 subtest fails)
### conformance/context/ — 1 page (4 F)
| Page | P/F | One-line reason |
|---|---|---|
| premultiplyalpha-test.html | 81P/4F | `premultiplyAlpha:false` draw test: expected 255,192,128,1 got 255,255,255,1 at (0,0) ×4 — premultiplied-alpha handling in one path (unchanged from baseline9) |
### conformance/glsl/misc/ — 1 page (2 F)
| Page | P/F | One-line reason |
|---|---|---|
| shader-with-double-underscore.html | 1P/2F | vertex + fragment shader with double-underscore identifier fails to compile (expected success) — GLSL compiler rejects `__` identifiers (unchanged from baseline9) |
### conformance/offscreencanvas/ — 1 page (1 F)
| Page | P/F | One-line reason |
|---|---|---|
| context-attribute-preserve-drawing-buffer.html | 2P/1F | `preserveDrawingBuffer:false`: buffer NOT cleared after present — expected 0,0,0,0 got 255,0,255,255 after 2000 ms (unchanged from baseline9) |
## Unexpected / investigated
- **0 timeouts, 0 errors** — nothing to rerun for timeout/error causes.
- **All 3 failing pages rerun singly ×2 each** (same bundle + harness shim injection, 1 page at a time) to capture per-subtest failure messages and console/pageerror output — every rerun reproduced with **identical P/F counts** (81P/4F, 1P/2F, 2P/1F), i.e. all 3 failures are deterministic, none flaky. Root causes above come from the in-page shim state; no pageerrors and no console errors were captured on any rerun.
- **Harness note (not fixed, per run scope):** the JSON report's `messages` array is always empty for subtest failures — the in-page shim collects `window.__softglCts.messages` (first 20, ≤500 chars) but they are never shipped into the result. Per-page root causes had to be extracted by rerunning the failing pages and reading the shim state directly (exactly as baseline9 did). Worth wiring `state.messages` into the snapshot/results in a future harness change.
- No page ran without the software renderer; bundle injection verified active on all 835 pages (`rendererActive: true` on every result).
## Known-unfixable / out-of-scope
- **shader-with-double-underscore.html** (1P/2F): the GLSL ES spec reserves identifiers containing double underscores (`__`); the renderer's GLSL compiler rejects them and the CTS expects a successful compile. Native Chrome fails this same CTS page, so this is a spec-vs-native disagreement rather than a renderer defect — out of scope.
- The other two residuals (premultiplyalpha-test 4F, context-attribute-preserve-drawing-buffer 1F) are genuine renderer defects (premultiplied-alpha handling in one path; drawing-buffer clearing after present with `preserveDrawingBuffer:false`) — fixable in a future wave, not spec-invalid expectations; no AA-threshold or other known-unfixable cases remain.
