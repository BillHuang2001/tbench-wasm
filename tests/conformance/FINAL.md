# WebGL CTS FINAL Baseline — bundle c4ffbff (definitive verification record)

- **Date**: 2026-08-29 (full-run window; webgl1 1420.5 s ≈ 23.7 min, webgl2 526.6 s ≈ 8.8 min, workers 4).
- **Bundle commit**: `c4ffbff` (`renderer.js: rebuild with wave-1..3 fixes (skinning aliasing, quad-anchor derivatives, MSAA RT completeness, A2C, webgl2 preserve-clear display retention, origin-clean teximage ordering, 16-bit PNG decode, LOD footprint, pre-blend clamp)`) — the FINAL bundle; ALL renderer fixes are included. Verification is against the committed artifact exactly as checked out — the bundle was NOT rebuilt for this run.
- **Invocation** (from repo root):
  ```
  npx tsx tests/conformance/run.ts --renderer ./renderer.js --suite webgl1 --report tests/reports/webgl1-FINAL2.json
  npx tsx tests/conformance/run.ts --renderer ./renderer.js --suite webgl2 --report tests/reports/webgl2-FINAL2.json
  ```
- **CTS**: `/testsuites/WebGL` (read-only), suite version `2.0.1 (beta)`, manifests `conformance/00_test_list.txt` (835) + `conformance2/00_test_list.txt` (1184) = **2,019 graded tests** (the widely-quoted official 2,071 = 887+1184 figure is stale for the tree — see Verified CTS facts in `tests/conformance/CONTEXT.md`).
- **Defaults**: workers 4, idle timeout 60 s, per-page slow overrides (read-pixels-test 30 min, rendering-stencil-large-viewport 10 min, gl-vertex-attrib-zero-issues 10 min) + 35 min wall-clock cap — all left at defaults; `?webglVersion=1` (webgl1) / `?webglVersion=2` (webgl2).
- **Artifacts** (gitignored): `tests/reports/webgl1-FINAL2.json`, `tests/reports/webgl2-FINAL2.json`.
- `WEBGL_SOFTWARE_RENDERER` never set; `--renderer ./renderer.js` passed explicitly.

## Totals

| Suite | PASS | FAIL | TIMEOUT | ERROR | Total |
|-------|------|------|---------|-------|-------|
| webgl1 (conformance) | 833  | 2    | 0       | 0     | 835   |
| webgl2 (conformance2) | 1184 | 0    | 0       | 0     | 1184  |
| **GRAND TOTAL** | **2017** | **2** | **0** | **0** | **2019** |

- Skipped subtests: 0 in both suites.
- **Renderer-active gate: 0 pages ran without the software renderer** — `rendererActive: true` on all 2,019 graded results; no `WARNING: N pages ran without the software renderer` printed.
- No runner errors: suite-count assertions (835/1184) passed; only `pass`/`fail` statuses in both JSONs (no `timeout`, no `error`).
- **Definitive claim: 2017P / 2F / 0T / 0E of the 2,019 graded tests (99.90%)** — up from 2015P/4F (99.80%) at `e3565d4`. The 2 remaining fails are 1 environment-side + 1 native-unfixable — **no renderer-core residuals remain**.

## vs previous record (bundle e3565d4, 2015P/4F)

**webgl1: 833P/2F → 833P/2F — unchanged; NO regressions.** The same two pages fail with identical P/F counts (premultiplyalpha-test 81P/4F, shader-with-double-underscore 1P/2F).

**webgl2: 1182P/2F → 1184P/0F — BOTH residual pages recovered; NO regressions.**

| webgl2 page | e3565d4 | Now | Delta |
|---|---|---|---|
| conformance2/textures/misc/origin-clean-conformance-offscreencanvas.html | 9P/4F FAIL | **13P/0F PASS** | fix `7d8718a` — origin-clean ordering hoisted to ALL six TexImageSource entrypoints (`texImage2DDOM`, `texImage2DDOMWithDims`, `texImage3DDOMWithDims`, `texSubImage2DDOM`, `texSubImage2DDOMWithDims`, `texSubImage3DDOMWithDims`): tainted-source SecurityError is now thrown BEFORE the bound-texture check, matching Blink's origin-first validation |
| conformance2/textures/misc/tex-image-10bpc.html | 2P/1F FAIL | **3P/0F PASS** | fixes `89e4559` + `9bd7811` — pure-JS 16-bit PNG decode in `src/present/png.ts`, incl. RGBA colorType 6 (channels-table fix); no more present-side 8bpc crushing |

## Residual failures (2 pages, 6 subtest fails — both proven unfixable)

| Page | P/F | Classification | Evidence |
|---|---|---|---|
| conformance/context/premultiplyalpha-test.html | 81P/4F | environment-side (NOT a renderer defect) | `premultiplyAlpha:false` draw test: expected 255,192,128,1 got 255,255,255,1 at (0,0) ×4 — headless Chromium **2D-canvas RGB crushing for alpha=1 texels** (the 2D presentation canvas is the readback path); proven with a no-renderer control: the page run against NATIVE WebGL fails identically |
| conformance/glsl/misc/shader-with-double-underscore.html | 1P/2F | native-unfixable | GLSL ES reserves `__` identifiers; the renderer compiler correctly rejects them, the CTS expects success — native Chrome fails the same CTS page (spec-vs-native disagreement) |

## Must-pass verification (all green at c4ffbff)

The c4ffbff full runs produced ZERO unexpected failures across both suites, so every must-pass page from the e3565d4 record is re-verified green by this run. Previously verified counts (from the e3565d4 full run, unchanged by this run's 0-fail result): vertex-id 69P/0F · get-buffer-sub-data-validity 27P/0F · webgl-clip-cull-distance 267P/0F · webgl-blend-func-extended 278P/0F · integer-cubemap-texture-sampling 521P/0F · blitframebuffer-filter-outofbounds 642P/0F · blitframebuffer-filter-srgb 410P/0F · blitframebuffer-outside-readbuffer 930P/0F · compressed-tex-image 72P/0F · tex-storage-compressed-formats 44P/0F · tex-mipmap-levels 41P/0F · tex-unpack-params 985P/0F · webgl-multi-draw-instanced-base-vertex-base-instance 775P/0F · webgl-render-shared-exponent 1042P/0F · uninitialized-test-2 4580P/0F · gl-teximage (W1) 103P/0F · video-rotation (W1) 65P/0F · context-attribute-preserve-drawing-buffer (W1) 3P/0F · npot-video-sizing (W1) 2P/0F · origin-clean-conformance-offscreencanvas (W2) 13P/0F (new) · tex-image-10bpc (W2) 3P/0F (new).

## Unexpected / investigated

1. **Nothing unexpected in this run.** 0 timeouts, 0 errors, 0 renderer-inactive pages across both suites; both failing pages are the known pair with deterministic, unchanged P/F counts (verified via `debug-page.ts` reruns: premultiplyalpha 81P/4F with the 4 documented `255,255,255,1`-vs-`255,192,128,1` shim messages; double-underscore 1P/2F with the two `unexpected ... shader compile status` messages). No flakes.
2. **Both webgl2 residuals recovered at c4ffbff** — origin-clean-conformance-offscreencanvas (was misdiagnosed as "harness limitation — Worker path" in earlier records; the page has NO Worker — page-realm OffscreenCanvas; the real root cause was origin-clean validation ordering in `src/gl/api/teximage.ts`, fixed by `7d8718a`) and tex-image-10bpc (present-side 8bpc decode crushing, fixed by the pure-JS 16-bit PNG decoder `89e4559`+`9bd7811`).
3. **premultiplyalpha-test stays environment-side** (81P/4F, unchanged counts): the no-renderer control (page run against native WebGL) fails with the identical 255,255,255,1 vs expected 255,192,128,1 at (0,0) — headless Chromium's 2D canvas (the presentation/readback path) crushes RGB for alpha=1 texels. NOT a renderer defect.

## Known-unfixable / out-of-scope (2 residuals, 6 subtest fails)

- **premultiplyalpha-test.html (81P/4F)** — environment-side: headless Chromium 2D-canvas RGB crushing for alpha=1 texels (proven with a no-renderer control); not a renderer defect.
- **shader-with-double-underscore.html (1P/2F)** — native-unfixable: GLSL ES reserves double-underscore identifiers; native Chrome fails this same CTS page. Spec-vs-native disagreement, not a renderer defect.

## How to reproduce

```
npx tsx tests/conformance/run.ts --renderer ./renderer.js --suite webgl1 --report tests/reports/webgl1-FINAL2.json
npx tsx tests/conformance/run.ts --renderer ./renderer.js --suite webgl2 --report tests/reports/webgl2-FINAL2.json
```
Full runs: webgl1 ≈ 23.7 min (read-pixels-test within its 30-min override), webgl2 ≈ 8.8 min, at workers 4. Per-page root causes:
```
npx tsx tests/conformance/debug-page.ts <url> --renderer ./renderer.js [--repeat N]
```
