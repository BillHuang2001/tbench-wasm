# WebGL CTS FINAL Baseline — bundle da02ec7 (wave-17 complete; handoff verification record)

- **Date**: 2026-08-28/29 (parallel run window 23:48Z – 00:12Z UTC; webgl1 1404.1 s ≈ 23.4 min, webgl2 516.1 s ≈ 8.6 min)
- **Bundle commit**: `da02ec7` (`renderer.js: rebuild with wave-16/17 fixes (compressed textures, clip-cull transport complete, tex-mipmap-levels, 21-float vertex record, small-page residuals)`) — the FINAL handoff bundle; includes EVERYTHING through wave-17 plus all prior waves.
- **Invocation** (from repo root):
  ```
  npx tsx tests/conformance/run.ts --renderer ./renderer.js --suite webgl1 --report tests/reports/webgl1-FINAL.json
  npx tsx tests/conformance/run.ts --renderer ./renderer.js --suite webgl2 --report tests/reports/webgl2-FINAL.json
  ```
- **CTS**: `/testsuites/WebGL` (read-only), suite version `2.0.1 (beta)`, manifests `conformance/00_test_list.txt` (835) + `conformance2/00_test_list.txt` (1184) = **2,019 graded tests**.
- **Defaults**: workers 4, idle timeout 60 s, per-page slow overrides (read-pixels-test 30 min, rendering-stencil-large-viewport 10 min, gl-vertex-attrib-zero-issues 10 min) + 35 min wall-clock cap — all left at defaults; `?webglVersion=1` (webgl1) / `?webglVersion=2` (webgl2).
- **Artifacts** (gitignored): `tests/reports/webgl1-FINAL.json`, `tests/reports/webgl2-FINAL.json` + `.log`.
- `WEBGL_SOFTWARE_RENDERER` never set; `--renderer ./renderer.js` passed explicitly.

## Totals

| Suite | PASS | FAIL | TIMEOUT | ERROR | Total |
|-------|------|------|---------|-------|-------|
| webgl1 (conformance) | 831  | 4    | 0       | 0     | 835   |
| webgl2 (conformance2) | 1182 | 2    | 0       | 0     | 1184  |
| **GRAND TOTAL** | **2013** | **6** | **0** | **0** | **2019** |

- Skipped subtests: 0 in both suites.
- **Renderer-active gate: 0 pages ran without the software renderer** — `rendererActive: true` on all 2,019 results; no `WARNING: N pages ran without the software renderer`.
- No runner errors: suite-count assertions (835/1184) passed; only `pass`/`fail` statuses in both JSONs (no `timeout`, no `error`). Runner exit codes 1 (failures present — expected semantics; no exit-2 conditions).
- **Handoff verification claim**: **2013P / 6F / 0T / 0E of the 2,019 graded tests (99.70%)** — up from 1996P/23F (98.86%) at baseline10+baseline5 (bundle b56a985). All 5 regression clusters reported in baseline5 are fixed; exactly one NEW regression (video-rotation, root cause below).

## vs baseline10 (webgl1, 832P/3F, bundle b56a985) / baseline5 (webgl2, 1164P/20F, bundle b56a985)

**webgl1: 832P/3F → 831P/4F (−1 pass, +1 fail) — ONE regression: `video-rotation` 65P/0F → 49P/16F (root cause in Unexpected #1).** The 3 baseline10 residuals fail with **identical** P/F counts (81P/4F, 1P/2F, 2P/1F); no other baseline10-passing page fails.

| webgl1 page | baseline10 | FINAL | Delta |
|---|---|---|---|
| conformance/textures/misc/video-rotation.html | 65P/0F PASS | 49P/16F FAIL | **REGRESSION** — rotated-video texSubImage2D INVALID_OPERATION (Unexpected #1) |
| conformance/context/premultiplyalpha-test.html | 81P/4F | 81P/4F | unchanged |
| conformance/glsl/misc/shader-with-double-underscore.html | 1P/2F | 1P/2F | unchanged |
| conformance/offscreencanvas/context-attribute-preserve-drawing-buffer.html | 2P/1F | 2P/1F | unchanged |

**webgl2: 1164P/20F → 1182P/2F (+18 pass, −18 fail) — 18 of the 20 baseline5 residual pages are now fully PASS; 2 remain with identical P/F counts; NO regressions** (no baseline5-passing page fails now).

| webgl2 page | baseline5 | FINAL | Delta |
|---|---|---|---|
| conformance2/textures/misc/compressed-tex-image.html | 40P/32F | **PASS** (72P/0F) | wave-16/17 ETC1/arity/PBO validation |
| conformance2/textures/misc/tex-storage-compressed-formats.html | 14P/30F | **PASS** (44P/0F) | wave-16/17 ETC2 texStorage support |
| conformance2/textures/misc/integer-cubemap-texture-sampling.html | 401P/120F | **PASS** (521P/0F) | raster integer readPixels missing-component fill (a351f46) |
| conformance2/textures/misc/tex-mipmap-levels.html | 39P/2F | **PASS** (41P/0F) | virtual size + generateMipmap float filterability |
| conformance2/textures/misc/tex-image-and-sub-image-with-array-buffer-view-sub-source.html | 101P/12F | **PASS** (113P/0F) | wave-15/16/17 small-page fixes |
| conformance2/textures/misc/tex-storage-2d.html | 136P/2F | **PASS** (138P/0F) | 10F_11F_11F_REV R/B order + bytes-per-texel (2c87ce5/ac113b8) |
| conformance2/textures/misc/tex-input-validation.html | 80P/3F | **PASS** (83P/0F) | unsized RED rejection (ed43fe1) |
| conformance2/textures/misc/tex-image-with-bad-args.html | 8P/1F | **PASS** (9P/0F) | RED/UNSIGNED_SHORT error-code fix |
| conformance2/textures/misc/pbo-texture-uploads.html | 2P/2F | **PASS** (4P/0F) | PBO-bound DOM upload INVALID_OPERATION guard (24166f5) |
| conformance2/textures/misc/npot-video-sizing.html | 1P/1F | **PASS** (2P/0F) | VideoFrame.visibleRect sizing (1b3176f) |
| conformance2/textures/misc/generate-mipmap-with-large-base-level.html | 3P/2F | **PASS** (5P/0F) | BASE_LEVEL clamp (645fae3) |
| conformance2/misc/blend-integer.html | 14P/12F | **PASS** (18P/0F) | uvec outputTypeFamily enum fix |
| conformance2/misc/uninitialized-test-2.html | 4577P/3F | **PASS** (4580P/0F) | 2D_ARRAY depth halving exempted (d720440) |
| conformance2/rendering/framebuffer-render-to-layer.html | 1062P/30F | **PASS** (1362P/0F) | same uvec fix — 300 more subtests now execute and pass |
| conformance2/rendering/clear-func-buffer-type-match.html | 23P/2F | **PASS** (25P/0F) | clear() integer-attachment INVALID_OPERATION (645fae3) |
| conformance2/extensions/webgl-clip-cull-distance.html | 251P/1F | **PASS** (267P/0F) | full clip/cull distance transport (wave-16, 21-float header) |
| conformance2/programs/gl-get-frag-data-location.html | 11P/2F | **PASS** (13P/0F) | getFragDataLocation index guard fix |
| conformance2/state/gl-object-get-calls.html | 365P/1F | **PASS** (366P/0F) | same getFragDataLocation fix |
| conformance2/textures/misc/origin-clean-conformance-offscreencanvas.html | 9P/4F | 9P/4F | unchanged (harness limitation — see Known-unfixable) |
| conformance2/textures/misc/tex-image-10bpc.html | 2P/1F | 2P/1F | unchanged (10bpc PNG decode) |

## Residual failure clusters (6 pages, 23 subtest fails)

### webgl1 (4 pages, 23 F)
| Page | P/F | Classification | One-line root cause |
|---|---|---|---|
| conformance/textures/misc/video-rotation.html | 49P/16F | **renderer-bug — REGRESSION** | `src/gl/api/teximage.ts` `sourceDims()` (lines 1010–1017, changed by commit `1b3176f` wave-16) prefers `VideoFrame.visibleRect` for video sources; rotated videos (rotation metadata 90/270) report visibleRect 96×128 vs element 128×96 → `texSubImage2DDOM` (line 1406) validates sub-rect 96×128 against the 128×96 level (`commonTexSubValidation`, line 1411) → INVALID_OPERATION (0x501) → upload skipped → black texture. Affects texSubImage2D only (4 rotated files × 4 quadrants); 6-arg texImage2D path passes. Deterministic (full run + 2 single reruns identical). Fix un-routed — escalate to `./src`. |
| conformance/context/premultiplyalpha-test.html | 81P/4F | renderer-bug | `premultiplyAlpha:false` draw test: expected 255,192,128,1 got 255,255,255,1 at (0,0) ×4 — premultiplied-alpha handling in one path (unchanged since baseline9; no fix location recorded) |
| conformance/glsl/misc/shader-with-double-underscore.html | 1P/2F | native-unfixable | GLSL ES reserves `__` identifiers; renderer compiler rejects them, CTS expects success — native Chrome fails the same CTS page (spec-vs-native disagreement; unchanged) |
| conformance/offscreencanvas/context-attribute-preserve-drawing-buffer.html | 2P/1F | renderer-bug | `preserveDrawingBuffer:false`: buffer NOT cleared after present — expected 0,0,0,0 got 255,0,255,255 after 2000 ms (unchanged; no fix location recorded) |

### webgl2 (2 pages, 5 F)
| Page | P/F | Classification | One-line root cause |
|---|---|---|---|
| conformance2/textures/misc/origin-clean-conformance-offscreencanvas.html | 9P/4F | harness-limitation | texImage3D/texSubImage3D must throw for cross-origin / non-origin-clean sources inside an OffscreenCanvas **Worker**; Playwright `addInitScript` does not reach Worker globals → Worker path runs NATIVE WebGL (documented in `tests/CONTEXT.md`); Chrome-native fails this combination the same way (unchanged) |
| conformance2/textures/misc/tex-image-10bpc.html | 2P/1F | renderer-bug | 10bpc PNG crushed to 8bpc on decode: uniquePixels 3 < 7 (unchanged since baseline5; no fix location recorded) |

## Must-pass verification (extracted from the suite JSONs)

| Page | Expected | Actual | Verdict |
|---|---|---|---|
| conformance2/rendering/vertex-id.html | 69P/0F | 69P/0F | ✅ |
| conformance2/buffers/get-buffer-sub-data-validity.html | 27P/0F | 27P/0F | ✅ |
| conformance2/extensions/webgl-clip-cull-distance.html | 267P/0F | 267P/0F | ✅ |
| conformance2/extensions/webgl-blend-func-extended.html | 278P/0F | 278P/0F | ✅ |
| conformance2/textures/misc/integer-cubemap-texture-sampling.html | 521P/0F | 521P/0F | ✅ |
| conformance2/rendering/blitframebuffer-filter-outofbounds.html | 0F | 642P/0F | ✅ |
| conformance2/rendering/blitframebuffer-filter-srgb.html | 0F | 410P/0F | ✅ |
| conformance2/rendering/blitframebuffer-outside-readbuffer.html | 0F | 930P/0F | ✅ |
| conformance2/textures/misc/compressed-tex-image.html | 0F | 72P/0F | ✅ |
| conformance2/textures/misc/tex-storage-compressed-formats.html | 0F | 44P/0F | ✅ |
| conformance2/textures/misc/tex-mipmap-levels.html | 0F | 41P/0F | ✅ |
| conformance2/textures/misc/tex-unpack-params.html | 985P/0F | 985P/0F | ✅ |
| conformance2/extensions/webgl-multi-draw-instanced-base-vertex-base-instance.html | 0F | 775P/0F | ✅ |
| conformance2/extensions/webgl-render-shared-exponent.html | 1042P/0F | 1042P/0F | ✅ |
| conformance2/misc/uninitialized-test-2.html | 4580P/0F | 4580P/0F | ✅ |
| conformance/textures/misc/gl-teximage.html (W1) | 103P/0F | 103P/0F | ✅ |
| conformance/textures/misc/video-rotation.html (W1) | 65P/0F | 49P/16F | ❌ **REGRESSION** (root cause above) |

## Unexpected / investigated

1. **video-rotation 65P/0F → 49P/16F — the ONLY regression vs baseline10/baseline5.** Deterministic: identical 49P/16F in the full run and in 2 single-page reruns (`debug-page.ts --repeat 2`); per-file probe (browser dims + `VideoFrame.visibleRect` + `gl.getError()` + quadrant readbacks for all 8 resource files) shows: all 8 videos load and the 6-arg `texImage2D` path renders correct colors for every file; the `texImage2D(explicit 128×96, null)` + `texSubImage2D(video)` path errors 0x501 (INVALID_OPERATION) **only for the 4 files with rotation metadata** (video-rotation-90/-270, `vfRot` 90/270) whose `VideoFrame.visibleRect` is the swapped 96×128. Root cause = wave-16 commit `1b3176f` (VideoFrame.visibleRect sizing, which fixed npot-video-sizing 1P/1F → 2P/0F) — a trade-off regression: visibleRect is correct for texture *allocation* (6-arg form) but wrong for *sub-rect validation* against an explicitly sized level. Fix (un-routed, escalate to `./src`): use element display dims (`videoWidth`/`videoHeight`) for the texSubImage2D sub-rect, or clamp/scale the visible rect into the target sub-rect.
2. **All wave-15/16/17 claims hold** — every one of the 18 baseline5 residual pages that was targeted is now 0F; `framebuffer-render-to-layer` 1062P/30F → 1362P/0F additionally proves the uvec INVALID_OPERATION early-outs were masking ~300 subtests. `integer-cubemap-texture-sampling` 401P/120F → 521P/0F (a351f46), `webgl-clip-cull-distance` 251P/1F → 267P/0F (full 21-float record transport), `uninitialized-test-2` 4577P/3F → 4580P/0F (d720440).
3. **0 timeouts, 0 errors, 0 renderer-inactive pages** across both suites; no pageerrors captured on any failing page rerun; every failing page is deterministic (no flakes).

## Known-unfixable / out-of-scope

- **shader-with-double-underscore.html (1P/2F)** — GLSL ES spec reserves double-underscore identifiers; native Chrome fails this same CTS page. Spec-vs-native disagreement, not a renderer defect.
- **origin-clean-conformance-offscreencanvas.html (9P/4F)** — Worker-global OffscreenCanvas path cannot be intercepted by Playwright `addInitScript` (documented harness limitation in `tests/CONTEXT.md`); the Worker sub-tests run against native WebGL.
- All other residuals are genuine renderer defects (see tables); video-rotation is the only one with a precise file/line fix location; the three pre-existing webgl1 defects (premultiplyalpha, preserve-drawing-buffer) and tex-image-10bpc predate this bundle's waves with no fix locations recorded.

## How to reproduce

```
npx tsx tests/conformance/run.ts --renderer ./renderer.js --suite webgl1 --report tests/reports/webgl1-FINAL.json
npx tsx tests/conformance/run.ts --renderer ./renderer.js --suite webgl2 --report tests/reports/webgl2-FINAL.json
```
Full runs: webgl1 ≈ 23.4 min (read-pixels-test 209P/0F in 1155 s, rendering-stencil-large-viewport 8P/0F in 200 s, gl-vertex-attrib-zero-issues 33P/0F in 123 s — all within their overrides), webgl2 ≈ 8.6 min, at workers 4. Per-page root causes:
```
npx tsx tests/conformance/debug-page.ts <url> --renderer ./renderer.js [--repeat N]
```
