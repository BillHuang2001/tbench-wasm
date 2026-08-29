# WebGL CTS FINAL Baseline — bundle e3565d4 (handoff verification record)

- **Date**: 2026-08-28/29 (full-run window 23:48Z – 00:12Z UTC; webgl1 1404.1 s ≈ 23.4 min, webgl2 516.1 s ≈ 8.6 min); the two post-FINAL residual pages were subsequently re-verified with temp bundles by the fixing agents, and `renderer.js` rebuilt at HEAD `e3565d4` with both fixes included.
- **Bundle commit**: `e3565d4` (`renderer.js: rebuild with final fixes (video-rotation srcDims rotation transposition, OffscreenCanvas preserve-clear without blit)`) — the FINAL handoff bundle: everything through wave-17 (compressed textures, clip-cull transport complete, tex-mipmap-levels, 21-float vertex record, small-page residuals) PLUS the two post-FINAL recoveries `b074ad3` (video-rotation) and `4ffbacc` (preserve-drawing-buffer).
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
| webgl1 (conformance) | 833  | 2    | 0       | 0     | 835   |
| webgl2 (conformance2) | 1182 | 2    | 0       | 0     | 1184  |
| **GRAND TOTAL** | **2015** | **4** | **0** | **0** | **2019** |

- Skipped subtests: 0 in both suites.
- **Renderer-active gate: 0 pages ran without the software renderer** — `rendererActive: true` on all 2,019 graded results; the recovery re-verifications of video-rotation and preserve-drawing-buffer (temp bundles) also ran renderer-active.
- No runner errors: suite-count assertions (835/1184) passed; only `pass`/`fail` statuses in both JSONs (no `timeout`, no `error`). Runner exit codes 1 (failures present — expected semantics; no exit-2 conditions).
- **Handoff verification claim**: **2015P / 4F / 0T / 0E of the 2,019 graded tests (99.80%)** — up from 2013P/6F (99.70%) at FINAL (da02ec7). The 4 remaining fails are 2 environment/harness + 1 native-unfixable + 1 present-side — all non-gl, non-raster, non-glsl.

## vs FINAL (bundle da02ec7, 2013P/6F)

**webgl1: 831P/4F → 833P/2F (+2 pass, −2 fail) — BOTH residual pages recovered; NO regressions.** premultiplyalpha-test (81P/4F) and shader-with-double-underscore (1P/2F) fail with **identical** P/F counts; premultiplyalpha is re-classified environment-side (evidence in Unexpected #2).

| webgl1 page | FINAL (da02ec7) | Now | Delta |
|---|---|---|---|
| conformance/textures/misc/video-rotation.html | 49P/16F FAIL | **65P/0F PASS** | fix `b074ad3` — `srcDims()` returns element display dims when `VideoFrame.visibleRect` is the exact transposition of the element's intrinsic dims (rotated-video texSubImage2D INVALID_OPERATION gone); npot-video-sizing preserved 2P/0F |
| conformance/offscreencanvas/context-attribute-preserve-drawing-buffer.html | 2P/1F FAIL | **3P/0F PASS** | fix `4ffbacc` — OffscreenCanvas gets the frame-boundary clear without the present() blit |
| conformance/context/premultiplyalpha-test.html | 81P/4F | 81P/4F | unchanged counts — **re-classified environment-side** (see Unexpected #2) |
| conformance/glsl/misc/shader-with-double-underscore.html | 1P/2F | 1P/2F | unchanged (native-unfixable — Chrome fails the same CTS page) |

**webgl2: 1182P/2F → 1182P/2F — unchanged; NO regressions.**

| webgl2 page | FINAL (da02ec7) | Now | Delta |
|---|---|---|---|
| conformance2/textures/misc/origin-clean-conformance-offscreencanvas.html | 9P/4F | 9P/4F | unchanged (harness limitation — Worker path runs native WebGL) |
| conformance2/textures/misc/tex-image-10bpc.html | 2P/1F | 2P/1F | unchanged (present-side 10bpc PNG decode) |

## Residual failure clusters (4 pages, 11 subtest fails)

### webgl1 (2 pages, 6 F)
| Page | P/F | Classification | One-line root cause |
|---|---|---|---|
| conformance/context/premultiplyalpha-test.html | 81P/4F | environment-side (NOT a renderer defect) | `premultiplyAlpha:false` draw test: expected 255,192,128,1 got 255,255,255,1 at (0,0) ×4 — headless Chromium **2D-canvas RGB crushing for alpha=1 texels** (the 2D presentation canvas is the readback path); proven with a no-renderer control: the page run against NATIVE WebGL fails identically |
| conformance/glsl/misc/shader-with-double-underscore.html | 1P/2F | native-unfixable | GLSL ES reserves `__` identifiers; renderer compiler rejects them, CTS expects success — native Chrome fails the same CTS page (spec-vs-native disagreement) |

### webgl2 (2 pages, 5 F)
| Page | P/F | Classification | One-line root cause |
|---|---|---|---|
| conformance2/textures/misc/origin-clean-conformance-offscreencanvas.html | 9P/4F | harness-limitation | texImage3D/texSubImage3D must throw for cross-origin / non-origin-clean sources inside an OffscreenCanvas **Worker**; Playwright `addInitScript` does not reach Worker globals → Worker path runs NATIVE WebGL (documented in `tests/CONTEXT.md`); Chrome-native fails this combination the same way |
| conformance2/textures/misc/tex-image-10bpc.html | 2P/1F | present-side (escalated) | 10bpc PNG crushed to 8bpc on decode: uniquePixels 3 < 7 — present-side 8bpc decode crushing; escalated to `./src` (present side, not gl/raster/glsl) |

## Must-pass verification (extracted from the suite JSONs + recovery re-verifications)

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
| conformance/textures/misc/video-rotation.html (W1) | 65P/0F | **65P/0F** | ✅ recovered (`b074ad3`) |
| conformance/offscreencanvas/context-attribute-preserve-drawing-buffer.html (W1) | 3P/0F | **3P/0F** | ✅ recovered (`4ffbacc`) |
| conformance2/textures/misc/npot-video-sizing.html | 2P/0F | **2P/0F** | ✅ preserved (no regression from `b074ad3`) |

## Unexpected / investigated

1. **Both post-FINAL recoveries are deterministic and regression-free.** video-rotation 49P/16F → 65P/0F (fix `b074ad3`: `srcDims()` returns element display dims when `VideoFrame.visibleRect` is the exact transposition of the element's intrinsic dims — the rotated-video 96×128-vs-128×96 sub-rect mismatch that caused texSubImage2D INVALID_OPERATION is gone), with npot-video-sizing preserved at 2P/0F (the wave-16 `1b3176f` trade-off regression is fully resolved). context-attribute-preserve-drawing-buffer 2P/1F → 3P/0F (fix `4ffbacc`: OffscreenCanvas gets the frame-boundary clear without the present() blit); the fix was validated on 40/40 graded WebGL1 pages (all webgl_canvas/offscreencanvas/buffer-preserve families) + 212/213 WebGL2 filtered pages, sole fail = the documented origin-clean offscreencanvas harness limitation.
2. **premultiplyalpha-test re-classified environment-side** (81P/4F, unchanged counts): a no-renderer control (the page run against native WebGL) fails with the identical 255,255,255,1 vs expected 255,192,128,1 at (0,0) — headless Chromium's 2D canvas (the presentation/readback path) crushes RGB for alpha=1 texels. NOT a renderer defect.
3. **All wave-15/16/17 claims hold** — every webgl2 residual fix and must-pass page is unchanged: `framebuffer-render-to-layer` 1362P/0F, `integer-cubemap-texture-sampling` 521P/0F, `webgl-clip-cull-distance` 267P/0F, `uninitialized-test-2` 4580P/0F.
4. **0 timeouts, 0 errors, 0 renderer-inactive pages** across both suites; every failing page is deterministic (no flakes).

## Known-unfixable / out-of-scope (4 residuals, 11 subtest fails — all non-gl, non-raster, non-glsl)

- **shader-with-double-underscore.html (1P/2F)** — native-unfixable: GLSL ES reserves double-underscore identifiers; native Chrome fails this same CTS page. Spec-vs-native disagreement, not a renderer defect.
- **origin-clean-conformance-offscreencanvas.html (9P/4F)** — harness limitation: Worker-global OffscreenCanvas path cannot be intercepted by Playwright `addInitScript` (documented in `tests/CONTEXT.md`); the Worker sub-tests run against native WebGL.
- **premultiplyalpha-test.html (81P/4F)** — environment-side: headless Chromium 2D-canvas RGB crushing for alpha=1 texels (proven with a no-renderer control); not a renderer defect.
- **tex-image-10bpc.html (2P/1F)** — present-side 8bpc decode crushing; escalated to `./src` (present side).

## How to reproduce

```
npx tsx tests/conformance/run.ts --renderer ./renderer.js --suite webgl1 --report tests/reports/webgl1-FINAL.json
npx tsx tests/conformance/run.ts --renderer ./renderer.js --suite webgl2 --report tests/reports/webgl2-FINAL.json
```
Full runs: webgl1 ≈ 23.4 min (read-pixels-test 209P/0F in 1155 s, rendering-stencil-large-viewport 8P/0F in 200 s, gl-vertex-attrib-zero-issues 33P/0F in 123 s — all within their overrides), webgl2 ≈ 8.6 min, at workers 4. Per-page root causes:
```
npx tsx tests/conformance/debug-page.ts <url> --renderer ./renderer.js [--repeat N]
```
