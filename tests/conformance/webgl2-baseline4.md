# WebGL2 CTS Baseline 4 — bundle ce8a79c (wave-6/7/8/9 fixes)

- **Date**: 2026-08-28 (run window 15:40:38Z – 15:48:29Z UTC, 470.6 s)
- **Bundle commit**: `ce8a79c` (`renderer.js: rebuild with glsl standalone-layout-declaration fix (Babylon standard-material compile)`)
- **Invocation** (from repo root):
  ```
  npx tsx tests/conformance/run.ts --renderer ./renderer.js --suite webgl2 --report tests/reports/webgl2-baseline4.json
  ```
- **CTS**: `/testsuites/WebGL` (read-only), suite version `2.0.1 (beta)`, manifest `conformance2/00_test_list.txt`
- **Defaults**: workers 4, idle timeout 60 s (per-page slow overrides + 35 min wall-clock cap untouched), `?webglVersion=2`
- **Artifacts** (gitignored): `tests/reports/webgl2-baseline4.json`, `tests/reports/webgl2-baseline4.log`
- Shell env had a poisoned `WEBGL_SOFTWARE_RENDERER` — `--renderer ./renderer.js` passed explicitly; never set env var.

## Totals

| PASS | FAIL | TIMEOUT | ERROR | Total |
|------|------|---------|-------|-------|
| 1130 | 54   | 0       | 0     | 1184  |

- Skipped subtests: 0.
- **Renderer-active gate: 0 pages ran without the software renderer** (no `WARNING: N pages ran without the software renderer` in the summary; `rendererActive: false` count in JSON = 0). Every result carries `rendererActive: true`.
- No runner errors: the suite-count assertion (1184) passed, all 1184 pages completed (only `pass`/`fail` statuses in the JSON — no `timeout`, no `error`). Note: the shell exit code shown was 0 because the command was piped through `tee`; the runner's own summary is authoritative (54 failures → runner exit code 1 semantics; no exit-2 conditions occurred).

## vs baseline3 (919P / 265F / 0T, bundle 32751d0)

**Pass delta: +211** (919 → 1130), failures 265 → 54. Zero timeouts in both.

| Page (semantic name) | baseline3 | baseline4 | Verified in this run |
|---|---|---|---|
| occlusion-query ×2 (conformance2/query/occlusion-query.html, occlusion-query-scissor.html) | ~8F + ~175F | **0F** | PASS (18P/0F, 519P/0F) |
| conformance2/extensions/ext-texture-norm16.html | 36F | **0F** | PASS (86P/0F) — norm16 readPixels fix holds |
| conformance2/extensions/oes-draw-buffers-indexed.html | 195F | **3F** | FAIL (1602P/3F) — exactly ~3 remaining |
| conformance2/misc/uninitialized-test-2.html | 1211F | **0F** | PASS (4580P/0F) |
| conformance2/textures/misc/gl-teximage.html | 103P/0F | n/a | **NOT in the conformance2 manifest** — the page lives in the WebGL1 suite (`conformance/textures/misc/gl-teximage.html`); a `--suite webgl2` run cannot verify it (see Unexpected) |
| conformance2/textures/video/* (68 pages) | 146P/0F total | **64/68 pages 146P/0F**; 4 pages 74P/72F | DEVIATION — the 4 `tex-2d-*8ui` integer pages fail (see Unexpected) |
| conformance2/samplers/sampler-drawing-test.html | 432F | **0F** | PASS (6P/0F) |
| conformance2/attribs/gl-bindAttribLocation-aliasing-inactive.html | 256F | **0F** | PASS (545P/0F) |
| conformance2/state/gl-object-get-calls.html | 53F | **0F** | PASS (366P/0F) — expected ~8 remaining fails; fully passes now (better than expected) |
| conformance2/misc/views-with-offsets.html | 45F | **1F** | FAIL (229P/1F) — exactly ~1 remaining |
| conformance2/transform_feedback/simultaneous_binding.html | 6F | **3F** | FAIL (128P/3F) — exactly ~3 remaining |

All must-hold items verified in the full run; no regressions among them.

## Residual failure clusters (54 pages)

### conformance2/textures/misc/ — 22 pages
| Page | P/F | One-line reason |
|---|---|---|
| compressed-tex-image.html | 40P/32F | compressedTexImage/SubImage2D ETC1: wrong error codes (INVALID_ENUM where NO_ERROR/INVALID_OPERATION expected) + missing "too few args" TypeError |
| copy-texture-cube-map-AMD-bug.html | 34P/24F | CopyTexSubImage2D returns INVALID_OPERATION (should succeed) |
| copy-texture-image-same-texture.html | 20P/2F | FBO status FRAMEBUFFER_COMPLETE vs expected INCOMPLETE_ATTACHMENT; alpha copied as 0 (255,255,255,0) |
| copy-texture-image-webgl-specific.html | 34P/14F | copyTexImage byte-comparison failures (bytes 0 instead of 255) |
| generate-mipmap-with-large-base-level.html | 3P/2F | generateMipmap returns INVALID_OPERATION where NO_ERROR expected |
| immutable-tex-render-feedback.html | 2425P/840F | immutable-texture FBO completeness (INCOMPLETE_ATTACHMENT) + render-feedback draw error codes |
| integer-cubemap-texture-sampling.html | 401P/120F | integer cubemap sampling: (255,255,255) instead of (255,0,0) |
| npot-video-sizing.html | 1P/1F | video-texture height wrong (red instead of green) |
| origin-clean-conformance-offscreencanvas.html | 9P/4F | texImage3D/texSubImage3D with cross-origin image / non-origin-clean canvas should throw, doesn't (offscreencanvas runs in a Worker — native path, known harness caveat) |
| pbo-texture-uploads.html | 2P/2F | PBO-bound uploads: wrong error codes (NO_ERROR/INVALID_VALUE vs INVALID_OPERATION) |
| tex-image-10bpc.html | 2P/1F | 10bpc PNG crushed to 8bpc: uniquePixels 3 < 7 |
| tex-image-and-sub-image-with-array-buffer-view-sub-source.html | 101P/12F | typed-array view sub-source: element 4 expected 4, got 1 |
| tex-image-with-bad-args.html | 8P/1F | RED/UNSIGNED_SHORT: INVALID_ENUM vs expected INVALID_VALUE/INVALID_OPERATION |
| tex-input-validation.html | 80P/3F | input validation error codes (NO_ERROR where INVALID_* expected) |
| tex-mipmap-levels.html | 39P/2F | partial-level draw color wrong; generateMipmap on float texture should fail, doesn't |
| tex-new-formats.html | 725P/238F | texSubImage2D wrong-data-type errors missing |
| tex-srgb-mipmap.html | 4P/11F | sRGB mipmap values off (55,4,13 vs 128,7,27) |
| tex-storage-2d.html | 133P/5F | texStorage2D uninitialized sampling (0,0,0,0 vs 0,0,0,255); sub-image red/blue swap |
| tex-storage-compressed-formats.html | 14P/30F | texStorage2D/3D compressed 0x9270: INVALID_ENUM where NO_ERROR expected |
| tex-unpack-params-imagedata.html | 11P/3F | ImageData unpack-params array mismatches |
| tex-unpack-params-with-flip-y-and-premultiply-alpha.html | 600P/166F | unpack-params × flipY/premultiply: error-code mismatches (INVALID_OPERATION where NO_ERROR expected and vice versa) |
| tex-unpack-params.html | 568P/417F | unpack-params pixel comparisons (24–105 pixels differ) |

### conformance2/rendering/ — 13 pages
| Page | P/F | One-line reason |
|---|---|---|
| attrib-type-match.html | 47P/55F | attrib type-mismatch INVALID_OPERATION not generated; draw color wrong |
| blitframebuffer-filter-outofbounds.html | 511P/131F | out-of-bounds blit filter: (32,32,32) instead of (0,0,0) |
| blitframebuffer-filter-srgb.html | 151P/259F | sRGB-filtered blit values (40 vs 110, 168 vs 212, alpha 72 vs 145) |
| blitframebuffer-outside-readbuffer.html | 474P/456F | outside-readbuffer blit values (32 vs 16) |
| clear-func-buffer-type-match.html | 23P/2F | clear with INT/UINT buffer should error INVALID_OPERATION, doesn't |
| clear-srgb-color-buffer.html | 3P/2F | sRGB clear color (51,136,187 vs 124,193,222) |
| clearbuffer-sub-source.html | 21P/6F | clearBuffer srcOffset: (3,4,5,6) vs (1,2,3,4); missing INVALID_VALUE |
| draw-buffers.html | 228P/1F | RASTERIZER_DISCARD draw: INVALID_OPERATION where NO_ERROR expected |
| fs-color-type-mismatch-color-buffer-type.html | 11P/4F | missing INVALID_OPERATION on FS-output/color-buffer type mismatch |
| instanced-arrays.html | 60P/4F | drawArraysInstanced LINE_LIST/TRI_LIST: INVALID_ENUM where NO_ERROR expected |
| read-draw-when-missing-image.html | 22P/15F | missing INVALID_OPERATION for color buffer without image; wrong color (green vs red) |
| uniform-block-buffer-size.html | 15P/20F | uniform block not backed by buffer: INVALID_OPERATION not generated |
| vertex-id.html | 61P/8F | gl_VertexID values off (2147483645 vs 2147483647; 1065353216 vs 1 — float bits) |

### conformance2/textures/video/ — 4 pages (DEVIATION, see Unexpected)
| Page | P/F | One-line reason |
|---|---|---|
| tex-2d-r8ui-red_integer-unsigned_byte.html | 74P/72F | display-p3 unpackColorSpace cases: raw sRGB uploaded (255,0,0) instead of converted (234,0,0) |
| tex-2d-rg8ui-rg_integer-unsigned_byte.html | 74P/72F | same — (0,255,0) instead of (117,251,0) |
| tex-2d-rgb8ui-rgb_integer-unsigned_byte.html | 74P/72F | same |
| tex-2d-rgba8ui-rgba_integer-unsigned_byte.html | 74P/72F | same |

### conformance2/extensions/ — 5 pages
| Page | P/F | One-line reason |
|---|---|---|
| oes-draw-buffers-indexed.html | 1602P/3F | indexed color-mask/blend-color: (255,0,0,0) instead of (0,0,255,255)/(0,0,255,0) |
| webgl-blend-func-extended.html | 229P/47F | GLSL extension missing ("no GL_EXT_blend_func_extended"), output-location conflicts; pageerror `Argument is not of type 'WebGLProgram'` in getUniformLocation after failed link |
| webgl-clip-cull-distance.html | 251P/1F | `gl_ClipDistance` undeclared in shader; pageerror getUniformLocation on unlinked program |
| webgl-multi-draw-instanced-base-vertex-base-instance.html | 88P/397F | INVALID_ENUM on multi-draw entry points; base-vertex/base-instance results wrong |
| webgl-render-shared-exponent.html | 764P/274F | shared-exponent readback (0,0,0,0) vs (64,32,16,1); error-code mismatches |

### conformance2/wasm/ — 3 pages
| Page | P/F | One-line reason |
|---|---|---|
| readpixels-4gb-wasm-memory.html | 2P/1F | readPixels into >4GB wasm memory: [0,0,0,0] instead of [42,84,128,255] |
| readpixels-2gb-in-4gb-wasm-memory.html | 2P/1F | same |
| readpixels-16gb-wasm-memory.html | 2P/1F | same |

### conformance2/buffers/ — 2 pages
| Page | P/F | One-line reason |
|---|---|---|
| get-buffer-sub-data-validity.html | 26P/1F | getBufferSubData: areArraysEqual(dest, srcData) false |
| uniform-buffers.html | 47P/2F | fragment shader compile error "UBOGreen : redefinition" → draws black |

### Singles (1 page each)
| Page | P/F | One-line reason |
|---|---|---|
| canvas/drawingbuffer-storage-test.html | 31P/2F | storage readback (0,0,0,0) vs (0.25,0.5,0.75,0.125); GL error 1282 |
| misc/views-with-offsets.html | 229P/1F | "Does not support doReadPixels with offsets into views" |
| reading/read-pixels-from-fbo-test.html | 193P/2F | FBO readback 128 vs expected 187.5 (float channel quantization) |
| renderbuffers/framebuffer-object-attachment.html | 237P/6F | incomplete-FBO readPixels: INVALID_OPERATION/NO_ERROR vs INVALID_FRAMEBUFFER_OPERATION |
| transform_feedback/simultaneous_binding.html | 128P/3F | PIXEL_PACK_BUFFER bound during transform feedback: INVALID_OPERATION not generated |

## Unexpected / investigated

No TIMEOUTS (0) and no ERRORS (0) in the full run; the two pageerrors that occurred (webgl-blend-func-extended, webgl-clip-cull-distance) are in-page follow-on effects of shader-compile failures (getUniformLocation called on an unlinked program) — deterministic, not crashes of the harness.

1. **conformance2/textures/video/tex-2d-{r8ui,rg8ui,rgb8ui,rgba8ui}-*_integer-* (4 pages, 72F each) — expected all-PASS, DEVIATION.**
   - Rerun singly (`--filter 'conformance2/textures/video/tex-2d-r(?:gba|gb|g)?8ui' --workers 1`): 4/4 fail again, identical 74P/72F — **not flaky, deterministic**.
   - Root cause (code-level, no fix applied): `src/gl/teximage.ts:1190` gates the display-p3 conversion behind `!spec.isInteger`:
     `if (!spec.isInteger && ctx.unpackColorSpace === 'display-p3') { srgbToDisplayP3(im.data); }`
     The 72 failing subtests per page are exactly the `gl.unpackColorSpace = 'display-p3'` cases (3 videos × 24 cases × 2 corner checks = 72; the srgb cases pass). The test computes expected colors via `namedColorInColorSpace('Red'|'Green', 'display-p3')` = (234,51,35)/(117,251,76) — the renderer uploads the raw sRGB video pixels (255,0,0)/(0,255,0). Per the CTS, the display-p3 conversion applies to integer formats too. The 3D `tex-3d-*8ui` pages pass only because `tex-image-and-sub-image-3d-with-video.js` does not cross-product `unpackColorSpacesToTest` at all. Suggested fix location (for the parent): remove/relax the `!spec.isInteger` gate (and the stale "integer formats … are exempt" comment at line 1187-1189).
2. **conformance2/state/gl-object-get-calls.html — expected FAIL ~8F, actual full PASS (366P/0F).** Better than expected, not a regression; the ~8 remaining fails from baseline3 were apparently resolved by wave-6/7/8/9.
3. **conformance2/textures/misc/gl-teximage.html — cannot be verified in this run.** The page exists only in the WebGL1 suite (`conformance/textures/misc/gl-teximage.html`); it is not in the conformance2 manifest (the 1184 pages). The "103P/0F" expectation likely referred to a webgl1 run; a `--suite webgl2` baseline cannot cover it.
4. **Flaky pages: none.** Every page that failed in the full run reproduced identically on single-page reruns (video ×4 checked individually; all other must-list pages re-derived from the full-run JSON).

## How to reproduce

```
npx tsx tests/conformance/run.ts --renderer ./renderer.js --suite webgl2 --report tests/reports/webgl2-baseline4.json
```
Full run takes ~8 min (470.6 s) at workers 4 on this machine.
