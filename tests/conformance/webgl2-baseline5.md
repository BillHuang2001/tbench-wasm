# WebGL2 CTS Baseline 5 — bundle b56a985 (wave-12/13/14 fixes)

- **Date**: 2026-08-28 (run window 20:45:52Z – 20:53:57Z UTC, 484.6 s)
- **Bundle commit**: `b56a985` (`renderer.js: rebuild with wave-12/13/14 fixes (SRC1 dual-source routing, int/uint varying+output bit-pack, TF raw-bit capture, raster integer-encode bit-reinterpret, blend secondary plumbing, index-1 outputs)`)
- **Invocation** (from repo root):
  ```
  npx tsx tests/conformance/run.ts --renderer ./renderer.js --suite webgl2 --report tests/reports/webgl2-baseline5.json
  ```
- **CTS**: `/testsuites/WebGL` (read-only), suite version `2.0.1 (beta)`, manifest `conformance2/00_test_list.txt`
- **Defaults**: workers 4, idle timeout 60 s (per-page slow overrides + 35 min wall-clock cap untouched — no webgl2 page is slow-marked), `?webglVersion=2`
- **Artifacts** (gitignored): `tests/reports/webgl2-baseline5.json`, `tests/reports/webgl2-baseline5.log`
- Shell env had a poisoned `WEBGL_SOFTWARE_RENDERER` — `--renderer ./renderer.js` passed explicitly; never set env var.

## Totals

| PASS | FAIL | TIMEOUT | ERROR | Total |
|------|------|---------|-------|-------|
| 1164 | 20   | 0       | 0     | 1184  |

- Skipped subtests: 0.
- **Renderer-active gate: 0 pages ran without the software renderer** (no `WARNING: N pages ran without the software renderer` in the summary; `rendererActive: false` count in JSON = 0). Every result carries `rendererActive: true`.
- No runner errors: the suite-count assertion (1184) passed, all 1184 pages completed (only `pass`/`fail` statuses in the JSON — no `timeout`, no `error`). Runner exit code 1 (20 failures → expected semantics; no exit-2 conditions). Note: the shell exit code shown was 0 because the command was piped through `tee`; the runner's own summary is authoritative.

## vs baseline4 (1130P / 54F / 0T / 0E, bundle ce8a79c)

**Pass delta: +34** (1130 → 1164), failures 54 → 20. Zero timeouts/errors in both.

| Page (semantic name) | baseline4 | baseline5 | Verified in this run |
|---|---|---|---|
| conformance2/rendering/vertex-id.html | 61P/8F | **0F** | PASS (69P/0F) — gl_VertexID bits fixed |
| conformance2/buffers/get-buffer-sub-data-validity.html | 26P/1F | **0F** | PASS (27P/0F) |
| conformance2/extensions/webgl-blend-func-extended.html | 229P/47F | **0F** | PASS (278P/0F) — SRC1 dual-source routing works |
| conformance2/rendering/blitframebuffer-filter-outofbounds.html | 511P/131F | **0F** | PASS (642P/0F) |
| conformance2/rendering/blitframebuffer-filter-srgb.html | 151P/259F | **0F** | PASS (410P/0F) |
| conformance2/rendering/blitframebuffer-outside-readbuffer.html | 474P/456F | **0F** | PASS (930P/0F) |
| conformance2/textures/misc/tex-unpack-params.html | 568P/417F | **0F** | PASS (985P/0F) |
| conformance2/textures/misc/immutable-tex-render-feedback.html | 2425P/840F | **0F** | PASS (3265P/0F) |
| conformance2/extensions/webgl-multi-draw-instanced-base-vertex-base-instance.html | 88P/397F | **0F** | PASS (775P/0F) — better than the claimed 485P/0F |
| conformance2/extensions/webgl-render-shared-exponent.html | 764P/274F | **0F** | PASS (1042P/0F) |
| conformance2/rendering/attrib-type-match.html | 47P/55F | **0F** | PASS (102P/0F) |
| conformance2/rendering/uniform-block-buffer-size.html | 15P/20F | **0F** | PASS (35P/0F) |
| conformance2/rendering/read-draw-when-missing-image.html | 22P/15F | **0F** | PASS (37P/0F) |
| conformance2/textures/video/tex-2d-{r8ui,rg8ui,rgb8ui,rgba8ui}-*_integer-* (4 pages) | 74P/72F each | **0F** | PASS (146P/0F each) — display-p3 integer conversion fixed |
| conformance2/textures/misc/tex-new-formats.html | 725P/238F | **0F** | PASS (963P/0F) |
| conformance2/textures/misc/integer-cubemap-texture-sampling.html | 401P/120F | **120F** | FAIL (401P/120F) — UNCHANGED, NOT fixed (see Unexpected) |
| conformance2/misc/views-with-offsets.html | 229P/1F | **0F** | PASS (240P/0F) |
| conformance2/extensions/oes-draw-buffers-indexed.html | 1602P/3F | **0F** | PASS (1605P/0F) |
| conformance2/transform_feedback/simultaneous_binding.html | 128P/3F | **0F** | PASS (131P/0F) |
| conformance2/wasm/readpixels-{4gb,2gb-in-4gb,16gb}-wasm-memory.html (3 pages) | 2P/1F each | **0F** | PASS (3P/0F each) — >4GB wasm readPixels fixed |
| conformance2/buffers/uniform-buffers.html | 47P/2F | **0F** | PASS (49P/0F) |
| conformance2/canvas/drawingbuffer-storage-test.html | 31P/2F | **0F** | PASS (33P/0F) |
| conformance2/reading/read-pixels-from-fbo-test.html | 193P/2F | **0F** | PASS (195P/0F) |
| conformance2/renderbuffers/framebuffer-object-attachment.html | 237P/6F | **0F** | PASS (243P/0F) |
| conformance2/textures/misc/copy-texture-* (cube-map-AMD-bug / same-texture / webgl-specific) | 24F/2F/14F | **0F** | PASS (58P/0F, 22P/0F, 34P/0F) |
| conformance2/textures/misc/tex-srgb-mipmap.html | 4P/11F | **0F** | PASS (5P/0F) |
| conformance2/textures/misc/tex-unpack-params-imagedata.html | 11P/3F | **0F** | PASS (14P/0F) |
| conformance2/textures/misc/tex-unpack-params-with-flip-y-and-premultiply-alpha.html | 600P/166F | **0F** | PASS (766P/0F) |
| conformance2/rendering/clear-srgb-color-buffer.html | 3P/2F | **0F** | PASS (5P/0F) |
| conformance2/rendering/clearbuffer-sub-source.html | 21P/6F | **0F** | PASS (27P/0F) |
| conformance2/rendering/draw-buffers.html | 228P/1F | **0F** | PASS (229P/0F) |
| conformance2/rendering/fs-color-type-mismatch-color-buffer-type.html | 11P/4F | **0F** | PASS (15P/0F) |
| conformance2/rendering/instanced-arrays.html | 60P/4F | **0F** | PASS (64P/0F) |
| conformance2/textures/misc/tex-storage-2d.html | 133P/5F | **2F** | FAIL (136P/2F) — improved 5F→2F (see residual clusters) |

All 39 baseline4-failing pages that were expected to pass **do pass**; every must-hold item verified. 15 baseline4 pages still fail (below). **5 NEW failures** appeared (regressions — see Unexpected): `blend-integer`, `uninitialized-test-2`, `gl-get-frag-data-location`, `framebuffer-render-to-layer`, `gl-object-get-calls`. No baseline4-passing page other than these 5 fails now.

## Residual failure clusters (20 pages)

### conformance2/textures/misc/ — 13 pages
| Page | P/F | One-line reason |
|---|---|---|
| compressed-tex-image.html | 40P/32F | compressedTexImage/SubImage2D ETC1: wrong error codes (INVALID_ENUM where NO_ERROR/INVALID_OPERATION expected) + missing "too few args" TypeError (unchanged) |
| generate-mipmap-with-large-base-level.html | 3P/2F | generateMipmap returns INVALID_OPERATION where NO_ERROR expected (unchanged) |
| integer-cubemap-texture-sampling.html | 401P/120F | integer cubemap sampling: (255,255,255) instead of (255,0,0) — raster integer-encode bit-reinterpret did NOT fix (unchanged) |
| npot-video-sizing.html | 1P/1F | video-texture height wrong: (0,0) expected 0,255,0,255 was 255,0,0,255 (unchanged) |
| origin-clean-conformance-offscreencanvas.html | 9P/4F | texImage3D/texSubImage3D with cross-origin image / non-origin-clean canvas should throw, doesn't — OffscreenCanvas-in-Worker runs native (harness limitation; see Known-unfixable) (unchanged) |
| pbo-texture-uploads.html | 2P/2F | PBO-bound uploads: wrong error codes (NO_ERROR/INVALID_VALUE vs INVALID_OPERATION) (unchanged) |
| tex-image-10bpc.html | 2P/1F | 10bpc PNG crushed to 8bpc: uniquePixels 3 < 7 (unchanged) |
| tex-image-and-sub-image-with-array-buffer-view-sub-source.html | 101P/12F | typed-array view sub-source: 12× "Element 4: expected 4 (or 4.000000236555934), got 1" (unchanged) |
| tex-image-with-bad-args.html | 8P/1F | RED/UNSIGNED_SHORT: INVALID_ENUM vs expected INVALID_VALUE/INVALID_OPERATION (unchanged) |
| tex-input-validation.html | 80P/3F | input validation error codes (NO_ERROR where INVALID_* expected): RED/UNSIGNED_BYTE invalid internalFormat combo, RGBA→RGB565, RGBA→RGB (unchanged) |
| tex-mipmap-levels.html | 39P/2F | partial-level draw color wrong (expected 0,0,255,255 got 255,0,0,255); generateMipmap on float texture should fail, doesn't (unchanged) |
| tex-storage-2d.html | 136P/2F | TWO pre-existing upload bugs remain (baseline4 5F→2F; wave-12/13/14 fixed 3): (1) UNSIGNED_INT_10F_11F_11F_REV decode swaps R↔B (`src/gl/teximage.ts:663-667`) → redpixel upload samples blue; (2) `srcBytesPerTexel` default branch returns 4 for FLOAT/INT/UNSIGNED_INT regardless of component count (`teximage.ts:595-606`) → RGBA32F upload scrambled (changed: improved) |
| tex-storage-compressed-formats.html | 14P/30F | texStorage2D/3D compressed 0x9270: INVALID_ENUM where NO_ERROR expected (ASTC 0x9270-0x9276 rejected on 2D_ARRAY; TEXTURE_3D INVALID_ENUM vs INVALID_OPERATION) (unchanged) |

### conformance2/misc/ — 2 pages (both NEW regressions)
| Page | P/F | One-line reason |
|---|---|---|
| blend-integer.html | 14P/12F | uvec outputs misclassified as float: `outputTypeFamily` (`src/gl/draw.ts:1753-1764`) uses WRONG UNSIGNED_INT_VEC2/3/4 enum constants (0x8b5c-5e = FLOAT_MAT4/SAMPLER_1D/SAMPLER_2D; real = 0x8DC6-8) → GLES3 §4.2.1 output-type/attachment mismatch check fires INVALID_OPERATION on unsigned draws → readback = clear values. Blending itself is spec-silent for integer buffers. NEW |
| uninitialized-test-2.html | 4577P/3F | `allocateImmutableStorage` halves depth for TEXTURE_2D_ARRAY too (`src/gl/teximage.ts:1525` `if (!isCube) d >>= 1`; 2D_ARRAY depth must stay constant) → truncated layer views → `resolveAttachmentSurface` hits undefined layer → readPixels INVALID_OPERATION (3 fails, TEXTURE_2D_ARRAY RGBA8/RGBA8UI/RGBA8I line-568 combos; was 4580P/0F). NEW |

### conformance2/rendering/ — 2 pages
| Page | P/F | One-line reason |
|---|---|---|
| clear-func-buffer-type-match.html | 23P/2F | clear with INT/UINT buffer should error INVALID_OPERATION, doesn't (unchanged) |
| framebuffer-render-to-layer.html | 1062P/30F | same uvec bug as blend-integer: render-to-layer for the 10 unsigned formats (R8UI…RGB10_A2UI) × TEXTURE_2D/TEXTURE_3D raises INVALID_OPERATION ("No errors from render to TEXTURE_2D R8UI"). NEW |

### conformance2/extensions/ — 1 page
| Page | P/F | One-line reason |
|---|---|---|
| webgl-clip-cull-distance.html | 251P/1F | `gl_ClipDistance` undeclared in shader; pageerror `TypeError: Argument is not of type 'WebGLProgram'` at getUniformLocation on unlinked program aborts checkClipDistance (unchanged; page IS graded — see Unexpected) |

### conformance2/programs/ — 1 page
| Page | P/F | One-line reason |
|---|---|---|
| gl-get-frag-data-location.html | 11P/2F | getFragDataLocation returns -1 for ALL outputs: `src/gl/api/programs.ts:746-752` index-null guard is dead since 3e94d5d (EXT_blend_func_extended GLSL support always emits a numeric index, `src/glsl/semantics-decl.ts:594-595`) → `fragDataMaps` empty → `fmap.get(nm) ?? -1`. "expected: fragColor0->2, fragColor1->0, got: fragColor0->-1, fragColor1->-1". NEW |

### conformance2/state/ — 1 page
| Page | P/F | One-line reason |
|---|---|---|
| gl-object-get-calls.html | 365P/1F | same getFragDataLocation bug: `gl.getFragDataLocation(program, "fragColor") should be 0. Was -1.` (was 366P/0F). NEW |

## Unexpected / investigated

No TIMEOUTS (0) and no ERRORS (0) in the full run; the only pageerror (webgl-clip-cull-distance) is the in-page follow-on of the shader-compile failure (getUniformLocation on an unlinked program) — deterministic, not a harness crash.

1. **5 NEW failures — real regressions from wave-12/13/14, all deterministic.** Each rerun 3× singly (`npx tsx tests/conformance/debug-page.ts <url> --renderer ./renderer.js --repeat 3`): identical P/F and identical shim-message sets in all 3 runs — not flakes. Two clusters + one singlet:
   - **Cluster A — getFragDataLocation -1 for all outputs** (gl-get-frag-data-location 11P/2F, gl-object-get-calls 365P/1F): the `o.index !== null` guard in `src/gl/api/programs.ts:746-752` is dead since commit 3e94d5d (EXT_blend_func_extended/dual-source "index-1 outputs" work in this bundle) — `src/glsl/semantics-decl.ts:594-595` always emits a numeric `index` (`q.layout?.index ?? 0`), so every output is skipped and `fragDataMaps` stays empty. Fix: drop the index clause (skip only explicit index-1 outputs). **Root cause of the wave-13 SRC1 dual-source routing change.**
   - **Cluster B — INVALID_OPERATION on draws to unsigned-integer attachments** (blend-integer 14P/12F, framebuffer-render-to-layer 1062P/30F): `outputTypeFamily` (`src/gl/draw.ts:1753-1764`) uses wrong UNSIGNED_INT_VEC2/3/4 constants (0x8b5c/0x8b5d/0x8b5e = FLOAT_MAT4/SAMPLER_1D/SAMPLER_2D; real enums 0x8DC6/7/8) → uvec outputs get family 0 (float), and the GLES3 §4.2.1 output-type/attachment-family mismatch check (draw.ts:1929-1948, added in 4ae0525) fires INVALID_OPERATION on every uvec→UINT-attachment draw. ivec is unaffected (signed sub-runs pass; fs-color-type-mismatch still passes 15/15 because it only uses vec4). Fix: correct the three constants.
   - **uninitialized-test-2 4577P/3F** (was 4580P/0F): `allocateImmutableStorage` halves depth for TEXTURE_2D_ARRAY (`src/gl/teximage.ts:1525` `if (!isCube) d >>= 1` — only TEXTURE_3D halves; 2D_ARRAY depth stays constant per spec) → level layer views truncated → `resolveAttachmentSurface` (`src/gl/framebuffer-util.ts:339-378`) hits undefined layer → readPixels INVALID_OPERATION. TEXTURE_3D passes. Fix: exempt 2D_ARRAY from the halving.
   All four located defects are recorded here (fixes un-routed — the parent must route them to `./src`).
2. **conformance2/textures/misc/integer-cubemap-texture-sampling.html NOT fixed — 401P/120F identical to baseline4.** The raster integer-encode bit-reinterpret (wave 14) was expected to fix the (255,255,255)-vs-(255,0,0) integer cubemap sampling; it did not. The int/uint bit-pack work does manifest elsewhere (tex-new-formats 963P/0F, views-with-offsets 240P/0F, wasm ×3, texture-video integer pages), but integer-cubemap remains the one 120F page of the three indicator pages listed in the claim.
3. **webgl-clip-cull-distance.html MANIFEST VERDICT: GRADED.** `conformance2/extensions/00_test_list.txt` line 25 is `--min-version 2.0.1 webgl-clip-cull-distance.html` — a plain graded leaf (not commented out; `--min-version 2.0.1` evaluated against suite version `2.0.1 (beta)` → included). It IS one of the 1184 graded pages. The glsl-side manager's claim that it is "not graded" is **REFUTED by the manifest**. Run status unchanged from baseline4: 251P/1F (`gl_ClipDistance` undeclared).
4. **tex-storage-2d improved 5F→2F** — wave-12/13/14 fixed 3 of the 5 baseline4 failures; the remaining 2 are two distinct pre-existing upload bugs (see residual table; both since draft commit 287034c).
5. **Flaky pages: none.** Every failing page reproduced identically on single-page reruns (all 20 rerun once; the 5 new pages 3× each — 35 rerun runs total, zero anomalies, rendererActive=true everywhere).

## Known-unfixable / out-of-scope

- **origin-clean-conformance-offscreencanvas (9P/4F)** — the 4 failures need texImage3D/texSubImage3D to throw for cross-origin / non-origin-clean sources inside an OffscreenCanvas **Worker**. Playwright `addInitScript` does not reach Worker globals, so the Worker path executes against NATIVE WebGL (the known caveat documented in `tests/CONTEXT.md` for all `conformance/offscreencanvas/` pages). Harness limitation, not a renderer defect; Chrome-native fails this combination the same way.
- **wasm >4GB readPixels — no longer applicable**: all 3 `readpixels-*-wasm-memory` pages PASS (3P/0F) in this run.
- All other residuals are renderer defects with identified root causes and fix locations (see tables) — fixable in principle.

## How to reproduce

```
npx tsx tests/conformance/run.ts --renderer ./renderer.js --suite webgl2 --report tests/reports/webgl2-baseline5.json
```
Full run takes ~8 min (484.6 s) at workers 4. Per-page root causes (shim messages + FAIL lines + pageerror):
```
npx tsx tests/conformance/debug-page.ts <url> --renderer ./renderer.js [--repeat N]
```
