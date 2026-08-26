# WebGL Software Renderer (pure JS)

## Intent
A pure-JS (no GPU, no browser, no outside libraries at runtime) software implementation of the WebGL 1.0 and WebGL 2.0 APIs — including a GLSL ES 1.00/3.00 compiler, triangle rasterizer, and the full WebGL API surface — verified against the Khronos WebGL Conformance Test Suite (2,071 tests: 887 WebGL1 + 1184 WebGL2), the three.js visual regression suite, and the Babylon.js visual regression suite.

## Architecture Overview
- **`renderer.js`** (repo root): the BUILT browser-side entry bundle. Defines `window.__createSoftwareWebGLContext(canvas, attrs[, type]) → WebGLRenderingContext | WebGL2RenderingContext | null`. Built from `src/entry.ts` by `scripts/build.mjs` (esbuild IIFE). Committed to git; tests read it directly.
- **`src/`**: renderer implementation in TypeScript, zero runtime deps. Modules: `util/` (shared helpers), `glsl/` (compiler+linker), `raster/` (rasterizer, fragment ops, texture sampling, pixel formats), `gl/` (WebGL API contexts/objects/extensions/state), `present/` (canvas adapters, image sources). `src/context-intercept.ts` is the test-harness helper that routes `canvas.getContext('webgl'|'webgl2')` to `window.__createSoftwareWebGLContext` (NOT part of the bundle).
- **`tests/`**: verification harnesses — `conformance/` (Khronos CTS runner), `threejs/` (visual regression driver), `babylon/` (visual regression driver), `unit/` (vitest unit tests of src modules).
- **`scripts/`**: build tooling.
- **`vendor/babylon-cdn/`**: offline Babylon.js UMD bundles (fetched by `scripts/fetch-babylon-cdn.sh`) used by `tests/babylon/`.

## Key Facts (verified by investigation)
- **CTS counts match exactly**: `sdk/tests/conformance/00_test_list.txt` = **887** tests and `sdk/tests/conformance2/00_test_list.txt` = **1184** tests when parsed with official harness semantics at suite version 2.0.1 (all `--min-version` directives included). These are the 2,071 graded tests. `sdk/tests/deqp/` (885 WebGL2 pages) is NOT in the 2,071 count (optional).
- **Result reporting**: standalone CTS test pages report via `window.webglTestHarness.reportResults(locationPathname, success, msg, skipped)` per subtest and `notifyFinished(locationPathname)` at the end (cached by `js/js-test-pre.js` from `window.parent.webglTestHarness` at load — so the shim must be injected BEFORE page scripts, e.g. via Playwright `addInitScript`). Fallback: DOM scrape of "TEST COMPLETE: N PASS, M FAIL" in `#description`.
- **Version selection**: runner must append `?webglVersion=1` (conformance) / `?webglVersion=2` (conformance2, deqp) — `WebGLTestUtils.getUrlOptions().webglVersion` decides the context version (path inference is only a fallback).
- **List parsing**: 00_test_list.txt syntax = full-line comments (`#`, `;`, `//`), whitespace tokens, directives `--min-version <v>`, `--max-version <v>`, `--slow` (any order, inherit into sublists), paths ending `.txt` recurse. Version compare is numeric-dotted. Port the official `parseTestList` semantics exactly (see `sdk/tests/webgl-conformance-tests.html`).
- **Serving**: plain no-cache static HTTP works; origin-clean tests need the conformance-resources logo served from a SECOND origin (they honor an `imgUrl` URL override).
- **three.js** (v0.185 dev): `WebGLRenderer` requests **webgl2 only** (no WebGL1 fallback); needs graceful `getExtension()` handling for ~15+ extensions (EXT_color_buffer_float, WEBGL_clip_cull_distance, OES_texture_float_linear, EXT_color_buffer_half_float, WEBGL_multisampled_render_to_texture, WEBGL_render_shared_exponent, WEBGL_multi_draw, KHR_parallel_shader_compile, EXT_clip_control, EXT_texture_filter_anisotropic, EXT_texture_norm16, WEBGL_lose_context, WEBGL_debug_shaders, compressed formats). Its e2e suite: `test/e2e` via puppeteer, ~600 example pages, screenshots compared against `examples/screenshots/` references.
- **Babylon.js** (9.22.1): requests `webgl2` with fallback to `webgl`; visual tests via Playwright (`npm run test:visualization`) comparing screenshots to `packages/tools/tests/test/visualization/ReferenceImages/*.png` (threshold 0.035); scenes from `packages/tools/tests/test/visualization/config.json`; requires many extensions implemented (OES_vertex_array_object, OES_element_index_uint, OES_standard_derivatives, OES_texture_float(+linear), OES_texture_half_float(+linear), ANGLE_instanced_arrays, EXT_blend_minmax, EXT_color_buffer_float/half_float, EXT_frag_depth, EXT_sRGB, EXT_shader_texture_lod, EXT_texture_filter_anisotropic, EXT_texture_norm16, KHR_parallel_shader_compile, OES_draw_buffers_indexed, WEBGL_depth_texture, WEBGL_draw_buffers, WEBGL_lose_context, WEBGL_debug_renderer_info, compressed formats).
- Extensions listed in the CTS WebGL1 core list that MUST be implemented: OES_texture_float(+variants), OES_texture_half_float(+variants), OES_element_index_uint, OES_standard_derivatives, OES_vertex_array_object, OES_fbo_render_mipmap, ANGLE_instanced_arrays, EXT_blend_minmax, EXT_frag_depth, EXT_sRGB, EXT_shader_texture_lod, EXT_texture_filter_anisotropic, WEBGL_depth_texture, WEBGL_draw_buffers, WEBGL_blend_func_extended, WEBGL_lose_context, WEBGL_debug_renderer_info, WEBGL_debug_shaders.

## Execution Model (cross-module contract — child nodes MUST conform)
Defined in detail in `src/CONTEXT.md`. Summary: `glsl/` compiles shaders to JS functions (codegen) with a per-program introspection model (attributes/uniforms/varyings/uniformBlocks); vertex output = clip-space position + point size + packed varyings (Float32Array); `raster/` consumes post-VS vertex records, does primitive assembly + homogeneous clipping + interpolation + fragment execution + fragment ops; `gl/` owns all GL state and objects, drives vertex evaluation, and calls the rasterizer with a draw call; `present/` presents the drawing buffer to the canvas (browser: blit via native 2D context; headless Node: internal buffer). Pixel formats live in `raster/formats.ts` as the single shared format registry.

## Constraints
- Zero runtime dependencies; bundle must run in headless Chromium AND Node (no DOM required for core paths).
- No per-fragment allocation in hot paths (typed arrays; preallocated scratch buffers).
- GL error queue per spec; internal exceptions never propagate to the page.
- Files ≤ ~1000 lines; split modules by responsibility (single responsibility, low coupling, high cohesion).
- `renderer.js` is a build artifact but committed — `scripts/build.mjs` regenerates it.

## Test Strategy
- `tests/conformance/` is the PRIMARY gate (2,071 tests). Build it first and iterate the renderer against it.
- `tests/unit/` (vitest, Node) for fast feedback on pure modules (glsl compiler, formats, math, state).
- `tests/threejs/`, `tests/babylon/` — curated visual regression subsets (full suites are the grader's verification; our drivers must run representative pages and can run the full suites on demand).
- Foreign repos (`/testsuites/WebGL`, `/testsuites/three.js`, `/testsuites/Babylon.js`) are READ-ONLY.

## Routing Table
- `src/` → Renderer implementation (TS) + `context-intercept.ts` harness helper
- `tests/` → Verification harnesses (CTS runner, three.js/Babylon drivers, unit tests)
- `scripts/` → Build tooling (`build.mjs`, `fetch-babylon-cdn.sh`)
- `vendor/babylon-cdn/` → Offline Babylon.js UMD bundles
- `renderer.js` → Built browser entry bundle (generated from `src/entry.ts`)
- `package.json` → npm scripts (build, test, test:conformance, test:threejs, test:babylon)
- Foreign (read-only): `/testsuites/WebGL` → Khronos CTS; `/testsuites/three.js` → three.js; `/testsuites/Babylon.js` → Babylon.js
