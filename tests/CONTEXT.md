# tests/ — Verification Harnesses

## Intent
Verification of the software WebGL renderer against the Khronos CTS (primary gate: 2,071 tests), three.js and Babylon.js visual regression suites, plus fast Node unit tests of `src/` modules. All browser-based harnesses inject the built `renderer.js` via the `src/context-intercept.ts` helper (Playwright `addInitScript`), so NO real GPU is ever used.

## How injection works (all browser suites)
`src/context-intercept.ts` `buildInterceptScript()`: returns JS that (1) defines `window.__createSoftwareWebGLContext` from the `renderer.js` bundle, (2) overrides `HTMLCanvasElement.prototype.getContext` so `'webgl'|'webgl2'|'experimental-webgl'` route to it (passing the requested type as 3rd arg), and (3) falls through to the native implementation for other types ('2d' — which the renderer itself uses for presentation blits). Default renderer path `./renderer.js`; override with env `WEBGL_SOFTWARE_RENDERER`.

## Shared runner requirements (conformance, threejs, babylon)
- Serve test assets over HTTP (no file://): no-cache headers; plain MIME types OK. CTS origin-clean tests need the conformance-resources logo served from a cross-origin URL: the dual-loopback server binds the SAME port on both 127.0.0.1 and localhost, so the logo is served from the opposite loopback hostname (same port, different origin) — no second port needed.
- Launch Chromium headless (Playwright); the installed browser is `chromium-1217`. Consider `--disable-gpu`; the software renderer does all work in JS.
- Per-test timeout generous (CTS: 60s; visual: 120s+); record timeouts distinctly.
- Concurrency: renderer is CPU-bound; default ~4 parallel pages (configurable). Results must be collected in-process (never via DOM for high-volume suites).
- Output: JSON report (per-test: pass/fail/skip/timeout, subtests, time) + human summary; `npm run test:conformance` etc. exit non-zero on any failure.

## Conformance suite (tests/conformance/) — the PRIMARY gate
- **Target: exactly 2,071 tests = 887 (conformance/) + 1184 (conformance2/)** from `/testsuites/WebGL/sdk/tests` at suite version 2.0.1 (all `--min-version` directives included). Optional `--deqp` flag adds the 885-page deqp WebGL2 suite.
- Parse 00_test_list.txt files with the OFFICIAL harness semantics (see root CONTEXT.md "List parsing") — port `parseTestList` from `sdk/tests/webgl-conformance-tests.html`; verify the parse reproduces 887/1184 exactly; fail loudly otherwise.
- Result capture: inject `window.webglTestHarness = { reportResults(url, success, msg, skipped), notifyFinished(url) }` via addInitScript BEFORE page scripts (js-test-pre.js caches `window.parent.webglTestHarness` at load). Poll for completion (notifyFinished or DOM "TEST COMPLETE" fallback), timeout → fail. Aggregate subtests: pass = failures 0 AND not timed out; skipped subtests do not fail the test.
- URL: `http://127.0.0.1:<port>/sdk/tests/<path>?webglVersion=<1|2>` (conformance → 1; conformance2/deqp → 2) + `&imgUrl=http://localhost:<port>/<logo>` (same port, opposite loopback hostname — cross-origin) where needed.
- Report: JSON + summary table; track per-test history for regression hunting (results dir, gitignored).
- **Renderer-active gate**: every result carries `rendererActive` (factory presence, polled per page). A page that completes or times out while `window.__createSoftwareWebGLContext` is missing is forced FAIL (`renderer bundle not active (window.__createSoftwareWebGLContext missing) — page ran WITHOUT the software renderer`), and the summary prints `WARNING: N pages ran without the software renderer (bundle likely dead)` — a dead bundle can never silently pass as native WebGL again.
- The runner must work even with a stub renderer (all tests fail with clear RENDERER_NOT_FOUND-style errors) — plumbing is validated independently of renderer progress.

## three.js suite (tests/threejs/)
Drive a curated subset of `/testsuites/three.js/examples/*.html` (webgl2-only renderer): serve `examples/` + `build/` + `jsm/`, inject the renderer + deterministic rAF/random (mirror `test/e2e/deterministic-injection.js` approach), screenshot after N frames, compare against `/testsuites/three.js/examples/screenshots/*.png` references (the repo's own goldens) with a tolerant metric (see `test/e2e/image.js`). Start with ~20-40 representative pages (animation, materials, geometry, postprocessing, shadowmap, particles); `--full` flag to run more. Report per-page pass/fail with diff stats.

## Babylon suite (tests/babylon/)
Mirror Babylon's own visual regression flow: serve `packages/tools/babylonServer/public/` (empty.html + UMD `vendor/babylon-cdn/babylon.js`), load scenes from `packages/tools/tests/test/visualization/config.json` (LOCAL sceneFolder/sceneFilename entries only — remote snippet IDs need network), render N frames, screenshot, compare against `packages/tools/tests/test/visualization/ReferenceImages/*.png` (threshold 0.035, maxDiffPixelRatio as in `playwright.config.ts`). Curated subset (~15-30 scenes) + `--full`. Engine creation: `new BABYLON.Engine(canvas, ...)` with the options used by the suite (`useHighPrecisionFloats`, `forceSRGBBufferSupportState`, `powerPreference: 'high-performance'`).

## Unit suite (tests/unit/)
Vitest (Node, no browser): fast tests for pure `src/` modules — glsl lexer/parser/codegen (compile + run small shaders against expected outputs), formats decode/encode round-trips, math, raster edge-function/clipping math, state defaults, extension registry. These are the iteration loop during implementation; the CTS remains the gate. Import `src/` modules directly (no bundle).

## Constraints
- Never modify foreign repos (read-only). All goldens/servers come from them at runtime.
- Keep runner code ≤ ~1000 lines per file; shared Playwright/server helpers in `tests/conformance/` may be imported by other suites only via a shared module (escalate to parent if a truly common helper is needed — parent decides placement).
- Reports written under `tests/reports/` (gitignored).

## Known Issues / Gotchas (all browser suites)
- **`WEBGL_SOFTWARE_RENDERER` env poisoning**: if the env var points to a NONEXISTENT file, `buildInterceptScript()` silently returns the RENDERER_NOT_FOUND stub (getContext **throws** "Software renderer not found") — no bundle is injected, and every test fails with "Unable to fetch WebGL rendering context for Canvas" (or crashes with `Cannot read properties of null (reading 'createProgram')` on pages that don't null-check `gl`). The agent shell commonly has `WEBGL_SOFTWARE_RENDERER=/app/renderer.js` (a path that does not exist in worktrees) — ALWAYS pass `--renderer ./renderer.js` (or a real path) explicitly.
- **Silent native fallback — gated in ALL three browser suites**: a bundle that THROWS at load (e.g. the `chainToNative` cyclic-proto bug — see `src/gl/CONTEXT.md` Known Issues) never defines `window.__createSoftwareWebGLContext`. When bundle+wrapper are injected as ONE init script (the normal `buildInterceptScript` path), the wrapper never installs either, so `getContext` stays NATIVE and pages silently grade native SwiftShader WebGL with zero error signal (the 832/835 "PASS" webgl1 run was 100% native). Every driver now guards this with a factory-presence check (`typeof window.__createSoftwareWebGLContext === 'function'`); a page without the factory is forced FAIL with `renderer bundle not active (window.__createSoftwareWebGLContext missing) — page ran WITHOUT the software renderer` and the summary prints `WARNING: N pages ran without the software renderer (bundle likely dead)`. Per driver: conformance polls `rendererActive` in `pollSnapshot` and forces fail on ALL completion paths (harness-finished, DOM-fallback, timeout); threejs checks after each successful goto (deterministic status `renderer-inactive`, skips the screenshot comparison; a goto timeout with unverified factory also fails); babylon checks after goto before engine creation (fails the scene and skips the screenshot so a native frame can never match the golden).
- **OffscreenCanvas not intercepted**: `src/context-intercept.ts` patches only `HTMLCanvasElement.prototype.getContext`. `OffscreenCanvas.getContext` (used by ALL 12 tests under `conformance/offscreencanvas/`, e.g. `offscreencanvas-transfer-image-bitmap.html` does `new OffscreenCanvas(128,128).getContext('webgl', ...)`) is NOT routed to the software renderer. Currently masked by the native fallback; must be addressed when the software renderer becomes the gate. (Irrelevant to threejs/babylon — their pages use plain `<canvas>` only.)

## Routing Table
- `conformance/` → Khronos CTS runner (2,071-test gate), test-list parsing, harness shim, static server
- `threejs/` → three.js example-page visual regression driver
- `babylon/` → Babylon.js visual regression driver
- `unit/` → vitest unit tests for `src/` modules
