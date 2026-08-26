# tests/ — Verification Harnesses

## Intent
Verification of the software WebGL renderer against the Khronos CTS (primary gate: 2,071 tests), three.js and Babylon.js visual regression suites, plus fast Node unit tests of `src/` modules. All browser-based harnesses inject the built `renderer.js` via the `src/context-intercept.ts` helper (Playwright `addInitScript`), so NO real GPU is ever used.

## How injection works (all browser suites)
`src/context-intercept.ts` `buildInterceptScript()`: returns JS that (1) defines `window.__createSoftwareWebGLContext` from the `renderer.js` bundle, (2) overrides `HTMLCanvasElement.prototype.getContext` so `'webgl'|'webgl2'|'experimental-webgl'` route to it (passing the requested type as 3rd arg), and (3) falls through to the native implementation for other types ('2d' — which the renderer itself uses for presentation blits). Default renderer path `./renderer.js`; override with env `WEBGL_SOFTWARE_RENDERER`.

## Shared runner requirements (conformance, threejs, babylon)
- Serve test assets over HTTP (no file://): no-cache headers; plain MIME types OK. CTS origin-clean tests need the conformance-resources logo served from a second origin (they accept an `imgUrl` override).
- Launch Chromium headless (Playwright); the installed browser is `chromium-1217`. Consider `--disable-gpu`; the software renderer does all work in JS.
- Per-test timeout generous (CTS: 60s; visual: 120s+); record timeouts distinctly.
- Concurrency: renderer is CPU-bound; default ~4 parallel pages (configurable). Results must be collected in-process (never via DOM for high-volume suites).
- Output: JSON report (per-test: pass/fail/skip/timeout, subtests, time) + human summary; `npm run test:conformance` etc. exit non-zero on any failure.

## Conformance suite (tests/conformance/) — the PRIMARY gate
- **Target: exactly 2,071 tests = 887 (conformance/) + 1184 (conformance2/)** from `/testsuites/WebGL/sdk/tests` at suite version 2.0.1 (all `--min-version` directives included). Optional `--deqp` flag adds the 885-page deqp WebGL2 suite.
- Parse 00_test_list.txt files with the OFFICIAL harness semantics (see root CONTEXT.md "List parsing") — port `parseTestList` from `sdk/tests/webgl-conformance-tests.html`; verify the parse reproduces 887/1184 exactly; fail loudly otherwise.
- Result capture: inject `window.webglTestHarness = { reportResults(url, success, msg, skipped), notifyFinished(url) }` via addInitScript BEFORE page scripts (js-test-pre.js caches `window.parent.webglTestHarness` at load). Poll for completion (notifyFinished or DOM "TEST COMPLETE" fallback), timeout → fail. Aggregate subtests: pass = failures 0 AND not timed out; skipped subtests do not fail the test.
- URL: `http://127.0.0.1:<port>/sdk/tests/<path>?webglVersion=<1|2>` (conformance → 1; conformance2/deqp → 2) + `&imgUrl=http://127.0.0.1:<port2>/<logo>` where needed.
- Report: JSON + summary table; track per-test history for regression hunting (results dir, gitignored).
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

## Routing Table
- `conformance/` → Khronos CTS runner (2,071-test gate), test-list parsing, harness shim, static server
- `threejs/` → three.js example-page visual regression driver
- `babylon/` → Babylon.js visual regression driver
- `unit/` → vitest unit tests for `src/` modules
