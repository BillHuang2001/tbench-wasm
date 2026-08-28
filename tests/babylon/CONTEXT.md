# tests/babylon/ — Babylon.js Visual Regression Driver

## Intent
Runs curated Babylon.js scenes against the software WebGL renderer in headless Chromium (Playwright), screenshots each scene, and compares against Babylon's own golden references (`ReferenceImages/*.png`) using Playwright-equivalent pixelmatch semantics. Entry: `npx tsx tests/babylon/run.ts` (= `npm run test:babylon`). Works end-to-end even with a stub renderer (no `renderer.js`): scenes fail with clear RENDERER_NOT_FOUND errors, driver completes, report written, exit code 1.

## Pipeline (per run)
1. `scenes.ts` parses `/testsuites/Babylon.js/packages/tools/tests/test/visualization/config.json` (**UTF-8 BOM!**) and selects runnable entries: 11 `sceneFolder+sceneFilename` ("file" kind) + 7 `scriptToRun+functionToCall` ("script" kind). The 835 `playgroundId` snippet entries are OUT OF SCOPE (skipped; see Design Decisions).
2. `server.ts` serves `empty.html` + the 4 vendor UMD bundles (`vendor/babylon-cdn/`: babylon.js, babylon.gui.min.js, babylonjs.materials.min.js, babylonjs.loaders.min.js), and proxies `/cdn/*` → `BABYLON_CDN` (default `https://cdn.babylonjs.com`) with an on-disk cache. Scene assets exist ONLY on the CDN — the foreign repo has no scene files.
3. `driver.ts`: per scene — new browser context (viewport 600×400, matching golden dims), `addInitScript(buildInterceptScript())` from `src/context-intercept.ts` (renderer injection; stub when `renderer.js` missing), `goto empty.html`, init engine, create scene, render `renderCount||1` frames + `scene.isReady()` poll, `page.screenshot()` (always taken, even on scene error → blank page for diffing).
4. `compare.ts`: pixelmatch (real lib, devDependency — same library Playwright uses) with `threshold` 0.035 and `maxDiffPixelRatio` = `errorRatio/100 || SCREENSHOT_MAX_PIXEL/100 || 1.1/100`, mirroring upstream `toHaveScreenshot` options.
5. `run.ts`: CLI orchestration, worker pool, JSON report → `tests/reports/babylon-report.json`, failed screenshots + diff PNGs → `tests/reports/babylon/`, exit non-zero on any failure.

## API Surface
CLI: `run.ts [--filter REGEX] [--full] [--workers N] [--renderer PATH] [--list] [--limit N] [--no-cache] [--timeout SECONDS] [--out PATH] [--cdn URL] [--root PATH]`
Env: `WEBGL_SOFTWARE_RENDERER` (renderer bundle; default `./renderer.js` — inherited from `src/context-intercept.ts`), `BABYLON_ROOT` (foreign repo; default `/testsuites/Babylon.js`), `BABYLON_CDN` (default `https://cdn.babylonjs.com`), `BABYLON_PORT` (default 0 = ephemeral), `SCREENSHOT_THRESHOLD` (default 0.035), `SCREENSHOT_MAX_PIXEL` (default 1.1; per-test `errorRatio` beats it).

Modules:
- `run.ts` — CLI parse, scene selection, server + browser lifecycle, worker pool, report, exit code
- `scenes.ts` — `SceneEntry` model, `loadConfig()`, curated lists, `selectScenes({full, filter, limit})`
- `driver.ts` — `runScene(browser, serverUrl, entry, opts)` → `SceneResult` (in-page engine/scene/render protocol)
- `compare.ts` — `sanitizeReferenceName()`, `goldenPath()`, `compareScreenshots()`
- `server.ts` — `createServer({cdn, cacheDir, noCache})` → `{port, url, close}`
- `empty.html` — page template (canvas `#babylon-canvas`, 100%×100%, loads vendor UMDs)

## Default scene list
Curated default (16): file — Sponza, Windows cafe, Espilit, The car (`errorRatio` 5), Viper, Retail, Hill Valley, Heart, SpaceDeK (`errorRatio` 5), Flat2009; script — Fog, Polygon, Lines, Lens, Self shadowing, GUI (`errorRatio` 4). `--full` adds Mansion and Procedural textures (both `excludeFromAutomaticTesting: true` upstream). All 18 have goldens on disk (600×400). Selection validates golden existence and warns+skips missing ones.

## Constraints
- `/testsuites/Babylon.js` is READ-ONLY — goldens + config.json are read at runtime, never modified.
- Scene assets are CDN-only → served through the `/cdn` proxy; `--no-cache` disables the disk cache; on network outage each scene fails with a clear error and the driver still completes and reports.
- No playground snippets (835 entries) — out of scope by design decision.
- Deviations from upstream harness (documented, intentional): (1) engine created WITHOUT `failIfMajorPerformanceCaveat` (a software renderer would trip it); (2) no draco/ktx2/basis/gltf-validation decoder configs (not served) — scenes relying on them show asset/decoder errors, recorded per scene; (3) `useHighPrecisionFloats: true`, `forceSRGBBufferSupportState: true`, `powerPreference: "high-performance"`, `disableWebGL2Support: false` kept as upstream.
- Determinism recipe (must match upstream): seed `window.seed = 1` + sine-LCG `Math.random` BEFORE scene creation; `scene.useConstantAnimationDeltaTime = true`; `engine.getCaps().parallelShaderCompile = undefined`; `engine.enableOfflineSupport = false`; `engine.setDitheringState(false)`; `BABYLON.SceneLoader.ShowLoadingScreen = false`; `ForceFullSceneLoadingForIncremental = true`; disable `activeCamera.useAutoRotationBehavior`; viewport 600×400.
- Cache + report dirs live under `tests/reports/` (gitignored at repo root).
- Files ≤ ~1000 lines; zero runtime deps beyond devDeps (playwright, sharp, pixelmatch, tsx).

## Design Decisions
- **Vendored pixelmatch** (`pixelmatch.cjs` + `pixelmatch.d.ts`): byte-for-byte copy of playwright-core 1.59.1's `lib/third_party/pixelmatch.js` (mapbox/pixelmatch, ISC — Playwright's own copy, so diff counts match `toHaveScreenshot` semantics by construction). The npm `pixelmatch` devDependency is declared in package.json but NOT actually installed in this environment, so the spec's fallback rule (vendor playwright's copy with attribution) applies. The file is `.cjs` because the repo's `"type": "module"` would mis-parse the CJS `module.exports` file as ESM under tsx (vitest tolerates it via its transform pipeline; tsx does not); `compare.ts` loads it via `createRequire(import.meta.url)` typed by `pixelmatch.d.ts`. `compare.test.ts` has a parity test asserting identical diff counts between our copy and playwright-core's live copy on synthetic opaque images (identical / single-pixel diff / gradient / checkerboard, threshold 0.035; skips if playwright-core is absent). Invocation mirrors `playwright-core/lib/server/utils/comparators.js`: `pixelmatch(a, b, diff, w, h, { threshold })` — threshold is the ONLY option. `compareScreenshots` decodes both PNGs with sharp (`removeAlpha().ensureAlpha()` → RGBA opaque — note: `removeAlpha()` alone yields 3-channel RGB, which pixelmatch rejects), runs pixelmatch into a diff buffer, `pass = diffPixels <= maxDiffPixelRatio * w * h`; size mismatch or missing golden → explicit failure reason.
- **Golden naming**: `(referenceImage || title).replace(".png","") + ".png"`, then Playwright filepath sanitization (runs of chars outside `[A-Za-z0-9._-]` → `-`). Known upstream mismatch: 10 config `referenceImage` values use underscores while disk files use dashes (`ext_lights_ies`→`ext-lights-ies`, `pbr_mirror`, `nme_glow_*`, `openpbr_base_*`) — avoid in curated list; if selected they fail golden lookup and are reported as missing.
- **`/cdn` proxy** keeps the page same-origin (no CORS edge cases for `.babylon`/texture loads) and the on-disk cache makes repeat runs offline-capable. `rootPath` does NOT change the fetch URL (upstream semantics); it only prefixes `replaceUrl` rewrites.
- **scriptToRun semantics (ported verbatim from upstream `visualizationPlaywright.utils.ts`)**: fetch `root + scriptToRun`; asset rewrites `../../assets/`, `../../Assets/`, `/assets/` → `root + "/Assets/"`; `replace` = comma-split `[from,to]` pairs, literal first-occurrence `String.replace`; `replaceUrl` = comma-split regex sources, each globally replaced with `root + rootPath + source`; then `window.scene = eval(script + "\n" + functionToCall + "(engine)")`. `specificRoot` → `BABYLON.Tools.BaseUrl = root + specificRoot`.
- **Snippet support (835 tests) is future work** — recipe: fetch `https://snippet.babylonjs.com/{id}` JSON, `typescript.transpileModule` TS→JS, strip module syntax, scene-call detection order `Playground.CreateScene` → `delayCreateScene` → `delayLoadScene` → `CreateScene` → `createScene`, rewrite `textures/` → `https://playground.babylonjs.com/textures/`, apply per-test `replace` pairs.

## Test Strategy
- Smoke (plumbing proof): `npx tsx tests/babylon/run.ts --limit 3 --workers 2` with no `renderer.js` → 3 failures with RENDERER_NOT_FOUND, driver completes, report + screenshots written, exit 1.
- Unit: `npx vitest run tests/babylon` — `scenes.test.ts` (BOM parsing, kind detection, curated/full/filter selection, missing-golden skip), `compare.test.ts` (identical→pass, diff-within-ratio→pass, beyond-ratio→fail, threshold semantics, sanitization).
- Real renderer: run against built `renderer.js`, iterate renderer until scenes converge toward goldens (diffs > 0.035 threshold or > ratio still reported per scene).

## Known Issues (first real-renderer run, wave-5 bundle: 16 scenes, 0 passed)
- **ALL 16 screenshots were pure black (every scene, including the 11 that timed out)** — combination of two renderer bugs + one driver gap:
  1. `schedulePreserveClear` (`src/gl/draw.ts:331-372`) rAF-clear wipes the drawing buffer and re-presents BLACK after the render loop stops (`preserveDrawingBuffer:false` default). This wipe is CTS-required (buffer-preserve tests), so it cannot be removed — the DRIVER must create the Engine with `preserveDrawingBuffer: true` in the options (driver.ts engine-creation site). Proven by instrumentation: Lines draw #1 renders 9,899 pixels — EXACTLY the golden's white-pixel count — then the frame goes black before screenshot.
  2. Default-VAO duplication: the constructor VAO (state.ts:390) vs the separate `ctx._defaultVAO` (created at lost.ts:250 with a NULL element-array-buffer); first `bindVertexArray(null)` (webgl2.ts:920-922) switches state to the EAB-less VAO; Babylon's EAB cache then skips rebinding → INVALID_OPERATION queued silently → zero pixels on later draws. One-line fix: `ctx._state.vao = ctx._defaultVAO` after lost.ts:250 (+ rebuild renderer.js via scripts/build.mjs).
- **The 11 scene timeouts (ok:false, ready:false, ~120s) are hangs on a black path** — `scene.isReady()` never true / `executeWhenReady` never fires while frames present black; NOT slow-but-rendering (fog.png/lens.png are pure black, mean RGB 0). Fixes above (1)+(2) are expected to unblock readiness.
- **Lines (ok:true, ready:true, 1.4–2.4s, diff 0.0412 vs maxDiffPixelRatio 0.011) is a STRUCTURAL failure, not AA**: actual = 100% black; diffPixels = 9,899 = exactly the golden's own white pixels; bounding box spans the full frame. No subpixel component.
- **Harness gaps (tests/babylon — driver fixes)**:
  - `window.canvas` is NEVER set: driver.ts creates the Engine with a local `canvas` var; upstream sets `window.canvas = document.getElementById("babylon-canvas")` in `evaluateInitEngineForVisualization` (visualizationPlaywright.utils.ts:367). Cached `shadows.js`/`gui.js` scenes use bare `canvas` (e.g. `camera.attachControl(canvas, true)`) → `ReferenceError: canvas is not defined` (Self shadowing, GUI).
  - `empty.html` loads only the 4 vendor UMDs; upstream empty.html loads `earcut.min.js` (+ ammo.js, recast.js, cannon.js, Oimo.js, havok UMDs) BEFORE babylon.js. Babylon core uses the global `earcut` as default triangulator → `ReferenceError: earcut is not defined` (Polygon). No curated scene needs the physics engines (`enablePhysics` unused); earcut is the only required add.

## Routing Table
- `run.ts` → CLI entry: args, orchestration, worker pool, report, exit code
- `scenes.ts` → config.json parsing, scene model, curated lists, selection
- `driver.ts` → per-scene Playwright flow (inject, engine, scene, render, screenshot)
- `compare.ts` → screenshot comparison + reference-name sanitization
- `pixelmatch.cjs` → vendored pixelmatch (playwright-core 1.59.1 copy, ISC) — do not edit; keep byte-identical to playwright-core
- `pixelmatch.d.ts` → type declaration for the vendored pixelmatch copy
- `server.ts` → static serving + `/cdn` proxy + disk cache
- `empty.html` → page template (canvas + vendor UMD script order)
- Foreign (read-only): `/testsuites/Babylon.js` → config.json, ReferenceImages, upstream harness reference
