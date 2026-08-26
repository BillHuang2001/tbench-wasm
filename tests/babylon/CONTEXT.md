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
- **pixelmatch as a real devDependency** (it is what Playwright's `toHaveScreenshot` uses internally). Invocation must mirror `node_modules/playwright-core/lib/server/screenshotter.js` (options: threshold; includeAA/alpha settings) — verify parity against that source during implementation. `compareScreenshots` decodes both PNGs with sharp to raw RGBA, runs pixelmatch into a diff buffer, `pass = diffPixels <= maxDiffPixelRatio * w * h`; size mismatch or missing golden → explicit failure reason.
- **Golden naming**: `(referenceImage || title).replace(".png","") + ".png"`, then Playwright filepath sanitization (runs of chars outside `[A-Za-z0-9._-]` → `-`). Known upstream mismatch: 10 config `referenceImage` values use underscores while disk files use dashes (`ext_lights_ies`→`ext-lights-ies`, `pbr_mirror`, `nme_glow_*`, `openpbr_base_*`) — avoid in curated list; if selected they fail golden lookup and are reported as missing.
- **`/cdn` proxy** keeps the page same-origin (no CORS edge cases for `.babylon`/texture loads) and the on-disk cache makes repeat runs offline-capable. `rootPath` does NOT change the fetch URL (upstream semantics); it only prefixes `replaceUrl` rewrites.
- **scriptToRun semantics (ported verbatim from upstream `visualizationPlaywright.utils.ts`)**: fetch `root + scriptToRun`; asset rewrites `../../assets/`, `../../Assets/`, `/assets/` → `root + "/Assets/"`; `replace` = comma-split `[from,to]` pairs, literal first-occurrence `String.replace`; `replaceUrl` = comma-split regex sources, each globally replaced with `root + rootPath + source`; then `window.scene = eval(script + "\n" + functionToCall + "(engine)")`. `specificRoot` → `BABYLON.Tools.BaseUrl = root + specificRoot`.
- **Snippet support (835 tests) is future work** — recipe: fetch `https://snippet.babylonjs.com/{id}` JSON, `typescript.transpileModule` TS→JS, strip module syntax, scene-call detection order `Playground.CreateScene` → `delayCreateScene` → `delayLoadScene` → `CreateScene` → `createScene`, rewrite `textures/` → `https://playground.babylonjs.com/textures/`, apply per-test `replace` pairs.

## Test Strategy
- Smoke (plumbing proof): `npx tsx tests/babylon/run.ts --limit 3 --workers 2` with no `renderer.js` → 3 failures with RENDERER_NOT_FOUND, driver completes, report + screenshots written, exit 1.
- Unit: `npx vitest run tests/babylon` — `scenes.test.ts` (BOM parsing, kind detection, curated/full/filter selection, missing-golden skip), `compare.test.ts` (identical→pass, diff-within-ratio→pass, beyond-ratio→fail, threshold semantics, sanitization).
- Real renderer: run against built `renderer.js`, iterate renderer until scenes converge toward goldens (diffs > 0.035 threshold or > ratio still reported per scene).

## Routing Table
- `run.ts` → CLI entry: args, orchestration, worker pool, report, exit code
- `scenes.ts` → config.json parsing, scene model, curated lists, selection
- `driver.ts` → per-scene Playwright flow (inject, engine, scene, render, screenshot)
- `compare.ts` → screenshot comparison + reference-name sanitization
- `server.ts` → static serving + `/cdn` proxy + disk cache
- `empty.html` → page template (canvas + vendor UMD script order)
- Foreign (read-only): `/testsuites/Babylon.js` → config.json, ReferenceImages, upstream harness reference
