# src/present/ — Presentation & Image Sources

## Intent
Adapts the renderer's internal drawing buffer to the outside world and decodes DOM image sources for texture uploads. Two responsibilities:
1. **CanvasSurface** — blit the software RGBA8 drawing buffer onto an HTMLCanvasElement via its native 2D context (so headless-Chromium screenshots, `canvas.toDataURL`, and drawImage-from-canvas see frames), or hold a pure buffer when there is no canvas (Node, unit tests).
2. **ImageSource decoding** — convert HTMLImageElement / HTMLCanvasElement / HTMLVideoElement / ImageBitmap / ImageData to straight-alpha RGBA8 for texImage2D/texSubImage2D.

No DOM APIs are touched at module load; every DOM interaction is lazy and feature-detected so the bundle also runs in Node.

## API Surface
Exact names (gl/ and entry.ts import from `./present`):
- `CanvasSurface` — `{ width, height, getPixels(): Uint8Array /* RGBA8 straight */, present(): void, resize(w, h): void }`
- `createCanvasSurface(canvas: unknown): CanvasSurface` — factory: canvas-like object (has a `getContext` function) → `BrowserCanvasSurface`; anything else → `NodeCanvasSurface`.
- `BrowserCanvasSurface` (canvas: HTMLCanvasElement) — lazily acquires the native 2D context; `present()` does `putImageData(0, 0)` reusing a cached `ImageData`; auto-resizes if canvas.width/height drifted; degrades to buffer-only if 2D unavailable; never throws.
- `NodeCanvasSurface` — pure buffer; `present()` no-op.
- `DecodedImage` — `{ width, height, data: Uint8ClampedArray /* RGBA8 straight */ }`
- `ImageSource` — `HTMLImageElement | HTMLCanvasElement | HTMLVideoElement | ImageBitmap | ImageData | OffscreenCanvas`
- `DecodeResult` — `{ ok: true, image: DecodedImage } | { ok: false, reason: string }`
- `decodeImageSource(source: ImageSource): DecodeResult`
- `decodeImageData(source: ImageData): DecodeResult` — pure direct-copy path (no DOM; Node-testable; duck-typed `{width, height, data}` accepted)
- `isDecodableImageSource(value: unknown): boolean` — gl/ dispatch helper (buffer views vs DOM sources)

## Constraints
- **No DOM at module load**: all `document`/canvas/ImageData access happens inside functions, guarded by feature detection. DOM types are compile-time only (erased at runtime; the bundle runs in Node).
- **Zero allocation in the present() hot path**: one cached `ImageData` reused across presents; only `resize()` reallocates (buffer + ImageData).
- **`present()` never throws**: adapter errors are swallowed and degrade to no-op (the GL error queue is the only error channel to the page).
- **Straight RGBA8 everywhere**: decoded images and the surface buffer are always straight (non-premultiplied) RGBA8. UNPACK_PREMULTIPLY_ALPHA_WEBGL / UNPACK_FLIP_Y_WEBGL / UNPACK_COLORSPACE_CONVERSION handling belongs to gl/ teximage, not here.
- **Buffer ownership**: the surface owns its RGBA8 buffer; `getPixels()` returns it (same identity until `resize()`). gl/ renders directly into it (zero-copy) and must re-fetch after any `resize()`.
- Files ≤ ~1000 lines. Leaf module: no imports from `gl/`, `raster/`, `glsl/`; may use `util/` if needed.

## Design Decisions
- **Native 2D blit**: `canvas.getContext('2d')` + `putImageData`. The test-harness getContext override falls through to the native 2D context for `'2d'`, so a plain call works. Playwright `page.screenshot`/`toHaveScreenshot` and `canvas.toDataURL` composite the 2D canvas bitmap.
- **present() per draw**: gl/ calls `present()` after every draw/clear so the canvas bitmap is current at draw time (no explicit flush needed before screenshots taken in the same task). Caveat: for `preserveDrawingBuffer:false` contexts the next frame boundary re-presents the wiped (black) buffer — see Known Issues.
- **Buffer owned by the surface, gl/ renders in place**: the surface allocates `Uint8Array(w*h*4)` on resize; gl/ wraps `getPixels()` as the drawing-buffer color storage (RGBA8). `present()` auto-resizes to `canvas.width/height` if they differ (safety net for page-driven canvas resizes between draws).
- **ImageData reuse**: `new ImageData(view, w, h)` built once per resize from a view over the same buffer (constructor does not copy — use a `Uint8ClampedArray` view); putImageData copies only at blit time.
- **Uniform scratch-canvas decode**: all DOM sources decode through one lazily created scratch canvas (`drawImage` + `getImageData`). The scratch is a module-level singleton shared across ALL decodes and contexts, so `decodeViaScratch` resets the transform and clears it to transparent before every `drawImage` (a same-size decode would otherwise composite source-over the previous decode's pixels and corrupt straight-alpha readback). A canvas painted by our own 2D blit reads back correctly (the native 2D context holds the pixels). `SecurityError` (tainted source) and incomplete sources return `{ok:false}` — gl/ maps to INVALID_VALUE per spec.
- **ImageData direct path**: `ImageData.data` is already straight RGBA8 → direct copy without any DOM (works in Node unit tests).
- **No Node toDataURL**: a valid image URL needs PNG encoding (a real dependency, violating zero-deps); unit tests assert on `getPixels()` instead.

## Known Issues / Gotchas
- **`preserveDrawingBuffer:false` (WebGL default) → the 2D canvas bitmap is overwritten with BLACK at the next frame boundary** (verified by live probing with bundle md5 d950a9cf: a Babylon frame yields 9899 non-black pixels via readPixels right after render/endFrame, ALL BLACK after the page settles, and page.screenshot() pure black). present() itself is synchronous (`putImageData`, canvas.ts:168) and blits the FRAME immediately after every draw/clear/blit/clearBuffer that touches the default FB (`presentIfDefault`, ../gl/draw.ts:269-312, call sites 1936/1997/2286/2432). But for `preserveDrawingBuffer:false` gl then schedules a rAF-deferred wipe (`schedulePreserveClear`, ../gl/draw.ts:331-372 — rAF in browsers, setImmediate/microtask/setTimeout in Node) which (1) zero-fills the internal drawing buffer (`clearDefaultFramebufferForPreserve`, ../gl/draw.ts:381-391) and (2) RE-PRESENTS the cleared buffer (../gl/draw.ts:349-353) — so the 2D bitmap, and hence page.screenshot()/toHaveScreenshot, ends up black. This is NOT a present-layer bug: the internal wipe is spec-required (CTS `conformance/canvas/buffer-preserve-test.html`, readPixels after composite) and the re-present is REQUIRED by CTS `conformance/context/context-attribute-preserve-drawing-buffer.html` (drawImage(webglCanvas) after composite must read (0,0,0,255)); both are in the graded 887. Real browsers reconcile display-vs-readback via a separate compositor layer; a single 2D bitmap cannot, and the design chose CTS compliance over page display. **Visual-regression drivers (tests/threejs, tests/babylon) MUST create contexts with `preserveDrawingBuffer:true`** (three.js: `new WebGLRenderer({preserveDrawingBuffer:true})`; Babylon: `new Engine(canvas, antialias, {preserveDrawingBuffer:true})`) or every screenshot of a default context is black. Not triggered by endFrame/readPixels/getContext('2d') — only by the next rAF after a default-FB draw.
- `canvas.getContext('2d')` returns null if the canvas already has a native (browser) WebGL/other non-2D context → present() degrades to buffer-only (rendering still correct, just not visible). Degenerate case — handle by degrading, not failing.
- For HTMLImageElement always use `naturalWidth`/`naturalHeight` (intrinsic size), never CSS sizes (`clientWidth`/`offsetWidth`).
- `putImageData` does not scale: the buffer size must equal the canvas bitmap size; the auto-resize safety net handles drift.
- Incomplete/broken images must return `{ok:false}` immediately — never block waiting for load (spec: texImage2D with an incomplete source → INVALID_VALUE).
- Cross-origin images taint the scratch canvas → getImageData throws → mapped to `{ok:false}` (INVALID_VALUE). Setting `crossOrigin` is the page's job.
- The scratch 2D context is acquired as `getContext('2d')` with no attributes — transparent; `decodeViaScratch` clears it (setTransform + clearRect) before every drawImage, so the scratch must never be created with `{alpha:false}` (source-over over opaque black would force alpha 255).

## Test Strategy
- `tests/unit/` (Node, vitest): NodeCanvasSurface resize/getPixels; decodeImageData (pure copy); decodeImageSource failure modes with duck-typed fakes (incomplete image, unsupported source); createCanvasSurface dispatch (fake canvas-like vs plain object). The present tests import the real `createCanvasSurface`/`NodeCanvasSurface` (sibling `tests/unit/` node — read-only from here).
- Browser paths are verified by the CTS (`conformance/textures/`, `canvas/` tests) and the three.js/Babylon screenshot suites (frames visible to `toHaveScreenshot`).

## Status
Implementation complete: `canvas.ts` (BrowserCanvasSurface/NodeCanvasSurface/createCanvasSurface) and `image.ts` (decodeImageSource/decodeImageData/isDecodableImageSource) implemented per the design above; validated with `npx tsc --noEmit` (zero errors in src/present) and Node smoke checks. Browser 2D-blit path works but is overridden by gl's deferred preserve-clear re-present for `preserveDrawingBuffer:false` contexts (see Known Issues) — visual suites must use `preserveDrawingBuffer:true`.

## Routing Table
- No child directories (leaf node).
- `../gl/` → consumer: drawing-buffer presentation, texImage2D DOM-source decode (sibling — read-only, escalate writes to parent)
- `../entry.ts` → consumer: canvas → surface wiring (sibling — read-only, escalate writes to parent)
- `../raster/` → sibling: `formats.ts` owns the pixel-format encode/decode gl/ uses when wrapping `getPixels()` as a raster surface (read-only, escalate writes to parent)
