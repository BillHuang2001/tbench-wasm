# src/present/ — Presentation & Image Sources

## Intent
Adapts the renderer's internal drawing buffer to the outside world and decodes DOM image sources for texture uploads. Two responsibilities:
1. **CanvasSurface** — blit the software RGBA8 drawing buffer onto an HTMLCanvasElement via its native 2D context (so headless-Chromium screenshots, `canvas.toDataURL`, and drawImage-from-canvas see frames), or hold a pure buffer when there is no canvas (Node, unit tests).
2. **ImageSource decoding** — convert HTMLImageElement / HTMLCanvasElement / HTMLVideoElement / ImageBitmap / VideoFrame / ImageData to straight-alpha RGBA8 for texImage2D/texSubImage2D. Preferred path: byte-exact readback through a dedicated scratch NATIVE WebGL1 context; fallback: 2D drawImage + getImageData (also the ImageBitmap path).

No DOM APIs are touched at module load; every DOM interaction is lazy and feature-detected so the bundle also runs in Node.

## API Surface
Exact names (gl/ and entry.ts import from `./present`):
- `CanvasSurface` — `{ width, height, getPixels(): Uint8Array /* RGBA8 straight */, present(): void, resize(w, h): void }`
- `createCanvasSurface(canvas: unknown): CanvasSurface` — factory: canvas-like object (has a `getContext` function) → `BrowserCanvasSurface`; anything else → `NodeCanvasSurface`.
- `BrowserCanvasSurface` (canvas: HTMLCanvasElement) — lazily acquires the native 2D context; `present()` does `putImageData(0, 0)` reusing a cached `ImageData`; auto-resizes if canvas.width/height drifted; degrades to buffer-only if 2D unavailable; never throws.
- `NodeCanvasSurface` — pure buffer; `present()` no-op.
- `DecodedImage` — `{ width, height, data: Uint8ClampedArray /* RGBA8 straight */ }`
- `ImageSource` — `HTMLImageElement | HTMLCanvasElement | HTMLVideoElement | ImageBitmap | ImageData | OffscreenCanvas | VideoFrame`
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
- **present() per draw**: gl/ calls `present()` after every draw/clear so the canvas bitmap is always current (no explicit flush needed before screenshots). putImageData is a memcpy — acceptable for a software renderer.
- **Buffer owned by the surface, gl/ renders in place**: the surface allocates `Uint8Array(w*h*4)` on resize; gl/ wraps `getPixels()` as the drawing-buffer color storage (RGBA8). `present()` auto-resizes to `canvas.width/height` if they differ (safety net for page-driven canvas resizes between draws).
- **ImageData reuse**: `new ImageData(view, w, h)` built once per resize from a view over the same buffer (constructor does not copy — use a `Uint8ClampedArray` view); putImageData copies only at blit time.
- **Uniform scratch-canvas decode**: all DOM sources decode through one lazily created scratch canvas (`drawImage` + `getImageData`). The scratch is a module-level singleton shared across ALL decodes and contexts, so `decodeViaScratch` resets the transform and clears it to transparent before every `drawImage` (a same-size decode would otherwise composite source-over the previous decode's pixels and corrupt straight-alpha readback). A canvas painted by our own 2D blit reads back correctly (the native 2D context holds the pixels). `SecurityError` (tainted source) and incomplete sources return `{ok:false}` — gl/ maps to INVALID_VALUE per spec.
- **Native-WebGL-readback decode (primary path)**: `decodeViaNativeWebGL` uploads the source into a scratch NATIVE WebGL1 context (captured from `HTMLCanvasElement.prototype.getContext` at module scope, before the test-harness override is installed; invoked via `.call()` on a dedicated scratch canvas so it never recurses into the software renderer), attaches it to a framebuffer (`framebufferTexture2D`) and reads it back with `readPixels` → byte-exact top-down straight RGBA8. Why: the 2D `drawImage`+`getImageData` path is COLOR-MANAGED — sRGB→linear→sRGB round trips collapse RGB levels (~205 unique values instead of 256 on a 256-level ramp; CTS gl-teximage quantization) and ICC profiles are applied; the native GL upload with `UNPACK_COLORSPACE_CONVERSION_WEBGL = NONE` (re-applied on every call) is profile-ignored, which is what the WebGL spec requires (CTS gl-teximage ICC subtests). `UNPACK_FLIP_Y`/`UNPACK_PREMULTIPLY_ALPHA` are also forced off — gl/teximage.ts applies both itself. Readback is TOP-DOWN (row 0 = source top row): never y-flip. Any failure falls back to `decodeViaScratch` (2D), except tainted-source failures which propagate so gl/ throws the required SecurityError (native texImage2D's SecurityError message lacks the words security/taint/insecure, so the reason is prefixed with "security:" to keep gl/'s `/security|taint|insecure/i` mapping working). Result: CTS `gl-teximage.html` went 73P/30F → 103P/0F with zero regressions on the canvas tex-2d gates (1570P/0F WebGL1, 394P/0F WebGL2).
- **ImageBitmap stays on the 2D path**: native readback returns the bitmap's RAW STORAGE bytes, which are premultiplied for default-created bitmaps — and gl/teximage.ts premultiplies AGAIN (`bitmap.premultiply !== false` → premultiply:true), double-premultiplying. The 2D round trip always yields straight alpha, exactly what gl/ expects. Bitmaps created with `premultiplyAlpha:'none'` stay straight on both paths.
- **ImageData direct path**: `ImageData.data` is already straight RGBA8 → direct copy without any DOM (works in Node unit tests).
- **No Node toDataURL**: a valid image URL needs PNG encoding (a real dependency, violating zero-deps); unit tests assert on `getPixels()` instead.

## Known Issues / Gotchas
- `canvas.getContext('2d')` returns null if the canvas already has a native (browser) WebGL/other non-2D context → present() degrades to buffer-only (rendering still correct, just not visible). Degenerate case — handle by degrading, not failing.
- For HTMLImageElement always use `naturalWidth`/`naturalHeight` (intrinsic size), never CSS sizes (`clientWidth`/`offsetWidth`).
- **Videos must be accepted from readyState 1 (HAVE_METADATA)**: a video is frame-drawable (and natively uploadable) once metadata is loaded — the first frame is available at readyState 1. CTS `conformance/textures/misc/video-rotation.html` videos upload at readyState 1; a stricter gate (`readyState >= 2`, HAVE_CURRENT_DATA) rejected them → decode `{ok:false}` → gl/teximage.ts zero-filled the level → all-black quads (1P/64F). Accept `readyState >= 1`; still reject readyState 0 (HAVE_NOTHING).
- `putImageData` does not scale: the buffer size must equal the canvas bitmap size; the auto-resize safety net handles drift.
- Incomplete/broken images must return `{ok:false}` immediately — never block waiting for load (spec: texImage2D with an incomplete source → INVALID_VALUE).
- Cross-origin images taint the scratch canvas → getImageData throws → mapped to `{ok:false}` (INVALID_VALUE). Setting `crossOrigin` is the page's job.
- The scratch 2D context is acquired as `getContext('2d')` with no attributes — transparent; `decodeViaScratch` clears it (setTransform + clearRect) before every drawImage, so the scratch must never be created with `{alpha:false}` (source-over over opaque black would force alpha 255).
- The scratch NATIVE WebGL canvas must NEVER get a 2D context (a canvas that already has a 2D context returns null for `getContext('webgl')`) — it is a dedicated canvas separate from the 2D scratch.
- Setting `canvas.width`/`canvas.height` RESETS all WebGL context state (pixel store, scissor, etc.) — `decodeViaNativeWebGL` re-applies `UNPACK_FLIP_Y=false`, `UNPACK_PREMULTIPLY_ALPHA=false`, `UNPACK_COLORSPACE_CONVERSION_WEBGL=NONE` and `disable(SCISSOR_TEST)` on every call (cheap) rather than only after a resize. The scratch native context is created with `{alpha:true, premultipliedAlpha:false, preserveDrawingBuffer:true, antialias:false, depth:false, stencil:false}` — `alpha:false` would force alpha 255 on readback.
- The native-readback path needs a real browser WebGL context: in Node (no DOM) `getNativeGLContext()` returns null and every decode falls back to the 2D path, preserving the old behavior and `{ok:false}` contracts.

## Test Strategy
- `tests/unit/` (Node, vitest): NodeCanvasSurface resize/getPixels; decodeImageData (pure copy); decodeImageSource failure modes with duck-typed fakes (incomplete image, unsupported source); createCanvasSurface dispatch (fake canvas-like vs plain object). The present tests import the real `createCanvasSurface`/`NodeCanvasSurface` (sibling `tests/unit/` node — read-only from here).
- Browser paths are verified by the CTS (`conformance/textures/`, `canvas/` tests) and the three.js/Babylon screenshot suites (frames visible to `toHaveScreenshot`).

## Status
Implementation complete: `canvas.ts` (BrowserCanvasSurface/NodeCanvasSurface/createCanvasSurface) and `image.ts` (decodeImageSource/decodeImageData/isDecodableImageSource) implemented per the design above; validated with `npx tsc --noEmit` (zero errors) and `npm run test:unit` (115/115). Browser decode validated end-to-end: `conformance/textures/misc/gl-teximage.html` = 103P/0F (was 73P/30F with the color-managed 2D path), `conformance/textures/canvas/tex-2d-rgba-rgba-unsigned_byte.html` = 1570P/0F, `conformance2/textures/canvas/tex-2d-rgba8-rgba-unsigned_byte.html` = 394P/0F.

## Routing Table
- No child directories (leaf node).
- `../gl/` → consumer: drawing-buffer presentation, texImage2D DOM-source decode (sibling — read-only, escalate writes to parent)
- `../entry.ts` → consumer: canvas → surface wiring (sibling — read-only, escalate writes to parent)
- `../raster/` → sibling: `formats.ts` owns the pixel-format encode/decode gl/ uses when wrapping `getPixels()` as a raster surface (read-only, escalate writes to parent)
