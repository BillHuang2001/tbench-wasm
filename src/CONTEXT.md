# src/ — Software WebGL Renderer (implementation)

## Intent
Pure-JS software implementation of WebGL 1.0 (`WebGLRenderingContext`) and WebGL 2.0 (`WebGL2RenderingContext`): GLSL ES 1.00/3.00 compiler, triangle rasterizer, texture pipeline, and the full WebGL API surface. Compiled by `scripts/build.mjs` into the single-file bundle `renderer.js` (IIFE, ES2019, zero deps) that runs in headless Chromium and Node. Entry: `src/entry.ts`.

## API Surface
- `src/entry.ts` — defines `window.__createSoftwareWebGLContext(canvas, attrs, type)` (type: `'webgl' | 'webgl2' | 'experimental-webgl'`, optional; default `'webgl'`). Returns a new context for the canvas or null on conflict (spec: one context per canvas; `'webgl'` and `'experimental-webgl'` share the WebGL1 slot). Also exports `__createSoftwareWebGLContext` for Node (`module.exports`) for unit tests. Context classes: `WebGLRenderingContext` (WebGL1) and `WebGL2RenderingContext extends WebGLRenderingContext` (WebGL2). **Native instanceof compatibility** (`src/gl/native-chain.ts`): in browsers, every context/object class prototype is re-chained under the NATIVE global class of the same name at module load (`Object.setPrototypeOf` — prototype-only, never the constructor chain), so `gl instanceof WebGLRenderingContext` / `gl.createBuffer() instanceof WebGLBuffer` (native classes) are true — required by CTS `webgl-test-utils.js` (`isWebGLContext`) and `instanceof-test.js`. In browsers, a WebGL2 context is `instanceof WebGL2RenderingContext` but NOT `instanceof WebGLRenderingContext` (mirrors native Chromium); in Node the native globals are absent and the plain `extends` chain applies. WebGL2's prototype carries the full WebGL1+2 API + C1+C2 constants as OWN properties (installAll/installConstants on both prototypes), so the re-chain loses nothing.
- `src/context-intercept.ts` — test-harness helper (exists; NOT bundled). Injects `renderer.js` + a `HTMLCanvasElement.prototype.getContext` override into Playwright pages. It passes the requested context type as the 3rd arg to `__createSoftwareWebGLContext`.

## Constraints
- Zero runtime dependencies. ES2019-compatible output. No DOM required in core paths (canvas is an abstract surface via `present/`).
- Typed-array based; NO per-fragment allocation in hot paths (rasterizer, shader execution, texture sampling).
- GL error queue per spec (INVALID_ENUM/INVALID_VALUE/INVALID_OPERATION/OUT_OF_MEMORY/CONTEXT_LOST_WEBGL); internal exceptions must never leak to the page — catch at API boundary, report via GL errors.
- Each file ≤ ~1000 lines. Split by responsibility. Shared utilities live at the lowest common ancestor (`util/`).
- Do NOT implement the full API in one class file — split the WebGL1/WebGL2 contexts into focused modules (state, objects, validation, draw, getters) composed into the context classes.

## Cross-Module Contracts (MANDATORY — all child nodes must conform)

### 1. glsl/ → gl/, raster/ (shader compilation & program model)
Public API (exact names):
```ts
compileShader(source: string, opts: { type: 'VERTEX'|'FRAGMENT', version: 100|300,
              defines?: Record<string, string>, extensions?: Set<string> })
  → { ok: true, shader: Shader } | { ok: false, errors: { line: number, message: string }[] }
linkProgram(vs: Shader, fs: Shader) → { ok: true, program: Program } | { ok: false, log: string }
```
`Program` must expose (gl/ needs this for ALL API surface: getActiveAttrib/Uniform, bindAttribLocation, getUniformLocation, UBO offset queries, draw validation):
- `attributes: AttribInfo[]` — `{ name, location, type /*GLenum*/, size, components, integral }`
- `uniforms: UniformInfo[]` — `{ name, location /*index into uniform store*/, type, size, components, integral, blockIndex /*-1 if default block*/, sampler: boolean }`
- `uniformBlocks: UniformBlockInfo[]` — `{ name, index, size, activeUniforms: [{ name, offset, type, size, arrayStride, matrixStride, rowMajor }] }` (WebGL2)
- `varyings: VaryingInfo[]` — `{ name, type, components, flat }` (packed contiguously in vertex-out order)
- `vertex: { run(ctx: VertexExecCtx): void }`, `fragment: { run(ctx: FragmentExecCtx): void }` plus `fragment.usesDerivatives`, `fragment.usesFragDepth`, `fragment.outputs: { location, type }[]`
- `usesPointSize` (vertex writes gl_PointSize), `usesGLPointCoord` etc. as needed by raster.

Execution model:
- **Vertex**: `ctx.attribs` = array of TypedArray views (by location) or numbers (WebGL1 constant attribs) set up by gl/; `ctx.uniforms` = typed-array uniform store (per-program, laid out by location) + block stores; `ctx.vertexId`, `ctx.instanceId`. Outputs: `ctx.out.position = [x,y,z,w]` (clip space), `ctx.out.pointSize`, `ctx.out.varyings` (Float32Array packed per VaryingInfo order). `run()` may be called per instance (instancing: gl/ re-evaluates with instanceId set).
- **Fragment**: `ctx.varyings` = array of `{ v: Float32Array /*interpolated*/, ddx?: Float32Array, ddy?: Float32Array }` (derivative arrays valid only when `usesDerivatives`); `ctx.fragCoord = [x,y,z,w]`; `ctx.frontFacing`; `ctx.pointCoord`; `ctx.uniforms` (same store). Outputs: `ctx.out.color[location] = [r,g,b,a]` (location 0 = gl_FragColor; WebGL2 multiple outputs), `ctx.out.fragDepth` (if usesFragDepth).
- The compiled code must handle: precision qualifiers, `#extension` enable/disable, built-in functions per version, WebGL1 strictness (varying/attribute keywords, gl_FragColor vs gl_FragData exclusivity, no dynamic loop bounds issues per GLSL spec — constant/uniform bounds), WebGL2 features (in/out, flat, layout qualifiers, gl_VertexID/gl_InstanceID, textureLod, integer textures, UBOs, samplers with bindings).
- Compile errors must be reported with 1-based line numbers (CTS checks error text loosely but line numbers matter in some tests).
- **Codegen to JS is the required strategy** (compile GLSL → JS function source → `new Function`) for performance (visual regression suites render large scenes). Interpreters are not acceptable for the full pipeline.

### 2. gl/ → raster/ (draw pipeline)
gl/ evaluates vertices (attribute fetch, per-vertex `vertex.run`) into a packed buffer: per vertex `[px,py,pz,pw, pointSize, varyings...]` (varyings laid out per Program.varyings), then calls:
```ts
rasterizer.draw(dc: DrawCall): void
interface DrawCall {
  mode: GLenum /* TRIANGLES|TRIANGLE_STRIP|TRIANGLE_FAN|LINES|LINE_STRIP|LINE_LOOP|POINTS */
  count: number; first: number; instanceCount: number;
  vertices: Float32Array /* packed records */; vertexStride: number;
  program: Program; varyingsOffset: number; /* start of varyings in record */
  fb: FramebufferTarget; /* color[] + depth + stencil surfaces (see raster→gl) */
  viewport: {x,y,w,h}; depthRange: {near,far};
  scissor: {enabled, x, y, w, h}; cull: {enabled, face, frontFace};
  polygonOffset: {enabled, factor, units}; dither: boolean;
  colorMask: [boolean x4 per output]; blend: {enabled, srcRGB, dstRGB, srcAlpha, dstAlpha, eqRGB, eqAlpha, color};
  depthTest: {enabled, func}; depthMask: boolean; stencilTest: {enabled, front: StencilState, back: StencilState};
  sampleCoverage: {enabled, value, invert}; rasterizerDiscard: boolean;
  lineWidth: number; /* only 1.0 required, but wider must not crash */
}
```
raster/ does: primitive assembly (strip/fan/loop), homogeneous clipping (all 6 clip planes, interpolating position + varyings linearly in clip space — CTS clipping tests are strict), perspective-correct varying interpolation (with flat interpolation for flat varyings and integer/uint varyings), 2×2 fragment quads (derivatives; helper invocations must not write outputs), gl_PointCoord + point size + point sprite behavior (gl_PointCoord for POINTS; `conformance/rendering/point-*` tests), depth/stencil/scissor/blend/colorMask/dither per spec, and writes into the target surfaces. Fragment shader evaluation happens inside raster (it owns the fragment loop and quad generation).

### 3. raster/ → gl/ (surfaces, formats, textures)
- `raster/formats.ts` is the SINGLE shared pixel-format registry: for every WebGL1+2 internal format (RGBA8, RGB8, RGBA4, RGB5_A1, RGB565, LUMINANCE/LUMINANCE_ALPHA/ALPHA, DEPTH_COMPONENT16/24/32F, DEPTH24_STENCIL8, DEPTH_STENCIL, R8..RGBA32UI, R16F..RGBA32F, RGB10_A2, SRGB8_ALPHA8, RGBA16F etc.): `{ bytesPerPixel, components, decode(data, off) → [r,g,b,a] (float 0..1 or raw), encode(...), isDepth, isStencil, isFloat, isSigned, isInteger, normalized }`. Also source-format conversion tables for texImage2D/texSubImage2D (all WebGL1+2 source formats × internal formats), readPixels pack conversion, and renderbuffer format validation. gl/ allocates, uploads, and reads via this module.
- Surfaces (drawing buffer, renderbuffers, texture levels used as FBO attachments) are plain typed-array-backed objects: `{ width, height, format, data: Uint8Array|Float32Array|..., depthData?, stencilData? }` — raster writes them, gl reads them (readPixels) and presents them (present/).
- `raster/texture-sampler.ts`: `sampleTexture(img: TextureImage, state: SamplerState, u, v, w?, ...) → [r,g,b,a]` — implements NEAREST/LINEAR (+ mipmap filters NEAREST_MIPMAP_NEAREST etc.), wrap modes (REPEAT, CLAMP_TO_EDGE, MIRRORED_REPEAT), LOD (base/max level, min/max lod, textureLod bias), depth comparison (shadow samplers), anisotropy (EXT_texture_filter_anisotropic). `TextureImage = { levels: {width,height,depth,data}[], internalFormat, ... }`. Integer texture sampling returns raw integers (for integer samplers).
- The sampler must support 2D, CUBE, 3D, 2D_ARRAY textures, and WebGL2 texelFetch.

### 4. present/ (canvas & images)
- `CanvasSurface` abstraction: `{ width, height, getPixels(): Uint8Array (RGBA8), present(): void }`. Browser adapter: lazily obtains the canvas's native 2D context (fall through to original getContext for '2d') and `putImageData` after draws (so screenshots/canvas.toDataURL see the frame). Node adapter: pure buffer, `present()` no-op.
- `ImageSource` decoding for texImage2D/texSubImage2D with DOM sources (HTMLImageElement, HTMLCanvasElement, ImageData, HTMLVideoElement): decode to RGBA8 via 2D context drawImage + getImageData; headless-safe path when no DOM.
- Drawing buffer ownership: gl/ allocates the default framebuffer surface via raster/formats; present/ only reads it for presentation. `preserveDrawingBuffer:false` is an optimization hint — the buffer may persist (screenshots need the last frame visible).

### 5. gl/ state & objects
- `state.ts`: plain mutable state container (caps, blend, depth, stencil, viewport, scissor, colorMask, current program/buffers/textures/VAO, pixelStore, activeTexture, etc.) shared with raster via DrawCall. No logic, just data + defaults.
- Objects (WebGLBuffer, WebGLTexture, WebGLProgram, WebGLShader, WebGLFramebuffer, WebGLRenderbuffer, WebGLVertexArrayObject, WebGLSampler, WebGLQuery, WebGLSync, WebGLTransformFeedback, WebGLUniformLocation, WebGLActiveInfo, WebGLShaderPrecisionFormat) are thin JS classes validated via `validateObject()` (CTS `misc/type-conversion-test.html` requires objects to reject cross-context / fake objects).
- Context lifecycle: one context per (canvas, type-slot); `getContext('webgl2')` on a canvas with a 'webgl' context returns null and vice versa; canvas width/height changes resize the drawing buffer (state reset semantics per spec — verify against CTS `canvas/` tests); `WEBGL_lose_context` + `webglcontextlost/restored` events; `isContextLost()`; `getContextAttributes()` honoring alpha/depth/stencil/antialias/premultipliedAlpha/preserveDrawingBuffer/powerPreference/failIfMajorPerformanceCaveat.
- Extensions live in `gl/extensions/` — registry maps name → factory; `getExtension()` returns the singleton object or null; `getSupportedExtensions()` lists enabled ones. WebGL2-only extensions attach to WebGL2 contexts; WebGL1 extensions (OES_*/EXT_*/WEBGL_*/ANGLE_*) attach to WebGL1 (and where the spec says so, also WebGL2 — e.g. OES_texture_float etc. are core in WebGL2 and must NOT be exposed there; check each spec).
- WebGL2: VAOs, UBOs, transform feedback (with primitive mode validation + `transformFeedbackVaryings`), samplers, sync objects, queries (occlusion — must actually count samples for CTS occlusion tests), instanced draws, `drawBuffers`, integer textures/attachments (readPixels integer variants), multisample renderbuffers (`renderbufferStorageMultisample` — resolve on blitFramebuffer; antialiasing edges need not be perfect but CTS `multisampled_depth_*` tests must pass), `texStorage2D/3D`, `copyTexImage*`, `readBuffer`, `getFragDataLocation`, `vertexAttribDivisor`, `uniformBlockBinding`, `getIndexedParameter`, `getInternalformatParameter`, `getSyncParameter`, `getQueryParameter`, `getTransformFeedbackVarying`, `invalidateFramebuffer`, `blitFramebuffer` (color+depth+stencil, with filtering for color), `framebufferTextureLayer`.

## Design Decisions
- **Why JS codegen for GLSL**: correctness (CTS shader tests) AND performance (three.js/Babylon render scenes with tens of thousands of triangles; an interpreter would be too slow for the visual suites). Compile once at link time; the generated function closes over nothing mutable except the exec context.
- **Why 2×2 quads in raster**: derivatives (dFdx/dFdy, textureLod implicit derivatives) require neighbor fragments; GPUs always evaluate in quads. Raster always evaluates fragments in 2×2 quads (helper invocations discarded) so `usesDerivatives` needs no special path.
- **Why formats.ts is shared**: every module (texture upload, FBO allocation, readPixels, sampling, blending) needs format knowledge; one registry prevents drift.
- **Why present via native 2D blit**: headless Chromium screenshots (`toHaveScreenshot`, page.screenshot) composite the 2D canvas content; putImageData after each draw makes the last frame visible without preserveDrawingBuffer semantics games.
- **TypeScript + esbuild**: strict typing catches API-surface bugs in a codebase this size; esbuild produces a single dependency-free IIFE in milliseconds.
- **WASM not used**: pure JS with typed arrays meets the "pure-JS/WASM" requirement with far less toolchain risk; performance is adequate for the target suites.

## Known Issues / Gotchas
- `__createSoftwareWebGLContext(canvas, attrs)` (2-arg) is the contract from the objective; the 3rd arg (type) is an extension used by `src/context-intercept.ts`. When type is absent, default to 'webgl'.
- CTS `context-lost` tests exist in the WebGL1 list — implement lose/restore semantics (state reset, resources invalidated) per spec.
- `failIfMajorPerformanceCaveat: true` — decide behavior per CTS `context-attributes.html` expectations (software implementations may return null; check the test).
- Some CTS WebGL1 tests are gated `--min-version 2.0` in the list (suite-version gating) — they ARE part of the 887 and must run with `?webglVersion=1`.
- deqp suite (885 pages) is optional; each page has millions of subtests (one has 33M reportResults calls) — never route results through the DOM; count in-process.
- **Committed `renderer.js` may lag HEAD src/**: the bundle was last rebuilt at commit 32751d0; later merged branches (e.g. 229ba99 gl/draw.ts drawBuffers/blend/clear/readPixels changes, T1-A456 glsl fixes) are in src/ but NOT in the committed bundle (md5 d950a9cf855c5d3712ca7bb4ca5d7648). Tests that load `./renderer.js` exercise the 32751d0 snapshot, not HEAD src/. If a fix lands in src/ without a rebuild, rebuild via `scripts/build.mjs` (or verify with `git log 32751d0..HEAD -- src`). Line primitives are unaffected: `src/raster/lines.ts`/`rasterizer.ts`/`clip.ts` are unchanged since before the build.

## Routing Table
- `util/` → math (matrices/vectors for glsl codegen support and raster), typed-array helpers, small utilities, logging (debug hooks)
- `glsl/` → GLSL ES 1.00/3.00 lexer/parser/preprocessor/type-checker/builtins/codegen, linker, Program model (contract §1)
- `raster/` → primitive assembly, clipping, rasterization, fragment ops, texture sampler, formats.ts, surfaces (contract §2, §3)
- `gl/` → WebGLRenderingContext + WebGL2RenderingContext, state.ts, all WebGL object classes, extensions, validation (contract §5)
- `present/` → canvas adapters (browser 2D blit / Node buffer), image source decoding (contract §4)
- `entry.ts` → bundle entry: wires gl + present + canvas registry, defines `window.__createSoftwareWebGLContext`
- `context-intercept.ts` → test-harness helper (NOT bundled)
