# tests/unit/ — Vitest Unit Tests (Node, no browser)

## Intent
Fast, dependency-free unit tests for the pure `src/` modules — glsl compiler/linker, raster formats & pure raster math, util math, gl state defaults, present's Node surface. This is the iteration loop during implementation (seconds, not minutes); the Khronos CTS remains the gate. Tests import `src/` modules directly (no `renderer.js` bundle, no Playwright, no DOM).

## How to run
- `npm run test:unit` — `vitest run tests/unit` (exit non-zero on any failure)
- `npm run test:watch` — watch mode
- Single file: `npx vitest run tests/unit/intercept.test.ts`
- Config lives at repo root (`vitest.config.ts`: forks pool, include `tests/**/*.test.ts`, 120s timeout).

## Conventions
- **Naming**: one `<area>.test.ts` per src module area, mirroring `src/` layout. `helpers.ts` holds shared utilities (it is NOT a test file).
- **Imports**: static imports from `src/` at their final contract paths (see Contract Assumptions). No bundle, no globals.
- **GL enums**: use the `GL` constants from `./helpers` (spec-fixed numeric values); never import a src/gl constants module.
- **Determinism**: pure functions only, no network, no timers. Typed-array results compared with `expectArrayClose` (tolerance-aware; handles NaN/Infinity).
- **Status discipline**: tests are written against the FINAL contracts and are expected to fail until the corresponding `src/` module lands. Do NOT delete or skip them to make the suite green — the failures are the executable spec for src/ implementers. When a src module lands, its test file should flip to green; fix the test only if the src contract changed (update this file's Contract Assumptions in the same pass).
- **File size**: each test file ≤ ~1000 lines; split by module.

## Current status (at architecture time)
| File | Status | Turns green when |
|---|---|---|
| `intercept.test.ts` | ✅ PASSES today | already green |
| `glsl.test.ts` | ❌ module not found (`src/glsl/index`) | src/glsl lands |
| `formats.test.ts` | ❌ module not found (`src/raster/formats`) | src/raster/formats lands |
| `math.test.ts` | ❌ module not found (`src/util/math`) | src/util/math lands |
| `raster.test.ts` | ❌ module not found (`src/raster/index`) | src/raster lands |
| `state.test.ts` | ❌ module not found (`src/gl/state`) | src/gl/state lands |
| `present.test.ts` | ❌ module not found (`src/present/index`) | src/present lands |

Expected failure mode: vitest "Failed to resolve import" per file; the rest of the suite still runs. `npm run test:unit` exits non-zero until src/ implementation lands — that is the intended state.

## Contract Assumptions (coordination contract with src/)
Import paths and export names assumed from src/CONTEXT.md. If the src/ layout differs, update ONLY the import lines / call sites in the listed file (they are marked in each test file's header), and record the change here. The semantics under test are the src contracts.

| Test file | Import | Assumed exports |
|---|---|---|
| `glsl.test.ts` | `../../src/glsl/index` | `compileShader(source, {type:'VERTEX'\|'FRAGMENT', version:100\|300, ...}) → {ok:true, shader} \| {ok:false, errors:[{line, message}]}`; `linkProgram(vs, fs) → {ok:true, program} \| {ok:false, log}`; `Program.{attributes[{name,location,components}], uniforms[{name,location,components}], varyings[{name,components}], usesPointSize, vertex.run(ctx), fragment.run(ctx)}` with exec ctx per src/CONTEXT.md §1 |
| `formats.test.ts` | `../../src/raster/formats` | `getFormat(internalFormat) → PixelFormat \| undefined` with `{bytesPerPixel, components, decode(data, off) → [r,g,b,a], encode(data, off, r,g,b,a), isDepth, isStencil, isFloat, isSigned, isInteger, normalized}` (decode: floats 0..1 for normalized, raw for float/int, a=1 when no alpha; encode: floats in, quantizes); `convertPixels(srcFormat, dstFormat, src, dst, width, height, opts?: {flipY?, premultiplyAlpha?})` |
| `math.test.ts` | `../../src/util/math` | gl-matrix-style out-param helpers: `vec3.{dot, cross, length, normalize, add, sub, scale}`, `mat4.{identity, multiply, invert, perspective, translate}` (column-major 16-element) |
| `raster.test.ts` | `../../src/raster/index` | `edgeFunction(a, b, c) → signed area` (positive inside for CCW), `clipInterpolate(a, b, t) → linear interp`, `perspectiveCorrect(s, t, w, va, vb, vc)` (barycentric / perspective-correct) |
| `state.test.ts` | `../../src/gl/state` | `createState() → fresh state`; shape follows the DrawCall contract (§2): `blend{enabled,srcRGB,dstRGB,srcAlpha,dstAlpha,eqRGB,eqAlpha,color}`, `depthTest{enabled,func}`, `depthMask`, `cull{enabled,face,frontFace}`, `scissor{enabled,x,y,w,h}`, `stencilTest{enabled,front,back}`, `sampleCoverage{enabled,value,invert}`, `polygonOffset{enabled,factor,units}`, `colorMask[4]`, `clearColor/clearDepth/clearStencil`, `dither`, `rasterizerDiscard`, `lineWidth`, `activeTexture`, `pixelStore{packAlignment,unpackAlignment,flipY,premultiplyAlpha}`, `currentProgram` |
| `present.test.ts` | `../../src/present/index` | `createNodeSurface(width, height) → {width, height, getPixels(): Uint8Array (RGBA8), present(): void}` (present no-op in Node) |
| `intercept.test.ts` | `../../src/context-intercept` | (module exists today — no assumptions; uses explicit missing/temp renderer paths so it stays green after `renderer.js` is built) |

## Test Strategy
- `glsl.test.ts` — compile/link smoke (trivial v+f shaders), compile-error line numbers, WebGL1 `in`-keyword strictness vs v300 in/out, Program metadata, vertex exec (position/varyings/uniforms/gl_PointSize/gl_VertexID/gl_InstanceID), built-in function evaluation, fragment gl_FragColor.
- `formats.test.ts` — metadata (bpp/components/classification/storage) for ~16 core formats; encode/decode round-trips with quantization tolerances (RGBA8 exact-ish, RGB565/RGBA4/RGB5_A1 within bit depth, LUMINANCE(+ALPHA) semantics, DEPTH_COMPONENT16/24 f32, R16F f32, RGBA32F); `getTexImageConverter` per-texel source conversions (RGBA→LUMINANCE, RGB→RGBA8, RGBA→RGB565 — no flipY, that's gl/'s concern); implemented numeric helpers (halfToFloat/floatToHalf, sRGB round-trips, packDepth24Stencil/unpackDepth24).
- `math.test.ts` — vec3 dot/cross/length/normalize/add/sub/scale; mat4 identity/multiply/invert/perspective (near→z=-1, far→z=+1)/translate.
- `raster.test.ts` — edge-function sign/magnitude vs triangle interiors, clip interpolation endpoints/midpoints/long records, perspective-correct interpolation incl. equal-w and vertex degeneration.
- `state.test.ts` — GL spec defaults (clear values, masks, blend/depth/cull/stencil/scissor/polygonOffset/sampleCoverage, dither, rasterizerDiscard, lineWidth, activeTexture, pixelStore, currentProgram) + fresh-per-call independence.
- `present.test.ts` — Node surface dimensions, RGBA8 buffer size, no-op present.
- `intercept.test.ts` — `buildInterceptScript` (renderer-present: embeds source, routes webgl/webgl2/experimental-webgl through `__createSoftwareWebGLContext`, falls through for '2d'; renderer-missing: RENDERER_NOT_FOUND stub that throws), `getRendererPath` env handling, `assertRendererExists`.
- Not yet covered (add when APIs settle): ImageSource decoding, texture sampler, fragment-ops blending math, raster primitive assembly. `helpers.ts` is the home for a `makeContext()`-style harness once gl/ lands.

## Constraints
- Never import the built `renderer.js` bundle; always import `src/` modules directly.
- No browser APIs, no Playwright, no network — pure Node.
- Keep each file ≤ ~1000 lines.

## Routing Table
- `helpers.ts` → shared GL enum constants + `expectArrayClose` (and future shared harnesses)
- `intercept.test.ts` → context-intercept harness helper tests (passes today)
- `glsl.test.ts` → GLSL compiler/linker/Program model tests (fails until src/glsl)
- `formats.test.ts` → pixel format registry + conversion tests (fails until src/raster/formats)
- `math.test.ts` → vec/mat helper tests (fails until src/util/math)
- `raster.test.ts` → pure raster math tests (fails until src/raster)
- `state.test.ts` → GL state default tests (fails until src/gl/state)
- `present.test.ts` → Node canvas surface tests (fails until src/present)
