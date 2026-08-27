# tests/unit/ — Vitest Unit Tests (Node, no browser)

## Intent
Fast, dependency-free unit tests for the pure `src/` modules — glsl compiler/linker, raster formats & pure raster math, util math, gl state defaults, present's Node surface. This is the iteration loop during implementation (seconds, not minutes); the Khronos CTS remains the gate. Tests import `src/` modules directly (no `renderer.js` bundle, no Playwright, no DOM).

## How to run
- `npm run test:unit` — `vitest run tests/unit` (exit non-zero on any failure)
- `npm run test:watch` — watch mode
- Single file: `npx vitest run tests/unit/intercept.test.ts`
- Type-check: `npx tsc --noEmit` must report ZERO errors (tsconfig includes `tests/**/*.ts`).
- Config lives at repo root (`vitest.config.ts`: forks pool, include `tests/**/*.test.ts`, 120s timeout).

## Conventions
- **Naming**: one `<area>.test.ts` per src module area, mirroring `src/` layout. `helpers.ts` holds shared utilities (it is NOT a test file).
- **Imports**: static imports from `src/` at their final contract paths (see API Surface below). No bundle, no globals.
- **GL enums**: use the `GL` constants from `./helpers` (spec-fixed numeric values); never import a src/gl constants module.
- **Determinism**: pure functions only, no network, no timers. Typed-array results compared with `expectArrayClose` (tolerance-aware; handles NaN/Infinity).
- **Status discipline**: the suite is the executable spec for src/ and is fully green (95/95). Do NOT delete or skip tests to make the suite green; fix a test only if the src contract changed (update this file's API Surface in the same pass). A failing test is a regression signal — investigate src/ first, and report genuine src bugs rather than papering over them in the test.
- **File size**: each test file ≤ ~1000 lines; split by module.

## Current status (verified 2026-08: all src/ modules landed, suite fully green)
| File | Compile (`tsc`) | Runtime |
|---|---|---|
| `intercept.test.ts` | ✅ | ✅ 6/6 PASS |
| `state.test.ts` | ✅ | ✅ 17/17 PASS |
| `formats.test.ts` | ✅ | ✅ 20/20 PASS |
| `raster.test.ts` | ✅ | ✅ 18/18 PASS |
| `glsl.test.ts` | ✅ | ✅ 18/18 PASS |
| `math.test.ts` | ✅ | ✅ 11/11 PASS |
| `present.test.ts` | ✅ | ✅ 5/5 PASS |

`npm run test:unit` and `npm run typecheck` are both fully green. Suite total: 95 tests, all pass.

## API Surface (coordination contract with src/ — VERIFIED against the real modules)
Import paths and export names below are the ACTUAL src APIs the tests compile against (rewritten 2026-08 from the earlier assumed contracts). If src/ changes a shape, update the call sites in the listed file AND this table in the same pass.

| Test file | Import | Real exports under test |
|---|---|---|
| `glsl.test.ts` | `../../src/glsl/index` | `compileShader(source, {type:'VERTEX'\|'FRAGMENT', version:100\|300, defines?, extensions?}) → {ok:true, shader} \| {ok:false, errors:[{line, message}]}`; `linkProgram(vs, fs, opts?) → {ok:true, program} \| {ok:false, log}`. `Program.{attributes, uniforms, varyings, uniformBlocks, uniformMap, floatStore, intStore, scratchSize, intScratchSize, usesPointSize, usesGLPointCoord, usesFragCoord, usesFrontFacing, vertex: VertexStage, fragment: FragmentStage}`. `VertexExecCtx` = BaseExecCtx + `{attribs: AttribSource[], attribIndices: Int32Array, vertexId, instanceId, out:{position: Float32Array, pointSize, varyings: Float32Array}}`; `FragmentExecCtx` = BaseExecCtx + `{varyings: VaryingValues[], fragCoord: Float32Array, frontFacing, pointCoord, discarded, out:{color: Float32Array[], fragDepth}}`; BaseExecCtx = `{uniforms, intUniforms, blockStores, blockIntStores, textures, samplerStates, scratch, intScratch}`. Helpers in the test: `makeVertexCtx(program, attribs, opts)` / `makeFragmentCtx(program, opts)` build full structural ctxs (uniforms bound to `program.floatStore`). |
| `formats.test.ts` | `../../src/raster/formats` | `getFormat(internalFormat) → PixelFormatInfo \| null` (`{format, components, bytesPerPixel, storage: StorageKind('u8'\|'i8'\|'u16'\|'i16'\|'u32'\|'i32'\|'f32'\|'f16'), isColor, isDepth, isStencil, isFloat, isSigned, isInteger, isSRGB, normalized, decode(data, byteOffset, out: Float32Array): void, encode(data, byteOffset, r,g,b,a): void}` — decode fills caller's out, no return). Per-texel upload converters: `getTexImageConverter(srcFormat, srcType, internalFormat) → TexelConverter\|null` (`{srcComponents, read(src, byteOffset, out), write(dst, byteOffset, r,g,b,a)}`), `getTexelReader`, `getTexelWriter`, `getPackConverter(internalFormat, packFormat, packType) → PackConverter\|null` (`{convert(src, srcByteOffset, dst, dstByteOffset)}`). NO `convertPixels` (row flip/UNPACK_ALIGNMENT is gl/'s concern). Implemented numeric helpers: `halfToFloat`, `floatToHalf`, `sRGBToLinear`, `linearToSRGB`, `packDepth24Stencil`, `unpackDepth24`. Storage: 8-bit normalized → u8/i8; 16-bit → u16/i16; ALL float formats AND depth → f32 (so R16F/DEPTH_COMPONENT16/24 bpp=4). |
| `math.test.ts` | `../../src/util/math` | Flat top-level functions (NOT namespaced, NO gl-matrix-style `vec3.`/`mat4.` objects), Float32Array in/out, optional trailing `out` of exact length: `vec2/vec3/vec4/mat2/mat3/mat4` constructors, `mat2Identity/mat3Identity/mat4Identity(out)`, `vecAdd/vecSub/vecMul/vecDiv/vecScale/vecNeg`, `dot/cross/length/lengthSq/distance/distanceSq/normalize/reflect/refract/faceforward`, `vecAbs..vecSmoothstep` (per-component GLSL builtins), `mat4Mul/mat3Mul/mat2Mul`, `mat4MulVec4/mat3MulVec3/mat2MulVec2` (+ row-vector variants), `mat4Transpose/mat3/mat2`, `mat4Determinant/mat3/mat2`, `mat4Inverse/mat3/mat2`, `matrixCompMult`, `outerProduct`, `packUnorm2x16/packSnorm2x16/packHalf2x16`, `unpackUnorm2x16/unpackSnorm2x16/unpackHalf2x16`. NO `perspective`/`translate` — those tests were dropped (no API). |
| `raster.test.ts` | `../../src/raster/index` | Record layout: `RECORD_OFFSET_X/Y/Z/W/POINT_SIZE`, `VARYINGS_OFFSET`, `RECORD_HEADER_FLOATS`, `computeVertexStride(varyings)`, `writeVertexHeader(out, base, x,y,z,w, pointSize)`, `MAX_CLIPPED_VERTICES`. Clip: `clipPrimitive(buf, base, stride, count, scratch, out, outBase) → #vertices`, `pointIsVisible(buf, base, stride)`, `applyViewportTransform(buf, base, stride, count, viewport, depthRange)` (in-place; clip w preserved in slot 3). Triangles: `signedArea2(buf, i0, i1, i2, stride)` (window-space area ×2, sign = facing), `depthSlope`. NO `edgeFunction`/`clipInterpolate`/`perspectiveCorrect` — perspective-correct interpolation is INTERNAL to `rasterizeTriangle` (no exported helper; covered by conformance/visual suites). |
| `state.test.ts` | `../../src/gl/state` | `createDefaultState(version: 1\|2): State` (NOT `createState()`), `defaultLimits(): Limits`, `defaultVertexAttrib(): VertexAttribState`, `defaultVAOState(numAttribs): VAOState`, `KEEP` (0x1E00). State shape: capabilities in `caps.*` (BLEND, CULL_FACE, DEPTH_TEST, DITHER=true, POLYGON_OFFSET_FILL, SAMPLE_ALPHA_TO_COVERAGE, SAMPLE_COVERAGE, SCISSOR_TEST, STENCIL_TEST, RASTERIZER_DISCARD); `blend{srcRGB,dstRGB,srcAlpha,dstAlpha,eqRGB,eqAlpha,color}` (no `enabled`); `depth{func,mask,range}` (mask replaces depthMask); `stencil{front,back}` (full 7-field StencilState); top-level `cullFace/frontFace/scissor/viewport/polygonOffset{factor,units}/sampleCoverage{value,invert}/lineWidth/activeTexture` (unit INDEX 0 = TEXTURE0); `pixelStore.pack.alignment` / `pixelStore.unpack.{alignment,flipY,premultiplyAlpha,...}`; `currentProgram`, `colorMask`, `clearColor/clearDepth/clearStencil`, `uniformBuffers` (72 for v2, 0 for v1). |
| `present.test.ts` | `../../src/present/index` | `interface CanvasSurface {readonly width, height, getPixels(): Uint8Array (RGBA8), present(): void, resize(w,h): void}`; `class NodeCanvasSurface` (no-arg ctor; dimensions from `resize()`; `present()` no-op); `class BrowserCanvasSurface` (HTMLCanvasElement ctor); `createCanvasSurface(canvas: unknown): CanvasSurface` factory (structural: object with `getContext` fn → Browser, else Node). NO `createNodeSurface`. |
| `intercept.test.ts` | `../../src/context-intercept` | (module exists — no assumptions; uses explicit missing/temp renderer paths so it stays green after `renderer.js` is built) |

## Test Strategy
- `glsl.test.ts` — compile/link smoke (trivial v+f shaders), compile-error line numbers, WebGL1 `in`-keyword strictness vs v300 in/out, Program metadata (attributes/uniforms/varyings/fragment.outputs), vertex exec via `makeVertexCtx` (position/varyings/uniforms-by-location through `program.floatStore`/gl_PointSize/gl_VertexID/gl_InstanceID), built-in function evaluation, fragment gl_FragColor via `makeFragmentCtx`. Varying-link expectations are native-verified (probe: `src/glsl/probe-native-varying-link.mjs` + `src/glsl/probe-results.json`): FS READS a varying the VS never declares → link FAILS (log contains `not matched`); VS-extra varyings and FS-declared-but-unread varyings (v100 AND v300) → link OK.
- `formats.test.ts` — metadata (bpp/components/storage/classification) for ~16 core formats; encode/decode round-trips with quantization tolerances (RGBA8 exact-ish, RGB565/RGBA4/RGB5_A1 within bit depth, LUMINANCE(+ALPHA) semantics, DEPTH_COMPONENT16/24 f32, R16F f32, RGBA32F); `getTexImageConverter` per-texel source conversions (RGBA→LUMINANCE, RGB→RGBA8, RGBA→RGB565; no flipY — gl/'s concern); implemented numeric helpers (halfToFloat/floatToHalf incl. specials, sRGB round-trips, packDepth24Stencil/unpackDepth24) — PASS today.
- `math.test.ts` — vec dot/cross/length/normalize/add/sub/scale; mat4 identity/multiply-composition/invert/mat4MulVec4 transform/mat4Determinant/mat4Transpose (perspective/translate dropped — no API).
- `raster.test.ts` — `signedArea2` sign/magnitude/facing vs triangle interiors; `clipPrimitive` pass-through, plane-crossing interpolation (vertices on the clip plane), long varying records, fully-clipped → 0, line clipping; `applyViewportTransform` (center mapping, w preservation, depth-range mapping + clamping); `pointIsVisible`; record-layout constants/`computeVertexStride`/`writeVertexHeader` (PASS today). Perspective-correct interpolation is internal to `rasterizeTriangle` — no unit coverage; covered by conformance/visual suites.
- `state.test.ts` — GL spec defaults (caps, clear values, masks, blend/depth/cull/stencil/scissor/polygonOffset/sampleCoverage, dither, lineWidth, activeTexture, pixelStore, currentProgram, limits) + fresh-per-call independence + version-1-vs-2 differences (uniformBuffers sizing) — PASS today.
- `present.test.ts` — NodeCanvasSurface dimensions via resize(), RGBA8 buffer size (w*h*4), present() no-op; createCanvasSurface structural factory tests (Node for non-canvas, Browser for getContext-bearing object).
- `intercept.test.ts` — `buildInterceptScript` (renderer-present: embeds source, routes webgl/webgl2/experimental-webgl through `__createSoftwareWebGLContext`, falls through for '2d'; renderer-missing: RENDERER_NOT_FOUND stub that throws), `getRendererPath` env handling, `assertRendererExists`.
- Not yet covered (add when APIs settle): ImageSource decoding, texture sampler, fragment-ops blending math, raster primitive assembly, glsl uniform blocks / UBO exec. `helpers.ts` is the home for a `makeContext()`-style harness once gl/ lands.

## Constraints
- Never import the built `renderer.js` bundle; always import `src/` modules directly.
- No browser APIs, no Playwright, no network — pure Node.
- Keep each file ≤ ~1000 lines.
- Never modify anything under `src/` from tests (the API Surface table is the coordination channel).

## Routing Table
- `helpers.ts` → shared GL enum constants (incl. UNSIGNED_BYTE/FLOAT data types) + `expectArrayClose` (and future shared harnesses)
- `intercept.test.ts` → context-intercept harness helper tests (passes today)
- `state.test.ts` → GL state container default tests (PASSES today)
- `formats.test.ts` → pixel format registry + per-texel converter tests (4/20 pass; rest fail until formats registry populates)
- `raster.test.ts` → clip/signed-area/viewport-transform/record-layout tests (3/18 pass; rest fail until clip+triangles land)
- `glsl.test.ts` → GLSL compiler/linker/Program model tests (fails until src/glsl lands)
- `math.test.ts` → vec/mat helper tests (fails until src/util/math lands)
- `present.test.ts` → Node canvas surface tests (fails until src/present/canvas lands)
