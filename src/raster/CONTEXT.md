# src/raster/ — Rasterization Core

## Intent
Pure-JS software rasterization for WebGL 1.0/2.0 (no GPU, zero deps, no per-fragment allocation): consumes post-VS packed vertex records from `gl/` (contract §2 of `../CONTEXT.md`), does primitive assembly, homogeneous clipping, viewport transform, triangle/line/point rasterization with perspective-correct varying interpolation, fragment shader execution in 2×2 quads (derivatives), and fragment ops (scissor/sample-coverage/stencil/depth/blend/dither/sRGB/colorMask). Also owns the SINGLE shared pixel-format registry (`formats.ts`) and the texture sampler (`sampler.ts`) used by `gl/` and `glsl/` (contract §3). Leaf module: imports nothing from `gl/` or `glsl/` at runtime.

## API Surface (public contract — pinned; do not rename)
- `draw(dc: DrawCall): void` (index.ts) — the rasterizer entry. `DrawCall` (types.ts) keeps every field from contract §2 (mode/count/first/instanceCount/vertices/vertexStride/program/varyingsOffset/fb/viewport/depthRange/scissor/cull/polygonOffset/dither/colorMask/blend/depthTest/depthMask/stencilTest/sampleCoverage/rasterizerDiscard/lineWidth) plus: `textures` (per-unit `{img, state}` effective bindings) and `drawBuffers` (output location → color attachment index or −1 for NONE).
- DrawCall carries three fields beyond contract §2's list: `uniforms` (fragment uniform store), `uniformBlocks?` (WebGL2 UBO stores), `sampleCountRef?` (occlusion counter — see below).
- Vertex record layout (types.ts): `[x, y, z, w, pointSize, varyings...]`, `VARYINGS_OFFSET = 5`. Pre-viewport: x/y/z/w = clip space. Post-`applyViewportTransform`: x/y/z = window coords, **w preserved as clip w** (perspective + gl_FragCoord.w). `computeVertexStride(varyings)`, `writeVertexHeader(...)` helpers. Instanced addressing: instance i, vertex j → record `first + i*count + j` (gl/ packs all instances contiguously).
- `FragmentExecCtx` (types.ts) — the FULL `BaseExecCtx` set (uniforms = default-block float store, intUniforms = default-block int/uint/bool/sampler store, blockStores/blockIntStores = uniform-block data INDEXED BY BLOCK INDEX — raster converts the DrawCall's name-keyed `uniformBlocks` into index-keyed arrays, zero-filled when unbound; textures = per-unit images or null; samplerStates = effective per-unit state or DEFAULT_STATE; scratch/intScratch = codegen locals sized from program.scratchSize/intScratchSize) plus varyings `{v, ddx?, ddy?}`, fragCoord, frontFacing, pointCoord, out.color[loc], out.fragDepth, `discarded: boolean` (codegen sets on `discard`; raster resets before each invocation) and `tex: TextureEnv` (texture sampling — see below). glsl codegen compiles against this exact shape.
- `TextureEnv` (types.ts) — codegen-facing sampler: `units[unit] = {img, state}`; methods `sample2D/sample2DLod/sample3D/sample3DLod/sampleCube/sampleCubeLod/sample2DArray/sample2DArrayLod/sample2DShadow/sampleCubeShadow/sample2DArrayShadow/texelFetch2D/texelFetch3D/texelFetch2DArray` (unit, coords, explicit ∂/∂x ∂/∂y, bias or lod). All write into shared scratch `out` (float) + `outInt`/`outUint` (raw bits, same buffer). No allocation.

### Occlusion queries (`sampleCountRef` — WebGL2 SAMPLES_PASSED)
`DrawCall.sampleCountRef` is an optional out-param `{ value: number }` owned by gl/. The fragment pipeline increments it exactly once per sample that passes the stencil AND depth tests (i.e. every fragment that reaches the blending stage):
- Shaders NOT writing gl_FragDepth: incremented in `FragmentOpsImpl.test()` immediately after the depth-read test passes (before the fragment shader runs).
- Shaders writing gl_FragDepth: the depth test is deferred to `finalize()` (post-shader), so the increment happens there after the post-shader depth test passes.
- Shader-`discard`ed fragments ARE counted when the depth test already passed in test() (the count happens before the shader); for gl_FragDepth shaders a discarded fragment never reaches finalize() and is not counted.
- Helper invocations (outside-pixel quad lanes that exist only to provide derivatives) never count — runQuad only calls ops.test for inside pixels.
- `rasterizerDiscard` draws return before any fragment work: nothing is counted.
- Blending/dither/sRGB/colorMask never affect the counter. The counter is per-sample; the current rasterizer is single-sample (each fragment = 1 sample).
- `sampleTexture(img, state, coord: SampleCoord, bias, out)` + `sampleTextureLod` + `sampleTextureShadow` + `texelFetch` (sampler.ts) — generic core; `SampleCoord = {v, dx?, dy?}` (dx/dy optional; zero-filled scratch when unused).
- `formats.ts` — `getFormat`, `defineFormat`, predicates (`isColorFormat` etc.), `isValidRenderbufferFormat(f, 1|2)`, `isValidTextureInternalFormat`, `getTexImageConverter(srcFmt, srcType, dstFmt)`, `getPackConverter(intFmt, packFmt, packType)`, `halfToFloat/floatToHalf/sRGBToLinear/linearToSRGB/packDepth24Stencil/unpackDepth24`. Registry populated at module init via `defineFormat()` for every entry in `ALL_INTERNAL_FORMATS` (WebGL1 unsized formats get concrete sized storage entries under their own GLenum keys).
- `surface.ts` — `createSurface`, `getDepthData`, `getStencilData`, predicates.
- `fragment-ops.ts` — `FragmentOpsImpl` (implements `FragmentOps`), `runQuad(rs, qx, qy, inside)`, clear/blit helpers, `blendColor` (exported for unit tests).
- `clip.ts` — `clipPrimitive` (6 planes, interpolates all fields linearly in clip space), `pointIsVisible`, `applyViewportTransform`, `MAX_CLIPPED_VERTICES = 7`.
- `triangles.ts`/`lines.ts`/`points.ts` — `rasterizeTriangle(buf, i0, i1, i2, stride, rs)`, `rasterizeLine`, `rasterizePoint` (one primitive per call, window-space records).
- Impl limits (types.ts): `ALIASED_POINT_SIZE_RANGE = [1, 1024]`, `ALIASED_LINE_WIDTH_RANGE = [1, 1]` — gl/ MUST import these for getParameter (single source of truth).
- `gl-enums.ts` — the GL constants raster interprets (values spec-fixed; gl/ keeps its own larger table).

## Constraints
- Zero runtime deps; no per-fragment allocation anywhere (all scratch in `RasterState`/caller `out` arrays; quad buffers allocated once per draw).
- Each file ≤ ~1000 lines. Split by responsibility (current layout already does).
- **Surface rows: row 0 = BOTTOM** (GL window coords, y up). present/ flips for display; readPixels does NOT flip. Texture rows: row 0 = bottom too (upload flip is gl/'s concern).
- GL error queue is gl/'s domain — raster never throws to the page; internal exceptions must be caught by gl/ at the API boundary.
- Raster's program view (`RasterProgram` in types.ts) is a STRUCTURAL subset of glsl's `Program` — glsl must ensure its Program is assignable (import the type from `../raster`); `VaryingInfo` is owned here, glsl imports it.

## Design Decisions
- **2×2 quads (MANDATORY when `usesDerivatives`)**: fragment shader always evaluates in 2×2 quads; outside pixels are helper invocations (shader runs, outputs discarded) so inside pixels get correct ddx/ddy (differences across the quad) — matches the parent design decision and makes derivatives work for both dFdx and implicit-LOD texture(). Fast per-pixel path (runFragment) allowed only when `!usesDerivatives` (spec-correct; helpers are semantically irrelevant then).
- **Flat varyings**: provoking vertex = LAST vertex of the primitive (GLES). rasterizer copies provoking values into ALL vertices BEFORE clipping so clip's linear interpolation is harmless. Integer varyings are flat; their bits travel in float records (codegen reinterprets via Int32Array views).
- **Two-phase fragment ops**: `test()` (scissor → coverage → stencil test → depth READ) runs before shading; `finalize()` (depth write + stencil zpass + blend + sRGB + mask + write) runs after, so `discard` suppresses ALL writes and gl_FragDepth shaders get a post-shader depth test.
- **Depth/stencil storage**: DEPTH_COMPONENT* → Float32Array (even 16/24-bit); DEPTH*_STENCIL* → split planes (Float32 depth + Uint8 stencil); packed form only materializes at readPixels via `getPackConverter`. All float formats stored as Float32Array (16F/R11F/RGB9_E5 included); packed GL forms exist only at upload/readPixels boundaries.
- **Polygon offset**: per-fragment offset = m·factor + r·units, m = max(|dz/dx|,|dz/dy|) (constant per triangle), r = 2^-24. Triangles only.
- **Dither**: the DITHER state is accepted (gl.enable/getParameter) but the algorithm is a NO-OP — the spec leaves it implementation-defined, and all shipping implementations (ANGLE, SwiftShader, native drivers) do not dither; CTS exact-value readbacks (e.g. EXT_texture_norm16) expect plain round-to-nearest, so any position-dependent dither breaks them. **Sample coverage** (single-sample): deterministic per-fragment hash threshold ≈ value (invert flips); documented approximation — refine against CTS `sample-coverage` if it fails.
- **sRGB**: sampler decodes sRGB texels to linear; blending on SRGB8_ALPHA8 targets happens in linear space (decode dst → blend → re-encode).
- **Lines**: diamond-exit rule for width 1 (critical for three.js wireframe screenshots vs GPU references); width > 1 approximated as quads (must not crash).
- **Points**: squares; size clamped to ALIASED_POINT_SIZE_RANGE at rasterization; pointCoord = (pixelCenter − (center − s/2))/s; half-open coverage so a size-1 point at an integer center still covers its pixel.
- **Unwritten fragment outputs** treated as (0,0,0,1) (spec leaves undefined; ANGLE-compatible choice).
- **Texture LOD**: λ = log₂(max ρx, ρy), ρx = max(|∂u/∂x|·w, |∂v/∂x|·h) (per-axis texel footprint); clamp to [minLod, maxLod] + bias; mip filter selects level(s). Anisotropy: ρ_aniso = max(ρx,ρy)/min(aniso, max(ρx,ρy)/min(ρx,ρy)). Incomplete textures → (0,0,0,1). Cube maps seamless.
- **formats decode/encode take an `out` param** (deviation from the illustrative `→ [r,g,b,a]` in contract §3 — required by the no-allocation rule). Decode is a correctness path only; the sampler reads raw typed arrays with format-class fast paths.

## Known Issues / Gotchas
- Compressed formats (S3TC/ETC/ASTC/PVRTC) are OUT OF SCOPE initially — extension tests skip when getExtension returns null; three.js/Babylon degrade gracefully. Do not block on them.
- deqp exactness tests (lines/points) are optional (not in the 2,071 graded count); the graded CTS line/point tests are tolerant, but three.js wireframes ARE graded by screenshot diff — diamond-exit must be genuinely implemented, not approximated.
- `gl_FragCoord.w` = 1/w_clip (perspective-correct interpolation of 1/w).
- Clipping: triangle results are fan-able polygons (up to 7 verts) — rasterize as fan from vertex 0.
- `dc.varyingsOffset` must equal `VARYINGS_OFFSET` (5) — kept as a field for contract fidelity; draw() should trust it.
- SampleCoord for 2D_ARRAY: v[2] = layer (not filtered/wrapped); shadow 2D: v = [u,v], ref separate.
- FBO multisample renderbuffers: raster draws single-sample surfaces; gl/ resolves via `blitFramebuffer` (blitColorSurface with 'nearest'/'linear').
- Line clip (count==2, clipPrimitive): segment-parameter path — one-inside → 2 verts (inside + intersection), both-inside → 2, same-side fully-outside → 0, degenerate/point-touch → 0, and a segment straddling the volume (both endpoints outside, interior crosses it) → interior segment (2 verts, GLES 2.0 §2.13). Do NOT change straddling to 0 — visible lines would vanish. `tests/unit/raster.test.ts` "clips a LINE primitive" fails its first assertion because its "both endpoints outside" data (x=-2 → x=+2) straddles the volume but expects 0 — test-data error; correction pending in tests/unit (read-only sibling, escalated).
- **Float/R11F readback machinery is READY but not yet routed by gl/**: `getPackConverter` in formats-convert.ts handles float internal formats (RGBA32F/RGBA16F/R16F/RG16F/R11F/RGB9_E5) → FLOAT/HALF_FLOAT/UNSIGNED_INT_10F_11F_11F_REV/UNSIGNED_INT_5_9_9_9_REV, incl. `pack11/pack10/pack9E5` encode helpers (formats.ts). But `src/gl/draw.ts` `executeReadPixels` bypasses it for float surfaces (local `makeLocalPack` lacks packed-float types) and `src/gl/api/draw.ts` `readComboOK` rejects (RGBA, FLOAT) reads from R32F/RG32F/R11F — gl-side follow-up required for ext-color-buffer-float (19F) and format-r11f-g11f-b10f (68F) to flip.
- **EXT_texture_norm16**: raster registry + converters are complete (R16/RG16/RGB16/RGBA16 + SNORM entries in formats.ts; upload exact, round-trip verified). 36 remaining failures are gl-side: INVALID_OPERATION on renderbuffer bindings/copyTexImage2D and `was 0,0,0,0` RTT readbacks in testNorm16Render (gl-owned follow-up).
- **Large-viewport allocation** (`conformance/rendering/rendering-stencil-large-viewport.html`, WebGL1): resizes the drawing buffer to 32767×32767 ≈ 4.29GB (RGBA8) — legal per MAX_VIEWPORT_DIMS. VERIFIED PASSING (8/8, ~212s) on the current machine; the allocation does not OOM here. If a future machine OOMs, cap the surface allocation in `surface.ts` WITHOUT changing the page's assertions (it only checks a few pixels' stencil behavior).
## Test Strategy
- `tests/unit` (vitest, Node): formats (encode/decode round-trips, converters, half/sRGB helpers), sampler (filters, wraps, LOD selection, shadow compare, texelFetch, integer raw bits), clip (6-plane correctness, interpolation), triangles (top-left rule, perspective interpolation, flat), lines (diamond-exit patterns), fragment ops (blend math incl. sRGB, stencil sequences), rasterizer (draw pipeline end-to-end vs CPU reference).
- Pipeline-level coverage: the temporary pipeline integration self-tests that once lived in `src/raster/` (`__selfcheck.ts`, `__dbg.ts`) were removed. Pipeline-level behavior — triangle fill, diamond-exit lines, blending, stencil, scissor, occlusion counting, instancing — lives in `tests/unit/raster-pipeline.test.ts` (being added in parallel) and in the CTS/visual suites.
- Primary gate: `tests/conformance` (2,071 CTS tests) — raster correctness is judged there (`rendering/`, `glsl/`, `textures/`, `drawing/`, `state/` groups).
- three.js/Babylon visual suites exercise raster performance (no per-fragment allocation matters — tens of thousands of triangles).

## Routing Table
- `types.ts` → shared types: DrawCall, record layout + helpers, FragmentExecCtx, TextureEnv, Surface/FramebufferTarget, draw-state interfaces, RasterState, impl limits
- `gl-enums.ts` → GL constants raster interprets
- `formats.ts` → pixel-format registry + upload/readPixels converters + half/sRGB helpers
- `sampler.ts` → texture sampling core (LOD, filters, projectToFace) + createTextureEnv; split across `sampler-raw.ts` (raw texel taps) and `sampler-env.ts` (codegen-facing env)
- `surface.ts` → surface creation/view resolution
- `fragment-ops.ts` → FragmentOps (scissor/coverage/stencil/depth/blend/dither/sRGB/mask), runQuad quad driver, clear + blit helpers
- `clip.ts` → homogeneous clipping + viewport transform
- `triangles.ts` → triangle rasterization (top-left rule, perspective interpolation, polygon offset)
- `lines.ts` → line rasterization (diamond-exit; wide-line quads)
- `points.ts` → point rasterization (size clamp, pointCoord)
- `rasterizer.ts` → draw() driver: assembly, instances, flat fixup, cull, dispatch
- `index.ts` → public API re-exports
