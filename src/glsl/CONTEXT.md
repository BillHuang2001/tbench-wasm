# src/glsl/ — GLSL ES 1.00 / 3.00 Compiler & Linker

## Intent
Compiles GLSL ES shader source to **JavaScript** (codegen — the REQUIRED strategy) and links vertex+fragment pairs into the `Program` model consumed by `gl/` (API surface, uniform/attribute management, draw-time vertex evaluation) and `raster/` (varying interpolation, fragment execution) per cross-module contract §1 in `../CONTEXT.md`. Zero runtime dependencies; must compile every valid shader in the Khronos CTS `glsl/` subtree (313 WebGL1 tests), the `conformance2/glsl3/` subtree, and all shaders used by three.js/Babylon examples, and reject invalid ones with precise 1-based-line errors.

## API Surface (exact — contract §1)
```ts
compileShader(source: string, opts: {
  type: 'VERTEX' | 'FRAGMENT'; version: 100 | 300;
  defines?: Record<string, string>; extensions?: Set<string>;
}) → { ok: true; shader: Shader } | { ok: false; errors: { line: number; message: string }[] }
linkProgram(vs: Shader, fs: Shader, opts?: LinkOptions)
  → { ok: true; program: Program } | { ok: false; log: string }
```
- `compiler.ts`: `compileShader`/`linkProgram`, `Shader` (type/version/source/infoLog/extensions/ast/info), `ShaderInfo` (resolved `AttributeDecl`/`VaryingDecl`/`UniformDecl`/`UniformBlockDecl`/`OutputDecl` + `ShaderUses` capability flags), `LinkOptions`/`LinkLimits` (limits, `attribBindings`, `transformFeedback`).
- `program.ts`: `Program` (attributes/uniforms/uniformBlocks/varyings, `vertex.run`, `fragment.run`+`usesDerivatives`+`usesFragDepth`+`outputs`, capability flags, `uniformMap`, `floatStore`/`intStore`, scratch sizes, `transformFeedbackVaryings`), info interfaces (`AttribInfo`/`UniformInfo`/`UniformBlockInfo`/`UniformBlockMemberInfo`/`VaryingInfo`), exec contexts (`BaseExecCtx`/`VertexExecCtx`/`FragmentExecCtx`).
- `types.ts`: `GLSLType` tagged union, `Precision`, qualifiers, declared helpers (`toGLenum`, `typeComponents`, `isIntegral`, `isFloat`, `isSampler`, `typeSize`, `typeEquals`, `typeName`).
- `ast.ts`: full AST type definitions (TranslationUnit, decls, statements, expressions with `resolvedType`/`constValue`/`lvalue` annotation slots).
- `index.ts`: re-exports all of the above. Phase 1 status: these files are complete type contracts; function bodies throw `not implemented` (implemented in Phase 2).

## Architecture — Compilation Pipeline
```
compileShader:  preprocessor → lexer → parser(+parser-expr) → semantics → Shader{ast, info}
linkProgram:    matching → attrib/uniform/block layout (std140) → limits → codegen → Program
```
- `preprocessor.ts` → `lexer.ts` → `parser.ts` (+ `parser-expr.ts`) → `semantics.ts` (type checking, symbol tables, overload resolution, const folding, precision rules, extension gating, ShaderInfo) → `linker.ts` (varying matching, location/store/block layout, transform feedback validation, codegen orchestration, Program assembly).
- `builtins/` — builtin SIGNATURE tables only (name → overloads → {params, ret, extension gate}): `100.ts`, `300.ts`, `extensions.ts` (GL_OES_standard_derivatives → dFdx/dFdy, GL_EXT_shader_texture_lod → texture2DLodEXT family (fragment only), GL_EXT_frag_depth → gl_FragDepthEXT, GL_EXT_draw_buffers → gl_FragData writes + gl_MaxDrawBuffers). Builtin VARIABLES (gl_Position/gl_PointSize/gl_FragCoord/gl_FrontFacing/gl_PointCoord/gl_FragColor/gl_FragData/gl_VertexID/gl_InstanceID/gl_FragDepth(EXT)/gl_Max*) live with the builtin tables too; gl_Max* VALUES = WebGL minimums (must match gl/ getParameter; see Known Issues).
- `codegen/` — JS generation: `expressions.ts` (operator/builtin lowering incl. dual-number templates), `statements.ts` (control flow, discard), `vertex.ts`, `fragment.ts` (stage assembly), `runtime.ts` (the shared `R` helper object: complex builtin implementations + texture sampling wrappers that forward to `raster/texture-sampler.ts` `sampleTexture`).

## Design Decisions

### 1. Codegen model (REQUIRED): GLSL → JS at link time
- Each stage compiles to a single JS function body via `new Function('ctx', 'R', body)`; `Program.vertex.run = (ctx) => fn(ctx, R)` (closure binds the shared runtime `R` — never embedded per shader).
- **Vectors/matrices/structs are flattened to scalar locals** (vec4 → 4 JS number variables; swizzles are compile-time index remaps) — zero allocation, fastest execution.
- **All functions are inlined** (recursion is illegal in GLSL); params become renamed locals; returns become jumps (labeled blocks). Control flow (if/for/while/do/switch) lowers to native JS; break/continue map directly. GLSL `return` from an inlined function → jump to that function's epilogue label.
- **Local arrays** (non-const, dynamically indexed) map to fixed offsets in `ctx.scratch`/`ctx.intScratch` (per-draw preallocated; sizes on `Program`); statically-indexed small arrays may become scalars. Complex builtins (mat inverse, etc.) write results into scratch too. **No per-vertex/per-fragment allocation anywhere.**
- Uniform/attrib/varying access is baked constant-offset indexing; block reads bake the block index + std140 byte offset (binding changes handled by gl/ remapping `ctx.blockStores[blockIndex]` per draw — gl/ must always provide a view, zero-filled fallback when unbound).

### 2. Derivative handling — dual-number mode
- `usesDerivatives` = explicit dFdx/dFdy/fwidth **OR any implicit-LOD texture function** (texture2D/texture with no explicit LOD). Texture functions with explicit LOD never trigger it.
- In dual mode, **every float expression lowers to a (v, dx, dy) triple of scalars**; arithmetic is inlined (add = 3 adds; mul = product rule; builtins have per-builtin dual templates). Integer/uint/bool expressions never carry duals. Non-dual shaders compile to plain scalars (fast path).
- Varying loads in dual mode read `ctx.varyings[i].v/ddx/ddy` (raster provides ddx/ddy for ALL varyings whenever `usesDerivatives`; flat varyings read dx=dy=0). gl_FragCoord = constant duals (dx(x)=1, dy(x)=0). Vertex shaders never use dual mode (no derivatives in vertex; implicit-LOD vertex texture sampling uses LOD 0).

### 3. Uniform store layout
- Default block: **two per-program stores** — `floatStore` (Float32Array) and `intStore` (Int32Array), shared by both stages via the exec ctx. **vec4-slot packing**: each uniform occupies `ceil(components/4)` consecutive slots (matC = C slots, column-major), `UniformInfo.location` = start slot, no cross-uniform sharing (conservative; WebGL1 model, WebGL2-legal).
- int/uint/bool/samplers live in `intStore` (uint values stored as int32 bits; generated code uses unsigned JS semantics: `>>> 0` for uint ops, truncating division). Samplers hold the texture-unit binding.
- Uniforms are FLATTENED leaves (getActiveUniform semantics): arrays = one entry (`'u[0]'`, size N); structs = per-member entries (`'u.m'`, `'u[0].m'`, `'u[2].m'`); UBO members included with `blockIndex >= 0`, `location = -1`. `uniformMap` (getUniformLocation lookup) contains ONLY default-block paths incl. bare array names → first element; UBO members and invalid paths → null.
- **UBOs** (WebGL2): std140 layout computed at link (offsets/arrayStride/matrixStride/size per `UniformBlockMemberInfo`; block size rounded to 16; one block entry per array element, named `'b[0]'`...). NOT in the default stores — read through `ctx.blockStores[blockIndex]`/`blockIntStores[blockIndex]`.

### 4. Vertex attribute fetch
- gl/ performs a per-draw **dense extraction** of each enabled attribute (stride removed, draw-order, format conversion) into `ctx.attribs[loc]` (reused scratch); constant attribs pass as number or 4-element view. `ctx.attribIndices[loc]` = fetch index (vertex position in draw for divisor-0; `floor(instanceId/divisor)` for instanced; 0 for constants). Codegen reads `attribs[loc][attribIndices[loc]*components + c]`. `ctx.vertexId` (gl_VertexID) = first+i (drawArrays) or element value (drawElements) — independent of the fetch index.

### 5. Fragment output / discard model
- Shader writes `ctx.out.color[location]` / `ctx.out.fragDepth`; **raster commits after run() returns**, so 2×2 helper invocations naturally never write (no special codegen path). `discard` compiles to `{ ctx.discarded = true; return; }` (exits loops too); raster skips committing discarded fragments.

### 6. Version & extension rules
- `opts.version` = context max (100/300); declared `#version` > context → compile error. Keyword/literal rules differ per version: `attribute`/`varying` (1.00) vs `in`/`out` (3.00); `^^` only 1.00; bitwise ops/`switch`/`uint`/f-suffix/`flat`/`centroid`/`layout()` only 3.00; WebGL1 samplers = sampler2D/Cube only (GL_OES_texture_3D is NOT a WebGL extension — no sampler3D in 1.00).
- `#extension` validated against `opts.extensions`; unsupported/unknown extension with require/enable → compile error. Extension-name macros (GL_OES_standard_derivatives) defined after successful enable (spec behavior — verify against CTS).
- ES 1.00 fragment shaders must declare a default float precision (`precision mediump float;`) — CTS tests this. ES 3.00 fragment: no default float precision either (int defaults mediump — verify per spec).
- Permissive loop handling: NO appendix-A loop-bound rejection in 1.00 (appendix A is "may fail" — accepting everything is CTS-safe).
- ES 3.00 implicit conversions per spec (int→float, uint→float, and CHECK whether int→uint is implicit — implementer must verify against spec + CTS).
- No `fma` (not in ES 3.00). No `precise` semantics (accepted, ignored). Precision qualifiers accepted but execution is uniformly highp (float64) — more precise than required, which is conformant.

### 7. Error reporting
- 1-based line numbers (post-#line remapping); Khronos format `ERROR: 0:<line>: <message>`; `Shader.infoLog` = '' on success; link failures return one formatted log. CTS checks line numbers in some tests — keep them exact.

## Constraints
- Every file ≤ ~1000 lines; split by responsibility (parser → `parser.ts` + `parser-expr.ts` if needed).
- Strict TypeScript; zero runtime deps; ES2019-compatible generated output (generated JS uses only plain var/function constructs; no allocations in the per-invocation path).
- Codegen runs ONLY at link time (WebGL semantics: shader compile errors are separate from link errors).
- Dependency direction: `codegen/runtime.ts` imports `../raster/texture-sampler.ts` (sampling) and `program.ts` type-imports raster's `TextureImage`/`SamplerState` — ONE-WAY glsl→raster runtime edge (type-only back-reference from raster's DrawCall to `Program` is erased). No import cycles.

## Known Issues / Gotchas
- **Integral varyings vs Float32 packing**: contract §2 packs vertex records into Float32Array — int varyings > 2^24 lose precision. gl/ + raster/ must add a bit-exact side channel for integral varyings (they can detect via `VaryingInfo.type`); verify CTS glsl3 int-varying tests.
- **Conservative vec4 uniform packing** may accept programs CTS `overflow-uniforms-*` expects to fail (packing is impl-defined; verify against those tests, tighten packing if needed).
- **gl_Max* builtin values = WebGL minimums** — must stay consistent with gl/ `getParameter` values (CTS compares in some shader tests).
- **uint arithmetic in JS**: generated code must use unsigned semantics (`>>> 0`, truncating division, unsigned comparisons) — subtle bug source; unit-test heavily.
- **fragment default precision**: 1.00 fragment requires explicit float precision declaration (CTS `shader-with-precision-*` tests).
- **`#extension` unknown**: follow the objective (error); verify exact CTS expectation (spec allows warn for `enable` of unsupported in some cases).
- **getUniformLocation**: bare array name returns first element; UBO members → null (absent from uniformMap); malformed names → null (not an error).
- **getActiveUniform**: structs flattened; arrays one entry `[0]` + size; UBO members included (blockIndex ≥ 0, location -1).
- **gl_FragColor + gl_FragData both written** → link error (1.00); gl_FragData only with GL_EXT_draw_buffers; gl_FragDepthEXT only with GL_EXT_frag_depth (1.00); gl_FragDepth only in 3.00 fragment.
- **Sampler unit conflicts** (two active samplers of different types on one unit) and exceeding MAX_COMBINED_TEXTURE_IMAGE_UNITS → link errors (WebGL2 spec).
- **texture2DLodEXT etc. are fragment-only** (EXT_shader_texture_lod) — vertex use is a compile error.
- `^^` is 1.00-only; 3.00 must reject it. Octal literals: ES 1.00 allows `0`-prefixed; 3.00 removes them (verify).
- `#line` remaps error lines AND __LINE__/__FILE__; `#if` evaluates int expressions (defined(), GL_ES, __VERSION__, extension macros).
- discard inside a loop must terminate the WHOLE shader → `ctx.discarded = true; return;` (never a loop-level break).
- Struct type equality is BY NAME; two structs with identical members but different names are distinct types (CTS tests this).
- WebGL1 vertex texture units: minimum is 0 but software supports 16 — keep limits consistent with gl/.

## Test Strategy
- `tests/unit/` (vitest, Node): per-stage unit tests — preprocessor macros (pasting, prescan, recursion suppression), lexer tokens per version, parser error line numbers, semantics errors (type mismatches, extension gating, precision rules), std140 offsets, uniformMap paths, gl_Max* values, codegen smoke tests (compile & run generated functions against expected values), dual-number derivative correctness (dFdx of expressions vs analytic), uint arithmetic edge cases (2^32-1), discard/loop behavior.
- CTS gates (must pass 100%): `sdk/tests/conformance/glsl/` (313 tests) and `sdk/tests/conformance2/glsl3/`, run via `tests/conformance/`.
- Regression: compile the full three.js/Babylon example shader sets without errors (catches missing builtins/features).

## Routing Table
- `compiler.ts` → entry API: compileShader/linkProgram, Shader/ShaderInfo/LinkOptions/LinkLimits (Phase 1: type stubs)
- `program.ts` → Program model, info types, exec contexts (Phase 1: type stubs)
- `types.ts` → GLSL type system + declared helpers (Phase 1: types + stubs)
- `ast.ts` → AST definitions (Phase 1: complete)
- `index.ts` → public re-exports (Phase 1: complete)
- `preprocessor.ts` → # directives, macros, token pasting/stringize, __LINE__/__FILE__/__VERSION__, extension macros (pending)
- `lexer.ts` → tokens, version-dependent keywords/literals (pending)
- `parser.ts` → declarations & statements; `parser-expr.ts` → expressions (pending)
- `semantics.ts` → symbol tables, type checking, overload resolution, const folding, precision/extension rules, ShaderInfo (pending)
- `linker.ts` → varying matching, attrib/uniform/block layout, std140, limits, transform feedback, codegen orchestration (pending)
- `builtins/` → builtin signature tables: `100.ts`, `300.ts`, `extensions.ts` (pending)
- `codegen/` → JS codegen: `expressions.ts` (incl. dual-number templates), `statements.ts`, `vertex.ts`, `fragment.ts`, `runtime.ts` (R helpers; imports raster/texture-sampler.ts) (pending)
- `../raster/texture-sampler.ts` → texture sampling runtime + TextureImage/SamplerState types (sibling — READ-ONLY from here; writes escalate to `../`)
- `../gl/` → consumer of Program/Shader (sibling — read-only)

## Status
Phase 1 complete: public API, type system, AST and exec-context contracts in place (compiler.ts/program.ts bodies throw 'not implemented'). Phase 2 pending: implement pipeline modules per Routing Table, then verify against the CTS gates above.
