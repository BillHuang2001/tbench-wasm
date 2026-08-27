# src/glsl/codegen/ — GLSL → JS codegen (link-time, inlined)

## Intent
Compiles annotated GLSL ASTs (vertex + fragment) into ONE JS function body per stage (`new Function('ctx','R', body)`), per cross-module contract §1 in `../CONTEXT.md`. The LINKER (linker.ts) computes `CodegenLayout` (uniform vec4 slots, std140 block offsets, varying packing, attrib/output locations, `uses`, `structNames`) and calls `generateVertexStage`/`generateFragmentStage` (this dir) — the codegen NEVER computes layout itself. `R` = shared runtime helper object (runtime.ts; imports raster/texture-sampler.ts). Zero per-invocation allocation: locals flatten to scalar JS vars, arrays go to `ctx.scratch`/`ctx.intScratch` fixed offsets, ALL user functions are INLINED at call sites.

## API Surface
- `index.ts` — the codegen↔linker contract: `CodegenLayout`, `UniformSlot`, `BlockMemberLayout`, `VaryingLayout`, `StageCodegenResult`, `Value` (the expression currency: `{v, dx?, dy?, pre?}`; dual mode = dFdx/dFdy triples), re-exports `generateVertexStage` (vertex.ts), `generateFragmentStage` (fragment.ts), `R` (runtime.ts).
- `env.ts` — `CodegenEnv`: var/scratch/temp allocation, `walk` storage resolution (`resolveLocal` → `globalInfo`), `emitLValue`, `dualWrite`, param frames (`pushParamFrame`/`makeParamLocal`).
- `expressions.ts` (+`expr-builtins.ts`, `expr-ctor.ts`) — expression emission incl. `emitCall` (user→builtin→ctor routing), `emitLValue`, the `walk`er (throw site for unknown identifiers).
- `statements.ts` — statement emission, control flow, `discard`; `functions.ts` — the USER-FUNCTION INLINER (IIFE-wrapped inlines, param frames, scratch copy-in/out for array params).
- `vertex.ts` / `fragment.ts` — stage assembly; `runtime.ts` — `R` helpers (math/pack/bitfield/texture wrappers).
- Diagnostic tools (NOT bundled): `diag-cts-shaders.ts` (ogles/build/struct/equal cluster measurement; run `npx tsx src/glsl/codegen/diag-cts-shaders.ts`), `selftest-*.ts` (committed tsx validation scripts).

## Constraints
- Generated JS must be ES2019, no per-fragment/per-invocation allocation, no `new Function` runtime closure over mutable state (only `ctx` + `R`).
- `layout.uses.derivatives` → dual-number mode: every float value carries (v, dx, dy); `Value.pre` must stay PURE (side effects fold into `v`).
- gl/ MUST supply `ctx.blockStores[i]`/`ctx.intBlockStores[i]` views (zero-filled when unbound) — codegen emits NO null guards.
- Codegen exceptions are caught by the linker (`linker: codegen failed: <msg>`) — stacks are swallowed; reproduce via a scratch script that calls `generateVertexStage`/`generateFragmentStage` directly with a hand-built layout (see diag tools / Known Issues).

## Known Issues / Gotchas (diagnosed 2026-03, ogles functions/ cluster — NOT yet fixed)
- **BUG A — overloaded user functions: name-keyed registry picks the LAST definition** (`functions.ts` `installUserFunctions`: `fns.set(d.prototype.name, d)`). Overloads overwrite each other; `emitUserCall` inlines the last-registered body regardless of which overload semantics resolved. Manifests as `linker: codegen failed: Cannot read properties of undefined (reading 'v')` at `functions.ts:232` (`names.map((n,c) => ... temps[c].v)`) when the last definition has MORE param components than the call's args (e.g. scalar call `is_all(ret, true)` inlined against the array overload's body → `temps[1..3]` undefined). 19 ogles pairs hit this (all `*_array`/`*_bigarray` function tests: bvec4/ivec4/mat4/vec4 variants of `is_all` scalar + array overloads, array overload defined last). Even when component counts align it is a silent MISCOMPILE (wrong body inlined). Fix direction: key the registry by signature or route the call to the resolved overload; consider a defensive arity/component check in the bind loop.
- **BUG B — user GLOBAL variables unsupported in codegen**: semantics accepts `float gray = 0.0;` at file scope (legal ES 1.00, const initializer) but codegen's `walk` (expressions.ts:545 `throw new Error('codegen: unknown identifier ...')`) has no storage surface for non-const globals — `resolveLocal` misses, `globalInfo` only knows uniform/block/attrib/varying/output/builtin/const. `linker: codegen failed: codegen: unknown identifier 'gray'`. Hit by `void_empty_empty_void_empty` (ogles functions/; also reproduces with `float gray;` without initializer; const globals fold fine). Fix direction: allocate a per-stage storage surface (scratch slot) + emit initializer once, or reject at semantics with a clear `linker: ... not supported` message (rejection alone would fail the positive CTS page).
- linker.ts:1320 catches codegen throws and keeps only `e.message` — to debug, call `generateVertexStage`/`generateFragmentStage` directly with a hand-built `CodegenLayout` (uniformSlots/varyings/attribLocations/outputLocations Maps; `uses` all-false; version 100) and print `e.stack` (used for both bugs above).
- `selftest-predrop.ts` pins `Value.pre`-drop consumers; `selftest-struct-fixes.ts`/`selftest-ctor-fixes.ts` pin gl_DepthRange + runtime struct ==/!= + matrix-in-vector-ctor + structNames seeding (all green at HEAD).

## Test Strategy
- In-repo: `npx tsx src/glsl/codegen/selftest-<name>.ts` (codegen-expr 118, codegen-stmt 30, codegen-fn 21, codegen-stage 27, dual 31, dual-builtins 60, integration 59, predrop 21, struct-fixes, ctor-fixes — all green at HEAD).
- CTS gates: `sdk/tests/conformance/ogles/GL/functions/` (126 positive cases; 43/63 pairs link-green at HEAD — BUGs A+B account for all 20 failing pairs), plus `ogles/GL/build/`, `biuDepthRange/`, `struct/`, `equal/` (see diag-cts-shaders.ts).
- Full sweep measurement: `npx tsx /scratch/final_scan.mts` style enumeration over the functions/ dir (read-only foreign repo).

## Routing Table
- `index.ts` → contract + seam (Value, CodegenLayout, stage entry re-exports)
- `env.ts` → CodegenEnv (storage/scratch/temps, globalInfo, dualWrite, param frames)
- `expressions.ts` / `expr-builtins.ts` / `expr-ctor.ts` → expression emission + walker + emitCall routing
- `statements.ts` → statement emission; `functions.ts` → user-function inliner (BUG A lives here)
- `vertex.ts` / `fragment.ts` → stage assembly; `runtime.ts` → R helpers (imports `../raster/texture-sampler.ts`)
- `diag-cts-shaders.ts` → ogles cluster measurement tool; `selftest-*.ts` → regression scripts
- `../linker.ts` → CodegenLayout builder + codegen orchestration (sibling, read-only)
