# src/util/ — Shared Low-Level Foundation

## Intent
The bottom of the dependency DAG: pure, dependency-free helpers shared by every other module (`glsl/`, `raster/`, `gl/`, `present/`). No WebGL logic, no GL state, no DOM — small, exact, fast primitives whose semantics are authoritative across the codebase (GLSL math, typed-array plumbing, bit/float conversions, logging). All public API is stubbed (`throw new Error('not implemented')`) — signatures are the contract; implementation is pending.

## API Surface
- `index.ts` — barrel: `export *` from the four modules below. All consumers import from `../util`.
- `math.ts` — GLSL-style vec2/3/4 & mat2/3/4 math on `Float32Array`s, column-major (GLSL uniform layout). Componentwise arithmetic (add/sub/mul/div/scale/neg, generic `vecMap`/`vecMap2` for trig/exp/log), geometric (dot/cross/length/lengthSq/distance/distanceSq/normalize/reflect/refract/faceforward), exact GLSL componentwise builtins (clamp/min/max/mix/mod/fract/step/smoothstep/roundEven/abs/sign/floor/ceil/trunc/round/sqrt/inversesqrt/pow/atan2/exp/log/exp2/log2), matrix ops (mul ×3 sizes, mat×vec, vec×mat, transpose, determinant, inverse, matrixCompMult, outerProduct), WebGL2 packing builtins (pack*/unpack* 2x16 Unorm/Snorm/Half). **Every producing op takes an optional trailing `out` buffer (zero-alloc hot paths); inputs never mutated.**
- `typedarray.ts` — `TypedArrayKind` string registry ('int8'…'float64') ↔ constructors (`TYPED_ARRAY_CTORS`, `typedArrayCtor`, `makeTypedArray`), `viewOf` (aliasing views over ArrayBuffers), `copyTypedArray` (element-wise, cross-kind), `copyBytes` (memmove-style raw bytes), fill/zero, `bytesPerElement`/`byteLengthOf`, `typedArrayToBytes` (zero-copy byte reinterpretation), `cloneTypedArray`, `concatTypedArrays`, `isTypedArray` guard, `typedArrayKindOf`.
- `misc.ts` — exact GLSL scalar builtins (clamp, clamp01, fract, mod, mix, step, smoothstep, roundEven, sign), pow2 utils (isPow2, nextPow2, log2Int, alignUp), IEEE-754 bit reinterpretation (floatToIntBits/intBitsToFloat/floatToUintBits/uintBitsToFloat), binary16 pack/unpack (toHalfFloat/fromHalfFloat), GLSL ES 3.00 bitfield ops (bitfieldExtract, bitfieldInsert, bitCount, findLSB, findMSB), dev `assert`.
- `log.ts` — leveled logging: `setLogLevel`/`getLogLevel`/`isLogEnabled`, `createLogger(namespace)` → `{ debug, info, warn, error }`; levels debug < info < warn < error < silent; default 'warn'; all loggers honor the global level at call time.

## Constraints
- **ZERO imports from other `src/` modules** (bottom of DAG). Only stdlib: Math, console, typed arrays, ArrayBuffer, DataView. Must run in Node AND headless Chromium; no DOM.
- Pure functions; the only state is (a) the log level in `log.ts` and (b) a shared module-level DataView scratch in `misc.ts` for bit reinterpretation (documented; safe — JS single-threaded, non-reentrant).
- Zero-allocation hot paths: math ops take optional `out` buffers; callers (glsl codegen, raster) pass preallocated scratch — never allocate in per-fragment loops.
- Exact GLSL ES semantics for all GLSL-named functions (mod/fract/mix/clamp/step/smoothstep/roundEven/refract/bitfield ops…) — single source of truth; glsl codegen and raster must NOT reimplement them differently.
- Strict TS (tsconfig: strict, ES2022, ESNext, moduleResolution bundler). Files ≤ ~1000 lines.

## Design Decisions
- **Out-param convention**: trailing optional `out` (allocate if omitted) so glsl codegen emits calls into preallocated scratch buffers → zero per-fragment allocation in shader execution.
- **Column-major Float32Array matrices** matching GLSL uniform layout: uniform upload and GLSL matrix constructors need no transposition.
- **GLSL-named functions centralized in util** rather than inlined in codegen: one correct, unit-testable implementation (CTS shader tests are strict about e.g. mod/fract/roundEven semantics).
- **Generic vecMap/vecMap2** for the long tail of componentwise builtins (sin/cos/tanh/exp2…) instead of dozens of near-identical functions; dedicated functions only for hot or semantically tricky ops.
- **TypedArrayKind strings, not GL enums**: util must not know GL; gl/ maps GLenum ↔ kind.
- **Stub bodies now**: files define the complete public API contract (signatures + JSDoc with exact semantics); implementation is delegated to the Manager phase.

## Known Issues
- `smoothstep` with edge0 == edge1 is undefined in GLSL; our implementation returns 0 (no NaN).
- `normalize` of a zero vector returns the zero vector (undefined in GLSL; avoids NaN).
- Matrix `inverse` of singular matrices returns NaN/Inf entries (undefined in GLSL; no check — callers that care must guard).
- `toHalfFloat`/`fromHalfFloat` follow IEEE 754 binary16 (round-to-nearest-even, subnormals, ±Inf, NaN).
- `findLSB`/`findMSB` return -1 when no bits are set; `findMSB` handles negative ints per GLSL spec (MSB of complement).
- Bit-reinterpretation helpers share one DataView scratch — fine single-threaded, but never hold a result across a re-entrant call.

## Test Strategy
- Pure functions → easy vitest unit tests under `tests/unit/` (sibling — read-only from this node; escalate writes to parent). Priority coverage: GLSL-exact functions (mod/fract/mix/clamp/step/smoothstep/roundEven), matrix mul/inverse vs. known results, pack*/unpack* round-trips, half-float edge cases, bitfield ops.

## Routing Table
Leaf node — no child directories. Consumers: `../glsl/` (codegen runtime), `../raster/` (clip/interp math, formats), `../gl/` (state, objects, validation), `../present/`, `../entry.ts`. Tests: `../../tests/unit/` (sibling — read-only, escalate writes to parent).
