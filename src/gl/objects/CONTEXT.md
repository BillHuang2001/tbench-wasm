# src/gl/objects/ — WebGL object classes + aux records

## Intent
Thin JS classes for every WebGL object type (WebGLBuffer, WebGLTexture, WebGLShader, WebGLProgram, WebGLFramebuffer, WebGLRenderbuffer, WebGLVertexArrayObject, WebGLSampler, WebGLQuery, WebGLSync, WebGLTransformFeedback) plus the three aux record/handle classes (WebGLUniformLocation, WebGLActiveInfo, WebGLShaderPrecisionFormat), the per-context `Resources` tracker, `createObject`, `OBJECT_CLASSES`, and the bottom-of-module `chainToNative` wiring that makes instances `instanceof` the NATIVE browser classes of the same name (via `../native-chain.ts`).

## API Surface
- One file per class (`webgl-object.ts` abstract base with `_context`/`_deleted`; `webgl-buffer.ts` with `_target/_data/_size/_usage/_deletePending/_tfRangeBindings`; `webgl-texture.ts` with `_levels[faceIndex][level]`, `_params`, `_image: GLTextureImage`, `_deletePending`; `webgl-shader.ts` with `_translatedSource`; `webgl-program.ts` with `_uniformStore: DataView`, `_blockStores`, `_bindAttribLocations`, `_transformFeedbackVaryings`, `_tfBufferMode`, `_linkComplete`; `webgl-framebuffer.ts` with `_attachments`, `_status`, `_multisampled`; `webgl-renderbuffer.ts` with `_surface`, `_samples`; `webgl-vao.ts` holding VAOState; `webgl-sampler.ts`; `webgl-query.ts` with `_active/_result/_resultAvailable`; `webgl-sync.ts`; `webgl-transform-feedback.ts`).
- `aux.ts` — `WebGLUniformLocation {_program,_index,_name}` (opaque handle), `WebGLActiveInfo {size,type,name}`, `WebGLShaderPrecisionFormat {rangeMin,rangeMax,precision}` — NOT WebGLObjects (no delete/context tag, per spec); `validateUniformLocation(loc, program)` (null/undefined → null, non-WebGLUniformLocation → TypeError, cross-program → null).
- `index.ts` — barrel: re-exports all classes + `Resources` (track/untrack/invalidateAll), `createObject(ctx, ctor)`, `OBJECT_CLASSES` (11 concrete classes), and bottom-of-module `chainToNative` calls for the 11 concrete classes AND the 3 aux classes (lines ~104-118).

## Constraints
- Zero runtime deps; classes are plain data carriers — all validation lives in `../validation.ts` / api/ modules.
- `chainToNative` wiring MUST be in value position at module bottom; every class referenced there needs a VALUE import in the same module (see Known Issues).

## Known Issues
- **Bundle-load bug (CONFIRMED, current)** — the 3 aux classes are only RE-EXPORTED from `index.ts` (`export { WebGLUniformLocation, WebGLActiveInfo, WebGLShaderPrecisionFormat, validateUniformLocation } from './aux'`, lines 37-42) with NO value import. The bottom-of-module `chainToNative(WebGLUniformLocation.prototype, 'WebGLUniformLocation')` etc. (lines 116-118) therefore resolve to the ambient lib.dom GLOBAL declarations (why `tsc --noEmit` stays clean), not the local classes. esbuild then renames the local classes to `WebGLUniformLocation2`/`WebGLActiveInfo2`/`WebGLShaderPrecisionFormat2` in the bundle while leaving the call sites as global lookups. Consequences: in browsers `Object.setPrototypeOf(nativeProto, nativeProto)` → `TypeError: Cyclic __proto__ value` at bundle load → IIFE aborts before `globalThis.__createSoftwareWebGLContext` is assigned → browser suites silently run NATIVE WebGL (the context-intercept wrapper falls back to native getContext when the factory is missing — nothing asserts the factory loaded); in Node `require('./renderer.js')` throws `ReferenceError: WebGLUniformLocation is not defined`.
  **Minimal fix (VERIFIED in temp build + headless Chromium)**: add one value import to `index.ts` (keep the re-export; `export { X } from` + local import of the same name coexist fine):
  ```ts
  import { WebGLActiveInfo, WebGLShaderPrecisionFormat, WebGLUniformLocation } from './aux';
  ```
  After the fix: no `*2` class renames remain in the bundle, `require('./renderer.js')` loads, `typeof window.__createSoftwareWebGLContext === 'function'` in Chromium with zero pageerrors, and `gl instanceof WebGLRenderingContext` (native) is true.
  When fixed, DELETE this entry (and the matching one in `../CONTEXT.md`).
- **No unit-test coverage of this path**: `tests/unit` imports src/ modules directly and never value-imports this barrel (e.g. `src/gl/state.ts`'s objects import is type-only), so the broken global lookups never execute in Node — the suite passes 106/106 despite the bug.

## Routing Table
- `webgl-object.ts` → abstract WebGLObject base
- `webgl-buffer.ts`, `webgl-texture.ts`, `webgl-shader.ts`, `webgl-program.ts`, `webgl-framebuffer.ts`, `webgl-renderbuffer.ts`, `webgl-vao.ts`, `webgl-sampler.ts`, `webgl-query.ts`, `webgl-sync.ts`, `webgl-transform-feedback.ts` → concrete classes
- `aux.ts` → 3 aux classes + `validateUniformLocation`
- `index.ts` → barrel, `Resources`, `createObject`, `OBJECT_CLASSES`, `chainToNative` wiring
