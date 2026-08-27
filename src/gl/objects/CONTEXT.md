# src/gl/objects/ — WebGL object classes + aux records

## Intent
Thin JS classes for every WebGL object type (WebGLBuffer, WebGLTexture, WebGLShader, WebGLProgram, WebGLFramebuffer, WebGLRenderbuffer, WebGLVertexArrayObject, WebGLSampler, WebGLQuery, WebGLSync, WebGLTransformFeedback) plus the three aux record/handle classes (WebGLUniformLocation, WebGLActiveInfo, WebGLShaderPrecisionFormat), the per-context `Resources` tracker, `createObject`, `OBJECT_CLASSES`, and the bottom-of-module `chainToNative` wiring that makes instances `instanceof` the NATIVE browser classes of the same name (via `../native-chain.ts`).

## API Surface
- One file per class (`webgl-object.ts` abstract base with `_context`/`_deleted`; `webgl-buffer.ts` with `_target/_data/_size/_usage/_deletePending/_tfRangeBindings`; `webgl-texture.ts` with `_levels[faceIndex][level]`, `_params`, `_image: GLTextureImage`, `_deletePending`; `webgl-shader.ts` with `_translatedSource`; `webgl-program.ts` with `_uniformStore: DataView`, `_blockStores`, `_bindAttribLocations`, `_transformFeedbackVaryings`, `_tfBufferMode`, `_linkComplete`; `webgl-framebuffer.ts` with `_attachments`, `_status`, `_multisampled`; `webgl-renderbuffer.ts` with `_surface`, `_samples`; `webgl-vao.ts` holding VAOState; `webgl-sampler.ts`; `webgl-query.ts` with `_active/_result/_resultAvailable`; `webgl-sync.ts`; `webgl-transform-feedback.ts`).
- `aux.ts` — `WebGLUniformLocation {_program,_index,_name}` (opaque handle), `WebGLActiveInfo {size,type,name}`, `WebGLShaderPrecisionFormat {rangeMin,rangeMax,precision}` — NOT WebGLObjects (no delete/context tag, per spec); `validateUniformLocation(loc, program)` (null/undefined → null, non-WebGLUniformLocation → TypeError, cross-program → null).
- `index.ts` — barrel: value-imports + re-exports all classes (incl. the 3 aux classes from `./aux`), `Resources` (track/untrack/invalidateAll), `createObject(ctx, ctor)`, `OBJECT_CLASSES` (11 concrete classes), and bottom-of-module `chainToNative` calls for the 11 concrete classes AND the 3 aux classes.

## Constraints
- Zero runtime deps; classes are plain data carriers — all validation lives in `../validation.ts` / api/ modules.
- `chainToNative` wiring MUST be in value position at module bottom; every class referenced there needs a VALUE import in the same module — including the 3 aux classes, which are value-imported from `./aux` in `index.ts` so the calls bind to OUR classes rather than the ambient lib.dom globals of the same name (a re-export alone leaves esbuild resolving them as free globals → `Object.setPrototypeOf(nativeProto, nativeProto)` → `TypeError: Cyclic __proto__ value` at bundle load).

## Known Issues
- **No unit-test coverage of the native-chain wiring**: `tests/unit` imports src/ modules directly and never value-imports this barrel (e.g. `src/gl/state.ts`'s objects import is type-only), so the bottom-of-module `chainToNative` calls never execute in Node — the bundle-load path is only exercised by browser builds of `renderer.js`.

## Routing Table
- `webgl-object.ts` → abstract WebGLObject base
- `webgl-buffer.ts`, `webgl-texture.ts`, `webgl-shader.ts`, `webgl-program.ts`, `webgl-framebuffer.ts`, `webgl-renderbuffer.ts`, `webgl-vao.ts`, `webgl-sampler.ts`, `webgl-query.ts`, `webgl-sync.ts`, `webgl-transform-feedback.ts` → concrete classes
- `aux.ts` → 3 aux classes + `validateUniformLocation`
- `index.ts` → barrel, `Resources`, `createObject`, `OBJECT_CLASSES`, `chainToNative` wiring
