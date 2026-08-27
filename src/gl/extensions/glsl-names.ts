/**
 * src/gl/extensions/glsl-names.ts — WebGL extension → GLSL extension name plumbing.
 *
 * The GLSL preprocessor resolves `#extension <name> : require|enable` against
 * the `extensions` set passed to glsl.compileShader (src/glsl/preprocessor.ts
 * `handleExtension`). That set must contain the GLSL extension names
 * (GL_ANGLE_multi_draw, GL_OES_standard_derivatives, ...) — CTS shaders declare
 * the GLSL names, e.g. webgl-multi-draw.html `#extension GL_ANGLE_multi_draw :
 * require`, webgl-draw-buffers.html `#extension GL_EXT_draw_buffers : require`.
 * Feeding only the WebGL registry names (WEBGL_multi_draw, ...) rejects every
 * such directive with "extension '<name>' is not supported".
 *
 * Enablement semantics (CTS-verified): a GLSL extension is available ONLY for
 * WebGL extensions that have been ENABLED on the context — i.e. getExtension()
 * returned an object, recorded in the per-context singleton cache
 * `ctx._extensions` (extensions/index.ts getExtensionObject). Every CTS
 * extension page runs a "disabled" phase BEFORE getExtension() on the SAME
 * context and expects `#extension GL_* : require` to FAIL there, then an
 * "enabled" phase after getExtension() and expects it to COMPILE (e.g.
 * webgl-draw-buffers.html runShadersTestDisabled/runShadersTestEnabled;
 * oes-standard-derivatives.html, ext-frag-depth.html, ext-shader-texture-lod.html
 * follow the identical pattern). Feeding names for all SUPPORTED extensions
 * (getSupportedExtensions) would break every disabled phase, so the helper
 * below iterates the enabled cache instead. The cache is per-context and
 * version-gated by the registry (getExtensionObject returns null for the wrong
 * context version), so the union is automatically per-context correct: WebGL2
 * contexts never enable WebGL1-only extensions (WEBGL_draw_buffers,
 * OES_standard_derivatives, ...) and their GLSL names are never fed.
 *
 * The mapping covers every 'implement' WebGL extension whose GLSL counterpart
 * the glsl compiler knows (src/glsl/builtins/extensions.ts extension-gated
 * builtins + src/glsl/program.ts gl_DrawID). Extensions without a GLSL
 * counterpart (formats, state, objects — EXT_texture_filter_anisotropic,
 * OES_texture_float, WEBGL_depth_texture, EXT_sRGB, ...) map to nothing.
 */

import type { WebGLRenderingContext } from '../webgl1';

/**
 * WebGL extension name (EXTENSION_SPECS canonical — the getSupportedExtensions
 * name and the ctx._extensions cache key) → GLSL extension name(s) the glsl
 * compiler resolves. Sorted by WebGL name. Only entries whose GLSL name is
 * known to the compiler are listed.
 */
export const WEBGL_TO_GLSL_EXTENSIONS: Readonly<Record<string, readonly string[]>> = {
  // glsl: GL_ANGLE_instanced_arrays is the canonical GLSL counterpart of
  // ANGLE_instanced_arrays (adds gl_InstanceID to ES 1.00 vertex shaders);
  // glsl does not gate 1.00 gl_InstanceID behind it yet — feeding the name is
  // harmless and forward-compatible (no CTS shader requires it).
  ANGLE_instanced_arrays: ['GL_ANGLE_instanced_arrays'],
  // glsl builtins/extensions.ts: gl_FragDepthEXT.
  EXT_frag_depth: ['GL_EXT_frag_depth'],
  // glsl builtins/extensions.ts: texture2DLodEXT/texture2DGradEXT (fragment).
  EXT_shader_texture_lod: ['GL_EXT_shader_texture_lod'],
  // glsl builtins/extensions.ts: dFdx/dFdy/fwidth (fragment).
  OES_standard_derivatives: ['GL_OES_standard_derivatives'],
  // glsl: canonical GLSL counterpart (gl_ClipDistance/gl_CullDistance);
  // conformance2/extensions/webgl-clip-cull-distance.html compiles
  // `#extension GL_ANGLE_clip_cull_distance : require`. glsl's 300.ts does not
  // define the builtins yet — feeding the name is forward-compatible.
  WEBGL_clip_cull_distance: ['GL_ANGLE_clip_cull_distance'],
  // glsl builtins/extensions.ts: gl_FragData[4] + gl_MaxDrawBuffers = 4.
  WEBGL_draw_buffers: ['GL_EXT_draw_buffers'],
  // glsl program.ts: gl_DrawID (multi-draw subdraw index, constant per draw).
  WEBGL_multi_draw: ['GL_ANGLE_multi_draw'],
};

/**
 * The `extensions` set fed to glsl.compileShader for one context: the union of
 * the ENABLED WebGL extension names (registry names — the previous behavior,
 * kept for compatibility) and their mapped GLSL names. Computed at
 * compileShader call time; the caller snapshots it (KHR_parallel_shader_compile
 * deferred path) so sync and deferred compiles behave identically.
 */
export function shaderCompileExtensions(ctx: WebGLRenderingContext): Set<string> {
  const out = new Set<string>();
  for (const name of ctx._extensions.keys()) {
    out.add(name);
    const glslNames = WEBGL_TO_GLSL_EXTENSIONS[name];
    if (glslNames !== undefined) {
      for (const g of glslNames) out.add(g);
    }
  }
  return out;
}
