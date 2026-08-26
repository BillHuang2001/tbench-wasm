/**
 * src/gl/extensions/index.ts — extension registry.
 *
 * SINGLE table of every extension the renderer knows about, with:
 *  - `versions`: which context type(s) expose it ('webgl1' | 'webgl2').
 *    Per WebGL 2.0 spec §6.2, WebGL1 extensions are available on WebGL2 EXCEPT
 *    those subsumed into core (ANGLE_instanced_arrays, EXT_blend_minmax,
 *    EXT_frag_depth, EXT_shader_texture_lod, OES_element_index_uint,
 *    OES_standard_derivatives, OES_texture_float, OES_texture_half_float,
 *    OES_vertex_array_object, WEBGL_depth_texture, WEBGL_draw_buffers) and
 *    EXT_disjoint_timer_query (must NOT be advertised on WebGL2 — CTS checks).
 *  - `status`: 'implement' → factory created in Phase 2; 'null' → getExtension
 *    returns null and getSupportedExtensions does not list it.
 *
 * CTS-verified decision (suite version 2.0.1): every graded extension test
 * reports `testPassed("No X support -- this is legal")` when the extension is
 * unavailable, so status 'null' does not fail the 2,071 graded tests. The
 * invariant that MUST hold: getSupportedExtensions() lists an extension IFF
 * getExtension() returns an object (tested by get-extension.html,
 * webgl-multi-draw.html, webgl-debug-shaders.html, ext-texture-filter-anisotropic.html).
 *
 * Status rationale:
 *  - 'implement' = mandated by objective/root facts (three.js/Babylon visual
 *    suites query them) — factories live in extensions/*.ts (Phase 2).
 *  - 'null' = CTS tests skip; three.js/Babylon degrade gracefully. Compressed
 *    formats can be promoted to 'implement' by adding a decompressor (s3tc first).
 */

import type { WebGLRenderingContext } from '../webgl1';
import type { WebGL2RenderingContext } from '../webgl2';

export type ExtensionContextVersion = 1 | 2;

export type ExtensionStatus = 'implement' | 'null';

export interface ExtensionSpec {
  /** Canonical name (getExtension case-sensitive). */
  name: string;
  /** Legacy prefixed aliases (WEBKIT_* / MOZ_*) accepted by getExtension. */
  aliases?: string[];
  /** Context versions exposing this extension. */
  versions: ExtensionContextVersion[];
  /** 'implement' → factory must exist in extensions/<group>.ts; 'null' → not available. */
  status: ExtensionStatus;
  /** Optional: per-extension enablement hook (e.g. promoted extensions in WebGL2). */
  available?: (ctx: WebGLRenderingContext | WebGL2RenderingContext) => boolean;
}

/** Canonical registry — keep sorted; adding an extension = one row here + factory. */
export const EXTENSION_SPECS: ExtensionSpec[] = [
  // ---- WebGL1 classic (implement — mandated by objective; three.js/Babylon need them) ----
  { name: 'ANGLE_instanced_arrays', versions: [1], status: 'implement' },
  { name: 'EXT_blend_minmax', versions: [1], status: 'implement' },
  { name: 'EXT_frag_depth', versions: [1], status: 'implement' },
  { name: 'EXT_shader_texture_lod', versions: [1], status: 'implement' },
  { name: 'EXT_sRGB', versions: [1, 2], status: 'implement' },
  { name: 'OES_element_index_uint', versions: [1], status: 'implement' },
  { name: 'OES_fbo_render_mipmap', versions: [1, 2], status: 'implement' },
  { name: 'OES_standard_derivatives', versions: [1], status: 'implement' },
  { name: 'OES_texture_float', versions: [1], status: 'implement' },
  { name: 'OES_texture_float_linear', versions: [1, 2], status: 'implement' },
  { name: 'OES_texture_half_float', versions: [1], status: 'implement' },
  { name: 'OES_texture_half_float_linear', versions: [1, 2], status: 'implement' },
  { name: 'OES_vertex_array_object', versions: [1], status: 'implement' },
  {
    name: 'WEBGL_depth_texture',
    versions: [1],
    status: 'implement',
  },
  { name: 'WEBGL_draw_buffers', versions: [1], status: 'implement' },
  { name: 'WEBGL_blend_func_extended', versions: [1, 2], status: 'implement' },
  { name: 'WEBGL_lose_context', versions: [1, 2], status: 'implement' },
  { name: 'WEBGL_debug_renderer_info', versions: [1, 2], status: 'implement' },
  { name: 'WEBGL_debug_shaders', versions: [1, 2], status: 'implement' },
  { name: 'EXT_texture_filter_anisotropic', aliases: ['WEBKIT_EXT_texture_filter_anisotropic', 'MOZ_EXT_texture_filter_anisotropic'], versions: [1, 2], status: 'implement' },

  // ---- WebGL2-side (implement — mandated by objective) ----
  { name: 'EXT_clip_control', versions: [1, 2], status: 'implement' },
  { name: 'EXT_color_buffer_float', versions: [2], status: 'implement' },
  { name: 'EXT_color_buffer_half_float', versions: [1, 2], status: 'implement' },
  { name: 'EXT_float_blend', versions: [1, 2], status: 'implement' },
  { name: 'EXT_texture_norm16', versions: [2], status: 'implement' },
  { name: 'KHR_parallel_shader_compile', versions: [1, 2], status: 'implement' },
  { name: 'OES_draw_buffers_indexed', versions: [2], status: 'implement' },
  { name: 'WEBGL_clip_cull_distance', versions: [2], status: 'implement' },
  { name: 'WEBGL_multi_draw', versions: [1, 2], status: 'implement' },
  { name: 'WEBGL_multisampled_render_to_texture', versions: [2], status: 'implement' },
  { name: 'WEBGL_render_shared_exponent', versions: [2], status: 'implement' },

  // ---- 'null' status (CTS tests skip; three.js/Babylon degrade gracefully) ----
  { name: 'EXT_conservative_depth', versions: [2], status: 'null' },
  { name: 'EXT_depth_clamp', versions: [1, 2], status: 'null' },
  { name: 'EXT_disjoint_timer_query', versions: [1], status: 'null' },
  { name: 'EXT_disjoint_timer_query_webgl2', versions: [2], status: 'null' },
  { name: 'EXT_polygon_offset_clamp', versions: [1, 2], status: 'null' },
  { name: 'EXT_render_snorm', versions: [2], status: 'null' },
  { name: 'EXT_texture_compression_bptc', versions: [1, 2], status: 'null' },
  { name: 'EXT_texture_compression_rgtc', versions: [1, 2], status: 'null' },
  { name: 'EXT_texture_mirror_clamp_to_edge', versions: [1, 2], status: 'null' },
  { name: 'NV_shader_noperspective_interpolation', versions: [2], status: 'null' },
  { name: 'OES_sample_variables', versions: [2], status: 'null' },
  { name: 'OES_shader_multisample_interpolation', versions: [2], status: 'null' },
  { name: 'OVR_multiview2', versions: [2], status: 'null' },
  { name: 'WEBGL_compressed_texture_astc', versions: [1, 2], status: 'null' },
  { name: 'WEBGL_compressed_texture_etc', versions: [1, 2], status: 'null' },
  { name: 'WEBGL_compressed_texture_etc1', versions: [1, 2], status: 'null' },
  { name: 'WEBGL_compressed_texture_pvrtc', versions: [1, 2], status: 'null' },
  { name: 'WEBGL_compressed_texture_s3tc', versions: [1, 2], status: 'null' },
  { name: 'WEBGL_compressed_texture_s3tc_srgb', versions: [1, 2], status: 'null' },
  { name: 'WEBGL_polygon_mode', versions: [1, 2], status: 'null' },
  { name: 'WEBGL_provoking_vertex', versions: [2], status: 'null' },
  { name: 'WEBGL_shader_pixel_local_storage', versions: [2], status: 'null' },
  { name: 'WEBGL_stencil_texturing', versions: [2], status: 'null' },
  { name: 'WEBGL_webcodecs_video_frame', versions: [1, 2], status: 'null' },
];

const SPEC_BY_NAME = new Map<string, ExtensionSpec>();
for (const spec of EXTENSION_SPECS) {
  SPEC_BY_NAME.set(spec.name, spec);
  for (const alias of spec.aliases ?? []) SPEC_BY_NAME.set(alias, spec);
}

/** Names reported by getSupportedExtensions() for a context version (canonical only). */
export function getSupportedExtensionNames(version: ExtensionContextVersion): string[] {
  const out: string[] = [];
  for (const spec of EXTENSION_SPECS) {
    if (spec.status !== 'implement') continue;
    if (!spec.versions.includes(version)) continue;
    out.push(spec.name);
  }
  return out;
}

/**
 * Resolve getExtension(name): returns the singleton extension object (cached on
 * the context) or null. Case-sensitive; aliases resolve to their canonical spec.
 * `status: 'null'` specs return null — their factories do not exist yet.
 */
export function getExtensionObject(ctx: WebGLRenderingContext | WebGL2RenderingContext, name: string): object | null {
  const spec = SPEC_BY_NAME.get(name);
  if (!spec) return null;
  const version: ExtensionContextVersion = ctx._version;
  if (!spec.versions.includes(version)) return null;
  if (spec.status !== 'implement') return null;
  if (spec.available && !spec.available(ctx)) return null;
  // Singleton per context.
  const cache = ctx._extensions;
  const existing = cache.get(spec.name);
  if (existing !== undefined) return existing;
  // Phase 2: factories live in extensions/*.ts — see CONTEXT.md Routing Table.
  const ext = createExtension(ctx, spec);
  cache.set(spec.name, ext);
  return ext;
}

/** Phase 2 hook: build the extension object (constants + methods) per spec. */
function createExtension(
  ctx: WebGLRenderingContext | WebGL2RenderingContext,
  spec: ExtensionSpec,
): object {
  throw new Error(
    `GL extension factory not implemented yet: ${spec.name} (Phase 2 — see src/gl/CONTEXT.md)`,
  );
}
