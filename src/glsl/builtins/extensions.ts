/**
 * builtins/extensions.ts — extension-gated builtins for GLSL ES 1.00 shaders.
 *
 * Every entry here carries an `extension` tag; the semantics pass only makes
 * these visible when the shader enables the matching extension (via
 * `#extension <name> : require|enable`). When multiple extensions are enabled
 * their tables merge; the EXT_draw_buffers gl_FragData / gl_MaxDrawBuffers
 * entries OVERRIDE the core 1.00 values (see 100.ts) at the use site.
 *
 * Extension list (all are WebGL 1.0 extensions):
 * - GL_OES_standard_derivatives: dFdx/dFdy/fwidth (fragment stage only).
 * - GL_EXT_shader_texture_lod: Lod/Grad texture functions in the FRAGMENT
 *   stage (core 1.00 only has the Lod forms in vertex shaders).
 * - GL_EXT_frag_depth: gl_FragDepthEXT builtin variable.
 * - GL_EXT_draw_buffers: gl_FragData resized to gl_MaxDrawBuffers = 4.
 *
 * NOTE on shadow functions: WebGL 1.0 has no sampler2DShadow/samplerCubeShadow
 * types at all, so texture2DShadowLodEXT can never be called from a WebGL 1.0
 * shader — it is listed for completeness of the GL_EXT_shader_texture_lod
 * surface (the extension spec defines it), and the sampler type cannot be
 * declared anyway.
 */
import {
  F,
  gen,
  genType,
  sig,
  smp,
  V2,
  V3,
  V4,
  arr,
} from './types.js';
import type { BuiltinConstant, BuiltinSignature, BuiltinVariable } from './types.js';

export const extensionFunctions: BuiltinSignature[] = [
  /* ---------------------------------------------------------------- */
  /* GL_OES_standard_derivatives — fragment stage                     */
  /* ---------------------------------------------------------------- */
  ...gen('dFdx', genType, { extension: 'GL_OES_standard_derivatives', stage: 'FRAGMENT' }),
  ...gen('dFdy', genType, { extension: 'GL_OES_standard_derivatives', stage: 'FRAGMENT' }),
  ...gen('fwidth', genType, { extension: 'GL_OES_standard_derivatives', stage: 'FRAGMENT' }),

  /* ---------------------------------------------------------------- */
  /* GL_EXT_shader_texture_lod — fragment stage                       */
  /* ---------------------------------------------------------------- */
  sig('texture2DLodEXT', [smp('sampler2D'), V2, F], V4, {
    extension: 'GL_EXT_shader_texture_lod',
    stage: 'FRAGMENT',
  }),
  sig('texture2DProjLodEXT', [smp('sampler2D'), V3, F], V4, {
    extension: 'GL_EXT_shader_texture_lod',
    stage: 'FRAGMENT',
  }),
  sig('texture2DProjLodEXT', [smp('sampler2D'), V4, F], V4, {
    extension: 'GL_EXT_shader_texture_lod',
    stage: 'FRAGMENT',
  }),
  sig('textureCubeLodEXT', [smp('samplerCube'), V3, F], V4, {
    extension: 'GL_EXT_shader_texture_lod',
    stage: 'FRAGMENT',
  }),
  sig('texture2DGradEXT', [smp('sampler2D'), V2, V2, V2], V4, {
    extension: 'GL_EXT_shader_texture_lod',
    stage: 'FRAGMENT',
  }),
  sig('texture2DProjGradEXT', [smp('sampler2D'), V3, V2, V2], V4, {
    extension: 'GL_EXT_shader_texture_lod',
    stage: 'FRAGMENT',
  }),
  sig('texture2DProjGradEXT', [smp('sampler2D'), V4, V2, V2], V4, {
    extension: 'GL_EXT_shader_texture_lod',
    stage: 'FRAGMENT',
  }),
  sig('textureCubeGradEXT', [smp('samplerCube'), V3, V3, V3], V4, {
    extension: 'GL_EXT_shader_texture_lod',
    stage: 'FRAGMENT',
  }),
  sig('texture2DShadowLodEXT', [smp('sampler2DShadow'), V3, F], V4, {
    extension: 'GL_EXT_shader_texture_lod',
    stage: 'FRAGMENT',
  }),
];

export const extensionVariables: BuiltinVariable[] = [
  {
    name: 'gl_FragDepthEXT',
    type: F,
    stage: 'FRAGMENT',
    writable: true,
    extension: 'GL_EXT_frag_depth',
  },
  {
    name: 'gl_FragData',
    type: arr(V4, 4),
    stage: 'FRAGMENT',
    writable: true,
    extension: 'GL_EXT_draw_buffers',
  },
];

export const extensionConstants: BuiltinConstant[] = [
  // Overrides the core 1.00 value (1) when GL_EXT_draw_buffers is enabled.
  { name: 'gl_MaxDrawBuffers', value: 4, extension: 'GL_EXT_draw_buffers' },
];
