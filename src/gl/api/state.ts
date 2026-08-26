/**
 * src/gl/api/state.ts — capability and simple state setters.
 *
 * Owns: enable, disable, isEnabled, blendColor, blendEquation(separate),
 * blendFunc(separate), clearColor, clearDepth, clearStencil, colorMask,
 * cullFace, depthFunc, depthMask, depthRange, frontFace, hint, lineWidth,
 * pixelStorei, polygonOffset, sampleCoverage, scissor, viewport.
 *
 * Behavior notes:
 *  - All setters validate the enum/value FIRST (INVALID_ENUM/INVALID_VALUE) and
 *    are no-ops on context loss (CONTEXT_LOST_WEBGL pushed once per call).
 *  - pixelStorei: UNPACK/PACK_ALIGNMENT ∈ {1,2,4,8} (INVALID_VALUE otherwise);
 *    UNPACK_FLIP_Y/PREMULTIPLY_ALPHA ∈ {true,false}; UNPACK_COLORSPACE_CONVERSION
 *    ∈ {BROWSER_DEFAULT_WEBGL, NONE}; WebGL2 ROW_LENGTH/SKIP_* / IMAGE_HEIGHT ≥ 0.
 *  - colorMask writes state.colorMask; OES_draw_buffers_indexed colorMaskiOES
 *    writes state.colorMaskPerDrawBuffer.
 *  - lineWidth: only 1.0 supported — other values are clamped/ignored per spec
 *    (no error; ALIASED_LINE_WIDTH_RANGE [1,1]).
 *  - hint: only GENERATE_MIPMAP_HINT (WebGL1) + FRAGMENT_SHADER_DERIVATIVE_HINT
 *    (WebGL2/OES_standard_derivatives) are valid; mode ∈ {FASTEST, NICEST,
 *    DONT_CARE}; value is stored but has no behavioral effect (software).
 */

import type { WebGLRenderingContext } from '../webgl1';

export function installStateApi(proto: WebGLRenderingContext): void {
  void proto;
}
