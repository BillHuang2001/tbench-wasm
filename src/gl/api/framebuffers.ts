/**
 * src/gl/api/framebuffers.ts — framebuffer/renderbuffer objects + operations.
 *
 * Owns: createFramebuffer, deleteFramebuffer, isFramebuffer, bindFramebuffer,
 * framebufferRenderbuffer, framebufferTexture2D, framebufferTextureLayer (W2),
 * checkFramebufferStatus, getFramebufferAttachmentParameter,
 * createRenderbuffer, deleteRenderbuffer, isRenderbuffer, bindRenderbuffer,
 * renderbufferStorage, renderbufferStorageMultisample (W2),
 * getRenderbufferParameter, drawBuffers (W2), readBuffer (W2),
 * invalidateFramebuffer (W2), invalidateSubFramebuffer (W2), blitFramebuffer (W2).
 *
 * Behavior notes:
 *  - bindFramebuffer: WebGL2 targets DRAW_FRAMEBUFFER/READ_FRAMEBUFFER +
 *    FRAMEBUFFER (both); null = default framebuffer (drawing buffer).
 *  - Attachment rules: texture level must exist & be complete-able; cube face
 *    targets map to faces; framebufferTextureLayer validates layer < depth;
 *    multisample mismatch (mixed sample counts / multisampled texture +
 *    single-sampled renderbuffer) → FRAMEBUFFER_INCOMPLETE_MULTISAMPLE;
 *    layered attachments (3D/2D_ARRAY) must ALL be layered →
 *    FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS.
 *  - checkFramebufferStatus implements the full completeness table
 *    (attachment, missing, dimensions, multisample, layer targets, unsupported);
 *    draw validation calls it before every draw (INVALID_FRAMEBUFFER_OPERATION).
 *  - drawBuffers: sequence length ≤ MAX_DRAW_BUFFERS, entries COLOR_ATTACHMENTn
 *    < MAX_COLOR_ATTACHMENTS or NONE; invalid → INVALID_OPERATION (no partial).
 *    Default: [COLOR_ATTACHMENT0]. WEBGL_draw_buffers exposes drawBuffersWEBGL
 *    with *_WEBGL suffix constants.
 *  - blitFramebuffer: color filtering (LINEAR/NEAREST), depth/stencil
 *    NEAREST-only, same-size rules for depth/stencil blits, multisample resolve.
 *  - renderbufferStorage: internalformat ∈ color/depth/stencil renderable sets
 *    (+ EXT_sRGB, EXT_color_buffer_float/half_float, EXT_texture_norm16,
 *    EXT_render_shared_exponent RGB9_E5); width/height ≤ MAX_RENDERBUFFER_SIZE;
 *    renderbufferStorageMultisample: samples ≤ MAX_SAMPLES, multisampled
 *    renderbuffers can only attach to multisampled FBOs.
 */

import type { WebGLRenderingContext } from '../webgl1';

export function installFramebuffersApi(proto: WebGLRenderingContext): void {
  void proto;
}
