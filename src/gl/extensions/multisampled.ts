/**
 * src/gl/extensions/multisampled.ts — WEBGL_multisampled_render_to_texture (WebGL2).
 *
 * Attaches a TEXTURE (with a sample count) to a framebuffer attachment point, so
 * the FBO completeness rules treat the attachment as multisampled. The actual
 * multisample storage/resolve is the framebuffers agent's job: it reads
 * `texture._msaaSamples` and `fbo._multisampled` (both set here) in its
 * completeness + blitFramebuffer implementation.
 *
 * The spec file is absent from /testsuites/WebGL/extensions/ (the extension is
 * not part of the Khronos repo); semantics follow the extension's public IDL +
 * GLES3 framebufferTexture2DMultisample rules:
 *  - target must be FRAMEBUFFER (0x8D40) → INVALID_ENUM; a framebuffer must be
 *    bound → INVALID_OPERATION.
 *  - attachment ∈ COLOR_ATTACHMENT0..MAX_COLOR_ATTACHMENTS-1, DEPTH_ATTACHMENT,
 *    STENCIL_ATTACHMENT → INVALID_ENUM otherwise.
 *  - textarget ∈ TEXTURE_2D + cube faces → INVALID_ENUM.
 *  - texture must be a valid (non-null, same-context, non-deleted) WebGLTexture
 *    → INVALID_OPERATION otherwise; level < 0 → INVALID_VALUE.
 *  - samples < 0 → INVALID_VALUE; samples > MAX_SAMPLES → INVALID_OPERATION
 *    (mirrors WebGL2 renderbufferStorageMultisample).
 *  - renderbufferStorageMultisampleEXT delegates to the core WebGL2
 *    renderbufferStorageMultisample engine (same spec semantics): it validates
 *    internalformat/MAX_RENDERBUFFER_SIZE, allocates the actual surface
 *    (allocateRenderbufferSurface — depth planes filled to 1.0), records
 *    _samples, and invalidates FBO status caches.
 */

import type { WebGLRenderingContext } from '../webgl1';
import { C1 } from '../constants';
import { WebGLFramebuffer, WebGLRenderbuffer, WebGLTexture } from '../objects';
import { buildExtension, isLost } from './util';

const FRAMEBUFFER = 0x8d40;
const RENDERBUFFER = 0x8d41;
const TEXTURE_2D = 0x0de1;
const CUBE_POSITIVE_X = 0x8515;
const CUBE_NEGATIVE_Z = 0x851a;
const COLOR_ATTACHMENT0 = 0x8ce0;
const DEPTH_ATTACHMENT = 0x8d00;
const STENCIL_ATTACHMENT = 0x8d20;

/** TEXTURE_SAMPLES_EXT / TEXTURE_FIXED_SAMPLE_LOCATIONS_EXT (not in CExt). */
const TEXTURE_SAMPLES_EXT = 0x9130;
const TEXTURE_FIXED_SAMPLE_LOCATIONS_EXT = 0x9131;

/** True when textarget is TEXTURE_2D or a cube-map face. */
function isTextureTarget(t: number): boolean {
  return t === TEXTURE_2D || (t >= CUBE_POSITIVE_X && t <= CUBE_NEGATIVE_Z);
}

/** WEBGL_multisampled_render_to_texture factory (WebGL2 — registry versions: [2]). */
export function createWEBGLMultisampledRenderToTexture(ctx: WebGLRenderingContext): object {
  return buildExtension(
    {
      TEXTURE_SAMPLES_EXT,
      TEXTURE_FIXED_SAMPLE_LOCATIONS_EXT,
    },
    {
      framebufferTexture2DMultisampleEXT: (
        target: number, attachment: number, textarget: number,
        texture: WebGLTexture | null, level: number, samples: number,
      ): void => {
        const gl = ctx;
        if (isLost(gl)) return;
        if (target !== FRAMEBUFFER) {
          gl._errors.push(C1.INVALID_ENUM);
          return;
        }
        const maxColor = gl._state.limits.MAX_COLOR_ATTACHMENTS;
        const attachOk =
          (attachment >= COLOR_ATTACHMENT0 && attachment < COLOR_ATTACHMENT0 + maxColor) ||
          attachment === DEPTH_ATTACHMENT ||
          attachment === STENCIL_ATTACHMENT;
        if (!attachOk) {
          gl._errors.push(C1.INVALID_ENUM);
          return;
        }
        if (!isTextureTarget(textarget)) {
          gl._errors.push(C1.INVALID_ENUM);
          return;
        }
        const fbo = gl._state.drawFramebuffer;
        if (fbo === null || !(fbo instanceof WebGLFramebuffer)) {
          gl._errors.push(C1.INVALID_OPERATION);
          return;
        }
        if (texture === null || texture === undefined) {
          gl._errors.push(C1.INVALID_OPERATION);
          return;
        }
        if (!(texture instanceof WebGLTexture)) {
          throw new TypeError("Argument is not of type 'WebGLTexture'");
        }
        if (texture._context !== gl || texture._deleted) {
          gl._errors.push(C1.INVALID_OPERATION);
          return;
        }
        const lvl = level | 0;
        if (lvl < 0) {
          gl._errors.push(C1.INVALID_VALUE);
          return;
        }
        const smp = samples | 0;
        if (smp < 0) {
          gl._errors.push(C1.INVALID_VALUE);
          return;
        }
        if (smp > gl._state.limits.MAX_SAMPLES) {
          gl._errors.push(C1.INVALID_OPERATION);
          return;
        }
        // Attach like framebufferTexture2D + record the sample count where the
        // framebuffers agent's completeness reads it.
        texture._msaaSamples = smp;
        fbo._attachments.set(attachment, { type: 'texture', texture, level: lvl, face: textarget, layer: 0 });
        fbo._multisampled = smp > 0;
      },

      renderbufferStorageMultisampleEXT: (
        target: number, samples: number, internalformat: number, width: number, height: number,
      ): void => {
        const gl = ctx;
        if (isLost(gl)) return;
        if (target !== RENDERBUFFER) {
          gl._errors.push(C1.INVALID_ENUM);
          return;
        }
        const rb = gl._state.renderbuffer;
        if (rb === null || !(rb instanceof WebGLRenderbuffer)) {
          gl._errors.push(C1.INVALID_OPERATION);
          return;
        }
        const w = width | 0;
        const h = height | 0;
        if (w < 0 || h < 0) {
          gl._errors.push(C1.INVALID_VALUE);
          return;
        }
        const smp = samples | 0;
        if (smp < 0) {
          gl._errors.push(C1.INVALID_VALUE);
          return;
        }
        if (smp > gl._state.limits.MAX_SAMPLES) {
          gl._errors.push(C1.INVALID_OPERATION);
          return;
        }
        // Delegate to the core WebGL2 renderbufferStorageMultisample (same spec
        // semantics): it validates internalformat/MAX_RENDERBUFFER_SIZE, allocates
        // the actual surface (allocateRenderbufferSurface — depth planes filled to
        // 1.0), records _samples, and invalidates FBO status caches. Without the
        // allocation the attached depth renderbuffer resolves to no image →
        // FRAMEBUFFER_INCOMPLETE_ATTACHMENT (three.js multisampled render targets:
        // webgl_mirror, webgl_multisampled_renderbuffers).
        (gl as unknown as { renderbufferStorageMultisample(t: number, s: number, f: number, w: number, h: number): void })
          .renderbufferStorageMultisample(target, smp, internalformat, w, h);
      },
    },
  );
}
