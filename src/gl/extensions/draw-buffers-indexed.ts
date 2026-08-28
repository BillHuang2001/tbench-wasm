/**
 * src/gl/extensions/draw-buffers-indexed.ts — OES_draw_buffers_indexed (WebGL2).
 *
 * Per-drawbuffer blend state + color masks + blend enable. Storage:
 *  - `state.blendPerDrawBuffer: Map<buf, {srcRGB,dstRGB,srcAlpha,dstAlpha,eqRGB,eqAlpha}>`
 *    — written by blendFunciOES/blendFuncSeparateiOES/blendEquationiOES/
 *    blendEquationSeparateiOES (entries created lazily, seeded from the base
 *    blend state).
 *  - `state.colorMaskPerDrawBuffer: Map<buf, [r,g,b,a]>` — written by
 *    colorMaskiOES.
 *  - `state.blendEnablePerDrawBuffer: Map<buf, boolean>` — written by
 *    enableiOES/disableiOES. Not declared in the pinned State type: this module
 *    mirrors the map onto `ctx._state` (cast) so ctx._state-based consumers can
 *    find it without importing this module.
 * Per-buffer blend-enable model: buffer 0 always follows the global
 * `caps.BLEND` (enableiOES/disableiOES with index 0 write the cap directly, so
 * gl.isEnabled(gl.BLEND) and gl.enable/disable(gl.BLEND) already reflect buffer
 * 0 — no api/state.ts changes needed); buffers i>0 use their explicit entry
 * when present and fall back to `caps.BLEND` otherwise (see
 * `blendEnableForDrawBuffer`). The draw engine and getIndexedParameter
 * (api/buffers.ts, another agent) consume `state.blendPerDrawBuffer` /
 * `state.colorMaskPerDrawBuffer` / `blendEnablePerDrawBuffer` with base-state
 * fallback.
 *
 * Validation mirrors the WebGL2 parent methods (OES_draw_buffers_indexed is a
 * WebGL2 extension): buf < MAX_DRAW_BUFFERS → INVALID_VALUE; blend factors from
 * the WebGL2 factor set (SRC_ALPHA_SATURATE legal on the dst side in WebGL2)
 * plus SRC1_* when WEBGL_blend_func_extended is enabled; equations FUNC_ADD/
 * FUNC_SUBTRACT/FUNC_REVERSE_SUBTRACT/MIN/MAX. The core WebGL blend-factors
 * limitation (WebGL 1.0 spec §6.15 — CONSTANT_COLOR/ONE_MINUS_CONSTANT_COLOR
 * may not be paired with CONSTANT_ALPHA/ONE_MINUS_CONSTANT_ALPHA) is applied
 * per the extension addendum: (src, dst) for blendFunciOES and
 * (srcRGB, dstRGB) + (srcAlpha, dstAlpha) for blendFuncSeparateiOES →
 * INVALID_OPERATION (constFactorPairInvalid, same rule api/state.ts uses for
 * the non-indexed setters).
 */

import type { WebGLRenderingContext } from '../webgl1';
import { C1, C2 } from '../constants';
import { buildExtension, isLost } from './util';

const BLEND = 0x0be2;
const ZERO = 0x0000;
const ONE = 0x0001;
const CONSTANT_COLOR = 0x8001;
const ONE_MINUS_CONSTANT_COLOR = 0x8002;
const CONSTANT_ALPHA = 0x8003;
const ONE_MINUS_CONSTANT_ALPHA = 0x8004;

/** WebGL2 blend factors (src set — SRC_ALPHA_SATURATE is legal on the dst side in WebGL2). */
const BLEND_FACTORS: number[] = [
  ZERO,
  ONE,
  C1.SRC_COLOR,
  C1.ONE_MINUS_SRC_COLOR,
  C1.SRC_ALPHA,
  C1.ONE_MINUS_SRC_ALPHA,
  C1.DST_ALPHA,
  C1.ONE_MINUS_DST_ALPHA,
  C1.DST_COLOR,
  C1.ONE_MINUS_DST_COLOR,
  C1.SRC_ALPHA_SATURATE,
  CONSTANT_COLOR,
  ONE_MINUS_CONSTANT_COLOR,
  CONSTANT_ALPHA,
  ONE_MINUS_CONSTANT_ALPHA,
];

const BLEND_EQUATIONS: number[] = [
  C1.FUNC_ADD,
  C1.FUNC_SUBTRACT,
  C1.FUNC_REVERSE_SUBTRACT,
  C2.MIN,
  C2.MAX,
];

/** Per-draw-buffer blend enable (buffer i>0 overrides; buffer 0 always follows caps.BLEND). */
const blendEnableMaps = new WeakMap<WebGLRenderingContext, Map<number, boolean>>();
function blendEnableMap(ctx: WebGLRenderingContext): Map<number, boolean> {
  let m = blendEnableMaps.get(ctx);
  if (!m) {
    m = new Map();
    blendEnableMaps.set(ctx, m);
    // Mirror onto ctx._state under a documented name so ctx._state-based consumers
    // (draw engine, getIndexedParameter) can find it without importing this module.
    (ctx._state as unknown as { blendEnablePerDrawBuffer?: Map<number, boolean> }).blendEnablePerDrawBuffer = m;
  }
  return m;
}
/** Effective blend enable for draw buffer i (buffer 0 always follows the global cap). */
export function blendEnableForDrawBuffer(ctx: WebGLRenderingContext, buf: number): boolean {
  const caps = ctx._state.caps.BLEND;
  if (buf === 0) return caps;
  return blendEnableMap(ctx).get(buf) ?? caps;
}

/** Per-drawbuffer blend entry (lazily seeded from the base blend state). */
function blendEntry(ctx: WebGLRenderingContext, buf: number): {
  srcRGB: number; dstRGB: number; srcAlpha: number; dstAlpha: number; eqRGB: number; eqAlpha: number;
} {
  const s = ctx._state;
  let entry = s.blendPerDrawBuffer.get(buf);
  if (!entry) {
    entry = {
      srcRGB: s.blend.srcRGB,
      dstRGB: s.blend.dstRGB,
      srcAlpha: s.blend.srcAlpha,
      dstAlpha: s.blend.dstAlpha,
      eqRGB: s.blend.eqRGB,
      eqAlpha: s.blend.eqAlpha,
    };
    s.blendPerDrawBuffer.set(buf, entry);
  }
  return entry;
}

/** Validate buf < MAX_DRAW_BUFFERS (INVALID_VALUE). Returns -1 on failure. */
function bufIndex(ctx: WebGLRenderingContext, buf: number): number {
  const b = buf >>> 0; // WebIDL unsigned long
  if (b >= ctx._state.limits.MAX_DRAW_BUFFERS) {
    ctx._errors.push(C1.INVALID_VALUE);
    return -1;
  }
  return b;
}

/** Factor accepted for this context (SRC1_* gated on WEBGL_blend_func_extended). */
function factorOk(ctx: WebGLRenderingContext, f: number): boolean {
  if (BLEND_FACTORS.includes(f)) return true;
  if (ctx._extensions.has('WEBGL_blend_func_extended')) {
    return (
      f === 0x88f9 /* SRC1_COLOR_WEBGL */ ||
      f === 0x8589 /* SRC1_ALPHA_WEBGL */ ||
      f === 0x88fa /* ONE_MINUS_SRC1_COLOR_WEBGL */ ||
      f === 0x88fb /* ONE_MINUS_SRC1_ALPHA_WEBGL */
    );
  }
  return false;
}

/** CONSTANT_COLOR / ONE_MINUS_CONSTANT_COLOR factor check. */
function isConstColorFactor(f: number): boolean {
  return f === CONSTANT_COLOR || f === ONE_MINUS_CONSTANT_COLOR;
}

/** CONSTANT_ALPHA / ONE_MINUS_CONSTANT_ALPHA factor check. */
function isConstAlphaFactor(f: number): boolean {
  return f === CONSTANT_ALPHA || f === ONE_MINUS_CONSTANT_ALPHA;
}

/**
 * WebGL 1.0 spec §6.15 ("Blending With Constant Color") / GLES3 §4.1.7: a
 * CONSTANT_COLOR/ONE_MINUS_CONSTANT_COLOR factor may not be paired with a
 * CONSTANT_ALPHA/ONE_MINUS_CONSTANT_ALPHA factor across a blend (a, b) pair →
 * INVALID_OPERATION. The OES_draw_buffers_indexed spec addendum applies the
 * core WebGL blend-factors limitation to the new entrypoints; api/state.ts
 * enforces the same rule for the non-indexed setters.
 */
function constFactorPairInvalid(a: number, b: number): boolean {
  return (
    (isConstColorFactor(a) && isConstAlphaFactor(b)) ||
    (isConstAlphaFactor(a) && isConstColorFactor(b))
  );
}

/** OES_draw_buffers_indexed factory (WebGL2 — registry versions: [2]). */
export function createOESDrawBuffersIndexed(ctx: WebGLRenderingContext): object {
  // Prime the per-drawbuffer blend-enable map so ctx._state.blendEnablePerDrawBuffer
  // exists as soon as the extension is enabled.
  blendEnableMap(ctx);
  return buildExtension({}, {
    enableiOES: (target: number, index: number): void => {
      const gl = ctx;
      if (isLost(gl)) return;
      if (target !== BLEND) {
        gl._errors.push(C1.INVALID_ENUM);
        return;
      }
      const b = bufIndex(gl, index);
      if (b < 0) return;
      if (b === 0) {
        gl._state.caps.BLEND = true;
      } else {
        blendEnableMap(gl).set(b, true);
      }
    },

    disableiOES: (target: number, index: number): void => {
      const gl = ctx;
      if (isLost(gl)) return;
      if (target !== BLEND) {
        gl._errors.push(C1.INVALID_ENUM);
        return;
      }
      const b = bufIndex(gl, index);
      if (b < 0) return;
      if (b === 0) {
        gl._state.caps.BLEND = false;
      } else {
        blendEnableMap(gl).set(b, false);
      }
    },

    blendEquationiOES: (buf: number, mode: number): void => {
      const gl = ctx;
      if (isLost(gl)) return;
      const b = bufIndex(gl, buf);
      if (b < 0) return;
      if (!BLEND_EQUATIONS.includes(mode)) {
        gl._errors.push(C1.INVALID_ENUM);
        return;
      }
      const e = blendEntry(gl, b);
      e.eqRGB = mode;
      e.eqAlpha = mode;
    },

    blendEquationSeparateiOES: (buf: number, modeRGB: number, modeAlpha: number): void => {
      const gl = ctx;
      if (isLost(gl)) return;
      const b = bufIndex(gl, buf);
      if (b < 0) return;
      if (!BLEND_EQUATIONS.includes(modeRGB) || !BLEND_EQUATIONS.includes(modeAlpha)) {
        gl._errors.push(C1.INVALID_ENUM);
        return;
      }
      const e = blendEntry(gl, b);
      e.eqRGB = modeRGB;
      e.eqAlpha = modeAlpha;
    },

    blendFunciOES: (buf: number, src: number, dst: number): void => {
      const gl = ctx;
      if (isLost(gl)) return;
      const b = bufIndex(gl, buf);
      if (b < 0) return;
      if (!factorOk(gl, src) || !factorOk(gl, dst)) {
        gl._errors.push(C1.INVALID_ENUM);
        return;
      }
      if (constFactorPairInvalid(src, dst)) {
        gl._errors.push(C1.INVALID_OPERATION);
        return;
      }
      const e = blendEntry(gl, b);
      e.srcRGB = src;
      e.dstRGB = dst;
      e.srcAlpha = src;
      e.dstAlpha = dst;
    },

    blendFuncSeparateiOES: (buf: number, srcRGB: number, dstRGB: number, srcAlpha: number, dstAlpha: number): void => {
      const gl = ctx;
      if (isLost(gl)) return;
      const b = bufIndex(gl, buf);
      if (b < 0) return;
      if (!factorOk(gl, srcRGB) || !factorOk(gl, dstRGB) || !factorOk(gl, srcAlpha) || !factorOk(gl, dstAlpha)) {
        gl._errors.push(C1.INVALID_ENUM);
        return;
      }
      if (constFactorPairInvalid(srcRGB, dstRGB) || constFactorPairInvalid(srcAlpha, dstAlpha)) {
        gl._errors.push(C1.INVALID_OPERATION);
        return;
      }
      const e = blendEntry(gl, b);
      e.srcRGB = srcRGB;
      e.dstRGB = dstRGB;
      e.srcAlpha = srcAlpha;
      e.dstAlpha = dstAlpha;
    },

    colorMaskiOES: (buf: number, r: boolean, g: boolean, b: boolean, a: boolean): void => {
      const gl = ctx;
      if (isLost(gl)) return;
      const bi = bufIndex(gl, buf);
      if (bi < 0) return;
      gl._state.colorMaskPerDrawBuffer.set(bi, [!!r, !!g, !!b, !!a]);
    },
  });
}
