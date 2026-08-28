/**
 * src/gl/extensions/draw-buffers.ts — WEBGL_draw_buffers (WebGL1).
 *
 * drawBuffersWEBGL(buffers) mirrors the WebGL2 drawBuffers semantics with the
 * WebGL1 extension's additional default-framebuffer restriction. Validation
 * verified against the CTS test (conformance/extensions/webgl-draw-buffers.html):
 *  - Default framebuffer: only [BACK] or [NONE] (length 1) are legal; anything
 *    else (including [], [NONE, NONE], [COLOR_ATTACHMENT0_WEBGL]) →
 *    INVALID_OPERATION.
 *  - FBO: entries must be NONE or COLOR_ATTACHMENT0..MAX_COLOR_ATTACHMENTS-1
 *    (INVALID_ENUM otherwise; BACK on an FBO → INVALID_OPERATION per spec);
 *    COLOR_ATTACHMENTn values must be strictly increasing (INVALID_OPERATION);
 *    length ≤ MAX_DRAW_BUFFERS (INVALID_VALUE); short arrays are legal.
 *  - Stores into state.drawBuffers (default [COLOR_ATTACHMENT0]); the getters
 *    DRAW_BUFFER0..15 pnames (0x8825..0x8834) read it back — the getters.ts
 *    pre-check is patched to admit these pnames on WebGL1 once this extension
 *    is enabled.
 *
 * Constants: DRAW_BUFFER0..15_WEBGL (0x8825..0x8834), MAX_DRAW_BUFFERS_WEBGL
 * (0x8824), MAX_COLOR_ATTACHMENTS_WEBGL (0x8CDF), COLOR_ATTACHMENT0..15_WEBGL
 * (0x8CE0..0x8CEF) — all from CExt.
 */

import type { WebGLRenderingContext } from '../webgl1';
import { C1, CExt } from '../constants';
import { buildExtension, isLost } from './util';

const NONE = 0x0000;
const BACK = 0x0405;
const COLOR_ATTACHMENT0 = 0x8ce0;

function makeConstants(): Record<string, number> {
  const table: Record<string, number> = {
    MAX_DRAW_BUFFERS_WEBGL: CExt.MAX_DRAW_BUFFERS_WEBGL,
    MAX_COLOR_ATTACHMENTS_WEBGL: CExt.MAX_COLOR_ATTACHMENTS_WEBGL,
  };
  for (let i = 0; i < 16; i++) {
    table[`DRAW_BUFFER${i}_WEBGL`] = CExt[`DRAW_BUFFER${i}_WEBGL` as keyof typeof CExt] as number;
    table[`COLOR_ATTACHMENT${i}_WEBGL`] = CExt[`COLOR_ATTACHMENT${i}_WEBGL` as keyof typeof CExt] as number;
  }
  return table;
}

/** WEBGL_draw_buffers factory (WebGL1 only — registry versions: [1]). */
export function createWEBGLDrawBuffers(ctx: WebGLRenderingContext): object {
  return buildExtension(makeConstants(), {
    drawBuffersWEBGL: (buffers: number[]): void => {
      const gl = ctx;
      if (isLost(gl)) return;
      const s = gl._state;

      // WebIDL sequence<GLenum> conversion — array-like required, elements → GLenum.
      let arr: number[];
      try {
        arr = Array.from(buffers as unknown as ArrayLike<unknown>, (v) => {
          const n = Number(v);
          if (!Number.isFinite(n)) throw new TypeError('drawBuffersWEBGL: invalid GLenum in sequence');
          return n >>> 0; // GLenum = unsigned long
        });
      } catch {
        throw new TypeError('drawBuffersWEBGL: buffers is not a sequence<GLenum>');
      }

      const maxDrawBuffers = s.limits.MAX_DRAW_BUFFERS;
      const maxColorAttachments = s.limits.MAX_COLOR_ATTACHMENTS;
      if (arr.length > maxDrawBuffers) {
        gl._errors.push(C1.INVALID_VALUE);
        return;
      }
      if (arr.length === 0) {
        gl._errors.push(C1.INVALID_OPERATION);
        return;
      }

      if (s.drawFramebuffer === null) {
        // Default framebuffer: only [BACK] or [NONE].
        if (arr.length !== 1 || (arr[0] !== BACK && arr[0] !== NONE)) {
          gl._errors.push(C1.INVALID_OPERATION);
          return;
        }
        // Normalize BACK → COLOR_ATTACHMENT0 BEFORE storing: s.drawBuffers is
        // consumed by draw.ts as ATTACHMENT INDICES (db - COLOR_ATTACHMENT0 →
        // surface index); a raw BACK (0x0405) entry computes a negative index
        // and silently drops color writes. getters.ts DRAW_BUFFERi reports the
        // default framebuffer as BACK for any non-NONE first entry, so the
        // normalized storage is observably identical (see api/framebuffers.ts
        // drawBuffers for the same rule).
        s.drawBuffers = arr[0] === BACK ? [COLOR_ATTACHMENT0] : arr;
        return;
      }

      // FBO: NONE or COLOR_ATTACHMENT0..MAX_COLOR_ATTACHMENTS-1, strictly increasing.
      let last = -1;
      for (const b of arr) {
        if (b === NONE) continue;
        if (b === BACK) {
          gl._errors.push(C1.INVALID_OPERATION);
          return;
        }
        const idx = b - COLOR_ATTACHMENT0;
        if (idx < 0 || idx >= maxColorAttachments) {
          gl._errors.push(C1.INVALID_ENUM);
          return;
        }
        if (idx <= last) {
          gl._errors.push(C1.INVALID_OPERATION);
          return;
        }
        last = idx;
      }
      s.drawBuffers = arr;
    },
  });
}
