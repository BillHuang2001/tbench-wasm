/**
 * src/gl/extensions/instancing.ts — ANGLE_instanced_arrays (WebGL1).
 *
 * WebGL1 instanced drawing: drawArraysInstancedANGLE / drawElementsInstancedANGLE
 * / vertexAttribDivisorANGLE. The draw ENGINE (draw.ts executeDraw, implemented
 * by the draw agent) performs the deep validation (program linked, FBO complete,
 * ≥1 enabled attrib with divisor 0, index-buffer bounds, feedback loops); this
 * factory performs the API-surface validation (mode/count/type/offset/primcount,
 * attrib index) and assembles the DrawRequest exactly like the WebGL2
 * drawArraysInstanced path (api/draw.ts) does.
 *
 * While the engine is still stubbed (parallel wave), executeDraw throws — the
 * catch converts that to a spec-shaped INVALID_OPERATION so extension calls
 * never crash the page (the draw agent's engine replaces the stub later).
 */

import type { WebGLRenderingContext } from '../webgl1';
import { C1 } from '../constants';
import { executeDraw, type DrawRequest } from '../draw';
import { buildExtension, isLost } from './util';

const MODES: number[] = [
  C1.POINTS,
  C1.LINES,
  C1.LINE_LOOP,
  C1.LINE_STRIP,
  C1.TRIANGLES,
  C1.TRIANGLE_STRIP,
  C1.TRIANGLE_FAN,
];

/** Byte size of an index type (UNSIGNED_SHORT/UNSIGNED_INT; UNSIGNED_BYTE is WebGL2-only). */
function indexTypeSize(type: number): number | null {
  switch (type) {
    case C1.UNSIGNED_SHORT:
      return 2;
    case C1.UNSIGNED_INT:
      return 4;
    default:
      return null;
  }
}

/**
 * Dispatch an assembled instanced draw to the engine. Prefers the installed
 * WebGL2 method when present (defensive — this extension is WebGL1-only, so
 * the executeDraw path is the one actually used); the engine call is wrapped
 * so a still-stubbed engine degrades to INVALID_OPERATION instead of throwing.
 */
function dispatchDraw(ctx: WebGLRenderingContext, req: DrawRequest): void {
  try {
    executeDraw(ctx, req);
  } catch {
    // Engine stub (parallel wave) — spec-shaped error until the draw agent lands.
    ctx._errors.push(C1.INVALID_OPERATION);
  }
}

/** ANGLE_instanced_arrays factory. */
export function createANGLEInstancedArrays(ctx: WebGLRenderingContext): object {
  const ext = buildExtension(
    { VERTEX_ATTRIB_ARRAY_DIVISOR_ANGLE: 0x88fe },
    {
      drawArraysInstancedANGLE: (mode: number, first: number, count: number, primcount: number): void => {
        const gl = ctx;
        if (isLost(gl)) return;
        if (!MODES.includes(mode)) {
          gl._errors.push(C1.INVALID_ENUM);
          return;
        }
        const c = count | 0; // GLsizei (WebIDL long)
        const pc = primcount | 0;
        if (c < 0 || pc < 0) {
          gl._errors.push(C1.INVALID_VALUE);
          return;
        }
        dispatchDraw(gl, { mode, count: c, instanceCount: pc, firstOrOffset: first | 0, indexed: false });
      },

      drawElementsInstancedANGLE: (mode: number, count: number, type: number, offset: number, primcount: number): void => {
        const gl = ctx;
        if (isLost(gl)) return;
        if (!MODES.includes(mode)) {
          gl._errors.push(C1.INVALID_ENUM);
          return;
        }
        const c = count | 0;
        const pc = primcount | 0;
        if (c < 0 || pc < 0) {
          gl._errors.push(C1.INVALID_VALUE);
          return;
        }
        const size = indexTypeSize(type);
        const uintOk = type === C1.UNSIGNED_INT && gl._extensions.has('OES_element_index_uint');
        if (size === null && !uintOk) {
          gl._errors.push(C1.INVALID_ENUM);
          return;
        }
        const off = offset | 0; // GLintptr (WebIDL long long → keep integer part)
        if (off < 0) {
          gl._errors.push(C1.INVALID_VALUE);
          return;
        }
        if (off % (size ?? 4) !== 0) {
          gl._errors.push(C1.INVALID_OPERATION);
          return;
        }
        dispatchDraw(gl, { mode, count: c, instanceCount: pc, firstOrOffset: off, indexed: true, indexType: type });
      },

      vertexAttribDivisorANGLE: (index: number, divisor: number): void => {
        const gl = ctx;
        if (isLost(gl)) return;
        const i = index >>> 0; // WebIDL unsigned long
        if (i >= gl._state.limits.MAX_VERTEX_ATTRIBS) {
          gl._errors.push(C1.INVALID_VALUE);
          return;
        }
        gl._state.vao.attribs[i].divisor = divisor >>> 0;
      },
    },
  );
  return ext;
}
