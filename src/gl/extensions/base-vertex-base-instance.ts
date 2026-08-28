/**
 * src/gl/extensions/base-vertex-base-instance.ts — WEBGL_draw_instanced_base_
 * vertex_base_instance + WEBGL_multi_draw_instanced_base_vertex_base_instance
 * (WebGL2, registry versions [2]).
 *
 * Per the extension.xml IDL the methods live on the EXTENSION OBJECT (unlike
 * WEBGL_multi_draw whose methods are added to the context prototypes) — the
 * CTS page (conformance2/extensions/webgl-multi-draw-instanced-base-vertex-
 * base-instance.html) calls them exclusively as `ext.method(...)`, so no
 * prototype installation is needed.
 *
 * Semantics (GLES 3.2 §10.5 + ANGLE_base_vertex_base_instance):
 *  - baseVertex (GLint, indexed draws only) is ADDED to every element index
 *    for attribute fetch AND to gl_VertexID.
 *  - baseInstance (GLuint) is ADDED to the divisor-based instance attribute
 *    fetch index: element = baseInstance + floor(instanceId/divisor).
 *    gl_InstanceID is NOT offset (starts at 0 for every draw).
 *  - Out-of-range accesses (baseVertex/baseInstance pushing the fetch past the
 *    end of an enabled vertex array) do NOT generate an error here — robust
 *    buffer access yields 0 for the missing components and NO_ERROR (the CTS
 *    accepts [NO_ERROR, INVALID_OPERATION] for these cases; matches ANGLE's
 *    position, see KhronosGroup/WebGL#3764).
 *
 * The GLSL builtins gl_BaseVertex/gl_BaseInstance are deliberately NOT added
 * (extension spec: "The vertex shader builtins gl_BaseVertex and gl_BaseInstance
 * are not added") — shaders using them without an (unavailable) extension
 * directive must fail to compile, which the test verifies.
 */

import type { WebGLRenderingContext } from '../webgl1';
import { C1 } from '../constants';
import {
  executeDraw,
  executeMultiDrawArraysInstancedBaseInstance,
  executeMultiDrawElementsInstancedBaseVertexBaseInstance,
  validateDrawArrays,
  validateDrawElements,
} from '../draw';
import { toList } from '../api/draw';
import { buildExtension, isLost } from './util';

/**
 * Convert a WebIDL list argument (Int32List/Uint32List) to a typed array —
 * TypeError on junk (WebIDL conversion failures propagate to the page, exactly
 * like every other API method).
 */
function toInt32List(v: unknown, name: string): Int32Array {
  return toList(v as Int32Array | number[], Int32Array, name);
}

function toUint32List(v: unknown, name: string): Uint32Array {
  return toList(v as Uint32Array | number[], Uint32Array, name);
}

/** drawcount + list-length preconditions shared by both multi-draw methods. */
function checkDrawcount(
  ctx: WebGLRenderingContext,
  drawcount: number,
  lengths: number[],
  offsets: number[],
): number {
  const dc = drawcount | 0;
  if (dc < 0) { ctx._errors.push(C1.INVALID_VALUE); return -1; }
  if (dc === 0) return 0; // NO_ERROR no-op
  for (let i = 0; i < lengths.length; i++) {
    if ((offsets[i] >>> 0) + dc > lengths[i]) {
      ctx._errors.push(C1.INVALID_OPERATION);
      return -1;
    }
  }
  return dc;
}

// ---------------------------------------------------------------------------
// WEBGL_draw_instanced_base_vertex_base_instance
// ---------------------------------------------------------------------------

/**
 * drawArraysInstancedBaseInstanceWEBGL(mode, first, count, instanceCount,
 * baseInstance) — drawArraysInstanced validation + baseInstance (GLuint).
 */
export function createWEBGLDrawInstancedBaseVertexBaseInstance(ctx: WebGLRenderingContext): object {
  return buildExtension({}, {
    drawArraysInstancedBaseInstanceWEBGL: (
      mode: number, first: number, count: number,
      instanceCount: number, baseInstance: number,
    ): void => {
      if (isLost(ctx)) return;
      // mode/first/count/instanceCount validate exactly like drawArraysInstanced
      // (INVALID_ENUM mode → INVALID_VALUE negatives → common preconditions).
      const req = validateDrawArrays(ctx, mode, first | 0, count | 0, instanceCount | 0);
      if (!req) return;
      req.baseInstance = baseInstance >>> 0;
      try { executeDraw(ctx, req); } catch { ctx._errors.push(C1.INVALID_OPERATION); }
    },

    drawElementsInstancedBaseVertexBaseInstanceWEBGL: (
      mode: number, count: number, type: number, offset: number,
      instanceCount: number, baseVertex: number, baseInstance: number,
    ): void => {
      if (isLost(ctx)) return;
      const req = validateDrawElements(
        ctx, mode, count | 0, type, offset,
        { instanceCount: instanceCount | 0 },
      );
      if (!req) return;
      req.baseVertex = baseVertex | 0;      // GLint
      req.baseInstance = baseInstance >>> 0; // GLuint
      try { executeDraw(ctx, req); } catch { ctx._errors.push(C1.INVALID_OPERATION); }
    },
  });
}

// ---------------------------------------------------------------------------
// WEBGL_multi_draw_instanced_base_vertex_base_instance
// ---------------------------------------------------------------------------

/**
 * multiDrawArraysInstancedBaseInstanceWEBGL(mode, firsts, firstsOffset, counts,
 * countsOffset, instanceCounts, instanceCountsOffset, baseInstances,
 * baseInstancesOffset, drawcount) — multi-draw list validation (drawcount < 0
 * → INVALID_VALUE; offset + drawcount > list length → INVALID_OPERATION;
 * drawcount == 0 → NO_ERROR no-op) then the validate-all-subdraws-first engine.
 */
export function createWEBGLMultiDrawInstancedBaseVertexBaseInstance(ctx: WebGLRenderingContext): object {
  return buildExtension({}, {
    multiDrawArraysInstancedBaseInstanceWEBGL: (
      mode: number,
      firsts: unknown, firstsOffset: number,
      counts: unknown, countsOffset: number,
      instanceCounts: unknown, instanceCountsOffset: number,
      baseInstances: unknown, baseInstancesOffset: number,
      drawcount: number,
    ): void => {
      if (isLost(ctx)) return;
      const firstsArr = toInt32List(firsts, 'Int32List');
      const countsArr = toInt32List(counts, 'Int32List');
      const instArr = toInt32List(instanceCounts, 'Int32List');
      const baseInstArr = toUint32List(baseInstances, 'Uint32List');
      const dc = checkDrawcount(ctx, drawcount, [firstsArr.length, countsArr.length, instArr.length, baseInstArr.length],
        [firstsOffset, countsOffset, instanceCountsOffset, baseInstancesOffset]);
      if (dc <= 0) return; // 0 = no-op; -1 = error already pushed
      try {
        executeMultiDrawArraysInstancedBaseInstance(
          ctx, mode,
          firstsArr, firstsOffset, countsArr, countsOffset,
          instArr, instanceCountsOffset, baseInstArr, baseInstancesOffset, dc,
        );
      } catch { ctx._errors.push(C1.INVALID_OPERATION); }
    },

    multiDrawElementsInstancedBaseVertexBaseInstanceWEBGL: (
      mode: number,
      counts: unknown, countsOffset: number,
      type: number,
      offsets: unknown, offsetsOffset: number,
      instanceCounts: unknown, instanceCountsOffset: number,
      baseVertices: unknown, baseVerticesOffset: number,
      baseInstances: unknown, baseInstancesOffset: number,
      drawcount: number,
    ): void => {
      if (isLost(ctx)) return;
      const countsArr = toInt32List(counts, 'Int32List');
      const offsetsArr = toInt32List(offsets, 'Int32List');
      const instArr = toInt32List(instanceCounts, 'Int32List');
      const baseVertsArr = toInt32List(baseVertices, 'Int32List'); // GLint — may be negative
      const baseInstArr = toUint32List(baseInstances, 'Uint32List');
      const dc = checkDrawcount(ctx, drawcount,
        [countsArr.length, offsetsArr.length, instArr.length, baseVertsArr.length, baseInstArr.length],
        [countsOffset, offsetsOffset, instanceCountsOffset, baseVerticesOffset, baseInstancesOffset]);
      if (dc <= 0) return; // 0 = no-op; -1 = error already pushed
      try {
        executeMultiDrawElementsInstancedBaseVertexBaseInstance(
          ctx, mode,
          countsArr, countsOffset, type, offsetsArr, offsetsOffset,
          instArr, instanceCountsOffset,
          baseVertsArr, baseVerticesOffset, baseInstArr, baseInstancesOffset, dc,
        );
      } catch { ctx._errors.push(C1.INVALID_OPERATION); }
    },
  });
}
