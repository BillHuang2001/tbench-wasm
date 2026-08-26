/**
 * src/gl/draw.ts — the DRAW PIPELINE ENGINE (internal; api/draw.ts delegates here).
 *
 * Pipeline per draw call (contract §2 with raster/):
 *  1. Validate: current program linked, FBO complete, enabled attribs have
 *     buffers (WebGL1 generic attrib 0 fallback), index buffer bound + type
 *     valid for indexed draws, no feedback loops, transform feedback active
 *     rules (primitive mode consistency, no default FBO + TF, no non-TF-varying
 *     draws while active, no query/TF conflicts).
 *  2. Attribute fetch: for each enabled attrib, read components from the bound
 *     buffer (byte offset = attrib.offset + vertexIndex * stride; WebGL2
 *     integer attribs via vertexAttribIPointer stay integer; constant attribs
 *     when no buffer bound) → per-vertex Float32Array input (or int views for
 *     integer attribs).
 *  3. Vertex evaluation: for each vertex (× instanceCount, divisor-based index
 *     advance), call program.vertex.run(ctx) with ctx.attribs/uniforms/
 *     vertexId/instanceId; collect output records: [px,py,pz,pw, pointSize,
 *     varyings...] (packed per Program.varyings; contract §2 record layout).
 *  4. Indexed draws: fetch indices from ELEMENT_ARRAY_BUFFER (UNSIGNED_BYTE in
 *     WebGL2, UNSIGNED_SHORT, UNSIGNED_INT with OES_element_index_uint/WebGL2;
 *     byte-offset + index offset validation, index bounds vs buffer size,
 *     MAX_ELEMENT_INDEX). drawRangeElements validates start/end against indices.
 *  5. Transform feedback capture: when active, write the captured varyings of
 *     each processed primitive into the bound TRANSFORM_FEEDBACK_BUFFER ranges
 *     (interleaved or separate per bufferMode) and count primitives; rasterizer
 *     is bypassed (rasterizerDiscard-equivalent) — occlusion queries do NOT
 *     count TF-only draws.
 *  6. Occlusion queries: when ANY_SAMPLES_PASSED(*) active, the rasterizer is
 *     asked to report whether any fragment passed the depth test — the raster
 *     feedback hook (contract: raster exposes a `sampleCount` callback on the
 *     DrawCall; gl sums into state.activeQueries) must be wired in Phase 2.
 *  7. Rasterize: build the DrawCall (contract §2) and call rasterizer.draw().
 *
 * All engine functions take the context and mutate only context-owned state.
 */

import type { WebGLRenderingContext } from './webgl1';
import type { GLenum, GLint, GLintptr, GLsizei, GLuint } from './types';

/** A fully validated, assembled draw request (before rasterizer call). */
export interface DrawRequest {
  mode: GLenum;
  count: GLsizei;
  instanceCount: GLsizei;
  /** For non-indexed: first vertex. For indexed: offset into index buffer (bytes). */
  firstOrOffset: GLint | GLintptr;
  indexed: boolean;
  /** For indexed: UNSIGNED_BYTE | UNSIGNED_SHORT | UNSIGNED_INT. */
  indexType?: GLenum;
  /** For drawRangeElements: [start, end] inclusive index range. */
  range?: [GLuint, GLuint];
}

/**
 * Execute an assembled draw request: attribute fetch → vertex evaluation →
 * record packing → TF capture → rasterizer.draw (steps above).
 * @internal engine — called by api/draw.ts after validation.
 */
export function executeDraw(ctx: WebGLRenderingContext, req: DrawRequest): void {
  void ctx;
  void req;
  throw new Error('GL stub: draw pipeline (Phase 2 — see src/gl/CONTEXT.md)');
}

/** clear(mask): scissor-respecting clear of color/depth/stencil of the draw target. */
export function executeClear(ctx: WebGLRenderingContext, mask: GLuint): void {
  void ctx;
  void mask;
  throw new Error('GL stub: clear (Phase 2)');
}

/** readPixels with pack-state (alignment/rowLength/skip) + format conversions. */
export function executeReadPixels(
  ctx: WebGLRenderingContext,
  x: GLint, y: GLint, width: GLsizei, height: GLsizei,
  format: GLenum, type: GLenum, pixels: ArrayBufferView,
): void {
  void ctx; void x; void y; void width; void height; void format; void type; void pixels;
  throw new Error('GL stub: readPixels (Phase 2)');
}

/** blitFramebuffer (color with filter, depth/stencil nearest-only). */
export function executeBlitFramebuffer(
  ctx: WebGLRenderingContext,
  srcX0: GLint, srcY0: GLint, srcX1: GLint, srcY1: GLint,
  dstX0: GLint, dstY0: GLint, dstX1: GLint, dstY1: GLint,
  mask: GLuint, filter: GLenum,
): void {
  void ctx; void srcX0; void srcY0; void srcX1; void srcY1;
  void dstX0; void dstY0; void dstX1; void dstY1; void mask; void filter;
  throw new Error('GL stub: blitFramebuffer (Phase 2)');
}

/** clearBuffer* (WebGL2): clear one attachment of the draw target. */
export function executeClearBuffer(
  ctx: WebGLRenderingContext,
  buffer: GLenum, drawbuffer: GLint, values: Float32Array | Int32Array | Uint32Array | null,
  depth?: number, stencil?: number,
): void {
  void ctx; void buffer; void drawbuffer; void values; void depth; void stencil;
  throw new Error('GL stub: clearBuffer* (Phase 2)');
}
