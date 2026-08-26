/**
 * src/gl/api/webgl2.ts — WebGL2-only object methods (queries, sync, samplers,
 * VAOs, transform feedback bindings).
 *
 * Owns: beginQuery, endQuery, createQuery, deleteQuery, isQuery, getQuery,
 * getQueryParameter; fenceSync, isSync, deleteSync, clientWaitSync, waitSync,
 * getSyncParameter; createSampler, deleteSampler, isSampler, bindSampler,
 * samplerParameterf, samplerParameteri, getSamplerParameter; createVertexArray,
 * deleteVertexArray, isVertexArray, bindVertexArray; createTransformFeedback,
 * deleteTransformFeedback, isTransformFeedback, bindTransformFeedback,
 * beginTransformFeedback, endTransformFeedback, pauseTransformFeedback,
 * resumeTransformFeedback.
 *
 * Behavior notes:
 *  - Occlusion queries MUST count real samples: the draw engine (draw.ts)
 *    sums rasterizer-reported passing fragments into state.activeQueries; the
 *    result is available immediately after endQuery (synchronous renderer).
 *  - getQuery(target, CURRENT_QUERY) returns the active query; getQueryParameter
 *    supports QUERY_RESULT / QUERY_RESULT_AVAILABLE.
 *  - Sync objects are signaled immediately (synchronous); clientWaitSync with
 *    timeout 0 returns CONDITION_SATISFIED (already signaled); with
 *    MAX_CLIENT_WAIT_TIMEOUT_WEBGL = 0, waitSync with timeout > 0 is still a
 *    no-op that returns immediately (see CONTEXT.md Known Issues — verify CTS
 *    `conformance2/sync/` expectations in Phase 2).
 *  - Samplers: samplerParameter validates the same pname/value table as
 *    texParameter; a sampler bound to a unit overrides the texture's params.
 *  - VAOs: bindVertexArray(null) restores the default VAO; deleting a bound VAO
 *    unbinds it (default VAO bound); OES_vertex_array_object shares this engine
 *    through the OES_-suffixed wrappers.
 *  - Transform feedback: beginTransformFeedback validates primitive mode ∈
 *    {POINTS, LINES, TRIANGLES} and that the current program has TF varyings
 *    (INVALID_OPERATION); active TF forbids binding new TF buffers; pause/
 *    resume toggle state.transformFeedback.paused; endTransformFeedback
 *    finalizes the primitives-written counter.
 */

import type { WebGL2RenderingContext } from '../webgl2';

export function installWebGL2Api(proto: WebGL2RenderingContext): void {
  void proto;
}
