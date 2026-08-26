/**
 * src/gl/objects/webgl-query.ts — WebGLQuery (WebGL2 occlusion queries).
 *
 * ANY_SAMPLES_PASSED / ANY_SAMPLES_PASSED_CONSERVATIVE targets. Sample counts
 * are fed by the rasterizer through the draw-pipeline feedback hook (see
 * CONTEXT.md "Occlusion queries" design note) — the query must report a real
 * sample count for CTS occlusion tests, not a guess.
 */

import { WebGLObject } from './webgl-object';
import type { GLenum } from '../types';

export class WebGLQuery extends WebGLObject {
  /** Target the query was created for (first beginQuery binds it). */
  _target: GLenum = 0;
  /** True while the query is active (beginQuery … endQuery). */
  _active = false;
  /** Number of samples that passed (accumulated; updated at endQuery). */
  _result = 0;
  /** QUERY_RESULT_AVAILABLE. */
  _resultAvailable = false;
  /** EXT_disjoint_timer_query_webgl2: TIME_ELAPSED/TIMESTAMP result in ns. */
  _isTimerQuery = false;
}
