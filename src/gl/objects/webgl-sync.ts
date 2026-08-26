/**
 * src/gl/objects/webgl-sync.ts — WebGLSync (WebGL2 sync objects).
 *
 * The renderer is synchronous, so a fence sync is SIGNALED immediately and
 * clientWaitSync returns CONDITION_SATISFIED|ALREADY_SIGNALED (or
 * TIMEOUT_EXPIRED per the timeout argument per spec — with MAX_CLIENT_WAIT_TIMEOUT
 * 0 any nonzero timeout yields TIMEOUT_EXPIRED; the spec permits implementations
 * to return TIMEOUT_EXPIRED for any timeout since the wait may be unbounded… but
 * since work completes synchronously, ALREADY_SIGNALED is the honest result for
 * timeout=0 and waits ≤ 0; see CONTEXT.md Known Issues for the CTS expectation).
 */

import { WebGLObject } from './webgl-object';
import type { GLenum, GLuint64 } from '../types';

export class WebGLSync extends WebGLObject {
  /** SYNC_FENCE. */
  _condition: GLenum = 0;
  /** 0 or SYNC_FLUSH_COMMANDS_BIT. */
  _flags: GLenum = 0;
  /** SYNC_STATUS — signaled immediately (synchronous renderer). */
  _signaled = true;
  /** Monotonic id for ordering assertions (fenceSync ordering tests). */
  _id = 0;
  /** Timeout bookkeeping (clientWaitSync). */
  _waitStarted: GLuint64 = 0;
}
