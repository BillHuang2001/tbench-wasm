/**
 * src/gl/objects/webgl-sync.ts — WebGLSync (WebGL2 sync objects).
 *
 * The renderer is synchronous (all GL work completes during the call), but the
 * spec still requires a sync object to become SIGNALED only after control
 * returns to the event loop (CTS sync-webgl-specific.html spin-loops and
 * asserts UNSIGNALED). fenceSync creates the object UNSIGNALED and schedules a
 * setTimeout(0) that flips `_signaled`; clientWaitSync(…, 0) before the flip
 * returns TIMEOUT_EXPIRED, after it ALREADY_SIGNALED. See api/webgl2.ts.
 */

import { WebGLObject } from './webgl-object';
import type { GLenum, GLuint64 } from '../types';

export class WebGLSync extends WebGLObject {
  /** SYNC_FENCE. */
  _condition: GLenum = 0;
  /** 0 or SYNC_FLUSH_COMMANDS_BIT. */
  _flags: GLenum = 0;
  /** SYNC_STATUS — flips to SIGNALED on the event loop after fenceSync. */
  _signaled = false;
  /** Monotonic id for ordering assertions (fenceSync ordering tests). */
  _id = 0;
  /** Timeout bookkeeping (clientWaitSync). */
  _waitStarted: GLuint64 = 0;
}
