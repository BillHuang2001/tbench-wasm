/**
 * src/gl/objects/webgl-object.ts — base class for all WebGL objects.
 *
 * Every WebGL object carries a reference to its creating context (the "private
 * context tag") used for cross-context validation, plus a deleted flag. Objects
 * are plain JS classes — the WebGL spec defines them as opaque types; the CTS
 * type-conversion tests require rejecting fake objects and cross-context objects.
 */

import type { WebGLRenderingContext } from '../webgl1';

export abstract class WebGLObject {
  /** Creating context — validation tag (never exposed to the page). */
  declare _context: WebGLRenderingContext;
  /** True after delete*() — further use generates INVALID_OPERATION. */
  _deleted = false;

  protected constructor(context: WebGLRenderingContext) {
    Object.defineProperty(this, '_context', {
      value: context,
      writable: false,
      enumerable: false,
      configurable: false,
    });
  }
}
