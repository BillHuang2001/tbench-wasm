/**
 * src/gl/extensions/util.ts — shared helpers for extension factories.
 *
 *  - `isLost`: context-loss guard mirroring the api/ modules (silent no-op
 *    while lost — NO error, per CTS context-lost.html).
 *  - `ctorOf`: cast an object class with a protected constructor so it can be
 *    passed to validateObject/createObject (same pattern as api/buffers.ts).
 *  - `buildExtension`: assemble an extension object from a constant table
 *    (enumerable, non-writable — spec) plus methods (exact arity preserved by
 *    the callers' function definitions).
 */

import type { WebGLRenderingContext } from '../webgl1';
import { installConstants } from '../constants';
import type { WebGLObject } from '../objects';

/**
 * Context-loss guard: no-op while lost WITHOUT generating an error (CTS
 * context-lost.html asserts NO_ERROR after every void/extension call while
 * lost — the single CONTEXT_LOST_WEBGL is delivered via getError's lost-epoch,
 * lost.ts).
 */
export function isLost(ctx: WebGLRenderingContext): boolean {
  return ctx._isLost;
}

/**
 * Cast a class with a protected constructor so it can be passed to
 * validateObject/createObject (same pattern as api/buffers.ts). The parameter
 * is typed via `prototype` because TypeScript rejects protected-constructor
 * classes as construct-signature arguments.
 */
export function ctorOf<T extends WebGLObject>(
  cls: { prototype: T },
): new (context: WebGLRenderingContext) => T {
  return cls as unknown as new (context: WebGLRenderingContext) => T;
}

/** Assemble an extension object: constants + methods (both enumerable). */
export function buildExtension(
  constants: Record<string, number>,
  methods?: Record<string, (...args: never[]) => unknown>,
): object {
  const obj: Record<string, unknown> = {};
  installConstants(obj, constants);
  if (methods) {
    for (const key of Object.keys(methods)) obj[key] = methods[key];
  }
  return obj;
}
