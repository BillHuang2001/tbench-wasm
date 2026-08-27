/**
 * src/gl/validation.ts — shared argument / object validation helpers.
 *
 * Emulates WebIDL + WebGL spec validation for arguments:
 *  1. WebIDL conversion failures (plain object where a WebGL object is expected)
 *     THROW a TypeError, exactly like browser bindings (CTS type-conversion tests).
 *  2. Valid WebGL objects from ANOTHER context, or already-deleted objects, are
 *     rejected with a GL INVALID_OPERATION error and the call is a no-op.
 *  3. `null` is accepted where the spec allows it (bind/delete null → valid).
 *
 * Every public API method validates via these helpers before touching state.
 */

import type { WebGLRenderingContext } from './webgl1';
import { WebGLObject } from './objects/webgl-object';
import { C1 } from './constants';

/**
 * Validate a WebGL object argument.
 * @returns the object cast to T, or null when the argument is null/undefined.
 * @throws TypeError when the argument is a non-null value that is not a WebGLObject
 *         (WebIDL interface conversion failure).
 *         NOTE: the context pushes INVALID_OPERATION and returns null for
 *         cross-context or deleted objects — it does NOT throw.
 */
export function validateObject<T extends WebGLObject>(
  ctx: WebGLRenderingContext,
  obj: unknown,
  ctor: new (...args: never[]) => T,
): T | null {
  if (obj === null || obj === undefined) return null;
  if (!(obj instanceof ctor)) {
    // Cross-class or fake object: browsers throw TypeError at the WebIDL boundary.
    throw new TypeError(`Argument is not of type '${ctor.name}'`);
  }
  const glObj = obj as WebGLObject;
  if (glObj._context !== ctx) {
    ctx._errors.push(C1.INVALID_OPERATION);
    return null;
  }
  if (glObj._deleted) {
    ctx._errors.push(C1.INVALID_OPERATION);
    return null;
  }
  return obj as T;
}

/**
 * Validate a NON-NULLABLE WebGL object argument (WebIDL: null/undefined → the
 * binding THROWS a TypeError before the implementation runs — CTS
 * bad-arguments-test / null-object-behaviour). Cross-context/deleted objects
 * keep the GL INVALID_OPERATION rejection (no throw).
 */
export function validateNonNullableObject<T extends WebGLObject>(
  ctx: WebGLRenderingContext,
  obj: unknown,
  ctor: new (...args: never[]) => T,
): T | null {
  if (obj === null || obj === undefined) {
    throw new TypeError(`Argument is not of type '${ctor.name}'`);
  }
  return validateObject(ctx, obj, ctor);
}

/** Validate an argument that may be null (bind/delete semantics — null is legal). */
export function validateNullableObject<T extends WebGLObject>(
  ctx: WebGLRenderingContext,
  obj: unknown,
  ctor: new (...args: never[]) => T,
): T | null {
  return validateObject(ctx, obj, ctor);
}

/** True when `v` is a non-negative integer (GLsizei/GLintptr/GLuint style args). */
export function isNonNegativeInt(v: unknown): v is number {
  return typeof v === 'number' && Number.isFinite(v) && v >= 0 && Math.floor(v) === v;
}

/** True when `v` is a finite number (GLfloat/GLclampf style args). */
export function isFiniteNumber(v: unknown): v is number {
  return typeof v === 'number' && Number.isFinite(v);
}

/** True when `v` is a GLenum (finite number) — range checks are per-call. */
export function isGLenum(v: unknown): v is GLenum {
  return typeof v === 'number' && Number.isFinite(v);
}

/**
 * Check an argument is an ArrayBufferView (or ArrayBuffer where the spec allows).
 * Throws TypeError for wrong types (WebIDL), matching browser behavior.
 */
export function requireBufferData(v: unknown, argName: string): ArrayBufferView | ArrayBuffer {
  if (v instanceof ArrayBuffer || ArrayBuffer.isView(v)) return v;
  throw new TypeError(`${argName} is not an ArrayBufferView`);
}

/**
 * Check an argument is a DOMString. Per WebIDL, DOMString conversion is
 * ToString(argument) — null → "null", undefined → "undefined", NO TypeError
 * (CTS type-conversion-test calls shaderSource(shader, null) expecting no
 * exception). Only a Symbol would throw, which String() cannot express — the
 * WebIDL TypeError for Symbol is not reachable through String() and no CTS
 * test exercises it.
 */
export function requireString(v: unknown, argName: string): string {
  if (typeof v === 'string') return v;
  return String(v); // WebIDL DOMString conversion
}
