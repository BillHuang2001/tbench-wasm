/**
 * src/gl/objects/aux.ts — WebGLUniformLocation, WebGLActiveInfo, WebGLShaderPrecisionFormat.
 *
 * These are NOT WebGLObjects (no delete, no context tag) — they are opaque
 * handles / plain records per the WebGL spec:
 *  - WebGLUniformLocation: opaque handle; only valid for the program it came from.
 *  - WebGLActiveInfo: { size, type, name } result record.
 *  - WebGLShaderPrecisionFormat: { rangeMin, rangeMax, precision } result record.
 */

import type { GLenum, GLint, GLuint } from '../types';
import type { WebGLProgram } from './webgl-program';

export class WebGLUniformLocation {
  /** Program this location belongs to — cross-program use → INVALID_OPERATION. */
  declare _program: WebGLProgram;
  /** Index into Program.uniforms (glsl/ contract §1). */
  declare _index: number;
  /** Base name (without array suffix) for validation of uniform* type mismatches. */
  declare _name: string;

  constructor(program: WebGLProgram, index: number, name: string) {
    this._program = program;
    this._index = index;
    this._name = name;
  }
}

export class WebGLActiveInfo {
  size: GLint;
  type: GLenum;
  name: string;

  constructor(size: GLint, type: GLenum, name: string) {
    this.size = size;
    this.type = type;
    this.name = name;
  }
}

export class WebGLShaderPrecisionFormat {
  rangeMin: GLint;
  rangeMax: GLint;
  precision: GLint;

  constructor(rangeMin: GLint, rangeMax: GLint, precision: GLint) {
    this.rangeMin = rangeMin;
    this.rangeMax = rangeMax;
    this.precision = precision;
  }
}

/** WebIDL-validated uniform location argument: null allowed, non-WebGLUniformLocation throws TypeError. */
export function validateUniformLocation(loc: unknown, program: WebGLProgram | null): WebGLUniformLocation | null {
  if (loc === null || loc === undefined) return null;
  if (!(loc instanceof WebGLUniformLocation)) {
    throw new TypeError(`Argument is not of type 'WebGLUniformLocation'`);
  }
  if (program !== null && (loc as WebGLUniformLocation)._program !== program) {
    return null; // caller pushes INVALID_OPERATION
  }
  return loc as WebGLUniformLocation;
}

export type { GLuint };
