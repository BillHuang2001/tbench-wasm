/**
 * src/gl/objects/webgl-shader.ts — WebGLShader.
 */

import { WebGLObject } from './webgl-object';
import type { GLenum } from '../types';

/** Result of compileShader (glsl/ contract §1). */
export interface ShaderCompileResult {
  ok: boolean;
  errors?: { line: number; message: string }[];
}

export class WebGLShader extends WebGLObject {
  /** VERTEX_SHADER | FRAGMENT_SHADER. */
  _type: GLenum = 0;
  /** Source string (set by shaderSource). */
  _source = '';
  /** COMPILE_STATUS. */
  _compileStatus = false;
  /** Info log ('' when ok). */
  _infoLog = '';
  /** Result of glsl/ compileShader — the compiled program model (contract §1). */
  _compiled: ShaderCompileResult | null = null;
  /** Translated source cache for WEBGL_debug_shaders. */
  _translatedSource: string | null = null;
}
