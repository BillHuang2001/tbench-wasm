/**
 * src/gl/objects/webgl-program.ts — WebGLProgram.
 *
 * The program is the link between gl/ and glsl/: at link time gl/ calls
 * `linkProgram(vs._compiled, fs._compiled)` (contract §1) and stores the
 * resulting `Program` model here along with the per-program uniform store
 * (typed-array storage laid out per Program.uniforms[].location — negotiated
 * with glsl/; see CONTEXT.md Design Decisions).
 */

import { WebGLObject } from './webgl-object';
import type { WebGLShader } from './webgl-shader';
import type { GLenum, GLuint } from '../types';

/** glsl/ program model (contract §1) — structural, filled by glsl/. */
export interface ProgramModel {
  attributes: { name: string; location: GLuint; type: GLenum; size: number; components: number; integral: boolean }[];
  uniforms: { name: string; location: number; type: GLenum; size: number; components: number; integral: boolean; blockIndex: number; sampler: boolean }[];
  uniformBlocks: {
    name: string;
    index: number;
    size: number;
    activeUniforms: { name: string; offset: number; type: GLenum; size: number; arrayStride: number; matrixStride: number; rowMajor: boolean }[];
  }[];
  varyings: { name: string; type: GLenum; components: number; flat: boolean }[];
  vertex: { run(ctx: unknown): void };
  fragment: {
    run(ctx: unknown): void;
    usesDerivatives: boolean;
    usesFragDepth: boolean;
    outputs: { location: number; type: GLenum }[];
  };
  usesPointSize: boolean;
  usesGLPointCoord: boolean;
  usesFragData: boolean; // WebGL1 gl_FragData (drawBuffers interplay)
}

export class WebGLProgram extends WebGLObject {
  /** Currently attached shaders. */
  _attachedShaders: Set<WebGLShader> = new Set();
  /** LINK_STATUS. */
  _linkStatus = false;
  /** Info log. */
  _infoLog = '';
  /** glsl/ program model — null until link succeeds (contract §1). */
  _program: ProgramModel | null = null;
  /** Per-program uniform storage (DataView over ArrayBuffer) — layout by glsl/. */
  _uniformStore: DataView | null = null;
  /** Uniform block stores: blockIndex → DataView (allocated at link). */
  _blockStores: (DataView | null)[] = [];
  /** Pending bindAttribLocation (applied at next link). */
  _bindAttribLocations: Map<string, GLuint> = new Map();
  /** transformFeedbackVaryings (applied at next link). */
  _transformFeedbackVaryings: string[] | null = null;
  /** TRANSFORM_FEEDBACK_BUFFER_MODE for TF varyings. */
  _tfBufferMode: GLenum = 0;
  /** True while this program is in useProgram — delete is deferred. */
  _inUse = false;
  /** True while this program is active in transform feedback — delete deferred. */
  _inTransformFeedback = false;
  /** Used by getProgramParameter/COMPLETION_STATUS_KHR: true when link "done" (always, synchronous). */
  _linkComplete = true;
}
