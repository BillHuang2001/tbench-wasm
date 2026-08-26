/**
 * src/gl/api/programs.ts — shader/program lifecycle, linking, introspection.
 *
 * Owns: createShader, deleteShader, isShader, shaderSource, compileShader,
 * createProgram, deleteProgram, isProgram, attachShader, detachShader,
 * linkProgram, useProgram, validateProgram, getShaderParameter,
 * getProgramParameter, getShaderInfoLog, getProgramInfoLog, getShaderSource,
 * getShaderPrecisionFormat, getAttribLocation, getUniformLocation,
 * bindAttribLocation, getActiveAttrib, getActiveUniform, getUniform +
 * WebGL2: uniformBlockBinding, getUniformBlockIndex, getActiveUniformBlockParameter,
 * getActiveUniformBlockName, getUniformIndices, getActiveUniformsiv,
 * getFragDataLocation, transformFeedbackVaryings, getTransformFeedbackVarying.
 *
 * Pipeline (contract §1 with glsl/):
 *  - compileShader → glsl.compileShader(source, {type, version, defines, extensions});
 *    stores ShaderCompileResult on the shader; COMPILE_STATUS + info log set.
 *  - linkProgram → glsl.linkProgram(vs, fs); on success stores ProgramModel,
 *    allocates the per-program uniform store (typed-array layout per
 *    Program.uniforms[].location — see CONTEXT.md Design Decisions) and block
 *    stores; applies pending bindAttribLocations + transformFeedbackVaryings.
 *  - useProgram: validates program is linked (INVALID_OPERATION otherwise);
 *    state.currentProgram = program.
 *  - getActiveAttrib/Uniform return WebGLActiveInfo {size, type, name} (arrays
 *    report size = array length, name WITHOUT '[0]' suffix); getUniformLocation
 *    supports array-name suffixing ('u[2]'); getUniform reads the uniform store.
 *  - getShaderPrecisionFormat: HIGH_FLOAT/MEDIUM_FLOAT/LOW_FLOAT/HIGH_INT/...;
 *    vertex shaders: full precision; fragment: LOW_FLOAT (range 0..0? — report
 *    per spec min: rangeMin 127, rangeMax 127, precision 23 for highp float).
 *  - KHR_parallel_shader_compile: COMPLETION_STATUS_KHR reports true
 *    (synchronous compile) for both shaders and programs.
 */

import type { WebGLRenderingContext } from '../webgl1';

export function installProgramsApi(proto: WebGLRenderingContext): void {
  void proto;
}
