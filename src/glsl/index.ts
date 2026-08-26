/**
 * index.ts — public re-exports of src/glsl/.
 *
 * Consumers (gl/, raster/, tests) import from './glsl' or './glsl/index':
 *   compileShader, linkProgram, Shader, ShaderInfo, LinkOptions, LinkLimits
 *   (compiler.ts), Program + all info types + exec contexts (program.ts),
 *   the GLSL type system (types.ts) and the AST definitions (ast.ts).
 */
export * from './types.js';
export * from './ast.js';
export * from './program.js';
export * from './compiler.js';
