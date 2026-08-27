/**
 * fragment.ts — FRAGMENT stage assembly (cross-module contract §1; task C4).
 *
 * generateFragmentStage(ast, layout) lowers the shader's `main` into a JS
 * function body (the linker compiles it via new Function('ctx','R', body) and
 * installs it as Program.fragment.run). Identical plumbing to vertex.ts —
 * see that file for the assembly steps; the fragment stage differs only in
 * the storage surface the env resolves (varying READS via ctx.varyings[i].v,
 * outputs via ctx.out.color[loc] / ctx.out.fragDepth / ctx.discarded).
 *
 * The generated body touches exactly these ctx fields:
 *   reads:  ctx.varyings[i].v, ctx.fragCoord, ctx.frontFacing, ctx.pointCoord,
 *           ctx.uniforms, ctx.intUniforms, ctx.blockStores, ctx.blockIntStores,
 *           ctx.scratch, ctx.intScratch, ctx.tex (sampler helpers + out arrays)
 *   writes: ctx.out.color[loc], ctx.out.fragDepth, ctx.discarded,
 *           ctx.scratch, ctx.intScratch
 * (matching FragmentExecCtx in program.ts — see the C4 selftest for the exact
 * field shapes the linker/gl/raster must provide).
 */
import type { FunctionDefinition, TranslationUnit } from '../ast.js';
import { CodegenEnv } from './env.js';
import { installUserFunctions } from './functions.js';
import { emitStatements } from './statements.js';
import type { CodegenLayout, StageCodegenResult } from './index.js';

/** The shader's entry function (defensive — semantics rejects its absence). */
function findMain(ast: TranslationUnit): FunctionDefinition {
  for (const d of ast.declarations) {
    if (d.kind === 'function-definition' && d.prototype.name === 'main') return d;
  }
  throw new Error('linker: no main');
}

/** Generate the fragment stage body (writes ctx.out.color[loc], ctx.out.fragDepth, ctx.discarded). */
export function generateFragmentStage(ast: TranslationUnit, layout: CodegenLayout): StageCodegenResult {
  const env = new CodegenEnv('FRAGMENT', layout);
  // Seed user struct type names so `Foo(...)` calls resolve to the struct
  // ctor (emitStructCtor) instead of falling through to builtin resolution.
  for (const s of layout.structNames ?? []) env.structNames.add(s);
  // Dual-number mode: every FLOAT value carries (v, dx, dy) screen-space
  // derivatives (see env.ts DUAL MODE + the dualWrite/varyingReadDual hooks).
  // The raster supplies ctx.varyings[i].ddx/ddy whenever usesDerivatives.
  env.dual = layout.uses.derivatives;
  installUserFunctions(ast, env); // BEFORE main emission — calls must inline
  const main = findMain(ast);
  const lines = emitStatements(main.body.body, env);
  const body = (env.temps.length ? `var ${env.temps.join(', ')};` : '') + '\n' + lines.join('\n');
  return { body, scratchSize: env.scratchSize, intScratchSize: env.intScratchSize };
}
