/**
 * vertex.ts — VERTEX stage assembly (cross-module contract §1; task C4).
 *
 * generateVertexStage(ast, layout) lowers the shader's `main` into a JS
 * function body (the linker compiles it via new Function('ctx','R', body) and
 * installs it as Program.vertex.run). All the machinery lives BELOW this
 * file — env.ts (storage access, locals, scratch), expressions.ts,
 * statements.ts (emitStatements/FnEmitContext) and functions.ts
 * (installUserFunctions) — this file only:
 *   1. creates the stage env,
 *   2. installs the user functions (BEFORE main emission so calls inline),
 *   3. emits main's statements (no FnEmitContext — main's `return` is a bare
 *      `return;`),
 *   4. assembles the body ('var t0, t1, ...;' temps header + statement lines)
 *      and reports the scratch sizes the generated body needs.
 *
 * The generated body touches exactly these ctx fields:
 *   reads:  ctx.attribs, ctx.attribIndices, ctx.uniforms, ctx.intUniforms,
 *           ctx.blockStores, ctx.blockIntStores, ctx.vertexId, ctx.instanceId,
 *           ctx.scratch, ctx.intScratch
 *   writes: ctx.out.position, ctx.out.pointSize, ctx.out.varyings,
 *           ctx.scratch, ctx.intScratch
 * (matching VertexExecCtx in program.ts — see the C4 selftest for the exact
 * field shapes the linker/gl/ must provide).
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

/** Generate the vertex stage body (writes ctx.out.position/pointSize/varyings). */
export function generateVertexStage(ast: TranslationUnit, layout: CodegenLayout): StageCodegenResult {
  const env = new CodegenEnv('VERTEX', layout);
  // Seed user struct type names so `Foo(...)` calls resolve to the struct
  // ctor (emitStructCtor) instead of falling through to builtin resolution.
  for (const s of layout.structNames ?? []) env.structNames.add(s);
  installUserFunctions(ast, env); // BEFORE main emission — calls must inline
  const main = findMain(ast);
  const lines = emitStatements(main.body.body, env);
  const body = (env.temps.length ? `var ${env.temps.join(', ')};` : '') + '\n' + lines.join('\n');
  return { body, scratchSize: env.scratchSize, intScratchSize: env.intScratchSize };
}
