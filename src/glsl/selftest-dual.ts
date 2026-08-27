/**
 * selftest-dual.ts — dual-number FRAGMENT codegen checks (tasks C5a1 + C5a2).
 *
 * In dual mode (layout.uses.derivatives → env.dual) every FLOAT value
 * carries (v, dx, dy) — value + screen-space derivatives as JS expression
 * strings. The C5a1 section pins the STORAGE + READS + WRITES + dFdx/dFdy/
 * fwidth layer (value flow restricted to what works WITHOUT the C5a2
 * arithmetic dual templates). The C5a2 section (below) pins the ARITHMETIC
 * dual templates: + - * / % and unary minus, compound *= /= %=, ternary
 * triples, int→float conversions, assignment-expression duals, dual-aware
 * constructors, and the loop accumulator — each verified ANALYTICALLY
 * against the expected derivative.
 *
 * Each shader compiles (compileShader → annotated AST), lowers through
 * generateFragmentStage (env.dual = layout.uses.derivatives), runs via
 * `new Function('ctx','R', body)`, and the resulting ctx.out.color is
 * asserted. ctx.varyings entries carry v + ddx + ddy (the raster supplies
 * the derivative arrays whenever usesDerivatives — the generated code reads
 * them directly, no guards).
 *
 * Run: npx tsx src/glsl/selftest-dual.ts
 *
 * Prints "OK" and exits 0 on success.
 */
import type { TranslationUnit } from './ast.js';
import type { Expr } from './ast.js';
import type { GLSLType } from './types.js';
import { compileShader } from './compiler.js';
import type { ShaderUses } from './compiler.js';
import { generateFragmentStage } from './codegen/index.js';
import type { CodegenLayout, UniformSlot, VaryingLayout } from './codegen/index.js';
import { R } from './codegen/runtime.js';
import { CodegenEnv } from './codegen/env.js';
import { emitExpr } from './codegen/expressions.js';

let failures = 0;
let checks = 0;

function check(cond: boolean, msg: string): void {
  checks++;
  if (!cond) {
    failures++;
    console.error('FAIL: ' + msg);
  }
}

/* ------------------------------------------------------------------ */
/* Layout helpers (derivatives ALWAYS on — dual mode)                  */
/* ------------------------------------------------------------------ */

function dualUses(): ShaderUses {
  return {
    pointSize: false,
    fragCoord: false,
    frontFacing: false,
    pointCoord: false,
    fragDepth: false,
    vertexId: false,
    instanceId: false,
    derivatives: true,
  };
}

function baseLayout(version: 100 | 300, extra?: Partial<CodegenLayout>): CodegenLayout {
  return {
    version,
    uniformSlots: new Map(),
    blocks: new Map(),
    blockIndices: new Map(),
    varyings: new Map(),
    attribLocations: new Map(),
    outputLocations: new Map([['gl_FragColor', 0]]),
    uses: dualUses(),
    ...extra,
  };
}

const slot = (store: 'float' | 'int', s: number, stride = 0): UniformSlot => ({ store, slot: s, stride });
const vg = (index: number, offset: number, components: number, elemComponents: number, flat = false): VaryingLayout =>
  ({ index, offset, components, elemComponents, flat });

/* ------------------------------------------------------------------ */
/* Driver: compileShader → generateFragmentStage → new Function → run  */
/* ------------------------------------------------------------------ */

/** A fragment ctx with one varying (v + ddx + ddy arrays) and float/int stores. */
function fragCtx(res: { scratchSize: number; intScratchSize: number }): Record<string, any> {
  return {
    uniforms: new Float32Array(64),
    intUniforms: new Int32Array(64),
    blockStores: [],
    blockIntStores: [],
    scratch: new Float32Array(Math.max(res.scratchSize, 16)),
    intScratch: new Int32Array(Math.max(res.intScratchSize, 16)),
    out: { color: [new Float32Array(4)], fragDepth: 0 },
    discarded: false,
    fragCoord: new Float32Array([0, 0, 0, 0]),
    frontFacing: true,
    pointCoord: new Float32Array([0, 0]),
    varyings: [],
  };
}

interface DualRunResult {
  body: string;
  ctx: Record<string, any>;
  scratchSize: number;
  intScratchSize: number;
}

function runDual(
  src: string,
  layout: CodegenLayout,
  ctxExtra: Record<string, unknown> = {},
): DualRunResult {
  const r = compileShader(src, {
    type: 'FRAGMENT',
    version: layout.version,
    extensions: new Set(['GL_OES_standard_derivatives']),
  });
  if (!r.ok) throw new Error(`compile failed: ${JSON.stringify(r.errors)}`);
  const res = generateFragmentStage(r.shader.ast as TranslationUnit, layout);
  const fn = new Function('ctx', 'R', res.body);
  const ctx = { ...fragCtx(res), ...ctxExtra };
  fn(ctx, R);
  return { body: res.body, ctx, scratchSize: res.scratchSize, intScratchSize: res.intScratchSize };
}

/* ------------------------------------------------------------------ */
/* 1. dFdx/dFdy of a (non-flat) float varying == the raster ddx/ddy    */
/* ------------------------------------------------------------------ */

{
  const layout = baseLayout(100, { varyings: new Map([['v', vg(0, 0, 1, 1)]]) });
  const r = runDual(
    `#extension GL_OES_standard_derivatives : enable
     precision mediump float;
     varying float v;
     void main() { gl_FragColor.x = dFdx(v); gl_FragColor.y = dFdy(v); }`,
    layout,
    { varyings: [{ v: new Float32Array([0.5]), ddx: new Float32Array([2.0]), ddy: new Float32Array([3.0]) }] },
  );
  const c = r.ctx.out.color[0];
  check(c[0] === 2.0 && c[1] === 3.0, `dFdx(v)==ddx, dFdy(v)==ddy (got [${Array.from(c).join(', ')}])`);
  check(
    r.body.includes('ctx.varyings[0].ddx[0]') && r.body.includes('ctx.varyings[0].ddy[0]'),
    `dual varying reads use direct .ddx/.ddy access (body: ${r.body.slice(0, 120)}...)`,
  );
}

/* ------------------------------------------------------------------ */
/* 2. FLAT varying: dFdx/dFdy must ignore the supplied ddx/ddy → 0     */
/* ------------------------------------------------------------------ */

{
  const layout = baseLayout(100, { varyings: new Map([['v', vg(0, 0, 1, 1, true)]]) });
  const r = runDual(
    `#extension GL_OES_standard_derivatives : enable
     precision mediump float;
     varying float v;
     void main() { gl_FragColor.x = dFdx(v); gl_FragColor.y = dFdy(v); }`,
    layout,
    { varyings: [{ v: new Float32Array([0.5]), ddx: new Float32Array([9.0]), ddy: new Float32Array([9.0]) }] },
  );
  const c = r.ctx.out.color[0];
  check(c[0] === 0 && c[1] === 0, `flat varying: dFdx/dFdy == 0 (got [${Array.from(c).join(', ')}])`);
}

/* ------------------------------------------------------------------ */
/* 3. dFdx(uniform) == 0 (constant duals)                              */
/* ------------------------------------------------------------------ */

{
  const layout = baseLayout(100, { uniformSlots: new Map([['u', slot('float', 0)]]) });
  const r = runDual(
    `#extension GL_OES_standard_derivatives : enable
     precision mediump float;
     uniform float u;
     void main() { gl_FragColor.x = dFdx(u); }`,
    layout,
    { uniforms: new Float32Array([7.0]) },
  );
  check(r.ctx.out.color[0][0] === 0, `dFdx(uniform) == 0 (got ${r.ctx.out.color[0][0]})`);
}

/* ------------------------------------------------------------------ */
/* 4. gl_FragCoord constant duals: dFdx(x)==1, dFdy(y)==1              */
/* ------------------------------------------------------------------ */

{
  const layout = baseLayout(100);
  const r = runDual(
    `#extension GL_OES_standard_derivatives : enable
     precision mediump float;
     void main() { gl_FragColor.x = dFdx(gl_FragCoord.x); gl_FragColor.y = dFdy(gl_FragCoord.y); }`,
    layout,
    { fragCoord: new Float32Array([10, 20, 0.5, 1]) },
  );
  const c = r.ctx.out.color[0];
  check(c[0] === 1 && c[1] === 1, `dFdx(gl_FragCoord.x)==1, dFdy(gl_FragCoord.y)==1 (got [${Array.from(c).join(', ')}])`);
}

/* ------------------------------------------------------------------ */
/* 5. Local chain: assignment writes the triple                        */
/* ------------------------------------------------------------------ */

{
  const layout = baseLayout(100, { varyings: new Map([['v', vg(0, 0, 1, 1)]]) });
  const r = runDual(
    `#extension GL_OES_standard_derivatives : enable
     precision mediump float;
     varying float v;
     void main() { float t = v; gl_FragColor.x = dFdx(t); }`,
    layout,
    { varyings: [{ v: new Float32Array([0.5]), ddx: new Float32Array([2.0]), ddy: new Float32Array([3.0]) }] },
  );
  check(
    r.ctx.out.color[0][0] === 2.0,
    `'float t = v; dFdx(t)' == v.ddx (got ${r.ctx.out.color[0][0]})`,
  );
  check(
    r.body.includes('var t = ctx.varyings[0].v[0], t_dx = ctx.varyings[0].ddx[0], t_dy = ctx.varyings[0].ddy[0];'),
    `decl init emits the triple var line (body: ${r.body.slice(0, 150)}...)`,
  );
}

/* ------------------------------------------------------------------ */
/* 6. Compound += updates all three planes                             */
/* ------------------------------------------------------------------ */

{
  const layout = baseLayout(100, { varyings: new Map([['v', vg(0, 0, 1, 1)]]) });
  const r = runDual(
    `#extension GL_OES_standard_derivatives : enable
     precision mediump float;
     varying float v;
     void main() { float t = v; t += v; gl_FragColor.x = dFdx(t); }`,
    layout,
    { varyings: [{ v: new Float32Array([0.5]), ddx: new Float32Array([2.0]), ddy: new Float32Array([3.0]) }] },
  );
  check(
    r.ctx.out.color[0][0] === 4.0,
    `'t += v; dFdx(t)' == 2*v.ddx (got ${r.ctx.out.color[0][0]})`,
  );
}

/* ------------------------------------------------------------------ */
/* 7. ++ on a float local: constant delta — duals unchanged            */
/* ------------------------------------------------------------------ */

{
  const layout = baseLayout(100, { varyings: new Map([['v', vg(0, 0, 1, 1)]]) });
  const r = runDual(
    `#extension GL_OES_standard_derivatives : enable
     precision mediump float;
     varying float v;
     void main() { float t = v; ++t; gl_FragColor.x = dFdx(t); }`,
    layout,
    { varyings: [{ v: new Float32Array([0.5]), ddx: new Float32Array([2.0]), ddy: new Float32Array([3.0]) }] },
  );
  check(
    r.ctx.out.color[0][0] === 2.0,
    `'++t; dFdx(t)' == v.ddx (got ${r.ctx.out.color[0][0]})`,
  );
}

/* ------------------------------------------------------------------ */
/* 8. inout param (ES 3.00): write-back carries the param's triple     */
/* ------------------------------------------------------------------ */

{
  const layout = baseLayout(300, {
    varyings: new Map([['v', vg(0, 0, 1, 1)]]),
    outputLocations: new Map([['oColor', 0]]),
  });
  const r = runDual(
    `#version 300 es
     precision mediump float;
     in float v;
     out vec4 oColor;
     void bump(inout float x) { x += v; }
     void main() { float t = v; bump(t); oColor.x = dFdx(t); }`,
    layout,
    { varyings: [{ v: new Float32Array([0.5]), ddx: new Float32Array([2.0]), ddy: new Float32Array([3.0]) }] },
  );
  check(
    r.ctx.out.color[0][0] === 4.0,
    `inout param write-back: 'bump(t); dFdx(t)' == 2*v.ddx (got ${r.ctx.out.color[0][0]})`,
  );
}

/* ------------------------------------------------------------------ */
/* 9. out param (ES 3.00): write-back carries the param's triple       */
/* ------------------------------------------------------------------ */

{
  const layout = baseLayout(300, {
    varyings: new Map([['v', vg(0, 0, 1, 1)]]),
    outputLocations: new Map([['oColor', 0]]),
  });
  const r = runDual(
    `#version 300 es
     precision mediump float;
     in float v;
     out vec4 oColor;
     void setv(out float x) { x = v; }
     void main() { float t = 0.0; setv(t); oColor.x = dFdx(t); }`,
    layout,
    { varyings: [{ v: new Float32Array([0.5]), ddx: new Float32Array([2.0]), ddy: new Float32Array([3.0]) }] },
  );
  check(
    r.ctx.out.color[0][0] === 2.0,
    `out param write-back: 'setv(t); dFdx(t)' == v.ddx (got ${r.ctx.out.color[0][0]})`,
  );
}

/* ------------------------------------------------------------------ */
/* 10. Local ARRAY: 3-plane scratch, dynamic index                     */
/* ------------------------------------------------------------------ */

{
  const layout = baseLayout(100, {
    uniformSlots: new Map([['u_i', slot('int', 0)]]),
    varyings: new Map([['v', vg(0, 0, 1, 1)]]),
  });
  const src = `#extension GL_OES_standard_derivatives : enable
     precision mediump float;
     uniform int u_i;
     varying float v;
     void main() { float a[2]; a[0] = v; a[1] = v; gl_FragColor.x = dFdx(a[u_i]); }`;
  for (const i of [0, 1]) {
    const r = runDual(src, layout, {
      intUniforms: Int32Array.from([i]),
      varyings: [{ v: new Float32Array([0.5]), ddx: new Float32Array([2.0]), ddy: new Float32Array([3.0]) }],
    });
    check(
      r.ctx.out.color[0][0] === 2.0,
      `local array a[${i}] with dynamic index: dFdx(a[u_i]) == v.ddx (got ${r.ctx.out.color[0][0]})`,
    );
    if (i === 0) {
      check(
        r.scratchSize === 6,
        `dual float array scratch is 3 planes × 2 elements (got scratchSize ${r.scratchSize})`,
      );
    }
  }
}

/* ------------------------------------------------------------------ */
/* 11. fwidth == |ddx| + |ddy|                                         */
/* ------------------------------------------------------------------ */

{
  const layout = baseLayout(100, { varyings: new Map([['v', vg(0, 0, 1, 1)]]) });
  const r = runDual(
    `#extension GL_OES_standard_derivatives : enable
     precision mediump float;
     varying float v;
     void main() { gl_FragColor.x = fwidth(v); }`,
    layout,
    { varyings: [{ v: new Float32Array([0.5]), ddx: new Float32Array([2.0]), ddy: new Float32Array([-3.0]) }] },
  );
  check(
    r.ctx.out.color[0][0] === 5.0,
    `fwidth(v) == |ddx| + |ddy| (got ${r.ctx.out.color[0][0]})`,
  );
}

/* ------------------------------------------------------------------ */
/* 12. vec2 local via member writes (no ctor — C5a2 scope)             */
/* ------------------------------------------------------------------ */

{
  const layout = baseLayout(100, { varyings: new Map([['v', vg(0, 0, 1, 1)]]) });
  const r = runDual(
    `#extension GL_OES_standard_derivatives : enable
     precision mediump float;
     varying float v;
     void main() { vec2 p; p.x = v; p.y = v; gl_FragColor.x = dFdx(p.x); }`,
    layout,
    { varyings: [{ v: new Float32Array([0.5]), ddx: new Float32Array([2.0]), ddy: new Float32Array([3.0]) }] },
  );
  check(
    r.ctx.out.color[0][0] === 2.0,
    `vec2 local member writes: dFdx(p.x) == v.ddx (got ${r.ctx.out.color[0][0]})`,
  );
}

/* ------------------------------------------------------------------ */
/* C5a2: ARITHMETIC dual templates (+ - * / % , unary minus)           */
/* ------------------------------------------------------------------ */

/* 13. Product rule with a uniform: dFdx(x * u) == u.ddx (x ddx=2, u=3 → 6) */
{
  const layout = baseLayout(100, {
    uniformSlots: new Map([['u', slot('float', 0)]]),
    varyings: new Map([['x', vg(0, 0, 1, 1)]]),
  });
  const r = runDual(
    `#extension GL_OES_standard_derivatives : enable
     precision mediump float;
     uniform float u;
     varying float x;
     void main() { gl_FragColor.x = dFdx(x * u); gl_FragColor.y = dFdx(u * x); }`,
    layout,
    { uniforms: new Float32Array([3.0]), varyings: [{ v: new Float32Array([0.5]), ddx: new Float32Array([2.0]), ddy: new Float32Array([0]) }] },
  );
  const c = r.ctx.out.color[0];
  check(
    c[0] === 6 && c[1] === 6,
    `product rule: dFdx(x*u) == u.ddx == 6 (got [${Array.from(c).join(', ')}])`,
  );
}

/* 14. Product rule, BOTH factors varying: dFdx(x*y) == adx*bv + av*bdx */
{
  const layout = baseLayout(100, {
    varyings: new Map([['x', vg(0, 0, 1, 1)], ['y', vg(1, 1, 1, 1)]]),
  });
  const r = runDual(
    `#extension GL_OES_standard_derivatives : enable
     precision mediump float;
     varying float x; varying float y;
     void main() { gl_FragColor.x = dFdx(x * y); }`,
    layout,
    { varyings: [
      { v: new Float32Array([0.5]), ddx: new Float32Array([2.0]), ddy: new Float32Array([0]) },
      { v: new Float32Array([0.5]), ddx: new Float32Array([4.0]), ddy: new Float32Array([0]) },
    ] },
  );
  check(
    r.ctx.out.color[0][0] === 3,
    `product rule both varying: dFdx(x*y) == adx*bv + av*bdx == 3 (got ${r.ctx.out.color[0][0]})`,
  );
}

/* 15. Quotient rule: dFdx(x/u) == adx/u (x v=8 ddx=2, u=4 → 0.5) */
{
  const layout = baseLayout(100, {
    uniformSlots: new Map([['u', slot('float', 0)]]),
    varyings: new Map([['x', vg(0, 0, 1, 1)]]),
  });
  const r = runDual(
    `#extension GL_OES_standard_derivatives : enable
     precision mediump float;
     uniform float u;
     varying float x;
     void main() { gl_FragColor.x = dFdx(x / u); gl_FragColor.y = dFdx(u / x); }`,
    layout,
    { uniforms: new Float32Array([4.0]), varyings: [{ v: new Float32Array([8.0]), ddx: new Float32Array([2.0]), ddy: new Float32Array([0]) }] },
  );
  const c = r.ctx.out.color[0];
  check(
    c[0] === 0.5 && c[1] === -0.125,
    `quotient rule: dFdx(x/u) == 0.5, dFdx(u/x) == (du*av - u*adx)/av^2 == -0.125 (got [${Array.from(c).join(', ')}])`,
  );
}

/* 16. Subtraction is linear; unary minus negates the triple */
{
  const layout = baseLayout(100, {
    varyings: new Map([['x', vg(0, 0, 1, 1)], ['y', vg(1, 1, 1, 1)]]),
  });
  const r = runDual(
    `#extension GL_OES_standard_derivatives : enable
     precision mediump float;
     varying float x; varying float y;
     void main() { gl_FragColor.x = dFdx(x - y); gl_FragColor.y = dFdx(-x); }`,
    layout,
    { varyings: [
      { v: new Float32Array([0.5]), ddx: new Float32Array([2.0]), ddy: new Float32Array([0]) },
      { v: new Float32Array([0.5]), ddx: new Float32Array([4.0]), ddy: new Float32Array([0]) },
    ] },
  );
  const c = r.ctx.out.color[0];
  check(
    c[0] === -2 && c[1] === -2,
    `linear + unary minus: dFdx(x-y) == -2, dFdx(-x) == -2 (got [${Array.from(c).join(', ')}])`,
  );
}

/* 17. Compound *= and /= update all three planes (product/quotient rule) */
{
  const layout = baseLayout(100, {
    uniformSlots: new Map([['u', slot('float', 0)]]),
    varyings: new Map([['x', vg(0, 0, 1, 1)]]),
  });
  const r = runDual(
    `#extension GL_OES_standard_derivatives : enable
     precision mediump float;
     uniform float u;
     varying float x;
     void main() { float t = x; t *= u; gl_FragColor.x = dFdx(t); t = x; t /= u; gl_FragColor.y = dFdx(t); }`,
    layout,
    { uniforms: new Float32Array([3.0]), varyings: [{ v: new Float32Array([8.0]), ddx: new Float32Array([2.0]), ddy: new Float32Array([0]) }] },
  );
  const c = r.ctx.out.color[0];
  check(
    c[0] === 6 && Math.abs(c[1] - 2 / 3) < 1e-6,
    `compound: 't *= u; dFdx(t)' == u.ddx == 6; 't /= u; dFdx(t)' == ddx/u == 2/3 (got [${Array.from(c).join(', ')}])`,
  );
}

/* 18. Float '%' operator dual formula (direct emitExpr — the compiler
 *      rejects float '%' per GLSL, so the permissive lowering is checked
 *      here): dx = adx − floor(av/bv)·bdx. */
{
  const e = new CodegenEnv('FRAGMENT', baseLayout(100));
  e.dual = true;
  e.declareLocal('x', { kind: 'scalar', base: 'float' });
  const lit2: Expr = { kind: 'literal', loc: { line: 1, column: 0 }, value: 2.0, literalType: 'float', resolvedType: { kind: 'scalar', base: 'float' }, constValue: 2.0 };
  const idX: Expr = { kind: 'identifier', loc: { line: 1, column: 0 }, name: 'x', resolvedType: { kind: 'scalar', base: 'float' } };
  const v = emitExpr({ kind: 'binary', loc: { line: 1, column: 0 }, op: '%', left: idX, right: lit2, resolvedType: { kind: 'scalar', base: 'float' } }, e)[0];
  const t = e.temps.length ? `var ${e.temps.join(', ')}; ` : '';
  const res = new Function('x', 'x_dx', `${t}return ${v.dx};`)(5.5, 2) as number;
  check(
    res === 2,
    `float '%': dx == adx - floor(av/bv)*bdx == 2 (got ${res}; dx string '${v.dx}')`,
  );
}

/* 19. Compound %= (dualWrite '%' template): t = x; t %= 2.0 → dFdx(t) == ddx */
{
  const layout = baseLayout(100, { varyings: new Map([['x', vg(0, 0, 1, 1)]]) });
  const r = runDual(
    `#extension GL_OES_standard_derivatives : enable
     precision mediump float;
     varying float x;
     void main() { float t = x; t %= 2.0; gl_FragColor.x = dFdx(t); gl_FragColor.y = t; }`,
    layout,
    { varyings: [{ v: new Float32Array([5.5]), ddx: new Float32Array([2.0]), ddy: new Float32Array([0]) }] },
  );
  const c = r.ctx.out.color[0];
  check(
    c[0] === 2 && c[1] === 1.5,
    `'t %= 2.0; dFdx(t)' == ddx == 2, t == 5.5 % 2 == 1.5 (got [${Array.from(c).join(', ')}])`,
  );
}

/* ------------------------------------------------------------------ */
/* C5a2: ternary triples, conversions, assignment duals, ctors, loops   */
/* ------------------------------------------------------------------ */

/* 20. Ternary selects each plane: dFdx(gl_FrontFacing ? a : b) */
{
  const layout = baseLayout(100, {
    varyings: new Map([['a', vg(0, 0, 1, 1)], ['b', vg(1, 1, 1, 1)]]),
  });
  const src = `#extension GL_OES_standard_derivatives : enable
     precision mediump float;
     varying float a; varying float b;
     void main() { gl_FragColor.x = dFdx(gl_FrontFacing ? a : b); }`;
  const varyings = [
    { v: new Float32Array([1.0]), ddx: new Float32Array([2.0]), ddy: new Float32Array([0]) },
    { v: new Float32Array([3.0]), ddx: new Float32Array([5.0]), ddy: new Float32Array([0]) },
  ];
  const rT = runDual(src, layout, { varyings });
  const rF = runDual(src, layout, { frontFacing: false, varyings });
  check(
    rT.ctx.out.color[0][0] === 2 && rF.ctx.out.color[0][0] === 5,
    `ternary: dFdx(front ? a : b) == a.ddx == 2 / b.ddx == 5 (got ${rT.ctx.out.color[0][0]} / ${rF.ctx.out.color[0][0]})`,
  );
}

/* 21. int→float conversion is a constant dual: dFdx(float(ui)) == 0 */
{
  const layout = baseLayout(100, {
    uniformSlots: new Map([['ui', slot('int', 0)]]),
  });
  const r = runDual(
    `#extension GL_OES_standard_derivatives : enable
     precision mediump float;
     uniform int ui;
     void main() { gl_FragColor.x = dFdx(float(ui)); }`,
    layout,
    { intUniforms: Int32Array.from([7]) },
  );
  check(
    r.ctx.out.color[0][0] === 0,
    `int→float: dFdx(float(ui)) == 0 (got ${r.ctx.out.color[0][0]})`,
  );
}

/* 22. Constructors: vec4 concat, vec3 concat, mat2 concat, broadcast */
{
  const layout = baseLayout(100, {
    varyings: new Map([
      ['x', vg(0, 0, 1, 1)], ['y', vg(1, 1, 1, 1)], ['z', vg(2, 2, 1, 1)], ['w', vg(3, 3, 1, 1)],
    ]),
  });
  const varyings = [
    { v: new Float32Array([1.0]), ddx: new Float32Array([2.0]), ddy: new Float32Array([0]) },
    { v: new Float32Array([3.0]), ddx: new Float32Array([4.0]), ddy: new Float32Array([0]) },
    { v: new Float32Array([5.0]), ddx: new Float32Array([6.0]), ddy: new Float32Array([0]) },
    { v: new Float32Array([7.0]), ddx: new Float32Array([8.0]), ddy: new Float32Array([0]) },
  ];
  const r = runDual(
    `#extension GL_OES_standard_derivatives : enable
     precision mediump float;
     varying float x; varying float y; varying float z; varying float w;
     void main() {
       vec4 q4 = vec4(x, y, z, w);
       vec3 q3 = vec3(x, y, z);
       mat2 m = mat2(x, y, z, w);
       vec4 b = vec4(w);
       gl_FragColor.x = dFdx(q4.x);
       gl_FragColor.y = dFdx(q4.z);
       gl_FragColor.z = dFdx(q3.y);
       gl_FragColor.w = dFdx(m[0][0]);
     }`,
    layout,
    { varyings },
  );
  const c = r.ctx.out.color[0];
  check(
    c[0] === 2 && c[1] === 6 && c[2] === 4 && c[3] === 2,
    `ctors: vec4(x,y,z,w) → dFdx(q4.x)==2, dFdx(q4.z)==6; vec3(x,y,z) → dFdx(q3.y)==4; mat2(x,y,z,w) → dFdx(m[0][0])==2 (got [${Array.from(c).join(', ')}])`,
  );
  // Broadcast: every component of vec4(v) carries v's dual.
  const r2 = runDual(
    `#extension GL_OES_standard_derivatives : enable
     precision mediump float;
     varying float v;
     void main() { vec4 q = vec4(v); gl_FragColor.x = dFdx(q.x); gl_FragColor.y = dFdx(q.w); }`,
    baseLayout(100, { varyings: new Map([['v', vg(0, 0, 1, 1)]]) }),
    { varyings: [{ v: new Float32Array([1.0]), ddx: new Float32Array([2.0]), ddy: new Float32Array([0]) }] },
  );
  const c2 = r2.ctx.out.color[0];
  check(
    c2[0] === 2 && c2[1] === 2,
    `broadcast ctor: vec4(v) — dFdx(q.x) == dFdx(q.w) == v.ddx == 2 (got [${Array.from(c2).join(', ')}])`,
  );
}

/* 23. Assignment EXPRESSION duals: dFdx(t = v) == v.ddx (and the write ran) */
{
  const layout = baseLayout(100, { varyings: new Map([['v', vg(0, 0, 1, 1)]]) });
  const r = runDual(
    `#extension GL_OES_standard_derivatives : enable
     precision mediump float;
     varying float v;
     void main() { float t; gl_FragColor.x = dFdx(t = v); gl_FragColor.y = t; }`,
    layout,
    { varyings: [{ v: new Float32Array([0.5]), ddx: new Float32Array([2.0]), ddy: new Float32Array([3.0]) }] },
  );
  const c = r.ctx.out.color[0];
  check(
    c[0] === 2 && c[1] === 0.5,
    `assignment expression: dFdx(t = v) == v.ddx == 2, t == 0.5 (got [${Array.from(c).join(', ')}])`,
  );
  check(
    r.body.includes('(t = ctx.varyings[0].v[0], t_dx = ctx.varyings[0].ddx[0]'),
    `assignment expression emits the triple composite before the value (body: ${r.body.slice(0, 120)}...)`,
  );
}

/* 24. Int/float mix: the int literal carries no duals — dFdx(v + 1) == v.ddx */
{
  const layout = baseLayout(100, { varyings: new Map([['v', vg(0, 0, 1, 1)]]) });
  const r = runDual(
    `#extension GL_OES_standard_derivatives : enable
     precision mediump float;
     varying float v;
     void main() { float t = v + 1; gl_FragColor.x = dFdx(t); }`,
    layout,
    { varyings: [{ v: new Float32Array([0.5]), ddx: new Float32Array([2.0]), ddy: new Float32Array([3.0]) }] },
  );
  check(
    r.ctx.out.color[0][0] === 2,
    `int/float mix: dFdx(v + 1) == v.ddx == 2 (got ${r.ctx.out.color[0][0]})`,
  );
}

/* 25. Loop accumulator: dFdx of a sum over iterations == 3*v.ddx */
{
  const layout = baseLayout(100, { varyings: new Map([['v', vg(0, 0, 1, 1)]]) });
  const r = runDual(
    `#extension GL_OES_standard_derivatives : enable
     precision mediump float;
     varying float v;
     void main() { float s = 0.0; for (int i = 0; i < 3; i++) { s += v; } gl_FragColor.x = dFdx(s); }`,
    layout,
    { varyings: [{ v: new Float32Array([0.5]), ddx: new Float32Array([2.0]), ddy: new Float32Array([0]) }] },
  );
  check(
    r.ctx.out.color[0][0] === 6,
    `loop accumulator: dFdx(s) == 3*v.ddx == 6 (got ${r.ctx.out.color[0][0]})`,
  );
}

/* ------------------------------------------------------------------ */
/* Report + exit                                                       */
/* ------------------------------------------------------------------ */

console.log(`codegen-dual selftest: ${checks} checks, ${failures} failure(s)`);
if (failures > 0) {
  console.error(`${failures} check(s) FAILED`);
  process.exit(1);
}
console.log('OK');
