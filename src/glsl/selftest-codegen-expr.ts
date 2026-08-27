/**
 * selftest-codegen-expr.ts — direct unit checks for the expression-codegen
 * modules (codegen/expressions.ts core + codegen/expr-ctor.ts constructors +
 * codegen/expr-builtins.ts builtin lowering, all NON-dual).
 *
 * Run: npx tsx src/glsl/selftest-codegen-expr.ts
 *
 * The selftest builds AST nodes by hand (semantics-independent: every node
 * carries `resolvedType`, literals carry `constValue`) and asserts EXACT
 * emitted JS strings plus a handful of mini-evals via `new Function`.
 * It also smoke-tests `compileShader` (implemented) by walking the annotated
 * AST of two tiny shaders and emitting their main assignment.
 *
 * Prints "OK" and exits 0 on success.
 */
import type { Expr, LiteralExpr, TranslationUnit, ExternalDecl, FunctionDefinition } from './ast.js';
import type { GLSLType } from './types.js';
import { compileShader } from './compiler.js';
import { CodegenEnv } from './codegen/env.js';
import { emitExpr, emitLValue } from './codegen/expressions.js';
import { emitStatements } from './codegen/statements.js';
import type { Value } from './codegen/index.js';
import type { CodegenLayout } from './codegen/index.js';
import { R } from './codegen/runtime.js';

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
/* Type factories                                                      */
/* ------------------------------------------------------------------ */

const fT = (): GLSLType => ({ kind: 'scalar', base: 'float' });
const iT = (): GLSLType => ({ kind: 'scalar', base: 'int' });
const uT = (): GLSLType => ({ kind: 'scalar', base: 'uint' });
const bT = (): GLSLType => ({ kind: 'scalar', base: 'bool' });
const vT = (base: 'float' | 'int' | 'uint' | 'bool', size: 2 | 3 | 4): GLSLType => ({ kind: 'vector', base, size });
const mT = (cols: 2 | 3 | 4, rows: 2 | 3 | 4): GLSLType => ({ kind: 'matrix', cols, rows });
const sT = (sampler: string): GLSLType => ({ kind: 'sampler', sampler } as GLSLType);

/* ------------------------------------------------------------------ */
/* Hand-built AST helpers (semantics has already annotated these)       */
/* ------------------------------------------------------------------ */

const loc = { line: 1, column: 0 };

function lit(v: number | boolean, t: GLSLType): LiteralExpr {
  return {
    kind: 'literal',
    loc,
    value: v,
    literalType: (t.kind === 'scalar' ? t.base : 'float') as 'int' | 'uint' | 'float' | 'bool',
    resolvedType: t,
    constValue: v,
  };
}

function ident(name: string, t: GLSLType): Expr {
  return { kind: 'identifier', loc, name, resolvedType: t };
}

function bin(op: string, l: Expr, r: Expr, t: GLSLType): Expr {
  return { kind: 'binary', loc, op: op as never, left: l, right: r, resolvedType: t };
}

function call(name: string, args: Expr[], t: GLSLType): Expr {
  return {
    kind: 'call',
    loc,
    callee: { kind: 'identifier', loc, name, resolvedType: { kind: 'void' } },
    args,
    resolvedType: t,
  };
}

function mem(obj: Expr, name: string, t: GLSLType): Expr {
  return { kind: 'member', loc, object: obj, name, resolvedType: t };
}

function tern(cond: Expr, a: Expr, b: Expr, t: GLSLType): Expr {
  return { kind: 'ternary', loc, cond, whenTrue: a, whenFalse: b, resolvedType: t };
}

/* ------------------------------------------------------------------ */
/* Layout + env factories                                              */
/* ------------------------------------------------------------------ */

function baseLayout(version: 100 | 300, extraUniforms?: [string, { store: 'float' | 'int'; slot: number; stride: number }][]): CodegenLayout {
  const uniformSlots = new Map<string, { store: 'float' | 'int'; slot: number; stride: number }>([
    ['u_color', { store: 'float', slot: 0, stride: 0 }],
    ['u_i', { store: 'int', slot: 4, stride: 0 }],
    ...(extraUniforms ?? []),
  ]);
  return {
    version,
    uniformSlots,
    blocks: new Map(),
    blockIndices: new Map(),
    varyings: new Map(),
    attribLocations: new Map([['a_pos', 0]]),
    outputLocations: new Map([['gl_FragColor', 0]]),
    uses: {
      pointSize: false,
      fragCoord: false,
      frontFacing: false,
      pointCoord: false,
      fragDepth: false,
      vertexId: false,
      instanceId: false,
      derivatives: false,
    },
  };
}

function env(stage: 'VERTEX' | 'FRAGMENT', version: 100 | 300, extraUniforms?: Parameters<typeof baseLayout>[1]): CodegenEnv {
  return new CodegenEnv(stage, baseLayout(version, extraUniforms));
}

/** Eval ONE Value (its pres run first; env temps declared). No locals. */
function evalV(v: Value, e: CodegenEnv): number {
  const decl = e.temps.length ? `var ${e.temps.join(', ')}; ` : '';
  const pres = v.pre && v.pre.length ? v.pre.join('; ') + '; ' : '';
  return new Function('ctx', 'R', `${decl}${pres}return ${v.v};`)({}, R) as number;
}

/** Eval ONE Value with named local params (bare JS names from declareLocal). */
function evalParams(v: Value, e: CodegenEnv, params: string[], args: number[]): number {
  const decl = e.temps.length ? `var ${e.temps.join(', ')}; ` : '';
  const pres = v.pre && v.pre.length ? v.pre.join('; ') + '; ' : '';
  return new Function('ctx', 'R', ...params, `${decl}${pres}return ${v.v};`)({}, R, ...args) as number;
}

/* ------------------------------------------------------------------ */
/* 1. Constructors (expr-ctor.ts)                                      */
/* ------------------------------------------------------------------ */

{
  const e = env('VERTEX', 100);
  const v = emitExpr(lit(4294967295, uT()), e)[0];
  check(v.v === '4294967295', `uint literal 4294967295 emits '4294967295' (got '${v.v}')`);
}

{
  const e = env('VERTEX', 100);
  const v = emitExpr(bin('+', lit(1, iT()), lit(0.5, fT()), fT()), e)[0];
  check(v.v === '(1 + 0.5)', `int+float binary → '(1 + 0.5)' (got '${v.v}')`);
}

{
  const e = env('VERTEX', 100);
  e.declareLocal('a', vT('float', 3));
  e.declareLocal('b', vT('float', 3));
  const vals = emitExpr(bin('+', ident('a', vT('float', 3)), ident('b', vT('float', 3)), vT('float', 3)), e);
  check(vals.length === 3, 'vec3 add → 3 components');
  check(vals[1].v === '(a__1 + b__1)', `vec3 add comp 1 → '(a__1 + b__1)' (got '${vals[1].v}')`);
}

{
  const e = env('VERTEX', 100);
  e.declareLocal('m', mT(2, 2));
  e.declareLocal('v', vT('float', 2));
  const vals = emitExpr(bin('*', ident('m', mT(2, 2)), ident('v', vT('float', 2)), vT('float', 2)), e);
  check(vals[0].v === '((m__0 * (v__0)) + (m__2 * (v__1)))', `mat2*vec2 comp0 (got '${vals[0].v}')`);
  check(vals[1].v === '((m__1 * (v__0)) + (m__3 * (v__1)))', `mat2*vec2 comp1 (got '${vals[1].v}')`);
}

{
  const e = env('VERTEX', 100);
  e.declareLocal('v', vT('float', 4));
  const vals = emitExpr(mem(ident('v', vT('float', 4)), 'yz', vT('float', 2)), e);
  check(vals[0].v === 'v__1' && vals[1].v === 'v__2', `swizzle v.yz → v__1, v__2 (got ${vals.map((x) => x.v).join(', ')})`);
}

{
  const e = env('VERTEX', 100);
  const v = emitExpr(tern(lit(true, bT()), lit(1.0, fT()), lit(2.0, fT()), fT()), e)[0];
  check(v.v === '(true ? (1.0) : (2.0))', `ternary → '(true ? (1.0) : (2.0))' (got '${v.v}')`);
}

{
  const e = env('VERTEX', 100);
  const vals = emitExpr(call('vec4', [call('vec2', [lit(1.0, fT()), lit(2.0, fT())], vT('float', 2))], vT('float', 4)), e);
  check(
    vals.map((x) => x.v).join(',') === '1.0,2.0,0.0,1.0',
    `vec4(vec2) pads 0s + final 1 (got ${vals.map((x) => x.v).join(',')})`,
  );
}

{
  const e = env('VERTEX', 100);
  e.declareLocal('m', mT(2, 2));
  const vals = emitExpr(call('vec4', [ident('m', mT(2, 2))], vT('float', 4)), e);
  check(
    vals.map((x) => x.v).join(',') === 'm__0,m__1,m__2,m__3',
    `vec4(mat2) flattens column-major per GLSL ES 1.00 §5.4.2 (got ${vals.map((x) => x.v).join(',')})`,
  );
}

{
  const e = env('VERTEX', 100);
  const vals = emitExpr(call('mat3', [lit(5.0, fT())], mT(3, 3)), e);
  check(
    vals[0].v === '5.0' && vals[1].v === '0.0' && vals[4].v === '5.0' && vals[8].v === '5.0',
    `mat3(scalar) → diagonal (got ${vals.map((x) => x.v).join(',')})`,
  );
}

{
  const e = env('VERTEX', 100);
  const v = emitExpr(call('int', [lit(2.7, fT())], iT()), e)[0];
  check(v.v === '((2.7) | 0)', `int(2.7) → '((2.7) | 0)' (got '${v.v}')`);
  check(evalV(v, e) === 2, `int(2.7) evaluates to 2`);
}

{
  const e = env('VERTEX', 100);
  const v = emitExpr(call('uint', [lit(-1, iT())], uT()), e)[0];
  check(v.v === '((-1) >>> 0)', `uint(-1) → '((-1) >>> 0)' (got '${v.v}')`);
  check(evalV(v, e) === 4294967295, `uint(-1) evaluates to 4294967295`);
}

{
  const e = env('VERTEX', 100);
  const v = emitExpr(call('bool', [lit(0.0, fT())], bT()), e)[0];
  check(v.v === '(0.0 !== 0.0)', `bool(0.0) → '(0.0 !== 0.0)' (got '${v.v}')`);
  check(!evalV(v, e), `bool(0.0) evaluates to false`);
}

{
  const e = env('VERTEX', 100);
  const v = emitExpr(call('float', [lit(true, bT())], fT()), e)[0];
  check(v.v === '(true ? 1.0 : 0.0)', `float(true) → '(true ? 1.0 : 0.0)' (got '${v.v}')`);
}

{
  const e = env('VERTEX', 100);
  const vals = emitExpr(call('vec2', [lit(3.0, fT())], vT('float', 2)), e);
  check(vals[0].v === '3.0' && vals[1].v === '3.0', `vec2(scalar) splats (got ${vals.map((x) => x.v).join(',')})`);
}

{
  const e = env('VERTEX', 100);
  const vals = emitExpr(
    call('ivec4', [lit(1, iT()), lit(2, iT()), lit(3, iT()), lit(4, iT())], vT('int', 4)),
    e,
  );
  check(vals.map((x) => x.v).join(',') === '1,2,3,4', `ivec4(1,2,3,4) (got ${vals.map((x) => x.v).join(',')})`);
}

{
  const e = env('VERTEX', 100);
  const vals = emitExpr(
    call('mat2', [lit(1.0, fT()), lit(2.0, fT()), lit(3.0, fT()), lit(4.0, fT())], mT(2, 2)),
    e,
  );
  check(
    vals.map((x) => x.v).join(',') === '1.0,2.0,3.0,4.0',
    `mat2(4 scalars) column-major (got ${vals.map((x) => x.v).join(',')})`,
  );
}

{
  const e = env('VERTEX', 100);
  e.declareLocal('m3', mT(3, 3));
  const vals = emitExpr(call('mat2', [ident('m3', mT(3, 3))], mT(2, 2)), e);
  check(
    vals.map((x) => x.v).join(',') === 'm3__0,m3__1,m3__3,m3__4',
    `mat2(mat3) overlapping copy (got ${vals.map((x) => x.v).join(',')})`,
  );
}

{
  const e = env('VERTEX', 100);
  const S: GLSLType = { kind: 'struct', name: 'S', members: [{ name: 'a', type: fT() }, { name: 'b', type: vT('float', 2) }] };
  e.structNames.add('S');
  const vals = emitExpr(
    call('S', [lit(1.0, fT()), call('vec2', [lit(2.0, fT()), lit(3.0, fT())], vT('float', 2))], S),
    e,
  );
  check(
    vals.map((x) => x.v).join(',') === '1.0,2.0,3.0',
    `struct ctor S(1.0, vec2(2.0, 3.0)) → 1.0,2.0,3.0 (got ${vals.map((x) => x.v).join(',')})`,
  );
}

/* ------------------------------------------------------------------ */
/* 2. Math builtins (expr-builtins.ts)                                 */
/* ------------------------------------------------------------------ */

{
  const e = env('VERTEX', 100);
  e.declareLocal('x', fT());
  const v = emitExpr(call('sin', [ident('x', fT())], fT()), e)[0];
  check(v.v === 'Math.sin(x)', `sin(x) → 'Math.sin(x)' (got '${v.v}')`);
}

{
  const e = env('VERTEX', 100);
  e.declareLocal('x', fT());
  e.declareLocal('y', fT());
  const v = emitExpr(call('atan', [ident('y', fT()), ident('x', fT())], fT()), e)[0];
  check(v.v === 'Math.atan2(y, x)', `atan(y,x) → 'Math.atan2(y, x)' (got '${v.v}')`);
}

{
  const e = env('VERTEX', 100);
  const v = emitExpr(call('mod', [lit(5.5, fT()), lit(2.0, fT())], fT()), e)[0];
  check(
    v.v === '((5.5) - (2.0) * Math.floor((5.5) / (2.0)))',
    `mod(5.5, 2.0) exact form (got '${v.v}')`,
  );
  check(evalV(v, e) === 1.5, `mod(5.5, 2.0) === 1.5`);
}

{
  // round is an ES 3.00 builtin (absent from the 1.00 table).
  const e = env('VERTEX', 300);
  const v = emitExpr(call('round', [lit(2.5, fT())], fT()), e)[0];
  check(v.v === 'Math.round(2.5)', `round(2.5) → 'Math.round(2.5)' (got '${v.v}')`);
  check(evalV(v, e) === 3, `round(2.5) === 3`);
}

{
  const e = env('VERTEX', 300);
  const v = emitExpr(call('roundEven', [lit(2.5, fT())], fT()), e)[0];
  check(v.v.includes('Math.round(2.5)') && v.v.includes('t0 & 1'), `roundEven(2.5) shape (got '${v.v}')`);
  check(evalV(v, e) === 2, `roundEven(2.5) === 2 (got ${evalV(v, e)})`);
}
{
  const e = env('VERTEX', 300);
  const v = emitExpr(call('roundEven', [lit(3.5, fT())], fT()), e)[0];
  check(evalV(v, e) === 4, `roundEven(3.5) === 4 (got ${evalV(v, e)})`);
}

{
  const e = env('VERTEX', 100);
  const v = emitExpr(bin('+', lit(4294967295, uT()), lit(1, uT()), uT()), e)[0];
  check(v.v === '((4294967295) + (1)) >>> 0', `uint add wraps (got '${v.v}')`);
  check(evalV(v, e) === 0, `(0xFFFFFFFF + 1) >>> 0 === 0`);
}

{
  const e = env('VERTEX', 100);
  e.declareLocal('a', iT());
  e.declareLocal('b', iT());
  const v = emitExpr(bin('/', ident('a', iT()), ident('b', iT()), iT()), e)[0];
  check(v.v === '((a) / (b)) | 0', `int div → '((a) / (b)) | 0' (got '${v.v}')`);
  check(evalParams(v, e, ['a', 'b'], [-7, 2]) === -3, `int (-7)/2 === -3`);
}

{
  const e = env('VERTEX', 100);
  e.declareLocal('a', uT());
  e.declareLocal('b', uT());
  const v = emitExpr(bin('*', ident('a', uT()), ident('b', uT()), uT()), e)[0];
  check(v.v === '(Math.imul(a, b)) >>> 0', `uint mul → '(Math.imul(a, b)) >>> 0' (got '${v.v}')`);
}

{
  const e = env('VERTEX', 300);
  e.declareLocal('a', fT());
  e.declareLocal('b', fT());
  const v = emitExpr(
    call('packHalf2x16', [call('vec2', [ident('a', fT()), ident('b', fT())], vT('float', 2))], uT()),
    e,
  )[0];
  check(v.v === '((R.packHalf2x16(a, b)) >>> 0)', `packHalf2x16 → '((R.packHalf2x16(a, b)) >>> 0)' (got '${v.v}')`);
  check(evalParams(v, e, ['a', 'b'], [1.0, -2.0]) === 0xc0003c00, `packHalf2x16(1, -2) === 0xC0003C00`);
}

{
  const e = env('VERTEX', 100);
  e.declareLocal('m', mT(2, 2));
  const vals = emitExpr(call('inverse', [ident('m', mT(2, 2))], mT(2, 2)), e);
  check(
    vals[0].pre !== undefined && vals[0].pre[0].includes('R.inv2('),
    `inverse(mat2) calls R.inv2 (got '${vals[0].pre?.[0]}')`,
  );
  check(vals[3].v === 'ctx.scratch[0 + 3]', `inverse(mat2) reads ctx.scratch (got '${vals[3].v}')`);
  check(e.scratchSize === 4, `inverse(mat2) allocates 4 scratch floats`);
}

{
  const e = env('VERTEX', 100);
  e.declareLocal('m4', mT(4, 4));
  const vals = emitExpr(call('inverse', [ident('m4', mT(4, 4))], mT(4, 4)), e);
  check(
    vals[0].pre !== undefined && vals[0].pre[0].includes('R.inv4(') && vals[0].pre[0].split(',').length === 18,
    `inverse(mat4) calls R.inv4 with 16 args (got '${vals[0].pre?.[0].slice(0, 60)}...')`,
  );
}

{
  const e = env('VERTEX', 100);
  const v = emitExpr(call('step', [lit(2.0, fT()), lit(1.0, fT())], fT()), e)[0];
  check(v.v === '((1.0) < (2.0) ? 0.0 : 1.0)', `step(2.0, 1.0) exact form (got '${v.v}')`);
  check(evalV(v, e) === 0, `step(2.0, 1.0) === 0.0 (x < edge)`);
  check(evalV(emitExpr(call('step', [lit(2.0, fT()), lit(3.0, fT())], fT()), e)[0], e) === 1, `step(2.0, 3.0) === 1.0`);
}

{
  const e = env('VERTEX', 100);
  const v = emitExpr(call('smoothstep', [lit(0.0, fT()), lit(1.0, fT()), lit(0.5, fT())], fT()), e)[0];
  check(evalV(v, e) === 0.5, `smoothstep(0, 1, 0.5) === 0.5`);
}

{
  const e = env('VERTEX', 100);
  const v = emitExpr(call('clamp', [lit(5.0, fT()), lit(0.0, fT()), lit(1.0, fT())], fT()), e)[0];
  check(evalV(v, e) === 1, `clamp(5, 0, 1) === 1`);
}

{
  const e = env('VERTEX', 100);
  const v = emitExpr(call('mix', [lit(0.0, fT()), lit(2.0, fT()), lit(0.25, fT())], fT()), e)[0];
  check(evalV(v, e) === 0.5, `mix(0, 2, 0.25) === 0.5`);
}

{
  const e = env('VERTEX', 100);
  const m = call('mat2', [lit(1.0, fT()), lit(2.0, fT()), lit(3.0, fT()), lit(4.0, fT())], mT(2, 2));
  const vv = call('vec2', [lit(5.0, fT()), lit(6.0, fT())], vT('float', 2));
  const vals = emitExpr(bin('*', m, vv, vT('float', 2)), e);
  check(evalV(vals[0], e) === 23 && evalV(vals[1], e) === 34, `mat2(1,2,3,4) * vec2(5,6) === (23, 34)`);
}

{
  const e = env('VERTEX', 100);
  const a = call('vec2', [lit(3.0, fT()), lit(4.0, fT())], vT('float', 2));
  const v = emitExpr(call('dot', [a, a], fT()), e)[0];
  check(v.v === '((3.0 * 3.0) + (4.0 * 4.0))', `dot(vec2(3,4), vec2(3,4)) exact (got '${v.v}')`);
  check(evalV(v, e) === 25, `dot((3,4),(3,4)) === 25`);
}

{
  const e = env('VERTEX', 100);
  const v = emitExpr(
    call('length', [call('vec2', [lit(3.0, fT()), lit(4.0, fT())], vT('float', 2))], fT()),
    e,
  )[0];
  check(v.v === 'Math.sqrt((3.0 * 3.0) + (4.0 * 4.0))', `length(vec2(3,4)) exact (got '${v.v}')`);
  check(evalV(v, e) === 5, `length((3,4)) === 5`);
}

{
  const e = env('VERTEX', 100);
  const v = emitExpr(
    call('distance', [call('vec2', [lit(0.0, fT()), lit(0.0, fT())], vT('float', 2)), call('vec2', [lit(3.0, fT()), lit(4.0, fT())], vT('float', 2))], fT()),
    e,
  )[0];
  check(evalV(v, e) === 5, `distance((0,0),(3,4)) === 5`);
}

{
  const e = env('VERTEX', 100);
  const vals = emitExpr(
    call('normalize', [call('vec2', [lit(3.0, fT()), lit(4.0, fT())], vT('float', 2))], vT('float', 2)),
    e,
  );
  const npre = vals[0].pre ? vals[0].pre[0] : '';
  check(
    npre.includes('Math.sqrt(') && npre.includes('3.0 * 3.0'),
    `normalize pre computes the length (got '${npre}')`,
  );
  check(evalV(vals[0], e) === 0.6, `normalize((3,4)).x === 0.6`);
}

{
  const e = env('VERTEX', 100);
  e.declareLocal('m', mT(2, 2));
  e.declareLocal('n', mT(2, 2));
  const vals = emitExpr(call('matrixCompMult', [ident('m', mT(2, 2)), ident('n', mT(2, 2))], mT(2, 2)), e);
  check(vals[1].v === '(m__1 * n__1)', `matrixCompMult per-element (got '${vals[1].v}')`);
}

{
  const e = env('VERTEX', 100);
  e.declareLocal('m', mT(2, 2));
  const vals = emitExpr(call('transpose', [ident('m', mT(2, 2))], mT(2, 2)), e);
  check(vals[1].v === 'm__2' && vals[2].v === 'm__1', `transpose(mat2) remaps off-diagonal (got ${vals.map((x) => x.v).join(',')})`);
}

{
  const e = env('VERTEX', 100);
  e.declareLocal('m', mT(2, 2));
  const v = emitExpr(call('determinant', [ident('m', mT(2, 2))], fT()), e)[0];
  check(
    v.v === '((m__0) * (m__3) - (m__1) * (m__2))',
    `det2 exact form (got '${v.v}')`,
  );
  const v2 = emitExpr(
    call('determinant', [call('mat2', [lit(1.0, fT()), lit(2.0, fT()), lit(3.0, fT()), lit(4.0, fT())], mT(2, 2))], fT()),
    e,
  )[0];
  check(evalV(v2, e) === -2, `det(mat2(1,2,3,4)) === -2`);
}

{
  // modf is an ES 3.00 builtin (absent from the 1.00 table).
  const e = env('VERTEX', 300);
  e.declareLocal('o', fT());
  const v = emitExpr(call('modf', [lit(3.7, fT()), ident('o', fT())], fT()), e)[0];
  check(
    v.pre !== undefined && v.pre[0] === 't0 = Math.trunc(3.7)' && v.pre[1] === '(o = t0)',
    `modf pre truncates + writes the out param (got ${JSON.stringify(v.pre)})`,
  );
  check(v.v === '((3.7) - t0)', `modf result = fract (got '${v.v}')`);
  const decl = e.temps.length ? `var ${e.temps.join(', ')}; ` : '';
  const pres = v.pre!.join('; ') + '; ';
  const res = new Function('ctx', 'R', 'o', `${decl}${pres}return [((3.7) - t0), o];`)({}, R, 0) as [number, number];
  check(Math.abs(res[0] - 0.7) < 1e-9 && res[1] === 3, `modf(3.7, o): fract 0.7, o === 3 (got ${res.join(',')})`);
}

{
  const e = env('VERTEX', 300);
  e.declareLocal('a', uT());
  e.declareLocal('b', uT());
  e.declareLocal('c', uT());
  const v = emitExpr(call('uaddCarry', [ident('a', uT()), ident('b', uT()), ident('c', uT())], uT()), e)[0];
  const ctx2 = { intScratch: new Int32Array(8), scratch: new Float32Array(8) } as never;
  const decl = e.temps.length ? `var ${e.temps.join(', ')}; ` : '';
  const pres = v.pre!.join('; ') + '; ';
  const res = new Function('ctx', 'R', 'a', 'b', 'c', `${decl}${pres}return [(ctx.intScratch[0]) >>> 0, c];`)(ctx2, R, 0xffffffff, 1, 0) as [number, number];
  check(res[0] === 0 && res[1] === 1, `uaddCarry(0xFFFFFFFF, 1, c): sum 0, carry 1 (got ${res.join(',')})`);
}

{
  const e = env('VERTEX', 300);
  e.declareLocal('a', uT());
  e.declareLocal('b', uT());
  e.declareLocal('c', uT());
  e.declareLocal('d', uT());
  const v = emitExpr(
    call('umulExtended', [ident('a', uT()), ident('b', uT()), ident('c', uT()), ident('d', uT())], { kind: 'void' }),
    e,
  )[0];
  const ctx2 = { intScratch: new Int32Array(8), scratch: new Float32Array(8) } as never;
  const decl = e.temps.length ? `var ${e.temps.join(', ')}; ` : '';
  const res = new Function('ctx', 'R', 'a', 'b', 'c', 'd', `${decl}return ${v.v}, [c, d];`)(ctx2, R, 0xffffffff, 2, 0, 0) as [number, number];
  check(res[0] === 1 && res[1] === 0xfffffffe, `umulExtended(0xFFFFFFFF, 2): hi 1, lo 0xFFFFFFFE (got ${res.map((x) => '0x' + x.toString(16)).join(',')})`);
}

/* ------------------------------------------------------------------ */
/* 3. Texture calls (expr-builtins.ts emitTextureCall)                 */
/* ------------------------------------------------------------------ */

const TEX_UNIFORM: [string, { store: 'float' | 'int'; slot: number; stride: number }] = [
  'u_tex',
  { store: 'int', slot: 0, stride: 0 },
];

{
  // Fragment textureLod (3.00) → ctx.tex.sample2DLod, result from ctx.tex.out.
  const e = env('FRAGMENT', 300, [TEX_UNIFORM]);
  e.declareLocal('P', vT('float', 2));
  const vals = emitExpr(
    call('textureLod', [ident('u_tex', sT('sampler2D')), ident('P', vT('float', 2)), lit(0.0, fT())], vT('float', 4)),
    e,
  );
  check(
    vals[0].pre !== undefined && vals[0].pre[0].includes('ctx.tex.sample2DLod('),
    `fragment textureLod → ctx.tex.sample2DLod (got '${vals[0].pre?.[0]}')`,
  );
  check(vals[0].v === 'ctx.tex.out[0]' && vals[3].v === 'ctx.tex.out[3]', `fragment textureLod reads ctx.tex.out`);
}

{
  // Vertex textureLod (3.00) → R.tex2DLod into ctx.scratch.
  const e = env('VERTEX', 300, [TEX_UNIFORM]);
  e.declareLocal('P', vT('float', 2));
  const vals = emitExpr(
    call('textureLod', [ident('u_tex', sT('sampler2D')), ident('P', vT('float', 2)), lit(0.0, fT())], vT('float', 4)),
    e,
  );
  check(
    vals[0].pre !== undefined && vals[0].pre[0].includes('R.tex2DLod(ctx, '),
    `vertex textureLod → R.tex2DLod(ctx, ...) (got '${vals[0].pre?.[0]}')`,
  );
  check(vals[0].v.startsWith('ctx.scratch['), `vertex textureLod reads ctx.scratch (got '${vals[0].v}')`);
}

{
  // Fragment texture2D (1.00) implicit LOD → ctx.tex.sample2D with bias 0.
  const e = env('FRAGMENT', 100, [TEX_UNIFORM]);
  e.declareLocal('P', vT('float', 2));
  const vals = emitExpr(call('texture2D', [ident('u_tex', sT('sampler2D')), ident('P', vT('float', 2))], vT('float', 4)), e);
  check(
    vals[0].pre !== undefined && vals[0].pre[0].includes('ctx.tex.sample2D(') && vals[0].pre[0].includes('implicit-LOD'),
    `fragment texture2D → ctx.tex.sample2D + implicit-LOD comment (got '${vals[0].pre?.[0]}')`,
  );
}

{
  // Fragment texture2DProj (1.00, vec3 P) divides by P.z first.
  const e = env('FRAGMENT', 100, [TEX_UNIFORM]);
  e.declareLocal('P', vT('float', 3));
  const vals = emitExpr(call('texture2DProj', [ident('u_tex', sT('sampler2D')), ident('P', vT('float', 3))], vT('float', 4)), e);
  check(
    vals[0].pre !== undefined && vals[0].pre[0].includes('((P__0) / (P__2))') && vals[0].pre[0].includes('((P__1) / (P__2))'),
    `texture2DProj divides by P.z (got '${vals[0].pre?.[0]}')`,
  );
}

{
  // Vertex texture2DLod (1.00, vertex-only) → R.tex2DLod with explicit LOD.
  const e = env('VERTEX', 100, [TEX_UNIFORM]);
  e.declareLocal('P', vT('float', 2));
  const vals = emitExpr(
    call('texture2DLod', [ident('u_tex', sT('sampler2D')), ident('P', vT('float', 2)), lit(1.0, fT())], vT('float', 4)),
    e,
  );
  check(
    vals[0].pre !== undefined && vals[0].pre[0].includes('R.tex2DLod(ctx, ctx.intUniforms[0 + 0], P__0, P__1, 1.0, ctx.scratch, 0)'),
    `vertex texture2DLod exact call (got '${vals[0].pre?.[0]}')`,
  );
}

{
  // texelFetch (isampler2D) → ctx.tex.texelFetch2D, result from outInt.
  const e = env('FRAGMENT', 300, [TEX_UNIFORM]);
  const vals = emitExpr(
    call('texelFetch', [ident('u_tex', sT('isampler2D')), call('ivec2', [lit(1, iT()), lit(2, iT())], vT('int', 2)), lit(0, iT())], vT('int', 4)),
    e,
  );
  check(
    vals[0].pre !== undefined && vals[0].pre[0].includes('ctx.tex.texelFetch2D('),
    `texelFetch → ctx.tex.texelFetch2D (got '${vals[0].pre?.[0]}')`,
  );
  check(vals[0].v === 'ctx.tex.outInt[0]', `isampler texelFetch reads outInt (got '${vals[0].v}')`);
}

{
  // texelFetch (usampler2D) → result wrapped to uint.
  const e = env('FRAGMENT', 300, [TEX_UNIFORM]);
  const vals = emitExpr(
    call('texelFetch', [ident('u_tex', sT('usampler2D')), call('ivec2', [lit(1, iT()), lit(2, iT())], vT('int', 2)), lit(0, iT())], vT('uint', 4)),
    e,
  );
  check(vals[0].v === '((ctx.tex.outUint[0]) >>> 0)', `usampler texelFetch wraps outUint (got '${vals[0].v}')`);
}

{
  // textureSize(sampler2D, lod) → R.textureSize into ctx.intScratch, ivec2 result.
  const e = env('FRAGMENT', 300, [TEX_UNIFORM]);
  const vals = emitExpr(
    call('textureSize', [ident('u_tex', sT('sampler2D')), lit(0, iT())], vT('int', 2)),
    e,
  );
  check(
    vals[0].pre !== undefined && vals[0].pre[0].includes('R.textureSize(ctx, ctx.intUniforms[0 + 0], 0, ctx.intScratch, 0)'),
    `textureSize exact call (got '${vals[0].pre?.[0]}')`,
  );
  check(vals[0].v === 'ctx.intScratch[0 + 0]' && vals[1].v === 'ctx.intScratch[0 + 1]', `textureSize reads intScratch`);
}

/* ------------------------------------------------------------------ */
/* 4. Storage paths (env.ts + expressions.ts walker)                   */
/* ------------------------------------------------------------------ */

{
  const e = env('VERTEX', 100);
  const vals = emitExpr(ident('a_pos', vT('float', 2)), e);
  check(
    vals[0].v.includes("(typeof ctx.attribs[0] === 'number'"),
    `attrib read carries the constant-attribute guard (got '${vals[0].v.slice(0, 60)}...')`,
  );
}

{
  const e = env('VERTEX', 100);
  const vals = emitExpr(ident('u_color', vT('float', 4)), e);
  check(vals[0].v === 'ctx.uniforms[0 + 0]' && vals[3].v === 'ctx.uniforms[0 + 3]', `uniform read (got '${vals[0].v}')`);
}

{
  const e = env('VERTEX', 100, [['u_mat', { store: 'float', slot: 8, stride: 0 }]]);
  const vals = emitExpr(ident('u_mat', mT(2, 2)), e);
  check(
    vals[0].v === 'ctx.uniforms[8 + 0 * 4 + 0]' && vals[2].v === 'ctx.uniforms[8 + 1 * 4 + 0]',
    `uniform mat2 column stride 4 (got ${vals.map((x) => x.v).join(',')})`,
  );
}

{
  const e = env('VERTEX', 100);
  const v = emitExpr(ident('u_i', uT()), e)[0];
  check(v.v === '((ctx.intUniforms[4 + 0]) >>> 0)', `uint uniform read wrapped (got '${v.v}')`);
}

{
  const e = env('VERTEX', 100);
  e.declareLocal('x', vT('float', 2));
  const lv = emitLValue(ident('x', vT('float', 2)), e);
  check(lv.targets[0] === 'x__0' && lv.targets[1] === 'x__1', `emitLValue local vec2 → ['x__0','x__1'] (got ${JSON.stringify(lv.targets)})`);
}

{
  const e = env('VERTEX', 100);
  const lv = emitLValue(ident('gl_Position', vT('float', 4)), e);
  check(lv.targets[0] === 'ctx.out.position[0]', `emitLValue gl_Position → ctx.out.position[0] (got '${lv.targets[0]}')`);
}

{
  const e = env('FRAGMENT', 100);
  const v = emitExpr(ident('gl_FragColor', vT('float', 4)), e)[0];
  check(v.v === 'ctx.out.color[0][0]', `gl_FragColor read → ctx.out.color[0][0] (got '${v.v}')`);
}

/* ------------------------------------------------------------------ */
/* 5. compileShader smoke (walk the annotated AST of real shaders)     */
/* ------------------------------------------------------------------ */

function firstAssignExpr(tu: TranslationUnit): Expr | null {
  const find = (decls: ExternalDecl[]): Expr | null => {
    for (const d of decls) {
      if (d.kind === 'function-definition') {
        for (const s of d.body.body) {
          if (s.kind === 'expr-stmt' && s.expr) return s.expr;
        }
      }
    }
    return null;
  };
  return find(tu.declarations);
}

{
  const r = compileShader('precision mediump float;\nvoid main() { gl_FragColor = vec4(1.0, 0.5, 0.25, 1.0); }', {
    type: 'FRAGMENT',
    version: 100,
  });
  check(r.ok === true, 'compileShader: fragment gl_FragColor shader compiles');
  if (r.ok) {
    const expr = firstAssignExpr(r.shader.ast);
    check(expr !== null && expr.kind === 'assign', 'fragment shader AST contains the assignment expression');
    if (expr) {
      const e = env('FRAGMENT', 100);
      const vals = emitExpr(expr, e);
      check(
        vals.length > 0 && vals[0].v.includes('ctx.out.color[0][0]'),
        `fragment assign emits ctx.out.color[0][0] (got '${vals[0]?.v}')`,
      );
    }
  }
}

{
  const r = compileShader('attribute vec2 a_pos;\nvoid main() { gl_Position = vec4(a_pos, 0.0, 1.0); }', {
    type: 'VERTEX',
    version: 100,
  });
  check(r.ok === true, 'compileShader: vertex gl_Position shader compiles');
  if (r.ok) {
    const expr = firstAssignExpr(r.shader.ast);
    check(expr !== null && expr.kind === 'assign', 'vertex shader AST contains the assignment expression');
    if (expr) {
      const e = env('VERTEX', 100);
      const vals = emitExpr(expr, e);
      check(
        vals.length > 0 && vals[0].v.includes('ctx.out.position[0]'),
        `vertex assign emits ctx.out.position[0] (got '${vals[0]?.v}')`,
      );
    }
  }
}

/* ------------------------------------------------------------------ */
/* 6. Compound assign & ++/-- in EXPRESSION contexts (regression)      */
/*    expressions.ts emitAssign/emitUnary used to fold LValue.prelude/ */
/*    copyBack ('; '-joined statements) inside parens — invalid JS —   */
/*    and the compound path received the parser's '+=' spelling.       */
/*    Every case below compiles a REAL shader, emits its main (or an   */
/*    inliner-style helper body) and RUNS the generated JS.            */
/* ------------------------------------------------------------------ */

function findFn(tu: TranslationUnit, name: string): FunctionDefinition {
  for (const d of tu.declarations) {
    if (d.kind === 'function-definition' && d.prototype.name === name) return d;
  }
  throw new Error(`no function '${name}' in shader`);
}

/** Compile + emit + run a main body; returns the JS body + mutated ctx. */
function runMainE(src: string, stage: 'VERTEX' | 'FRAGMENT', version: 100 | 300 = 100): { body: string; ctx: Record<string, any> } {
  const r = compileShader(src, { type: stage, version });
  if (!r.ok) throw new Error(`compile failed: ${JSON.stringify(r.errors)}`);
  const mainFn = findFn(r.shader.ast, 'main');
  const env = new CodegenEnv(stage, baseLayout(version));
  const stmts = emitStatements(mainFn.body.body, env);
  const body = [...(env.temps.length ? [`var ${env.temps.join(', ')};`] : []), ...stmts].join('\n');
  const fn = new Function('ctx', 'R', body);
  const ctx: Record<string, any> = {
    discarded: false,
    out: stage === 'VERTEX' ? { position: [0, 0, 0, 0], pointSize: 0 } : { color: [[0, 0, 0, 0]] },
    scratch: new Float32Array(Math.max(env.scratchSize, 16)),
    intScratch: new Int32Array(Math.max(env.intScratchSize, 16)),
  };
  fn(ctx, R);
  return { body, ctx };
}

/** Compile + emit + run an inliner-style helper body (retTemps + epilogue
 *  label, wrapped in the labeled block the inliner provides). */
function runFnE(src: string, fnName: string, stage: 'VERTEX' | 'FRAGMENT', version: 100 | 300): { body: string; ret: number } {
  const r = compileShader(src, { type: stage, version });
  if (!r.ok) throw new Error(`compile failed: ${JSON.stringify(r.errors)}`);
  const fn = findFn(r.shader.ast, fnName);
  const env = new CodegenEnv(stage, baseLayout(version));
  const lines = emitStatements(fn.body.body, env, {
    retTemps: ['r0'],
    epilogueLabel: 'EP_0',
    retType: { kind: 'scalar', base: 'float' },
  });
  const body = [...(env.temps.length ? [`var ${env.temps.join(', ')};`] : []), 'EP_0: {', ...lines.map((l) => '  ' + l), '}'].join('\n');
  const f = new Function('ctx', 'R', `${body}\nreturn r0;`);
  const ret = f(
    { out: { color: [[0, 0, 0, 0]] }, scratch: new Float32Array(16), intScratch: new Int32Array(16) },
    R,
  ) as number;
  return { body, ret };
}

{
  // `return x += 1.0;` inside a helper used by a FRAGMENT main writing
  // gl_FragColor (return-value context → statements.ts emitReturn → emitExpr).
  const { body, ret } = runFnE(
    `precision mediump float;
float bump() { float x = 1.0; return x += 1.0; }
void main() { gl_FragColor = vec4(bump(), 0.0, 0.0, 1.0); }`,
    'bump',
    'FRAGMENT',
    100,
  );
  check(ret === 2, `return x += 1.0; → 2.0 (got ${ret})`);
  check(
    body.includes('r0 = (x = x + 1.0);'),
    `return lowers to a clean 'r0 = (x = x + 1.0);' line (got:\n${body})`,
  );
}

{
  // Compound assign as a CALL ARG in a fragment main (gl_FragColor output).
  const { ctx } = runMainE(
    `precision mediump float;
void main() {
  float x = 1.0;
  gl_FragColor = vec4(x += 1.0, 0.0, 0.0, 1.0);
}`,
    'FRAGMENT',
  );
  check(
    ctx.out.color[0][0] === 2 && ctx.out.color[0][3] === 1,
    `gl_FragColor = vec4(x += 1.0, ...) → [2,0,0,1] (got [${ctx.out.color[0].join(', ')}])`,
  );
}

{
  // `a = b += c;` — the RHS (compound assign) is an EXPRESSION here.
  const { ctx, body } = runMainE(
    `void main() {
  float a = 1.0, b = 1.0, c = 2.0;
  a = b += c;
  gl_Position.x = a + b;
}`,
    'VERTEX',
  );
  check(ctx.out.position[0] === 6, `a = b += c; → a === b === 3 (got ${ctx.out.position[0]})`);
  check(body.includes('a = (b = b + c);'), `chain emits 'a = (b = b + c);' (got:\n${body})`);
}
{
  // Compound assign nested in a binary operand.
  const { ctx } = runMainE(
    `void main() {
  float a = 1.0, b = 1.0, c = 2.0;
  a = (b += c) * 2.0;
  gl_Position.x = a;
}`,
    'VERTEX',
  );
  check(ctx.out.position[0] === 6, `a = (b += c) * 2.0; → a === 6 (got ${ctx.out.position[0]})`);
}
{
  // Plain `=` chain (nested plain assigns through emitExpr).
  const { ctx } = runMainE(
    `void main() {
  float a = 1.0, b = 1.0, c = 2.0;
  a = b = c;
  gl_Position.x = a + b + c;
}`,
    'VERTEX',
  );
  check(ctx.out.position[0] === 6, `a = b = c; → all 2 (got ${ctx.out.position[0]})`);
}

{
  // Compound assign inside an if CONDITION.
  const { ctx } = runMainE(
    `void main() {
  float x = 1.0;
  float r = 0.0;
  if ((x += 1.0) > 2.0) { r = 10.0; } else { r = 20.0; }
  gl_Position.x = r + x;
}`,
    'VERTEX',
  );
  check(ctx.out.position[0] === 22, `if ((x += 1.0) > 2.0): x → 2.0, false branch → 22 (got ${ctx.out.position[0]})`);
}

{
  // ALL 10 compound ops in expression contexts (ES 3.00 int accumulator).
  const { ctx } = runMainE(
    `#version 300 es
void main() {
  int x = 0;
  int r = 0;
  r = (x += 5) + (x -= 2) + (x *= 3) + (x /= 2) + (x %= 4) + (x <<= 2) + (x >>= 1) + (x &= 7) + (x ^= 3) + (x |= 8);
  gl_Position.x = float(r);
  gl_Position.y = float(x);
}`,
    'VERTEX',
    300,
  );
  check(
    ctx.out.position[0] === 35 && ctx.out.position[1] === 11,
    `+= -= *= /= %= <<= >>= &= ^= |= → r === 35, x === 11 (got [${ctx.out.position[0]}, ${ctx.out.position[1]}])`,
  );
}

{
  // Dynamic-index compound on a LOCAL ARRAY in expression position — the
  // index temp is a PRELUDE that must fold as comma terms, not statements.
  const { ctx, body } = runMainE(
    `void main() {
  float a[4];
  a[0] = 2.0;
  int i = 0;
  gl_Position.x = (a[i] += 3.0) * 2.0;
}`,
    'VERTEX',
  );
  check(ctx.out.position[0] === 10, `(a[i] += 3.0) * 2.0 with a[0] = 2.0 → 10 (got ${ctx.out.position[0]})`);
  check(
    /\(t\d+ = i, \(ctx\.scratch/.test(body),
    `dyn-index prelude folds as '(tN = i, (...))' comma terms (got:\n${body})`,
  );
}

{
  // Prefix ++ / -- on a dynamic-index target (same prelude folding).
  const { ctx } = runMainE(
    `void main() {
  float a[4];
  a[0] = 1.0;
  int i = 0;
  gl_Position.x = ++a[i];
}`,
    'VERTEX',
  );
  check(ctx.out.position[0] === 2, `gl_Position.x = ++a[i]; → 2 (got ${ctx.out.position[0]})`);
}
{
  // uint ++ wraps (0xFFFFFFFF → 0).
  const { ctx } = runMainE(
    `#version 300 es
void main() {
  uint u = 4294967295u;
  gl_Position.x = float(++u);
}`,
    'VERTEX',
    300,
  );
  check(ctx.out.position[0] === 0, `uint ++u wraps 0xFFFFFFFF → 0 (got ${ctx.out.position[0]})`);
}

{
  // COPY-BACK: dynamic component of a FLAT vec4 local (ES 3.00) — compound,
  // plain `=` and prefix ++ all in expression position.
  const { ctx, body } = runMainE(
    `#version 300 es
void main() {
  vec4 v = vec4(1.0, 2.0, 3.0, 4.0);
  int i = 0;
  gl_Position.x = (v[i] += 10.0) * 2.0;
  gl_Position.y = v[0];
}`,
    'VERTEX',
    300,
  );
  check(
    ctx.out.position[0] === 22 && ctx.out.position[1] === 11,
    `(v[i] += 10.0) * 2.0 with readback → [22, 11] (got [${ctx.out.position[0]}, ${ctx.out.position[1]}])`,
  );
  check(
    /\(t\d+ = \(ctx\.scratch/.test(body) && /v__0 = \(ctx\.scratch/.test(body),
    `copy-back folds as comma terms after a value-preserving temp (got:\n${body})`,
  );
}
{
  const { ctx } = runMainE(
    `#version 300 es
void main() {
  vec4 v = vec4(1.0, 2.0, 3.0, 4.0);
  int i = 0;
  gl_Position.x = (v[i] = 7.0) * 3.0;
  gl_Position.y = ++v[i];
}`,
    'VERTEX',
    300,
  );
  check(
    ctx.out.position[0] === 21 && ctx.out.position[1] === 8,
    `(v[i] = 7.0) * 3.0 then ++v[i] → [21, 8] (got [${ctx.out.position[0]}, ${ctx.out.position[1]}])`,
  );
}

{
  // uint wrap on a dynamic-index compound (prelude + >>> 0 invariant).
  const { ctx } = runMainE(
    `#version 300 es
void main() {
  uint a[2];
  a[0] = 4294967295u;
  int i = 0;
  gl_Position.x = float((a[i] += 1u) == 0u ? 1.0 : 0.0);
}`,
    'VERTEX',
    300,
  );
  check(ctx.out.position[0] === 1, `uint (a[i] += 1u) wraps 0xFFFFFFFF → 0 (got ${ctx.out.position[0]})`);
}

{
  // Ternary arm and comma-expression contexts.
  const { ctx } = runMainE(
    `void main() {
  float x = 1.0;
  float y = 2.0;
  bool c = true;
  gl_Position.x = c ? (x += 10.0) : (y += 20.0);
  gl_Position.y = x;
}`,
    'VERTEX',
  );
  check(
    ctx.out.position[0] === 11 && ctx.out.position[1] === 11,
    `ternary arm (x += 10.0) → [11, 11] (got [${ctx.out.position[0]}, ${ctx.out.position[1]}])`,
  );
}
{
  const { ctx } = runMainE(
    `void main() {
  float x = 1.0;
  gl_Position.x = (x += 2.0, x) * 3.0;
}`,
    'VERTEX',
  );
  check(ctx.out.position[0] === 9, `(x += 2.0, x) * 3.0 → 9 (got ${ctx.out.position[0]})`);
}

{
  // Direct emitExpr checks (pure expressions layer, no statements).
  const e = env('VERTEX', 100);
  e.declareLocal('x', fT());
  const v = emitExpr(
    bin('*', { kind: 'assign', op: '+=', target: ident('x', fT()), value: lit(1.0, fT()), resolvedType: fT() } as never, lit(2.0, fT()), fT()),
    e,
  )[0];
  check(v.v === '((x = x + 1.0) * 2.0)', `compound in binary operand → '((x = x + 1.0) * 2.0)' (got '${v.v}')`);
  check(evalParams(v, e, ['x'], [1]) === 4, `(x += 1.0) * 2.0 evaluates to 4`);
}
{
  // Direct emitExpr: dyn-index compound — prelude lands in Value.pre.
  const e = env('VERTEX', 100);
  e.declareLocal('a', { kind: 'array', element: fT(), size: 4 });
  e.declareLocal('i', iT());
  const arrT: GLSLType = { kind: 'array', element: fT(), size: 4 };
  const v = emitExpr(
    {
      kind: 'assign',
      op: '+=',
      target: { kind: 'index', object: ident('a', arrT), index: ident('i', iT()), resolvedType: fT() },
      value: lit(3.0, fT()),
      resolvedType: fT(),
    } as never,
    e,
  )[0];
  check(
    v.pre !== undefined && /^t\d+ = i$/.test(v.pre[0]),
    `dyn-index compound prelude → Value.pre (got ${JSON.stringify(v.pre)})`,
  );
  const decl = e.temps.length ? `var ${e.temps.join(', ')}; ` : '';
  const pres = v.pre!.join('; ') + '; ';
  const res = new Function('ctx', 'R', 'i', `${decl}${pres}return ${v.v};`)({ scratch: new Float32Array([2, 0, 0, 0]), intScratch: new Int32Array(4) }, R, 0) as number;
  check(res === 5, `a[0] += 3.0 with a[0] = 2.0 → 5 (got ${res})`);
}

{
  // Return-context prelude: dynamic local array in `return a[i] += 3.0;`.
  const { body, ret } = runFnE(
    `precision mediump float;
float bump() {
  float a[4];
  a[0] = 2.0;
  int i = 0;
  return a[i] += 3.0;
}
void main() { gl_FragColor = vec4(bump(), 0.0, 0.0, 1.0); }`,
    'bump',
    'FRAGMENT',
    100,
  );
  check(ret === 5, `return a[i] += 3.0; (a[0] = 2.0) → 5 (got ${ret})`);
  check(
    /^\s*t\d+ = i;$/m.test(body) && !/\(t\d+ = i;/.test(body),
    `return-context prelude emits as a LINE before the value (got:\n${body})`,
  );
}

{
  // Return-context copy-back: dynamic component of a flat vec4 (ES 3.00).
  const { body, ret } = runFnE(
    `#version 300 es
precision mediump float;
float bump() {
  vec4 v = vec4(1.0, 2.0, 3.0, 4.0);
  int i = 0;
  return v[i] += 10.0;
}
void main() { }`,
    'bump',
    'FRAGMENT',
    300,
  );
  check(ret === 11, `return v[i] += 10.0; (v[0] = 1.0) → 11 (got ${ret})`);
  check(
    body.includes('v__0 = (ctx.scratch[0 + 0])'),
    `return-context copy-back folds as comma terms (got:\n${body})`,
  );
}

/* ------------------------------------------------------------------ */
/* Report + exit                                                       */
/* ------------------------------------------------------------------ */

console.log(`codegen-expr selftest: ${checks} checks, ${failures} failure(s)`);
if (failures > 0) {
  console.error(`${failures} check(s) FAILED`);
  process.exit(1);
}
console.log('OK');
