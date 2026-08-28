/**
 * statements.ts — GLSL→JS statement lowering (contract §1; the C3a task).
 *
 * emitStatements(stmts, env, fn?) returns the JS LINES for a statement list
 * (a fresh string[] — the caller owns the array; no buffer exists on env).
 * The generated lines assume the caller (stage assembly / inliner) has:
 *   - declared `var t0, t1, ...` (env.temps) at the top of the function body,
 *   - provided ctx with the scratch buffers sized to env.scratchSize /
 *     env.intScratchSize.
 *
 * EXPRESSION PROTOCOL (see expressions.ts): emitExpr returns one Value per
 * flat component. Value.pre = PURE materialization statements that must run
 * once, in order, before v evaluates. Statement emission follows the rule:
 *   - ONE-SHOT contexts (decl initializers, expr statements, return values,
 *     switch discriminants): emit deduped pres as lines FIRST, then use `v`.
 *   - PER-ITERATION contexts (if/while/do-while/for CONDITION and for UPDATE
 *     slots): pres are folded INLINE via foldPre — `(p1, p2, v)` — so they
 *     re-run exactly when GLSL re-evaluates the expression. Hoisting a cond
 *     into a temp before the loop would FREEZE loop-dependent conditions
 *     (e.g. `while (texture2D(u, P).a > 0.5)` with P varying per iteration),
 *     which is wrong; inline folding is the exact GLSL evaluation model.
 *   - LVALUE targets (assignments, ++/--): emitExpr folds LValue.prelude/
 *     copyBack INSIDE the parenthesized value, which is invalid JS in a
 *     statement context (and its compound-op path is broken — see
 *     emitExprStmt). Statements therefore bypass emitExpr and use the 4-step
 *     LVALUE PROTOCOL: deduped RHS pres as lines → `prelude` line → per-
 *     component `target = v;` → `copyBack` line. The for-update slot folds
 *     the same protocol into one comma expression (it must stay a single
 *     expression for `continue` semantics).
 *
 * SCOPE / REDECLARATION: GLSL forbids shadowing but allows SIBLING-scope
 * re-declarations (`for (int i = 0; ...) {} for (int i = 0; ...) {}` — each
 * for-init lives in its own scope). declareLocal registers ONE JS name per
 * GLSL name per stage, so a re-declaration reuses the existing registration:
 * the initializer re-initializes the shared JS var / scratch block, which is
 * exactly right because sibling scopes never overlap in time. A re-declaration
 * with a WIDER type (e.g. vec2 then vec3 in sibling scopes) cannot be
 * represented and throws a clear link-time error.
 *
 * RETURN / INLINING (C3b contract): with `fn` set, `return e;` compiles to
 *   retTemps[c] = <e component c>;   (pres first)
 *   break <epilogueLabel>;
 * The inliner must wrap the inlined body in `EP_N: { ... }` (labeled block)
 * and read retTemps after the block. `return;` in a void function emits only
 * the break. `discard` ALWAYS compiles to `ctx.discarded = true; return;` —
 * a bare return that terminates the whole stage function even when spliced
 * inside a loop or an inlined body (GLSL: discard ends the invocation).
 * `break`/`continue` map directly to JS — after inlining they hit the loop
 * enclosing the CALL SITE, matching GLSL semantics.
 */
import type {
  CaseLabelStmt,
  DeclStmt,
  Expr,
  ReturnStmt,
  Stmt,
  VarDeclarator,
} from '../ast.js';
import type { GLSLType } from '../types.js';
import { typeEquals } from '../types.js';
import {
  CodegenEnv,
  LocalVar,
  convertValue,
  flatComponents,
  foldPre,
  scalarBaseOf,
  unpackVaryingCell,
  packVaryingWrite,
  packVaryingCompound,
} from './env.js';
import { emitExpr, emitLValue, materialize, matrixCompoundMul } from './expressions.js';
import type { Value } from './index.js';

/** Context for an INLINED function body; absent ⇒ emitting `main`
 *  (return ⇒ plain `return;`). */
export interface FnEmitContext {
  /** One temp per return-value component (empty for void). */
  retTemps: string[];
  /** Dual-mode return slots per component: [dx, dy] temp names (null for
   *  non-float components / non-dual mode). The inliner allocates them so
   *  float return values carry their derivative triple. */
  retDualTemps?: ([string, string] | null)[];
  /** e.g. 'EP_3' — return compiles to `break EP_3;`. */
  epilogueLabel: string;
  /**
   * The inlined function's return type. Optional but recommended: when set,
   * return values are base-converted (int→uint wrap etc.) before the retTemp
   * assignment; without it values are emitted in their own type.
   */
  retType?: GLSLType;
}

/** Emit JS lines for a statement list (fresh array; caller owns it). */
export function emitStatements(stmts: Stmt[], env: CodegenEnv, fn?: FnEmitContext): string[] {
  const out: string[] = [];
  for (const s of stmts) emitStatement(s, env, fn, out);
  return out;
}

/* ------------------------------------------------------------------ */
/* Statement dispatch                                                   */
/* ------------------------------------------------------------------ */

function emitStatement(s: Stmt, env: CodegenEnv, fn: FnEmitContext | undefined, out: string[]): void {
  switch (s.kind) {
    case 'compound':
      for (const st of s.body) emitStatement(st, env, fn, out);
      break;
    case 'decl-stmt':
      emitDeclStmt(s, env, out);
      break;
    case 'expr-stmt':
      if (s.expr !== null) emitExprStmt(s.expr, env, out);
      break;
    case 'if':
      emitIf(s, env, fn, out);
      break;
    case 'for':
      emitFor(s, env, fn, out);
      break;
    case 'while': {
      const cond = condString(s.cond, env);
      const body: string[] = [];
      emitStatement(s.body, env, fn, body);
      out.push(`while (${cond}) {`);
      out.push(...indent(body));
      out.push('}');
      break;
    }
    case 'do-while': {
      const body: string[] = [];
      emitStatement(s.body, env, fn, body);
      out.push('do {');
      out.push(...indent(body));
      out.push(`} while (${condString(s.cond, env)});`);
      break;
    }
    case 'switch':
      emitSwitch(s, env, fn, out);
      break;
    case 'case':
      out.push(caseLabelString(s, env));
      break;
    case 'break':
      out.push('break;');
      break;
    case 'continue':
      out.push('continue;');
      break;
    case 'return':
      emitReturn(s, env, fn, out);
      break;
    case 'discard':
      // Whole-shader termination — even inside loops or inlined bodies.
      out.push('ctx.discarded = true; return;');
      break;
    case 'empty':
      break;
    default:
      throw new Error(`codegen: unsupported statement '${(s as { kind: string }).kind}'`);
  }
}

/* ------------------------------------------------------------------ */
/* Declarations                                                         */
/* ------------------------------------------------------------------ */

/** Declared type of one declarator: base type wrapped with its array dims
 *  (semantics guarantees the dims are positive integer constants). */
function declaratorType(base: GLSLType, d: VarDeclarator): GLSLType {
  let t = base;
  for (let i = d.arrayDims.length - 1; i >= 0; i--) {
    const dim = d.arrayDims[i];
    const cv = dim ? dim.constValue : undefined;
    const size = typeof cv === 'number' && Number.isInteger(cv) && cv > 0 ? cv : 1;
    t = { kind: 'array', element: t, size };
  }
  return t;
}

function emitDeclStmt(s: DeclStmt, env: CodegenEnv, out: string[]): void {
  const base = s.type.resolved;
  if (!base) throw new Error('codegen: declaration type unresolved (semantics must run first)');
  for (const d of s.declarators) {
    if (d.name === '') continue; // parser error-recovery placeholder
    const type = declaratorType(base, d);
    // Inlined-body locals first: the active frame materializes them with
    // per-call-site unique JS names (functions.ts), so a callee local can
    // NEVER alias a caller's same-named local (ogles functions pages — the
    // reuse path below would otherwise share the caller's JS names and the
    // IIFE's `var` hoisting would shadow the caller's values).
    const frameLv = env.frameLocal(d.name, type);
    let lv: LocalVar;
    if (frameLv !== null) {
      // Sibling-scope re-declaration inside the same inlined body: reuse the
      // frame's per-call-site var (same checks as the locals_ path below).
      if (!typeEquals(frameLv.type, type)) {
        if (frameLv.kind === 'scratch') {
          throw new Error(
            `codegen: sibling re-declaration of '${d.name}' with a different array type is unsupported`,
          );
        }
        if (flatComponents(frameLv.type) < flatComponents(type)) {
          throw new Error(
            `codegen: sibling re-declaration of '${d.name}' with a wider type is unsupported`,
          );
        }
      }
      lv = frameLv;
    } else {
      const existing = env.lookupLocal(d.name);
      if (existing !== null) {
        // Sibling-scope re-declaration: reuse the JS names / scratch block —
        // sibling scopes are disjoint in time, the initializer re-initializes.
        if (!typeEquals(existing.type, type)) {
          if (existing.kind === 'scratch') {
            throw new Error(
              `codegen: sibling re-declaration of '${d.name}' with a different array type is unsupported`,
            );
          }
          if (flatComponents(existing.type) < flatComponents(type)) {
            throw new Error(
              `codegen: sibling re-declaration of '${d.name}' with a wider type is unsupported`,
            );
          }
        }
        lv = existing;
      } else {
        env.declareLocal(d.name, type);
        lv = env.lookupLocal(d.name)!;
      }
    }
    if (d.init === null) {
      // Keep the declaration explicit even when uninitialized (flat locals
      // have JS bindings; scratch arrays are link-time allocations, nothing
      // to emit). Dual mode: float locals also declare their `_dx`/`_dy`
      // JS names (the env registered them; uninitialized duals are stale
      // until first written — GLSL locals are undefined until assigned too).
      if (lv.kind === 'flat') {
        const names = lv.compNames!.slice(0, flatComponents(type));
        if (env.dual && lv.dxNames) {
          const all: string[] = [];
          names.forEach((nm, c) => {
            all.push(nm);
            if (lv.dxNames![c]) all.push(lv.dxNames![c], `${nm}_dy`);
          });
          out.push(`var ${all.join(', ')};`);
        } else {
          out.push(`var ${names.join(', ')};`);
        }
      }
      continue;
    }
    const vals = convertPreserving(emitExpr(d.init, env), d.init.resolvedType!, type);
    emitPres(out, vals);
    if (lv.kind === 'scratch') {
      const store = lv.int ? 'ctx.intScratch' : 'ctx.scratch';
      if (env.dual && !lv.int) {
        // Dual planes: v at base, dx at base+blockSize, dy at base+2*blockSize.
        const n = flatComponents(lv.type);
        for (let c = 0; c < vals.length; c++) {
          out.push(`${store}[${lv.scratchBase} + ${c}] = ${vals[c].v};`);
          out.push(`${store}[${lv.scratchBase} + ${n} + ${c}] = ${vals[c].dx ?? '0'};`);
          out.push(`${store}[${lv.scratchBase} + ${2 * n} + ${c}] = ${vals[c].dy ?? '0'};`);
        }
      } else {
        for (let c = 0; c < vals.length; c++) {
          out.push(`${store}[${lv.scratchBase} + ${c}] = ${vals[c].v};`);
        }
      }
    } else {
      const names = lv.compNames!.slice(0, vals.length);
      if (env.dual && lv.dxNames) {
        const parts: string[] = [];
        names.forEach((nm, c) => {
          parts.push(`${nm} = ${vals[c].v}`);
          if (lv.dxNames![c]) {
            parts.push(`${lv.dxNames![c]} = ${vals[c].dx ?? '0'}`);
            parts.push(`${nm}_dy = ${vals[c].dy ?? '0'}`);
          }
        });
        out.push(`var ${parts.join(', ')};`);
      } else {
        out.push(`var ${names.map((n, c) => `${n} = ${vals[c].v}`).join(', ')};`);
      }
    }
  }
}

/* ------------------------------------------------------------------ */
/* if / for / switch                                                    */
/* ------------------------------------------------------------------ */

function emitIf(
  s: Extract<Stmt, { kind: 'if' }>,
  env: CodegenEnv,
  fn: FnEmitContext | undefined,
  out: string[],
): void {
  const cond = condString(s.cond, env);
  const thenLines: string[] = [];
  emitStatement(s.then, env, fn, thenLines);
  if (s.else === null) {
    out.push(`if (${cond}) {`);
    out.push(...indent(thenLines));
    out.push('}');
  } else {
    const elseLines: string[] = [];
    emitStatement(s.else, env, fn, elseLines);
    out.push(`if (${cond}) {`);
    out.push(...indent(thenLines));
    out.push('} else {');
    out.push(...indent(elseLines));
    out.push('}');
  }
}

function emitFor(
  s: Extract<Stmt, { kind: 'for' }>,
  env: CodegenEnv,
  fn: FnEmitContext | undefined,
  out: string[],
): void {
  // GLSL scopes the init declarators to the loop; the (unique) JS names are
  // declared here, before the JS loop. Non-declaration inits emit as
  // statements. The loop itself then has an empty init slot.
  if (s.init !== null) emitStatement(s.init, env, fn, out);
  const cond = s.cond !== null ? condString(s.cond, env) : 'true';
  const update = s.update !== null ? updateString(s.update, env) : '';
  const body: string[] = [];
  emitStatement(s.body, env, fn, body);
  out.push(`for (; ${cond};${update ? ' ' + update : ''}) {`);
  out.push(...indent(body));
  out.push('}');
}

function emitSwitch(
  s: Extract<Stmt, { kind: 'switch' }>,
  env: CodegenEnv,
  fn: FnEmitContext | undefined,
  out: string[],
): void {
  // The discriminant evaluates EXACTLY once in GLSL: pres run once before
  // the switch (materialized into a temp so the discriminant slot is pure).
  const v = emitExpr(s.expr, env)[0];
  let expr = v.v;
  if (v.pre && v.pre.length > 0) {
    const m = materialize([v], env)[0];
    out.push(`${m.pre![0]};`);
    expr = m.v;
  }
  out.push(`switch (${expr}) {`);
  const inner: string[] = [];
  if (s.body.kind === 'compound') {
    for (const st of s.body.body) emitStatement(st, env, fn, inner);
  } else {
    emitStatement(s.body, env, fn, inner);
  }
  out.push(...indent(inner));
  out.push('}');
}

/** `case <const>:` / `default:` — GLSL fallthrough matches JS exactly. */
function caseLabelString(s: CaseLabelStmt, env: CodegenEnv): string {
  if (s.value === null) return 'default:';
  const cv = s.value.constValue;
  const t = s.value.resolvedType;
  if (typeof cv === 'number' && Number.isInteger(cv) && t !== undefined) {
    return `case ${env.emitConstNumber(cv, t)}:`;
  }
  // Defensive: semantics guarantees folded integer labels.
  const v = emitExpr(s.value, env)[0];
  return `case ${v.pre && v.pre.length > 0 ? foldPre(v.pre, v.v) : v.v}:`;
}

/* ------------------------------------------------------------------ */
/* return                                                               */
/* ------------------------------------------------------------------ */

function emitReturn(
  s: ReturnStmt,
  env: CodegenEnv,
  fn: FnEmitContext | undefined,
  out: string[],
): void {
  if (s.value !== null) {
    let vals = emitExpr(s.value, env);
    if (fn && fn.retType) vals = convertPreserving(vals, s.value.resolvedType!, fn.retType);
    emitPres(out, vals);
    if (fn && vals.length === fn.retTemps.length) {
      for (let c = 0; c < vals.length; c++) {
        out.push(`${fn.retTemps[c]} = ${vals[c].v};`);
        const d = fn.retDualTemps?.[c];
        if (d) {
          out.push(`${d[0]} = ${vals[c].dx ?? '0'};`);
          out.push(`${d[1]} = ${vals[c].dy ?? '0'};`);
        }
      }
    } else {
      // main is void / defensive mismatch: emit the value lines, discard them.
      for (const v of vals) out.push(`${v.v};`);
    }
  }
  if (fn) out.push(`break ${fn.epilogueLabel};`);
  else out.push('return;');
}

/* ------------------------------------------------------------------ */
/* Shared helpers                                                       */
/* ------------------------------------------------------------------ */

/* ------------------------------------------------------------------ */
/* Expression statements: the lvalue protocol (4-step assignment)      */
/* ------------------------------------------------------------------ */

/**
 * One expression statement. Assignments and ++/-- are emitted via the
 * LVALUE PROTOCOL (pres lines → lvalue.prelude → target writes → lvalue.copyBack)
 * instead of emitExpr: expressions.ts's emitAssign folds prelude/copyBack
 * INSIDE the parenthesized value (`(t0 = i; target = ...)`) which is invalid
 * JS in a statement context, and its compound-op path receives the parser's
 * `'+='` spelling (pre-existing bug — see report). All other expressions
 * emit via emitExpr (their pres fold into comma expressions, valid anywhere).
 */
function emitExprStmt(e: Expr, env: CodegenEnv, out: string[]): void {
  if (e.kind === 'assign') {
    emitAssignStmt(e.target, e.value, e.op, env, out);
    return;
  }
  if (e.kind === 'unary' && (e.op === '++' || e.op === '--')) {
    // Same prelude/copyBack issue as assign: bypass emitUnary's fold.
    const lv = emitLValue(e.operand, env);
    if (lv.prelude) out.push(lv.prelude);
    const delta = e.op === '++' ? '1' : '-1';
    const base = scalarBaseOf(lv.type);
    if (base === null || base === 'bool') throw new Error('codegen: cannot increment a bool');
    for (let c = 0; c < lv.targets.length; c++) {
      const tgt = lv.targets[c];
      const bitKind = lv.bits !== undefined ? lv.bits[c] : false;
      if (bitKind) {
        // Packed int/uint varying cell (VERTEX): unpack the old value,
        // increment with the int32/uint32 wrap, repack (see packVaryingWrite).
        const wrap = bitKind === 'int' ? '| 0' : '>>> 0';
        out.push(`${tgt} = R.u2f(((${unpackVaryingCell(tgt, bitKind === 'int')} + ${delta}) ${wrap}));`);
      } else {
        out.push(
          base === 'float'
            ? `${tgt} = ${tgt} + ${delta};`
            : base === 'int'
              ? `${tgt} = ((${tgt} + ${delta}) | 0);`
              : `${tgt} = ((${tgt} + ${delta}) >>> 0);`,
        );
      }
    }
    if (lv.copyBack) out.push(lv.copyBack);
    return;
  }
  const vals = emitExpr(e, env);
  emitPres(out, vals);
  for (const v of vals) out.push(`${v.v};`);
}

/**
 * The 4-step assignment protocol:
 *   1. deduped Value pres (RHS) as lines first,
 *   2. `lvalue.prelude` (dynamic-index temps / spill copy-in) as one line,
 *   3. `targets[c] = <converted rhs>;` per flat component,
 *   4. `lvalue.copyBack` (spill copy-out) as one line.
 * Compound ops (`+=` etc.) lower per-component via compoundOpExpr — the
 * parser stores `'+='` and expressions.ts's compoundOp expects `'+'`.
 * EXCEPTION: mat×mat `*=` lowers to a MATRIX PRODUCT via matrixCompoundMul
 * (LHS snapshot + RHS materialization; dual aware) — see emitAssignStmt. */
function emitAssignStmt(
  target: Expr,
  value: Expr,
  op: string,
  env: CodegenEnv,
  out: string[],
): void {
  const lv = emitLValue(target, env);
  // mat×mat '*=' lowers to a MATRIX PRODUCT (matrixCompoundMul) — the
  // component-wise path below is only correct for scalar/vector targets and
  // mat×scalar broadcast (regression: matrix-compound-multiply CTS page
  // rendered black). The RAW (unconverted) RHS is passed at its own width;
  // its pres run inside the helper's materialization (emitPres skipped).
  const isMatMul =
    op !== '=' && op[0] === '*' && lv.type.kind === 'matrix' && value.resolvedType!.kind === 'matrix';
  const rawRhs = emitExpr(value, env);
  if (isMatMul) {
    if (lv.prelude) out.push(lv.prelude);
    const mm = matrixCompoundMul(env, lv.targets, lv.dualTargets, lv.type, rawRhs, value.resolvedType!);
    for (const p of mm.pre) out.push(`${p};`);
    for (const w of mm.writes) out.push(`${w};`);
  } else {
    const conv = convertPreserving(rawRhs, value.resolvedType!, lv.type);
    emitPres(out, conv);
    if (lv.prelude) out.push(lv.prelude);
    if (op === '=') {
      for (let c = 0; c < lv.targets.length; c++) {
        // Packed uint varying cell (VERTEX): store the value's BIT PATTERN via
        // R.u2f (see packVaryingWrite) — the statement form drops the value.
        if (lv.bits && lv.bits[c]) {
          out.push(`${lv.targets[c]} = R.u2f(${conv[c].v});`);
        } else if (env.dual && lv.dualTargets && lv.dualTargets[c]) {
          // Dual mode: write the whole triple as one comma statement
          // `(vslot = vv, dxslot = dxv, dyslot = dyv, vslot);`.
          out.push(`${env.dualWrite(lv.targets[c], lv.dualTargets[c], conv[c])};`);
        } else {
          out.push(`${lv.targets[c]} = ${conv[c].v};`);
        }
      }
    } else {
      const base = scalarBaseOf(lv.type);
      if (base === null || base === 'bool') {
        throw new Error('codegen: cannot compound-assign a non-scalar-shaped value');
      }
      const cop = op.replace('=', '');
      for (let c = 0; c < lv.targets.length; c++) {
        const bitKind = lv.bits !== undefined ? lv.bits[c] : false;
        // Packed int/uint varying cell (VERTEX): unpack the old value, apply
        // the op with the int32/uint32 wrap, repack (see packVaryingCompound).
        if (bitKind) {
          out.push(`${packVaryingCompound(cop, lv.targets[c], conv[c].v, bitKind === 'int')};`);
        } else if (env.dual && lv.dualTargets && base === 'float' && lv.dualTargets[c]) {
          // Dual mode, float target: linear ops (+=, -=) update all three planes
          // via dualWrite; non-linear compounds throw (C5a2 templates).
          out.push(`${env.dualWrite(lv.targets[c], lv.dualTargets[c], conv[c], cop)};`);
        } else {
          out.push(`${compoundOpExpr(cop, lv.targets[c], conv[c].v, base)};`);
        }
      }
    }
  }
  if (lv.copyBack) out.push(lv.copyBack);
}

/** convertValue drops Value.pre when it converts scalar bases — re-attach. */
function convertPreserving(vals: Value[], from: GLSLType, to: GLSLType): Value[] {
  const out = convertValue(vals, from, to);
  for (let i = 0; i < vals.length; i++) {
    const src = vals[i];
    if (out[i] !== src && src && src.pre && src.pre.length > 0) {
      out[i] = { ...out[i], pre: src.pre };
    }
  }
  return out;
}

/**
 * One component of a compound-assignment op — mirrors expressions.ts's
 * compoundOp (same formulas; that function is module-private and takes the
 * bare op). Returns a self-contained assignment expression (parens included),
 * valid both as a statement and inside a comma expression. The int/uint wrap
 * sits inside the outer parens so the whole assignment is an atom.
 */
function compoundOpExpr(op: string, target: string, rhs: string, base: string): string {
  const isU = base === 'uint';
  const isI = base === 'int';
  switch (op) {
    case '+':
      return isU
        ? `(${target} = (((${target}) + (${rhs})) >>> 0))`
        : isI
          ? `(${target} = (((${target}) + (${rhs})) | 0))`
          : `(${target} = ${target} + ${rhs})`;
    case '-':
      return isU
        ? `(${target} = (((${target}) - (${rhs})) >>> 0))`
        : isI
          ? `(${target} = (((${target}) - (${rhs})) | 0))`
          : `(${target} = ${target} - ${rhs})`;
    case '*':
      return isU
        ? `(${target} = ((Math.imul(${target}, ${rhs})) >>> 0))`
        : isI
          ? `(${target} = (((${target}) * (${rhs})) | 0))`
          : `(${target} = ${target} * ${rhs})`;
    case '/':
      return isU
        ? `(${target} = (((${target}) / (${rhs})) >>> 0))`
        : isI
          ? `(${target} = (((${target}) / (${rhs})) | 0))`
          : `(${target} = ${target} / ${rhs})`;
    case '%':
      return isU
        ? `(${target} = (((${target}) % (${rhs})) >>> 0))`
        : isI
          ? `(${target} = (((${target}) % (${rhs})) | 0))`
          : `(${target} = ${target} % ${rhs})`;
    case '<<':
      return isU
        ? `(${target} = (((${target}) << ((${rhs}) >>> 0)) >>> 0))`
        : `(${target} = (((${target}) << ((${rhs}) >>> 0)) | 0))`;
    case '>>':
      return isU
        ? `(${target} = (((${target}) >>> ((${rhs}) >>> 0)) >>> 0))`
        : `(${target} = (((${target}) >> ((${rhs}) >>> 0)) | 0))`;
    case '&':
      return isU
        ? `(${target} = (((${target}) & (${rhs})) >>> 0))`
        : `(${target} = (((${target}) & (${rhs})) | 0))`;
    case '^':
      return isU
        ? `(${target} = (((${target}) ^ (${rhs})) >>> 0))`
        : `(${target} = (((${target}) ^ (${rhs})) | 0))`;
    case '|':
      return isU
        ? `(${target} = (((${target}) | (${rhs})) >>> 0))`
        : `(${target} = (((${target}) | (${rhs})) | 0))`;
    default:
      throw new Error(`codegen: bad compound op '${op}'`);
  }
}

/** One JS expression for the for-update slot. The slot must stay a SINGLE
 *  expression (JS `continue` jumps to it — moving the update into the body
 *  would break GLSL continue semantics). Assignments / ++ / -- are emitted
 *  via the lvalue protocol folded into one comma expression: prelude and
 *  copyBack statements become comma terms, so pres re-run exactly once per
 *  iteration evaluation (the GLSL model). Non-assign updates fall back to
 *  condString. */
function updateString(e: Expr, env: CodegenEnv): string {
  const isAssign = e.kind === 'assign';
  const isIncDec = e.kind === 'unary' && (e.op === '++' || e.op === '--');
  if (!isAssign && !isIncDec) return condString(e, env);
  const lv = emitLValue(isAssign ? e.target : e.operand, env);
  const parts: string[] = [];
  if (lv.prelude) parts.push(preludeToComma(lv.prelude));
  if (isAssign) {
    const rawRhs = emitExpr(e.value, env);
    // mat×mat '*=' is a MATRIX PRODUCT (matrixCompoundMul) — intercept before
    // the component-wise lowering (regression: matrix-compound-multiply CTS
    // page rendered black). Snapshot/materialization statements fold as comma
    // terms, so pres re-run exactly once per iteration evaluation.
    const isMatMul =
      e.op !== '=' && e.op[0] === '*' && lv.type.kind === 'matrix' && e.value.resolvedType!.kind === 'matrix';
    const base = scalarBaseOf(lv.type);
    if (base === null || base === 'bool') {
      throw new Error('codegen: cannot compound-assign a non-scalar-shaped value');
    }
    if (isMatMul) {
      const mm = matrixCompoundMul(env, lv.targets, lv.dualTargets, lv.type, rawRhs, e.value.resolvedType!);
      for (const p of mm.pre) parts.push(p);
      for (const w of mm.writes) parts.push(w);
    } else {
      const conv = convertPreserving(rawRhs, e.value.resolvedType!, lv.type);
      for (let c = 0; c < lv.targets.length; c++) {
        const cv = conv[c];
        const rv = cv.pre && cv.pre.length > 0 ? foldPre(cv.pre, cv.v) : cv.v;
        // Dual mode, float target: dualWrite emits the triple update as one
        // comma term (linear ops only; non-linear compounds throw — C5a2).
        if (e.op === '=') {
          parts.push(
            env.dual && lv.dualTargets && lv.dualTargets[c]
              ? env.dualWrite(lv.targets[c], lv.dualTargets[c], { ...cv, v: rv })
              : `${lv.targets[c]} = ${rv}`,
          );
        } else {
          const cop = e.op.replace('=', '');
          parts.push(
            env.dual && lv.dualTargets && base === 'float' && lv.dualTargets[c]
              ? env.dualWrite(lv.targets[c], lv.dualTargets[c], { ...cv, v: rv }, cop)
              : compoundOpExpr(cop, lv.targets[c], rv, base),
          );
        }
      }
    }
  } else {
    const delta = e.op === '++' ? '1' : '-1';
    const base = scalarBaseOf(lv.type);
    if (base === null || base === 'bool') throw new Error('codegen: cannot increment a bool');
    for (const tgt of lv.targets) {
      parts.push(
        base === 'float'
          ? `${tgt} = ${tgt} + ${delta}`
          : base === 'int'
            ? `${tgt} = ((${tgt} + ${delta}) | 0)`
            : `${tgt} = ((${tgt} + ${delta}) >>> 0)`,
      );
    }
  }
  if (lv.copyBack) parts.push(preludeToComma(lv.copyBack));
  return parts.length === 1 ? parts[0] : `(${parts.join(', ')})`;
}

/** `'t0 = i; t1 = j;'` (emitLValue format) → `'t0 = i, t1 = j'` (comma terms). */
function preludeToComma(s: string): string {
  return s.replace(/; /g, ', ').replace(/;$/, '');
}

/** One JS expression for a condition/update slot: pres fold INLINE so they
 *  re-run per evaluation (GLSL evaluation model). */
function condString(e: Expr, env: CodegenEnv): string {
  const v = emitExpr(e, env)[0];
  return v.pre && v.pre.length > 0 ? foldPre(v.pre, v.v) : v.v;
}

/** Emit deduped pres (by array identity — multi-component results share one
 *  pre array) as statement lines, in order, before the value is used. */
function emitPres(out: string[], vals: Value[]): void {
  const seen = new Set<string[]>();
  for (const v of vals) {
    if (v.pre && v.pre.length > 0 && !seen.has(v.pre)) {
      seen.add(v.pre);
      for (const p of v.pre) out.push(`${p};`);
    }
  }
}

function indent(lines: string[]): string[] {
  return lines.map((l) => '  ' + l);
}
