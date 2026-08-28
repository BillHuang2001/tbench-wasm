/**
 * semantics-decl.ts — declaration rules, ShaderInfo assembly and the
 * ShaderUses scan (the layer on top of the semantics CORE).
 *
 * The CORE (semantics.ts) handles symbol tables, expressions, statements,
 * the global pre-pass and recursion detection. This module adds a SECOND
 * in-order pass over the translation unit that:
 * - enforces the storage-qualifier stage rules (attribute/varying/in/out per
 *   stage and version, 1.00 float-only attributes/varyings, 3.00 integral
 *   varyings must be flat, layout(location=) values);
 * - validates fragment outputs (1.00: gl_FragColor XOR gl_FragData with
 *   GL_EXT_draw_buffers index limits; 3.00: `out` must be vec4/ivec4/uvec4);
 * - assembles ShaderInfo (attributes/varyings/uniforms/uniformBlocks/outputs
 *   in DECLARATION ORDER) — the linker's input;
 * - scans the annotated AST for ShaderUses (pointSize/fragCoord/frontFacing/
 *   pointCoord/fragDepth/vertexId/instanceId/derivatives);
 * - exposes `analyze(ast, opts)` — the entry compileShader wires.
 *
 * Float-precision defaults are tracked in SemContext and checked DURING the
 * core pass (that is where declarations happen, in order); this pass replays
 * the precision statements only to read the defaults for ShaderInfo.
 */
import type {
  CallExpr, Expr, FunctionDefinition, GlobalVarDecl, IdentifierExpr,
  InterfaceBlockDecl, Stmt, StructDecl, TranslationUnit, TypeSpec, VarDeclarator,
} from './ast.js';
import type { GLSLType, Precision, StorageClass, TypeQualifiers } from './types.js';
import { isFloat, isIntegral, typeName } from './types.js';
import { analyzeProgram, SemContext } from './semantics.js';
import type { CompileError } from './compiler.js';
import type {
  AttributeDecl, OutputDecl, ShaderInfo, ShaderUses, UniformBlockDecl,
  UniformDecl, VaryingDecl,
} from './compiler.js';
import {
  builtinConstants, builtinSignatures, builtinVariables,
  extensionConstants, extensionFunctions, extensionVariables,
} from './builtins/index.js';

/* ------------------------------------------------------------------ */
/* Public entry                                                        */
/* ------------------------------------------------------------------ */

/**
 * Run the full semantic analysis of one shader (core pass + declaration
 * rules + ShaderInfo assembly). `opts.extensions` = the extensions ENABLED
 * in the shader (the preprocessor's result: #extension require|enable).
 * Returns the resolved declaration summaries on success, or the collected
 * 1-based errors.
 */
export function analyze(
  ast: TranslationUnit,
  opts: { type: 'VERTEX' | 'FRAGMENT'; extensions: Set<string> },
): { ok: true; info: ShaderInfo } | { ok: false; errors: CompileError[] } {
  const ctx = new SemContext(ast.version, opts.type, new Set(opts.extensions));
  analyzeProgram(ast, ctx);
  const info = analyzeDeclarations(ast, ctx);
  if (ctx.errors.length > 0) return { ok: false, errors: ctx.errors };
  return { ok: true, info };
}

/* ------------------------------------------------------------------ */
/* Declaration pass (in source order)                                  */
/* ------------------------------------------------------------------ */

function analyzeDeclarations(ast: TranslationUnit, ctx: SemContext): ShaderInfo {
  const info: ShaderInfo = {
    attributes: [],
    varyings: [],
    uniforms: [],
    uniformBlocks: [],
    outputs: [],
    uses: {
      pointSize: false,
      fragCoord: false,
      frontFacing: false,
      pointCoord: false,
      fragDepth: false,
      vertexId: false,
      instanceId: false,
      drawId: false,
      derivatives: false,
      depthRange: false,
    },
  };
  ctx.initDefaultPrecisions(); // replay precision statements in order
  for (const d of ast.declarations) {
    switch (d.kind) {
      case 'precision-decl':
        ctx.defaultPrecisions.set(d.base, d.precision);
        break;
      case 'global-var-decl':
        analyzeGlobalDecl(d, ctx, info);
        break;
      case 'interface-block':
        analyzeInterfaceBlock(d, ctx, info);
        break;
      case 'struct-decl':
        checkStructDecl(d, ctx);
        break;
      default:
        break; // functions/extensions/invariant: no ShaderInfo effect
    }
  }
  // Whole-translation-unit rules (reserved `gl_` prefixes, main() signature,
  // prototype-vs-definition parameter qualifiers).
  checkGlPrefix(ast, ctx);
  checkMainRules(ast, ctx);
  checkParamQualifierMatch(ast, ctx);
  scanUses(ast, ctx, info.uses, info);
  return info;
}

/* ------------------------------------------------------------------ */
/* Whole-translation-unit rules (reserved names, main, prototypes)     */
/* ------------------------------------------------------------------ */

/**
 * Every builtin name visible in this shader (variables, gl_Max* constants
 * and function names for the version + enabled extensions) — the whitelist
 * for the `gl_` reserved-prefix rule. Mirrors registerBuiltins (semantics.ts)
 * exactly: extension-gated entries count only when the extension is enabled.
 */
function builtinNameSet(ctx: SemContext): Set<string> {
  const names = new Set<string>();
  for (const v of builtinVariables(ctx.version)) {
    if (v.extension !== undefined && !ctx.enabledExtensions.has(v.extension)) continue;
    names.add(v.name);
  }
  for (const v of extensionVariables) {
    if (v.extension === undefined || ctx.enabledExtensions.has(v.extension)) names.add(v.name);
  }
  for (const c of builtinConstants(ctx.version)) names.add(c.name);
  for (const c of extensionConstants) {
    if (c.extension === undefined || ctx.enabledExtensions.has(c.extension)) names.add(c.name);
  }
  for (const s of builtinSignatures(ctx.version)) names.add(s.name);
  for (const s of extensionFunctions) {
    if (s.extension === undefined || ctx.enabledExtensions.has(s.extension)) names.add(s.name);
  }
  return names;
}

/**
 * GLSL ES §3.7 (reserved words): identifiers starting with `gl_` are reserved
 * for builtins — a user declaration of ANY `gl_*` name outside the visible
 * builtin set is a compile error (shader-with-invalid-identifier.frag.html,
 * ogles build identifier2). Runs over every declaration site: global
 * variables, struct types + members, interface blocks, function names +
 * parameters, and locals inside function bodies.
 */
function checkGlPrefix(ast: TranslationUnit, ctx: SemContext): void {
  const builtins = builtinNameSet(ctx);
  const check = (name: string, line: number): void => {
    if (name.startsWith('gl_') && !builtins.has(name)) {
      ctx.error(line, `'${name}' : identifiers starting with 'gl_' are reserved`);
    }
  };
  const walkStmt = (s: Stmt): void => {
    switch (s.kind) {
      case 'compound':
        for (const st of s.body) walkStmt(st);
        return;
      case 'decl-stmt':
        if (s.type.base.kind === 'struct-definition') {
          for (const m of s.type.base.members) check(m.name, m.loc.line);
        }
        for (const d of s.declarators) check(d.name, d.loc.line);
        return;
      case 'if':
        walkStmt(s.then);
        if (s.else !== null) walkStmt(s.else);
        return;
      case 'for':
        if (s.init !== null) walkStmt(s.init);
        walkStmt(s.body);
        return;
      case 'while':
      case 'do-while':
      case 'switch':
        walkStmt(s.body);
        return;
      default:
        return;
    }
  };
  for (const d of ast.declarations) {
    switch (d.kind) {
      case 'global-var-decl':
        if (d.type.base.kind === 'struct-definition') {
          for (const m of d.type.base.members) check(m.name, m.loc.line);
        }
        for (const dec of d.declarators) check(dec.name, dec.loc.line);
        break;
      case 'struct-decl':
        check(d.name, d.loc.line);
        for (const m of d.members) check(m.name, m.loc.line);
        break;
      case 'interface-block':
        check(d.blockName, d.loc.line);
        if (d.instanceName !== null) check(d.instanceName, d.loc.line);
        for (const m of d.members) check(m.name, m.loc.line);
        break;
      case 'function-prototype':
        check(d.name, d.loc.line);
        for (const p of d.params) check(p.name, p.loc.line);
        break;
      case 'function-definition':
        check(d.prototype.name, d.loc.line);
        for (const p of d.prototype.params) check(p.name, p.loc.line);
        walkStmt(d.body);
        break;
      default:
        break;
    }
  }
}

/** GLSL ES §6.1: `main` must take no parameters and return void. */
function checkMainRules(ast: TranslationUnit, ctx: SemContext): void {
  for (const d of ast.declarations) {
    if (d.kind !== 'function-definition' || d.prototype.name !== 'main') continue;
    if (d.prototype.params.length > 0) {
      ctx.error(d.loc.line, "'main' : main function cannot take parameters");
    }
    const rt = d.prototype.returnType;
    const isVoid = rt.base.kind === 'type-name' && rt.base.name === 'void';
    if (!isVoid) {
      ctx.error(d.loc.line, "'main' : main function must return void");
    }
  }
}

/** Type key of one param/return type (resolved name when available, else the
 *  AST base name). Used ONLY to pair a definition with its prototype —
 *  approximate matching is fine because the semantics core already rejected
 *  exact type mismatches. */
function paramTypeKey(t: TypeSpec): string {
  if (t.resolved !== undefined) return typeName(t.resolved);
  if (t.base.kind === 'type-name') return t.base.name;
  return '';
}

/** Normalized param storage for the qualifier-mismatch comparison: absent and
 *  `const`-only both mean a plain input parameter (`const int` ≡ `const in
 *  int` ≡ `in int` — the parser normalizes `const in` to `in`). */
function normParamStorage(s: StorageClass | undefined): string {
  return s === undefined || s === 'const' ? 'in' : s;
}

/**
 * GLSL ES §6.1.1: a function DEFINITION's parameter qualifiers must match its
 * prior PROTOTYPE declaration — `void f(inout int i);` followed by
 * `void f(in int i) {}` is an error (ogles build function9). The semantics
 * core matches signatures by type only (sameSignature), so the qualifier
 * comparison is done here at the AST level: for each definition, find the
 * preceding prototype with the same name, return type, arity and param
 * types; compare per-param storage.
 */
function checkParamQualifierMatch(ast: TranslationUnit, ctx: SemContext): void {
  const protos: { name: string; ret: string; params: { storage: StorageClass | undefined; key: string }[]; line: number }[] = [];
  for (const d of ast.declarations) {
    if (d.kind === 'function-prototype') {
      protos.push({
        name: d.name,
        ret: paramTypeKey(d.returnType),
        line: d.loc.line,
        params: d.params.map((p) => ({ storage: p.type.qualifiers.storage, key: paramTypeKey(p.type) })),
      });
      continue;
    }
    if (d.kind !== 'function-definition') continue;
    const def = d.prototype;
    const defParams = def.params.map((p) => ({ storage: p.type.qualifiers.storage, key: paramTypeKey(p.type) }));
    const defRet = paramTypeKey(def.returnType);
    for (const pr of protos) {
      if (pr.name !== def.name || pr.ret !== defRet || pr.params.length !== defParams.length) continue;
      let same = true;
      for (let i = 0; i < pr.params.length; i++) {
        if (pr.params[i].key !== defParams[i].key) {
          same = false;
          break;
        }
      }
      if (!same) continue;
      for (let i = 0; i < pr.params.length; i++) {
        if (normParamStorage(pr.params[i].storage) !== normParamStorage(defParams[i].storage)) {
          ctx.error(d.loc.line, `'${def.name}' : parameter qualifiers do not match the function declaration`);
          break;
        }
      }
      break;
    }
  }
}

/** Element type + outermost array size of one declarator (dims already folded
 * by the core pass; errors there fall back to size 1). */
function declaratorInfo(base: GLSLType, d: VarDeclarator): { element: GLSLType; arraySize: number } {
  if (d.arrayDims.length === 0) return { element: base, arraySize: 1 };
  const dim = d.arrayDims[0];
  const v = dim.constValue;
  return { element: base, arraySize: typeof v === 'number' && v > 0 ? v : 1 };
}

/** Full declared type of one declarator (array structure intact). */
function fullType(base: GLSLType, dims: Expr[]): GLSLType {
  let t = base;
  for (let i = dims.length - 1; i >= 0; i--) {
    const dim = dims[i];
    const v = dim.constValue;
    t = { kind: 'array', element: t, size: typeof v === 'number' && v > 0 ? v : 1 };
  }
  return t;
}

/** The precision of a declared type: explicit qualifier, else the default in
 * effect at the declaration (arrays unwrap to their element; structs/void
 * have no single precision → null). */
function precisionOf(q: TypeQualifiers, type: GLSLType, ctx: SemContext): Precision | null {
  if (q.precision !== undefined) return q.precision;
  let t = type;
  while (t.kind === 'array') t = t.element;
  switch (t.kind) {
    case 'scalar':
    case 'vector':
      return t.base === 'float' || t.base === 'int' ? (ctx.defaultPrecisions.get(t.base) ?? null) : null;
    case 'matrix':
      return ctx.defaultPrecisions.get('float') ?? null;
    case 'sampler':
      return ctx.defaultPrecisions.get(t.sampler) ?? null;
    default:
      return null; // struct / void
  }
}

/** 3.00 vertex inputs: float/int/uint scalar/vector/matrix (+ arrays of those). */
function isValidInputType(t: GLSLType): boolean {
  if (t.kind === 'array') return isValidInputType(t.element);
  if (t.kind === 'scalar' || t.kind === 'vector') return t.base !== 'bool';
  return t.kind === 'matrix';
}

/** 3.00 fragment outputs: exactly vec4 / ivec4 / uvec4 (per the OutputDecl
 * contract — FLOAT_VEC4 / INT_VEC4 / UINT_VEC4). */
function isVec4(t: GLSLType): boolean {
  return t.kind === 'vector' && t.size === 4 && (t.base === 'float' || t.base === 'int' || t.base === 'uint');
}

/** GLSL ES 3.00: integral varyings (vertex outs / fragment ins) must be flat. */
function checkFlatIntegral(ctx: SemContext, name: string, element: GLSLType, q: TypeQualifiers, line: number): void {
  if (isIntegral(element) && q.interpolation !== 'flat') {
    ctx.error(line, `'${name}' : integral varying variables must be declared with 'flat'`);
  }
}

function attrOf(name: string, element: GLSLType, arraySize: number, q: TypeQualifiers): AttributeDecl {
  return { name, type: element, arraySize, location: q.layout?.location ?? null };
}

function varyingOf(name: string, element: GLSLType, arraySize: number, q: TypeQualifiers): VaryingDecl {
  return {
    name,
    blockName: null,
    type: element,
    arraySize,
    // Integral varyings are flat-only (checked above); record flat as implied.
    flat: q.interpolation === 'flat' || isIntegral(element),
    centroid: q.centroid === true,
    noperspective: q.interpolation === 'noperspective',
    invariant: q.invariant === true,
    // Set to true by scanUses when the FRAGMENT shader reads the varying.
    used: false,
  };
}

/** True for bool scalars/vectors (`bool`, `bvec2..4`). */
function isBool(t: GLSLType): boolean {
  return (t.kind === 'scalar' || t.kind === 'vector') && t.base === 'bool';
}

/**
 * Struct-nesting depth of a type: 0 for non-structs (arrays unwrap to their
 * element), 1 + max(member depth) for structs. Memoized by struct name
 * (recursive structs are rejected at semantics, so the memo is safe; an
 * in-progress name guard prevents pathological loops).
 */
function structDepth(t: GLSLType, memo: Map<string, number>): number {
  if (t.kind === 'array') return structDepth(t.element, memo);
  if (t.kind !== 'struct') return 0;
  const cached = memo.get(t.name);
  if (cached !== undefined) return cached;
  memo.set(t.name, 0); // in-progress guard (defensive)
  let inner = 0;
  for (const m of t.members) {
    const md = structDepth(m.type, memo);
    if (md > inner) inner = md;
  }
  const depth = 1 + inner;
  memo.set(t.name, depth);
  return depth;
}

/** Bare `struct S { ... };` rules: empty struct + WebGL nesting limit. */
function checkStructDecl(d: StructDecl, ctx: SemContext): void {
  if (d.members.length === 0) {
    ctx.error(d.loc.line, `'struct' : structure must have at least one member`);
  }
  if (ctx.version !== 100) return;
  const members: { name: string; type: GLSLType }[] = [];
  for (const m of d.members) {
    const mt = m.type.resolved;
    if (mt !== undefined) members.push({ name: m.name, type: mt });
  }
  const t: GLSLType = { kind: 'struct', name: d.name, members };
  if (structDepth(t, new Map()) > 4) {
    ctx.error(d.loc.line, `'struct' : structure nesting exceeds the maximum of 4 levels`);
  }
}

/** Stage/storage rules + ShaderInfo collection for one global declaration. */
function analyzeGlobalDecl(d: GlobalVarDecl, ctx: SemContext, info: ShaderInfo): void {
  const q = d.type.qualifiers;
  const base = d.type.resolved;
  if (base === undefined) return; // type-resolution error already reported
  const loc = q.layout?.location;
  if (loc !== undefined && (!Number.isInteger(loc) || loc < 0)) {
    ctx.error(d.loc.line, "'layout(location=)' : location must be a non-negative integer");
  }
  // GLSL ES §4.5.3: precision qualifiers apply to float/int/sampler types
  // only — `mediump bool` / `mediump bvecN` are illegal on globals too
  // (conformance/glsl/misc/boolean_precision.html).
  if (q.precision !== undefined && isBool(base)) {
    ctx.error(d.loc.line, `'${q.precision}' : precision qualifiers are not allowed on bool types`);
  }
  if (d.type.base.kind === 'struct-definition' && base.kind === 'struct') {
    // Inline struct definitions: `struct S { ... } v;` / `struct { ... } v;` —
    // empty structs are illegal (ogles build struct7) and the WebGL 1.0
    // nesting limit applies (struct-nesting-exceeds-maximum.html).
    if (d.type.base.members.length === 0) {
      ctx.error(d.loc.line, `'struct' : structure must have at least one member`);
    }
    if (ctx.version === 100 && structDepth(base, new Map()) > 4) {
      ctx.error(d.loc.line, `'struct' : structure nesting exceeds the maximum of 4 levels`);
    }
  }
  for (const decl of d.declarators) {
    if (decl.name === '') continue; // parser error-recovery placeholder
    const { element, arraySize } = declaratorInfo(base, decl);
    const line = decl.loc.line;
    const name = decl.name;
    // GLSL ES 1.00 Appendix A §5: arrays of arrays are 3.00-only (ogles build
    // array1); array sizes must be CONSTANT INTEGRAL expressions — a float
    // const size is illegal even when integral-valued (ogles build array6).
    // Storage-qualified globals (uniform/varying/attribute) are read-only
    // inputs — initializers are illegal in 1.00 (ogles dataType2/dataType3/
    // varying3).
    if (ctx.version === 100) {
      if (decl.arrayDims.length > 1) {
        ctx.error(line, `'[' : arrays of arrays are not allowed in GLSL ES 1.00`);
      }
      for (const dim of decl.arrayDims) {
        const dt = dim.resolvedType;
        if (dt !== undefined && dt.kind === 'scalar' && dt.base === 'float') {
          ctx.error(dim.loc.line, 'array size must be a constant integer expression');
        }
      }
      if (decl.init !== null && (q.storage === 'uniform' || q.storage === 'varying' || q.storage === 'attribute')) {
        ctx.error(line, `'${name}' : ${q.storage} variables cannot be initialized in GLSL ES 1.00`);
      }
      // WebGL 1.0 limit: at most 4 levels of struct nesting.
      if ((base.kind === 'struct' || element.kind === 'struct') && structDepth(element, new Map()) > 4) {
        ctx.error(line, `'struct' : structure nesting exceeds the maximum of 4 levels`);
      }
    }
    switch (q.storage) {
      case 'attribute': // ES 1.00 vertex input (a 3.00 `attribute` never parses)
        if (ctx.stage !== 'VERTEX') {
          ctx.error(line, "'attribute' : only valid in vertex shaders");
        } else if (ctx.version === 100) {
          if (!isFloat(element)) {
            ctx.error(line, `'${name}' : attribute variables must have a float type in GLSL ES 1.00`);
          }
          // GLSL ES 1.00 Appendix A §5: attribute variables cannot be arrays
          // (conformance/glsl/misc/shader-with-attrib-array.vert.html, ogles
          // build attribute2_vert).
          if (arraySize !== 1) {
            ctx.error(line, `'${name}' : attribute variables cannot be arrays in GLSL ES 1.00`);
          }
        } else if (!isValidInputType(element)) {
          ctx.error(line, `'${name}' : attribute variables cannot have a boolean type`);
        }
        if (ctx.stage === 'VERTEX') info.attributes.push(attrOf(name, element, arraySize, q));
        break;
      case 'in': // ES 3.00: vertex input (attribute) or fragment input (varying)
        if (ctx.stage === 'VERTEX') {
          if (!isValidInputType(element)) {
            ctx.error(line, `'${name}' : attribute variables cannot have a boolean type`);
          }
          info.attributes.push(attrOf(name, element, arraySize, q));
        } else {
          checkFlatIntegral(ctx, name, element, q, line);
          info.varyings.push(varyingOf(name, element, arraySize, q));
        }
        break;
      case 'varying': // ES 1.00: both stages
        if (!isFloat(element)) {
          ctx.error(line, `'${name}' : varying variables must have a float type in GLSL ES 1.00`);
        }
        info.varyings.push(varyingOf(name, element, arraySize, q));
        break;
      case 'out': // ES 3.00: vertex output (varying) or fragment output
        if (ctx.stage === 'VERTEX') {
          checkFlatIntegral(ctx, name, element, q, line);
          info.varyings.push(varyingOf(name, element, arraySize, q));
        } else {
          if (arraySize !== 1 || !isVec4(element)) {
            ctx.error(line, `'${name}' : fragment shader outputs must be vec4, ivec4 or uvec4`);
          }
          info.outputs.push({
            name,
            index: null,
            location: q.layout?.location ?? null,
            type: element,
          });
        }
        break;
      case 'uniform': // default-block uniforms only (blocks are separate)
        info.uniforms.push({
          name,
          type: fullType(base, decl.arrayDims),
          precision: precisionOf(q, element, ctx),
          binding: q.layout?.binding ?? null,
        });
        break;
      default:
        break; // const / plain globals: no GL resource
    }
  }
}

/** ShaderInfo for an ES 3.00 interface block.
 *  - `uniform` storage (or absent storage): the UBO path — one UniformBlockDecl.
 *  - `out` (vertex) / `in` (fragment): a VARYING interface block — one
 *    ShaderInfo.varyings entry PER MEMBER, keyed '<instance>.<member>'
 *    (bare member name for instance-less blocks) with blockName = block name,
 *    so the linker can match members across stages whose instance names
 *    differ. These must NOT land in uniformBlocks.
 *  - other storage/stage combos (vertex `in`, fragment `out` blocks): recorded
 *    nowhere — the linker rejects them via the AST scan. */
function analyzeInterfaceBlock(d: InterfaceBlockDecl, ctx: SemContext, info: ShaderInfo): void {
  const q = d.qualifiers;
  const members: { name: string; type: GLSLType; precision: Precision | null; qualifiers: TypeQualifiers; line: number }[] = [];
  for (const m of d.members) {
    const mt = m.type.resolved;
    if (mt === undefined || mt.kind === 'void') continue; // error already reported
    members.push({
      name: m.name,
      type: mt,
      precision: precisionOf(m.type.qualifiers, mt, ctx),
      qualifiers: m.type.qualifiers,
      line: m.loc.line,
    });
  }
  let arraySize = 1;
  if (d.instanceName !== null && d.instanceName !== '') {
    const dim = d.arrayDims[0];
    const v = dim?.constValue;
    if (typeof v === 'number' && v > 0) arraySize = v;
  }
  const storage = q.storage;
  const isVaryingBlock =
    (storage === 'out' && ctx.stage === 'VERTEX') || (storage === 'in' && ctx.stage === 'FRAGMENT');
  if (!isVaryingBlock) {
    // Uniform blocks (or unsupported combos — recorded nowhere, linker rejects).
    if (storage === 'uniform' || storage === undefined) {
      const blk: UniformBlockDecl = {
        name: d.blockName,
        instanceName: d.instanceName,
        arraySize,
        binding: q.layout?.binding ?? null,
        members: members.map((m) => ({ name: m.name, type: m.type, precision: m.precision })),
      };
      info.uniformBlocks.push(blk);
    }
    return;
  }
  // Varying block: per-member varying entries (integral members must be flat).
  const prefix = d.instanceName !== null && d.instanceName !== '' ? `${d.instanceName}.` : '';
  for (const m of members) {
    let element = m.type;
    let memberArraySize = 1;
    while (element.kind === 'array') {
      memberArraySize *= element.size ?? 1;
      element = element.element;
    }
    const mq = m.qualifiers;
    checkFlatIntegral(ctx, m.name, element, mq, m.line);
    info.varyings.push({
      name: prefix + m.name,
      blockName: d.blockName,
      type: element,
      arraySize: arraySize * memberArraySize,
      flat: mq.interpolation === 'flat' || isIntegral(element),
      centroid: mq.centroid === true,
      noperspective: mq.interpolation === 'noperspective',
      invariant: q.invariant === true,
      // Set to true by scanUses when the FRAGMENT shader reads the member.
      used: false,
    });
  }
}

/* ------------------------------------------------------------------ */
/* ShaderUses scan (post-analysis AST walk)                            */
/* ------------------------------------------------------------------ */

/** ES 1.00 texture functions with IMPLICIT LOD (bias variants share names). */
const IMPLICIT_LOD_100: ReadonlySet<string> = new Set(['texture2D', 'texture2DProj', 'textureCube']);
/** ES 3.00 texture functions with IMPLICIT LOD (bias variants share names). */
const IMPLICIT_LOD_300: ReadonlySet<string> = new Set(['texture', 'textureProj', 'textureOffset', 'textureProjOffset']);
/** Derivative functions (fragment stage; gated by GL_OES_standard_derivatives in 1.00). */
const DERIVATIVE_NAMES: ReadonlySet<string> = new Set(['dFdx', 'dFdy', 'fwidth']);

const VEC4_FLOAT: GLSLType = { kind: 'vector', base: 'float', size: 4 };

interface FragOutputState {
  fragColorWritten: boolean;
  fragDataWritten: boolean;
  fragDataIndices: number[];
  fragDataLine: number;
}

/** The base identifier of an lvalue chain (member/index objects), plus the
 * FIRST index expression when the chain starts with one (gl_FragData[N]). */
function writeRoot(e: Expr): { root: IdentifierExpr; firstIndex: Expr | null } | null {
  let node = e;
  let firstIndex: Expr | null = null;
  for (;;) {
    if (node.kind === 'identifier') return { root: node, firstIndex };
    if (node.kind === 'member') {
      node = node.object;
      continue;
    }
    if (node.kind === 'index') {
      if (firstIndex === null) firstIndex = node.index;
      node = node.object;
      continue;
    }
    return null;
  }
}

/** The base identifier of a member/index chain (`b.c` → `b`, `b[0].c` → `b`,
 *  `b.c[0]` → `b`); null when the object is itself a member expression
 *  (nested struct access — the INNER member node then carries the varying
 *  key). Used to reconstruct the '<instance>.<member>' key of a
 *  varying-interface-block member read. */
function baseIdentifier(e: Expr): IdentifierExpr | null {
  let node = e;
  for (;;) {
    if (node.kind === 'identifier') return node;
    if (node.kind === 'index') {
      node = node.object;
      continue;
    }
    return null;
  }
}

/** Scan every expression of the shader for stage-specific builtin use. Only
 * SUCCESSFULLY resolved identifiers/calls count (error-recovery nodes don't
 * set resolvedType). ES 1.00 fragment outputs are finalized here too. */
function scanUses(ast: TranslationUnit, ctx: SemContext, uses: ShaderUses, info: ShaderInfo): void {
  const fragOut: FragOutputState = {
    fragColorWritten: false,
    fragDataWritten: false,
    fragDataIndices: [],
    fragDataLine: 1,
  };
  const written = new Set<IdentifierExpr>(); // write-root identifiers (not reads)

  const recordWrite = (root: IdentifierExpr, firstIndex: Expr | null): void => {
    if (root.resolvedType === undefined) return; // unresolved (error reported)
    const name = root.name;
    if (name === 'gl_PointSize') {
      uses.pointSize = true;
    } else if (name === 'gl_FragDepth' || name === 'gl_FragDepthEXT') {
      uses.fragDepth = true;
    } else if (ctx.version === 100 && ctx.stage === 'FRAGMENT') {
      if (name === 'gl_FragColor') {
        fragOut.fragColorWritten = true;
      } else if (name === 'gl_FragData') {
        fragOut.fragDataWritten = true;
        fragOut.fragDataLine = root.loc.line;
        if (firstIndex === null) return; // direct array write: permissive
        const cv = firstIndex.constValue;
        if (typeof cv === 'number' && Number.isInteger(cv)) {
          fragOut.fragDataIndices.push(cv);
        } else {
          ctx.error(root.loc.line, "'gl_FragData' : index must be a constant integer expression");
        }
      }
    }
  };

  /**
   * Scope replay for shadow detection (cross-scope shadowing is now legal in
   * the CORE — Scope.declare). The scan must mark a fragment varying USED
   * only when the identifier load actually RESOLVES to the varying (a
   * shadowing local/param/struct with the same name must not mark it). We
   * mirror the CORE's scope structure exactly: compound/if-substatement/for
   * scopes, params, and per-declarator declaration AFTER its initializer
   * (declaration-time scoping). The replay is exact for CLEAN shaders (every
   * declaration succeeded, so every registered name really shadows); for
   * shaders with compile errors the used-flag is unobservable (link is never
   * attempted).
   */
  const shadowStack: Set<string>[] = [];
  const pushScope = (): void => {
    shadowStack.push(new Set());
  };
  const popScope = (): void => {
    shadowStack.pop();
  };
  const isShadowed = (name: string): boolean => shadowStack.some((s) => s.has(name));
  const declareLocal = (name: string): void => {
    if (name !== '') shadowStack[shadowStack.length - 1].add(name);
  };

  const recordRead = (id: IdentifierExpr): void => {
    if (id.resolvedType === undefined) return;
    switch (id.name) {
      case 'gl_FragCoord':
        uses.fragCoord = true;
        break;
      case 'gl_FrontFacing':
        uses.frontFacing = true;
        break;
      case 'gl_PointCoord':
        uses.pointCoord = true;
        break;
      case 'gl_VertexID':
        uses.vertexId = true;
        break;
      case 'gl_InstanceID':
        uses.instanceId = true;
        break;
      case 'gl_DrawID':
        uses.drawId = true;
        break;
      case 'gl_DepthRange':
        // Builtin struct uniform (1.00 §7.6 / 3.00 §7.7, BOTH stages). Member
        // reads (gl_DepthRange.near) visit the base identifier here (the
        // `member` case descends into the object), so one case covers direct
        // struct uses and member reads alike.
        uses.depthRange = true;
        break;
      default:
        break;
    }
    // A fragment-varying value load marks the varying USED (the linker only
    // matches used fragment varyings against vertex outputs — native
    // behavior), but only when the load resolves to the varying: a local /
    // param / inner struct with the same name shadows it (Scope.declare now
    // allows cross-scope shadowing), so such loads must NOT mark it.
    if (!isShadowed(id.name)) markVaryingUsed(id.name);
  };

  /** Mark the fragment varying with ShaderInfo key `key` as READ. Keys: plain
   *  varyings + instance-less block members use the bare name; named block
   *  members '<instance>.<member>'. No-op in the vertex stage (the linker
   *  ignores `used` there) and for non-varying names. */
  const markVaryingUsed = (key: string): void => {
    if (ctx.stage !== 'FRAGMENT') return;
    const v = info.varyings.find((vv) => vv.name === key);
    if (v !== undefined) v.used = true;
  };

  const recordCall = (c: CallExpr): void => {
    if (c.resolvedType === undefined || c.callee.kind !== 'identifier') return;
    const name = c.callee.name;
    if (DERIVATIVE_NAMES.has(name)) {
      uses.derivatives = true;
    } else if (ctx.version === 100 ? IMPLICIT_LOD_100.has(name) : IMPLICIT_LOD_300.has(name)) {
      // Implicit-LOD lookups need screen-space derivatives (dual-number mode).
      uses.derivatives = true;
    }
  };

  const visit = (e: Expr): void => {
    switch (e.kind) {
      case 'literal':
        return;
      case 'identifier':
        if (!written.has(e)) recordRead(e);
        return;
      case 'unary': {
        if (e.op === '++' || e.op === '--') {
          const w = writeRoot(e.operand);
          if (w !== null) {
            // Pre/post increment READS the operand as well as writing it (the
            // old value must be loaded) — deliberately not added to `written`,
            // so the operand visit below counts as a varying read.
            recordWrite(w.root, w.firstIndex);
          }
        }
        visit(e.operand);
        return;
      }
      case 'binary':
        visit(e.left);
        visit(e.right);
        return;
      case 'assign': {
        const w = writeRoot(e.target);
        if (w !== null) {
          // Plain `=` is a pure write (target never read); compound
          // assignments (`+=` etc.) read the target as well — keep the root
          // OUT of `written` so the target visit counts as a varying read.
          if (e.op === '=') written.add(w.root);
          recordWrite(w.root, w.firstIndex);
        }
        visit(e.target);
        visit(e.value);
        return;
      }
      case 'ternary':
        visit(e.cond);
        visit(e.whenTrue);
        visit(e.whenFalse);
        return;
      case 'call':
        recordCall(e);
        visit(e.callee);
        for (const a of e.args) visit(a);
        return;
      case 'index':
        visit(e.object);
        visit(e.index);
        return;
      case 'member': {
        // Varying-interface-block member access: entries are keyed
        // '<instance>.<member>' — reconstruct the key from the object chain
        // and mark the varying used, UNLESS the whole chain is a pure
        // `=`-write target (base identifier in `written`) or the base is
        // shadowed by a local (a local struct instance named like the block
        // instance must not mark the varying). The base-identifier visit
        // below already covers plain varyings (incl. swizzles) and
        // instance-less block members (bare names).
        const base = baseIdentifier(e.object);
        if (base !== null && !written.has(base) && !isShadowed(base.name)) {
          markVaryingUsed(`${base.name}.${e.name}`);
        }
        visit(e.object);
        return;
      }
      case 'comma':
        for (const x of e.exprs) visit(x);
        return;
    }
  };

  const walkStmt = (s: Stmt): void => {
    switch (s.kind) {
      case 'compound':
        // Braced block: fresh scope (mirrors analyzeCompound).
        pushScope();
        for (const st of s.body) walkStmt(st);
        popScope();
        return;
      case 'decl-stmt':
        // Mirror the CORE's declaration order (semantics-stmt decl-stmt +
        // declareVariables): the struct TYPE is registered before the
        // declarators, and each declarator's name AFTER its dims+initializer
        // (declaration-time scoping — `float v = v;` reads the OUTER v).
        if (s.type.base.kind === 'struct-definition' && s.type.base.name !== null) {
          declareLocal(s.type.base.name);
        }
        for (const d of s.declarators) {
          for (const dim of d.arrayDims) visit(dim);
          if (d.init !== null) visit(d.init);
          declareLocal(d.name);
        }
        return;
      case 'expr-stmt':
        if (s.expr !== null) visit(s.expr);
        return;
      case 'if':
        visit(s.cond);
        // Selection substatements each get their own scope (mirrors
        // analyzeIf — shader-with-conditional-scoping.html).
        pushScope();
        walkStmt(s.then);
        popScope();
        if (s.else !== null) {
          pushScope();
          walkStmt(s.else);
          popScope();
        }
        return;
      case 'for':
        // The for-init/cond/update/body share the for's own scope (mirrors
        // semantics-stmt analyzeStatement 'for').
        pushScope();
        if (s.init !== null) walkStmt(s.init);
        if (s.cond !== null) visit(s.cond);
        if (s.update !== null) visit(s.update);
        walkStmt(s.body);
        popScope();
        return;
      case 'while':
        visit(s.cond);
        walkStmt(s.body);
        return;
      case 'do-while':
        walkStmt(s.body);
        visit(s.cond);
        return;
      case 'switch':
        visit(s.expr);
        walkStmt(s.body);
        return;
      case 'case':
        if (s.value !== null) visit(s.value);
        return;
      case 'return':
        if (s.value !== null) visit(s.value);
        return;
      case 'break':
      case 'continue':
      case 'discard':
      case 'empty':
        return;
    }
  };

  for (const d of ast.declarations) {
    switch (d.kind) {
      case 'global-var-decl':
        for (const decl of d.declarators) {
          for (const dim of decl.arrayDims) visit(dim);
          if (decl.init !== null) visit(decl.init);
        }
        break;
      case 'function-definition':
        // Function scope: params shadow global names (mirrors
        // analyzeFunctionBody — in-parameter-passed-as-inout-argument-and-
        // global.html, ogles build CorrectFull_vert).
        pushScope();
        for (const p of d.prototype.params) declareLocal(p.name);
        walkStmt(d.body);
        popScope();
        break;
      default:
        break;
    }
  }

  // ES 1.00 fragment outputs: gl_FragColor XOR gl_FragData (GL_EXT_draw_buffers
  // raises gl_MaxDrawBuffers from 1 to 4 — the builtin tables already resized
  // gl_FragData; the index rules are enforced here).
  if (ctx.version === 100 && ctx.stage === 'FRAGMENT') {
    const ext = ctx.enabledExtensions.has('GL_EXT_draw_buffers');
    const maxBuffers = ext ? 4 : 1;
    if (fragOut.fragDataWritten && fragOut.fragColorWritten) {
      ctx.error(fragOut.fragDataLine, "'gl_FragData' : cannot write both gl_FragColor and gl_FragData");
    }
    for (const idx of fragOut.fragDataIndices) {
      if (idx >= maxBuffers) {
        if (!ext) {
          ctx.error(fragOut.fragDataLine, `'gl_FragData' : index ${idx} requires GL_EXT_draw_buffers`);
        } else {
          ctx.error(fragOut.fragDataLine, `'gl_FragData' : index ${idx} out of range [0, ${maxBuffers - 1}]`);
        }
      }
      info.outputs.push({ name: `gl_FragData[${idx}]`, index: idx, location: null, type: VEC4_FLOAT });
    }
    if (fragOut.fragColorWritten) {
      info.outputs.push({ name: 'gl_FragColor', index: null, location: null, type: VEC4_FLOAT });
    }
  }
}
