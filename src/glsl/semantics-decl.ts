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
  InterfaceBlockDecl, Stmt, TranslationUnit, VarDeclarator,
} from './ast.js';
import type { GLSLType, Precision, TypeQualifiers } from './types.js';
import { isFloat, isIntegral } from './types.js';
import { analyzeProgram, SemContext } from './semantics.js';
import type { CompileError } from './compiler.js';
import type {
  AttributeDecl, OutputDecl, ShaderInfo, ShaderUses, UniformBlockDecl,
  UniformDecl, VaryingDecl,
} from './compiler.js';

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
      derivatives: false,
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
      default:
        break; // structs/functions/extensions/invariant: no ShaderInfo effect
    }
  }
  scanUses(ast, ctx, info.uses, info);
  return info;
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
    type: element,
    arraySize,
    // Integral varyings are flat-only (checked above); record flat as implied.
    flat: q.interpolation === 'flat' || isIntegral(element),
    centroid: q.centroid === true,
    noperspective: q.interpolation === 'noperspective',
    invariant: q.invariant === true,
  };
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
  for (const decl of d.declarators) {
    if (decl.name === '') continue; // parser error-recovery placeholder
    const { element, arraySize } = declaratorInfo(base, decl);
    const line = decl.loc.line;
    const name = decl.name;
    switch (q.storage) {
      case 'attribute': // ES 1.00 vertex input (a 3.00 `attribute` never parses)
        if (ctx.stage !== 'VERTEX') {
          ctx.error(line, "'attribute' : only valid in vertex shaders");
        } else if (ctx.version === 100) {
          if (!isFloat(element)) {
            ctx.error(line, `'${name}' : attribute variables must have a float type in GLSL ES 1.00`);
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

/** ShaderInfo for an ES 3.00 uniform block. */
function analyzeInterfaceBlock(d: InterfaceBlockDecl, ctx: SemContext, info: ShaderInfo): void {
  const q = d.qualifiers;
  const members: { name: string; type: GLSLType; precision: Precision | null }[] = [];
  for (const m of d.members) {
    const mt = m.type.resolved;
    if (mt === undefined || mt.kind === 'void') continue; // error already reported
    members.push({ name: m.name, type: mt, precision: precisionOf(m.type.qualifiers, mt, ctx) });
  }
  let arraySize = 1;
  if (d.instanceName !== null && d.instanceName !== '') {
    const dim = d.arrayDims[0];
    const v = dim?.constValue;
    if (typeof v === 'number' && v > 0) arraySize = v;
  }
  const blk: UniformBlockDecl = {
    name: d.blockName,
    instanceName: d.instanceName,
    arraySize,
    binding: q.layout?.binding ?? null,
    members,
  };
  info.uniformBlocks.push(blk);
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
      default:
        break;
    }
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
            written.add(w.root);
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
          written.add(w.root);
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
      case 'member':
        visit(e.object);
        return;
      case 'comma':
        for (const x of e.exprs) visit(x);
        return;
    }
  };

  const walkStmt = (s: Stmt): void => {
    switch (s.kind) {
      case 'compound':
        for (const st of s.body) walkStmt(st);
        return;
      case 'decl-stmt':
        for (const d of s.declarators) {
          for (const dim of d.arrayDims) visit(dim);
          if (d.init !== null) visit(d.init);
        }
        return;
      case 'expr-stmt':
        if (s.expr !== null) visit(s.expr);
        return;
      case 'if':
        visit(s.cond);
        walkStmt(s.then);
        if (s.else !== null) walkStmt(s.else);
        return;
      case 'for':
        if (s.init !== null) walkStmt(s.init);
        if (s.cond !== null) visit(s.cond);
        if (s.update !== null) visit(s.update);
        walkStmt(s.body);
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
        walkStmt(d.body);
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
