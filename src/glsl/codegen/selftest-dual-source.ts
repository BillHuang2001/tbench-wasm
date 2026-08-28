/**
 * selftest-dual-source.ts — index-1 fragment-output routing (T1-A590).
 *
 * Dual-source blending (GL_EXT_blend_func_extended / WEBGL_blend_func_extended)
 * needs the SECONDARY fragment color — the output with blend index 1 — to land
 * in `ctx.out.secondary[loc]` (raster feeds it into the SRC1_* blend factors),
 * NOT `ctx.out.color[loc]` (which would clobber the primary). The linker builds
 * a name → index map (CodegenLayout.outputIndices) and codegen routes index-1
 * outputs to the secondary slot:
 *   - 1.00: gl_SecondaryFragColorEXT → location 0, index 1;
 *   - 3.00: `layout(location=N, index=1) out vec4 o1;` → location N, index 1.
 *
 * Every index-1 check FAILS on the pre-fix code (the secondary write lands in
 * `ctx.out.color[loc]`, clobbering the primary — observed CTS failure
 * "was 32,32,32,255" on webgl-blend-func-extended.html).
 *
 * Run: npx tsx src/glsl/codegen/selftest-dual-source.ts
 * Prints "selftest-dual-source: N checks" and exits 0 only when all pass.
 */
import { compileShader, linkProgram } from '../compiler.js';
import type { Program } from '../program.js';

let failures = 0;
let checks = 0;

function check(cond: boolean, msg: string): void {
  checks++;
  if (!cond) {
    failures++;
    console.error('FAIL: ' + msg);
  }
}

function compile(src: string, type: 'VERTEX' | 'FRAGMENT', version: 100 | 300, extensions?: Set<string>) {
  const r = compileShader(src, { type, version, extensions });
  if (!r.ok) throw new Error(`compile failed: ${JSON.stringify(r.errors)}`);
  return r.shader;
}

function fragmentCtx(prog: Program, extra: Record<string, unknown> = {}): any {
  return {
    uniforms: prog.floatStore,
    intUniforms: prog.intStore,
    blockStores: [],
    blockIntStores: [],
    scratch: new Float32Array(Math.max(prog.scratchSize, 16)),
    intScratch: new Int32Array(Math.max(prog.intScratchSize, 16)),
    out: { color: [new Float32Array(4)], secondary: [new Float32Array(4)], fragDepth: 0 },
    discarded: false,
    fragCoord: new Float32Array([0, 0, 0, 1]),
    frontFacing: true,
    pointCoord: new Float32Array([0, 0]),
    varyings: [],
    ...extra,
  };
}

/** Set a default-block float uniform from its Program-model name. */
function setUniform(prog: Program, name: string, vals: number[]): void {
  const u = prog.uniforms.find((x) => x.name === name);
  if (!u) throw new Error(`missing uniform ${name}`);
  for (let i = 0; i < vals.length; i++) prog.floatStore[u.location + i] = vals[i];
}

const PRIMARY = [0.25, 0.5, 0.75, 1.0];
const SECONDARY = [0.125, 0.375, 0.625, 0.875]; // exactly representable in float32

/* ------------------------------------------------------------------ */
/* 1. GLSL ES 1.00: gl_FragColor + gl_SecondaryFragColorEXT            */
/* ------------------------------------------------------------------ */

{
  const fsSrc = `#extension GL_EXT_blend_func_extended : enable
     precision highp float;
     uniform vec4 u_src0;
     uniform vec4 u_src1;
     void main() { gl_FragColor = u_src0; gl_SecondaryFragColorEXT = u_src1; }`;
  // gl/ feeds the WebGL REGISTRY name; the preprocessor aliases it to the GLSL
  // extension name (see preprocessor.ts WEBGL_TO_GLSL_EXTENSION_ALIASES).
  const fs = compile(fsSrc, 'FRAGMENT', 100, new Set(['WEBGL_blend_func_extended']));
  const vs = compile('attribute vec4 a;\nvoid main() { gl_Position = a; }', 'VERTEX', 100);
  const l = linkProgram(vs, fs);
  check(l.ok, `1.00: dual-source pair links (${l.ok ? '' : l.log})`);
  if (l.ok) {
    const prog = l.program;
    check(
      prog.fragment.outputs.length === 2 &&
        prog.fragment.outputs[0].location === 0 && prog.fragment.outputs[0].index === 0 &&
        prog.fragment.outputs[1].location === 0 && prog.fragment.outputs[1].index === 1,
      `1.00: outputs carry index 0 (gl_FragColor) and 1 (gl_SecondaryFragColorEXT) (got ${JSON.stringify(prog.fragment.outputs)})`,
    );
    setUniform(prog, 'u_src0', PRIMARY);
    setUniform(prog, 'u_src1', SECONDARY);
    const fctx = fragmentCtx(prog);
    prog.fragment.run(fctx);
    const c = fctx.out.color[0];
    const s = fctx.out.secondary[0];
    check(
      c[0] === PRIMARY[0] && c[1] === PRIMARY[1] && c[2] === PRIMARY[2] && c[3] === PRIMARY[3],
      `1.00: primary color[0] = u_src0 (got [${Array.from(c).join(', ')}])`,
    );
    check(
      s[0] === SECONDARY[0] && s[1] === SECONDARY[1] && s[2] === SECONDARY[2] && s[3] === SECONDARY[3],
      `1.00: secondary[0] = u_src1 (got [${Array.from(s).join(', ')}])`,
    );
    check(c[0] !== SECONDARY[0] && c[1] !== SECONDARY[1],
      `1.00: secondary write did NOT clobber color[0] (color[0] = [${Array.from(c).join(', ')}])`);
  }
}

// Without the extension (or its WebGL registry-name alias) gl_SecondaryFragColorEXT
// must NOT compile — pins the enablement gate AND the alias as the enabler.
{
  const bad = compileShader(
    '#extension GL_EXT_blend_func_extended : enable\nprecision highp float;\nvoid main() { gl_FragColor = vec4(1.0); gl_SecondaryFragColorEXT = vec4(2.0); }',
    { type: 'FRAGMENT', version: 100 },
  );
  check(!bad.ok, '1.00: gl_SecondaryFragColorEXT without the extension fails to compile');
}

/* ------------------------------------------------------------------ */
/* 2. GLSL ES 3.00: layout(location=0, index=0/1) out vec4             */
/* ------------------------------------------------------------------ */

{
  const fsSrc = `#version 300 es
     #extension GL_EXT_blend_func_extended : enable
     precision highp float;
     layout(location=0, index=0) out vec4 o0;
     layout(location=0, index=1) out vec4 o1;
     uniform vec4 u_src0;
     uniform vec4 u_src1;
     void main() { o0 = u_src0; o1 = u_src1; }`;
  const fs = compile(fsSrc, 'FRAGMENT', 300, new Set(['WEBGL_blend_func_extended']));
  const vs = compile('#version 300 es\nin vec4 a;\nvoid main() { gl_Position = a; }', 'VERTEX', 300);
  const l = linkProgram(vs, fs);
  check(l.ok, `3.00: dual-source pair links (${l.ok ? '' : l.log})`);
  if (l.ok) {
    const prog = l.program;
    check(
      prog.fragment.outputs.length === 2 &&
        prog.fragment.outputs[0].location === 0 && prog.fragment.outputs[0].index === 0 &&
        prog.fragment.outputs[1].location === 0 && prog.fragment.outputs[1].index === 1,
      `3.00: outputs carry index 0/1 (got ${JSON.stringify(prog.fragment.outputs)})`,
    );
    setUniform(prog, 'u_src0', PRIMARY);
    setUniform(prog, 'u_src1', SECONDARY);
    const fctx = fragmentCtx(prog);
    prog.fragment.run(fctx);
    const c = fctx.out.color[0];
    const s = fctx.out.secondary[0];
    check(
      c[0] === PRIMARY[0] && c[1] === PRIMARY[1] && c[2] === PRIMARY[2] && c[3] === PRIMARY[3],
      `3.00: primary color[0] = u_src0 (got [${Array.from(c).join(', ')}])`,
    );
    check(
      s[0] === SECONDARY[0] && s[1] === SECONDARY[1] && s[2] === SECONDARY[2] && s[3] === SECONDARY[3],
      `3.00: secondary[0] = u_src1 (got [${Array.from(s).join(', ')}])`,
    );
    check(c[0] !== SECONDARY[0] && c[1] !== SECONDARY[1],
      `3.00: secondary write did NOT clobber color[0] (color[0] = [${Array.from(c).join(', ')}])`);
  }
}

/* ------------------------------------------------------------------ */
/* 3. No regression: index-0-only outputs never touch ctx.out.secondary */
/* ------------------------------------------------------------------ */

{
  const vs = compile('#version 300 es\nin vec4 a;\nvoid main() { gl_Position = a; }', 'VERTEX', 300);
  const fs = compile(
    '#version 300 es\nprecision highp float;\nout vec4 o;\nvoid main() { o = vec4(1.0, 2.0, 3.0, 4.0); }',
    'FRAGMENT',
    300,
  );
  const l = linkProgram(vs, fs);
  check(l.ok, `3.00: plain index-0 output links (${l.ok ? '' : l.log})`);
  if (l.ok) {
    // No secondary array on the ctx: if codegen emitted any ctx.out.secondary
    // access this run throws.
    const fctx = fragmentCtx(l.program, { out: { color: [new Float32Array(4)], fragDepth: 0 } });
    l.program.fragment.run(fctx);
    check(
      fctx.out.color[0][0] === 1 && fctx.out.color[0][3] === 4,
      `3.00: index-0-only output writes color without secondary (got [${Array.from(fctx.out.color[0]).join(', ')}])`,
    );
  }
}

/* ------------------------------------------------------------------ */
/* Report + exit                                                       */
/* ------------------------------------------------------------------ */

console.log(`selftest-dual-source: ${checks} checks, ${failures} failure(s)`);
if (failures > 0) {
  console.error(`${failures} check(s) FAILED`);
  process.exit(1);
}
console.log('OK');
