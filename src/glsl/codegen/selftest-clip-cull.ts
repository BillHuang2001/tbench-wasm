/**
 * selftest-clip-cull.ts — GL_ANGLE_clip_cull_distance builtins
 * (gl_ClipDistance / gl_CullDistance float[8], gl_MaxCullDistances /
 * gl_MaxCombinedClipAndCullDistances) end to end: compile gating, codegen
 * lowering, and vertex-scratch round-trips.
 *
 * Every check FAILS on pre-fix code: the variables/constants were absent from
 * the 300 table (undeclared identifier), subPIdx threw "cannot index builtin",
 * and there was no scratch transport at all.
 *
 * Run: npx tsx src/glsl/codegen/selftest-clip-cull.ts
 * Prints "selftest-clip-cull: N checks" and exits 0 only when all pass.
 */
import { compileShader, linkProgram } from '../compiler.js';
import type { Program } from '../program.js';

const EXT = new Set(['GL_ANGLE_clip_cull_distance']);

let failures = 0;
let checks = 0;

function check(cond: boolean, msg: string): void {
  checks++;
  if (!cond) {
    failures++;
    console.error('FAIL: ' + msg);
  }
}

function tryCompile(src: string, type: 'VERTEX' | 'FRAGMENT', version: 100 | 300, extensions?: Set<string>) {
  return compileShader(src, { type, version, extensions });
}

function mustCompile(src: string, type: 'VERTEX' | 'FRAGMENT', version: 100 | 300, what: string, extensions?: Set<string>) {
  const r = tryCompile(src, type, version, extensions);
  check(r.ok, `${what} must compile (${r.ok ? '' : JSON.stringify(r.errors)})`);
  return r.ok ? r.shader : null;
}

function mustFail(src: string, type: 'VERTEX' | 'FRAGMENT', version: 100 | 300, what: string, extensions?: Set<string>) {
  const r = tryCompile(src, type, version, extensions);
  check(!r.ok, `${what} must FAIL to compile`);
  return r;
}

function fragmentCtx(prog: Program, extra: Record<string, unknown> = {}): any {
  return {
    uniforms: prog.floatStore,
    intUniforms: prog.intStore,
    blockStores: [],
    blockIntStores: [],
    scratch: new Float32Array(Math.max(prog.scratchSize, 16)),
    intScratch: new Int32Array(Math.max(prog.intScratchSize, 16)),
    out: { color: [new Float32Array(4)], fragDepth: 0 },
    discarded: false,
    fragCoord: new Float32Array([0, 0, 0, 1]),
    frontFacing: true,
    pointCoord: new Float32Array([0, 0]),
    varyings: [],
    ...extra,
  };
}

function vertexCtx(prog: Program, extra: Record<string, unknown> = {}): any {
  return {
    uniforms: prog.floatStore,
    intUniforms: prog.intStore,
    blockStores: [],
    blockIntStores: [],
    scratch: new Float32Array(Math.max(prog.scratchSize, 16)),
    intScratch: new Int32Array(Math.max(prog.intScratchSize, 16)),
    out: { position: new Float32Array(4), pointSize: 0, varyings: new Float32Array(32) },
    attribs: [],
    attribIndices: new Int32Array(16),
    vertexId: 0,
    instanceId: 0,
    ...extra,
  };
}

const vsTrivial = compileShader(
  `#version 300 es
   void main() { gl_Position = vec4(0.0); }`,
  { type: 'VERTEX', version: 300 },
);
if (!vsTrivial.ok) throw new Error('trivial VS failed to compile');

const fsTrivial = compileShader(
  `#version 300 es
   precision mediump float;
   layout(location=0) out vec4 o;
   void main() { o = vec4(1.0); }`,
  { type: 'FRAGMENT', version: 300 },
);
if (!fsTrivial.ok) throw new Error('trivial FS failed to compile');

/* ------------------------------------------------------------------ */
/* a/b. VS writes gl_ClipDistance[0..7] — compiles WITH the extension, */
/*      fails without it (undeclared identifier).                      */
/* ------------------------------------------------------------------ */

{
  const vsSrc = `#version 300 es
#extension GL_ANGLE_clip_cull_distance : require
in vec2 a_position;
void main()
{
  gl_Position = vec4(a_position, 0.0, 1.0);
  gl_ClipDistance[0] = dot(gl_Position, vec4( 1,  0, 0, 0.5));
  gl_ClipDistance[1] = dot(gl_Position, vec4(-1,  0, 0, 0.5));
  gl_ClipDistance[2] = dot(gl_Position, vec4( 0,  1, 0, 0.5));
  gl_ClipDistance[3] = dot(gl_Position, vec4( 0, -1, 0, 0.5));
  gl_ClipDistance[4] = gl_ClipDistance[0];
  gl_ClipDistance[5] = gl_ClipDistance[1];
  gl_ClipDistance[6] = gl_ClipDistance[2];
  gl_ClipDistance[7] = gl_ClipDistance[3];
}`;
  const vs = mustCompile(vsSrc, 'VERTEX', 300, 'a: VS writing gl_ClipDistance[0..7] with GL_ANGLE_clip_cull_distance', EXT);
  check(vs !== null, 'a: VS shader returned');
  const noExt = vsSrc.replace('#extension GL_ANGLE_clip_cull_distance : require\n', '');
  const r = mustFail(noExt, 'VERTEX', 300, 'b: same VS WITHOUT the #extension', undefined);
  check(
    r.errors !== undefined && r.errors.some((e) => e.message.includes('gl_ClipDistance')),
    `b: error must name gl_ClipDistance (${JSON.stringify(r.errors)})`,
  );
}

/* ------------------------------------------------------------------ */
/* c/d. FS reads gl_ClipDistance[0] compiles; FS WRITES fail at LINK.  */
/* ------------------------------------------------------------------ */

{
  const fsRead = `#version 300 es
#extension GL_ANGLE_clip_cull_distance : require
precision highp float;
out vec4 my_FragColor;
void main()
{
  my_FragColor = vec4(gl_ClipDistance[0] + gl_ClipDistance[4]);
}`;
  const fs = mustCompile(fsRead, 'FRAGMENT', 300, 'c: FS reading gl_ClipDistance with GL_ANGLE_clip_cull_distance', EXT);
  check(fs !== null, 'c: FS shader returned');
  const l = linkProgram(vsTrivial.shader, fs!);
  check(l.ok, `c: link of FS-reading program succeeds (${l.ok ? '' : l.log})`);

  const fsWrite = `#version 300 es
#extension GL_ANGLE_clip_cull_distance : require
precision highp float;
out vec4 my_FragColor;
void main()
{
  gl_ClipDistance[0] = 1.0;
  my_FragColor = vec4(1.0);
}`;
  // Semantics allows the write (stage BOTH, writable in the table); the
  // CODEGEN throws in the fragment stage → the link fails (CTS never writes
  // clip/cull distances in the fragment stage, so a link failure is fine).
  const fsW = tryCompile(fsWrite, 'FRAGMENT', 300, EXT);
  check(fsW.ok, 'd: FS WRITE compiles at the shader stage (semantics sees BOTH-stage writable)');
  if (fsW.ok) {
    const lw = linkProgram(vsTrivial.shader, fsW.shader);
    check(!lw.ok && lw.log.includes('read-only'), `d: FS write must fail at LINK with a read-only error (${lw.ok ? 'linked?!' : lw.log})`);
  }
}

/* ------------------------------------------------------------------ */
/* e. gl_CullDistance write/read equivalents (a-d).                    */
/* ------------------------------------------------------------------ */

{
  const vsSrc = `#version 300 es
#extension GL_ANGLE_clip_cull_distance : require
in vec2 a_position;
void main()
{
  gl_Position = vec4(a_position, 0.0, 1.0);
  gl_CullDistance[0] = dot(gl_Position, vec4(1, 0, 0, 1));
  gl_CullDistance[1] = dot(gl_Position, vec4(-1, 0, 0, 1));
  gl_CullDistance[2] = dot(gl_Position, vec4(0, 1, 0, 1));
  gl_CullDistance[3] = dot(gl_Position, vec4(0, -1, 0, 1));
  gl_CullDistance[4] = gl_CullDistance[0];
  gl_CullDistance[5] = gl_CullDistance[1];
  gl_CullDistance[6] = gl_CullDistance[2];
  gl_CullDistance[7] = gl_CullDistance[3];
}`;
  const vs = mustCompile(vsSrc, 'VERTEX', 300, 'e1: VS writing gl_CullDistance[0..7] with GL_ANGLE_clip_cull_distance', EXT);
  const noExt = vsSrc.replace('#extension GL_ANGLE_clip_cull_distance : require\n', '');
  mustFail(noExt, 'VERTEX', 300, 'e2: same VS WITHOUT the #extension', undefined);

  const fsRead = `#version 300 es
#extension GL_ANGLE_clip_cull_distance : require
precision highp float;
out vec4 my_FragColor;
void main()
{
  my_FragColor = vec4(gl_CullDistance[0] + gl_CullDistance[7]);
}`;
  const fs = mustCompile(fsRead, 'FRAGMENT', 300, 'e3: FS reading gl_CullDistance with GL_ANGLE_clip_cull_distance', EXT);
  const l = linkProgram(vsTrivial.shader, fs!);
  check(l.ok, `e3: link of FS-reading program succeeds (${l.ok ? '' : l.log})`);

  const fsWrite = `#version 300 es
#extension GL_ANGLE_clip_cull_distance : require
precision highp float;
out vec4 my_FragColor;
void main()
{
  gl_CullDistance[0] = 1.0;
  my_FragColor = vec4(1.0);
}`;
  const fsW = tryCompile(fsWrite, 'FRAGMENT', 300, EXT);
  check(fsW.ok, 'e4: FS WRITE compiles at the shader stage');
  if (fsW.ok) {
    const lw = linkProgram(vsTrivial.shader, fsW.shader);
    check(!lw.ok && lw.log.includes('read-only'), `e4: FS write must fail at LINK with a read-only error (${lw.ok ? 'linked?!' : lw.log})`);
  }
  check(vs !== null, 'e1: VS shader returned');
}

/* ------------------------------------------------------------------ */
/* f. gl_MaxCullDistances / gl_MaxCombinedClipAndCullDistances usable  */
/*    with the extension, undeclared without.                          */
/* ------------------------------------------------------------------ */

{
  const fsConst = `#version 300 es
#extension GL_ANGLE_clip_cull_distance : require
precision highp float;
out vec4 my_FragColor;
void main()
{
  my_FragColor = vec4(float(gl_MaxCullDistances), float(gl_MaxCombinedClipAndCullDistances), 0.0, 1.0);
}`;
  const fs = mustCompile(fsConst, 'FRAGMENT', 300, 'f1: constants with GL_ANGLE_clip_cull_distance', EXT);
  const l = linkProgram(vsTrivial.shader, fs!);
  check(l.ok, `f1: link succeeds (${l.ok ? '' : l.log})`);
  if (l.ok) {
    const p = l.program;
    const fctx = fragmentCtx(p);
    p.fragment.run(fctx);
    const c = fctx.out.color[0];
    check(
      c[0] === 8 && c[1] === 16,
      `f1: gl_MaxCullDistances=8, gl_MaxCombinedClipAndCullDistances=16 (got [${Array.from(c).join(', ')}])`,
    );
  }
  const noExt = fsConst.replace('#extension GL_ANGLE_clip_cull_distance : require\n', '');
  const r = mustFail(noExt, 'FRAGMENT', 300, 'f2: constants WITHOUT the #extension', undefined);
  check(
    r.errors !== undefined && r.errors.some((e) => e.message.includes('gl_MaxCullDistances')),
    `f2: error must name gl_MaxCullDistances (${JSON.stringify(r.errors)})`,
  );
}

/* ------------------------------------------------------------------ */
/* g. RUNTIME round-trip through the vertex scratch:                   */
/*    gl_ClipDistance[0] = dot(gl_Position, plane);                    */
/*    gl_ClipDistance[4] = gl_ClipDistance[0];  cd = gl_ClipDistance[4] */
/* ------------------------------------------------------------------ */

{
  const vs = mustCompile(
    `#version 300 es
#extension GL_ANGLE_clip_cull_distance : require
in vec2 a_position;
out float cd;
void main()
{
  gl_Position = vec4(a_position, 0.0, 1.0);
  gl_ClipDistance[0] = dot(gl_Position, vec4(1, 0, 0, 0.5));
  gl_ClipDistance[4] = gl_ClipDistance[0];
  cd = gl_ClipDistance[4];
}`,
    'VERTEX',
    300,
    'g: round-trip VS',
    EXT,
  );
  const l = linkProgram(vs!, fsTrivial.shader);
  check(l.ok, `g: link succeeds (${l.ok ? '' : l.log})`);
  if (l.ok) {
    const p = l.program;
    check(p.scratchSize >= 8, `g: program.scratchSize covers the 8-float clip store (got ${p.scratchSize})`);
    const vctx = vertexCtx(p, {
      attribs: [new Float32Array([0.25, 0.25, 0, 1])],
      attribIndices: new Int32Array([0]),
    });
    let threw: Error | null = null;
    try {
      p.vertex.run(vctx);
    } catch (e) {
      threw = e as Error;
    }
    check(threw === null, `g: vertex.run must not throw (${threw ? threw.message : ''})`);
    const cd = vctx.out.varyings[0];
    // dot(vec4(0.25,0.25,0,1), vec4(1,0,0,0.5)) = 0.25 + 0.5 = 0.75.
    check(Math.abs(cd - 0.75) < 1e-6, `g: varying cd carries the scratch round-trip value 0.75 (got ${cd})`);
    // Both the direct write (index 0) and the self-read copy (index 4)
    // landed in the scratch store.
    const scratch = vctx.scratch;
    check(
      Math.abs(scratch[0] - 0.75) < 1e-6 && Math.abs(scratch[4] - 0.75) < 1e-6,
      `g: scratch[0]=scratch[4]=0.75 (got [${Array.from(scratch.slice(0, 8)).join(', ')}])`,
    );
  }
}

/* ------------------------------------------------------------------ */
/* g2. RUNTIME dynamic index: gl_ClipDistance[i] with a uniform i      */
/*     strides the scratch correctly (const + dynamic mixed).          */
/* ------------------------------------------------------------------ */

{
  const vs = mustCompile(
    `#version 300 es
#extension GL_ANGLE_clip_cull_distance : require
uniform int i;
in vec2 a_position;
out float cd;
void main()
{
  gl_Position = vec4(a_position, 0.0, 1.0);
  gl_ClipDistance[i] = 0.5;
  cd = gl_ClipDistance[i];
}`,
    'VERTEX',
    300,
    'g2: dynamic-index VS',
    EXT,
  );
  const l = linkProgram(vs!, fsTrivial.shader);
  check(l.ok, `g2: link succeeds (${l.ok ? '' : l.log})`);
  if (l.ok) {
    const p = l.program;
    const iLoc = p.uniformMap.get('i')!.location;
    const vctx = vertexCtx(p, {
      attribs: [new Float32Array([0.25, 0.25, 0, 1])],
      attribIndices: new Int32Array([0]),
    });
    p.intStore[iLoc] = 3;
    p.vertex.run(vctx);
    check(
      Math.abs(vctx.out.varyings[0] - 0.5) < 1e-6 && Math.abs(vctx.scratch[3] - 0.5) < 1e-6,
      `g2: dynamic i=3 writes/reads scratch[3] (cd=${vctx.out.varyings[0]}, scratch=${Array.from(vctx.scratch.slice(0, 8)).join(',')})`,
    );
  }
}

/* ------------------------------------------------------------------ */
/* h. RUNTIME zero-default: VS does NOT write gl_ClipDistance; the FS  */
/*    reads gl_ClipDistance[0] into the color → 0.                     */
/* ------------------------------------------------------------------ */

{
  const fs = mustCompile(
    `#version 300 es
#extension GL_ANGLE_clip_cull_distance : require
precision highp float;
out vec4 my_FragColor;
void main()
{
  my_FragColor = vec4(gl_ClipDistance[0]);
}`,
    'FRAGMENT',
    300,
    'h: zero-default FS',
    EXT,
  );
  const l = linkProgram(vsTrivial.shader, fs!);
  check(l.ok, `h: link succeeds (${l.ok ? '' : l.log})`);
  if (l.ok) {
    const p = l.program;
    const fctx = fragmentCtx(p);
    let threw: Error | null = null;
    try {
      p.fragment.run(fctx);
    } catch (e) {
      threw = e as Error;
    }
    check(threw === null, `h: fragment.run must not throw (${threw ? threw.message : ''})`);
    const c = fctx.out.color[0];
    check(
      c[0] === 0 && c[1] === 0 && c[2] === 0 && c[3] === 0,
      `h: FS output is (0,0,0,0) — un-transported reads default to 0 (got [${Array.from(c).join(', ')}])`,
    );
  }
}

/* ------------------------------------------------------------------ */
/* i. v100 negative: the extension does not exist at ES 1.00 —         */
/*    gl_ClipDistance is undeclared even with the pragma enabled.      */
/* ------------------------------------------------------------------ */

{
  const r = mustFail(
    `#version 100
#extension GL_ANGLE_clip_cull_distance : require
void main()
{
  gl_Position = vec4(0.0);
  gl_ClipDistance[0] = 0.0;
}`,
    'VERTEX',
    100,
    'i: ES 1.00 shader with the extension pragma',
    EXT,
  );
  check(
    r.errors !== undefined && r.errors.some((e) => e.message.includes('gl_ClipDistance')),
    `i: error must name gl_ClipDistance (${JSON.stringify(r.errors)})`,
  );
}

/* ------------------------------------------------------------------ */
/* j. Whole-array identifier-level use: `float a[8]; a = gl_ClipDistance;` */
/*    (legal ESSL 3.00 array assignment) — component-wise via the       */
/*    const-index path (flatOff 0..7).                                 */
/* ------------------------------------------------------------------ */

{
  const vs = mustCompile(
    `#version 300 es
#extension GL_ANGLE_clip_cull_distance : require
in vec2 a_position;
out float cd;
void main()
{
  gl_Position = vec4(a_position, 0.0, 1.0);
  gl_ClipDistance[0] = 0.25;
  float a[8];
  a = gl_ClipDistance;
  cd = a[7] + a[0];
}`,
    'VERTEX',
    300,
    'j: whole-array assignment VS',
    EXT,
  );
  const l = linkProgram(vs!, fsTrivial.shader);
  check(l.ok, `j: link succeeds (${l.ok ? '' : l.log})`);
  if (l.ok) {
    const p = l.program;
    const vctx = vertexCtx(p, {
      attribs: [new Float32Array([0.25, 0.25, 0, 1])],
      attribIndices: new Int32Array([0]),
    });
    p.vertex.run(vctx);
    check(
      Math.abs(vctx.out.varyings[0] - 0.25) < 1e-6,
      `j: whole-array copy reads the scratch (a[0]=0.25, a[7]=0 → cd=0.25; got ${vctx.out.varyings[0]})`,
    );
  }
}

/* ------------------------------------------------------------------ */
/* Report + exit                                                       */
/* ------------------------------------------------------------------ */

console.log(`selftest-clip-cull: ${checks} checks, ${failures} failure(s)`);
if (failures > 0) {
  console.error(`${failures} check(s) FAILED`);
  process.exit(1);
}
console.log('OK');
