/**
 * Node repro for the sampler/float uniform ARRAY ELEMENT addressing bug fix
 * (src/gl/api/programs.ts + src/gl/api/uniforms.ts).
 *
 * Run: npx tsx src/gl/repro-uniform-array-elem.ts
 * (kept out of tests/ on purpose; objective asked for a scratch repro)
 */
import { __createSoftwareWebGLContext } from '../entry';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const gl: any = __createSoftwareWebGLContext(
  { width: 64, height: 64, getContext: () => null } as unknown as HTMLCanvasElement,
  {},
);

let failures = 0;
function check(desc: string, actual: unknown, expected: unknown): void {
  const a = JSON.stringify(actual);
  const e = JSON.stringify(expected);
  const ok = a === e;
  if (!ok) failures++;
  console.log(`${ok ? 'PASS' : 'FAIL'} ${desc}: got ${a}${ok ? '' : `, expected ${e}`}`);
}

function compile(gl: WebGLRenderingContext, vsSrc: string, fsSrc: string): WebGLProgram {
  const vs = gl.createShader(gl.VERTEX_SHADER)!;
  gl.shaderSource(vs, vsSrc);
  gl.compileShader(vs);
  if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
    throw new Error('VS compile failed: ' + gl.getShaderInfoLog(vs));
  }
  const fs = gl.createShader(gl.FRAGMENT_SHADER)!;
  gl.shaderSource(fs, fsSrc);
  gl.compileShader(fs);
  if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
    throw new Error('FS compile failed: ' + gl.getShaderInfoLog(fs));
  }
  const prog = gl.createProgram()!;
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    throw new Error('link failed: ' + gl.getProgramInfoLog(prog));
  }
  return prog;
}

const VS = `
attribute vec4 a_position;
void main() { gl_Position = a_position; }
`;

// ---- 1. sampler array: uniform1i(uloc('u[2]'), 5) must write element 2 ----
{
  const prog = compile(
    gl,
    VS,
    `precision mediump float;
     uniform sampler2D u[4];
     void main() { gl_FragColor = texture2D(u[0], vec2(0.0)); }`,
  );
  gl.useProgram(prog);
  const l0 = gl.getUniformLocation(prog, 'u[0]');
  const l1 = gl.getUniformLocation(prog, 'u[1]');
  const l2 = gl.getUniformLocation(prog, 'u[2]');
  const l3 = gl.getUniformLocation(prog, 'u[3]');
  const lbare = gl.getUniformLocation(prog, 'u');
  gl.uniform1i(l0, 0);
  gl.uniform1i(l1, 1);
  gl.uniform1i(l2, 5);
  gl.uniform1i(l3, 3);
  check('sampler u[0]', gl.getUniform(prog, l0), 0);
  check('sampler u[1]', gl.getUniform(prog, l1), 1);
  check('sampler u[2] (was the bug)', gl.getUniform(prog, l2), 5);
  check('sampler u[3]', gl.getUniform(prog, l3), 3);
  // bare-name location reads the WHOLE array
  const all = gl.getUniform(prog, lbare) as Int32Array;
  check('sampler u whole array', Array.from(all), [0, 1, 5, 3]);
  // uniform1iv with an array value starting at element location u[1]
  gl.uniform1iv(l1, [7, 8]);
  check('sampler u[1] after 1iv([7,8])', gl.getUniform(prog, l1), 7);
  check('sampler u[2] after 1iv([7,8])', gl.getUniform(prog, l2), 8);
  check('sampler u[0] untouched', gl.getUniform(prog, l0), 0);
  check('no GL error', gl.getError(), gl.NO_ERROR);
}

// ---- 2. float array: element + whole-array writes ----
{
  const prog = compile(
    gl,
    VS,
    `precision mediump float;
     uniform float f[3];
     void main() { gl_FragColor = vec4(f[0], f[1], f[2], 1.0); }`,
  );
  gl.useProgram(prog);
  const lf0 = gl.getUniformLocation(prog, 'f[0]');
  const lf1 = gl.getUniformLocation(prog, 'f[1]');
  const lf2 = gl.getUniformLocation(prog, 'f[2]');
  gl.uniform1f(lf1, 7);
  check('float f[1] (was the bug)', gl.getUniform(prog, lf1), 7);
  check('float f[0] untouched', gl.getUniform(prog, lf0), 0);
  gl.uniform1f(lf0, 1);
  gl.uniform1f(lf2, 3);
  check('float f[0]', gl.getUniform(prog, lf0), 1);
  check('float f[2]', gl.getUniform(prog, lf2), 3);
  // whole-array write via bare name (no regression)
  gl.uniform1fv(gl.getUniformLocation(prog, 'f'), [1, 2, 3]);
  check('float f[0] whole write', gl.getUniform(prog, lf0), 1);
  check('float f[1] whole write', gl.getUniform(prog, lf1), 2);
  check('float f[2] whole write', gl.getUniform(prog, lf2), 3);
  check('no GL error', gl.getError(), gl.NO_ERROR);
}

// ---- 3. vec2 array: uniform2f(uloc('v[1]'), a, b) ----
{
  const prog = compile(
    gl,
    VS,
    `precision mediump float;
     uniform vec2 v[2];
     void main() { gl_FragColor = vec4(v[0], v[1]); }`,
  );
  gl.useProgram(prog);
  const lv0 = gl.getUniformLocation(prog, 'v[0]');
  const lv1 = gl.getUniformLocation(prog, 'v[1]');
  gl.uniform2f(lv1, 4, 5);
  check('vec2 v[1] (was the bug)', Array.from(gl.getUniform(prog, lv1) as Float32Array), [4, 5]);
  check('vec2 v[0] untouched', Array.from(gl.getUniform(prog, lv0) as Float32Array), [0, 0]);
  gl.uniform2fv(lv0, [1, 2]);
  check('vec2 v[0] after 2fv', Array.from(gl.getUniform(prog, lv0) as Float32Array), [1, 2]);
  check('no GL error', gl.getError(), gl.NO_ERROR);
}

// ---- 4. mat4 array element: uniformMatrix4fv(uloc('m[1]'), false, ...) ----
// NOTE: mat4 chosen over mat2/mat3 — the glsl linker's known matrix
// under-allocation (rows<4 stores sized cols*rows instead of cols*4 floats)
// drops tail elements of mat2/mat3 arrays (src/CONTEXT.md Known Issues,
// glsl cluster). mat4's footprint equals its allocation, so the write path
// here is fully verifiable.
{
  const prog = compile(
    gl,
    VS,
    `precision mediump float;
     uniform mat4 m[2];
     void main() { gl_FragColor = vec4(m[0][0].x, m[1][0].x, 0.0, 1.0); }`,
  );
  gl.useProgram(prog);
  const lm0 = gl.getUniformLocation(prog, 'm[0]');
  const lm1 = gl.getUniformLocation(prog, 'm[1]');
  const m1 = [];
  for (let i = 0; i < 16; i++) m1.push(i + 1); // m[0] = 1..16
  const m2 = [];
  for (let i = 0; i < 16; i++) m2.push(100 + i); // m[1] = 100..115
  gl.uniformMatrix4fv(lm1, false, m2); // only m[1]
  check('mat4 m[1] (was the bug)', Array.from(gl.getUniform(prog, lm1) as Float32Array), m2);
  check('mat4 m[0] untouched', Array.from(gl.getUniform(prog, lm0) as Float32Array), m1.map(() => 0));
  gl.uniformMatrix4fv(lm0, false, m1); // whole run from m[0]
  check('mat4 m[0] after whole write', Array.from(gl.getUniform(prog, lm0) as Float32Array), m1);
  check('no GL error', gl.getError(), gl.NO_ERROR);
}

// ---- 5. int + bool arrays via element locations ----
{
  const prog = compile(
    gl,
    VS,
    `precision mediump float;
     uniform int iv[3];
     uniform bool bv[2];
     void main() { gl_FragColor = vec4(float(iv[0] + iv[1] + iv[2])); }`,
  );
  gl.useProgram(prog);
  const li1 = gl.getUniformLocation(prog, 'iv[1]');
  gl.uniform1i(li1, -3);
  check('int iv[1]', gl.getUniform(prog, li1), -3);
  const lb1 = gl.getUniformLocation(prog, 'bv[1]');
  gl.uniform1i(lb1, 1);
  check('bool bv[1]', gl.getUniform(prog, lb1), true);
  const lb0 = gl.getUniformLocation(prog, 'bv[0]');
  check('bool bv[0] untouched', gl.getUniform(prog, lb0), false);
  check('no GL error', gl.getError(), gl.NO_ERROR);
}

// ---- 6. overflow beyond the array must not spill into the next uniform ----
{
  const prog = compile(
    gl,
    VS,
    `precision mediump float;
     uniform float f[2];
     uniform float g;
     void main() { gl_FragColor = vec4(f[0], f[1], g, 1.0); }`,
  );
  gl.useProgram(prog);
  const lg = gl.getUniformLocation(prog, 'g');
  gl.uniform1f(lg, 42);
  // 1fv at f[1] with 4 values: only f[1] (1 value) may be written; f[2], f[3]
  // do not exist and must NOT clobber g.
  gl.uniform1fv(gl.getUniformLocation(prog, 'f[1]'), [7, 99, 98, 97]);
  check('overflow f[1]', gl.getUniform(prog, gl.getUniformLocation(prog, 'f[1]')), 7);
  check('overflow g not clobbered', gl.getUniform(prog, lg), 42);
  check('no GL error', gl.getError(), gl.NO_ERROR);
}

console.log(failures === 0 ? '\nALL CHECKS PASSED' : `\n${failures} CHECK(S) FAILED`);
process.exit(failures === 0 ? 0 : 1);
