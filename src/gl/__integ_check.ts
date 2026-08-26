/**
 * INTEGRATION SELF-TEST (scratch — deleted after run).
 * Exercises the full src/gl surface against the still-stubbed siblings
 * (glsl/raster/present throw "not implemented" — graceful-degradation paths
 * must be active). Run: npx tsx src/gl/__integ_check.ts
 */
import { __createSoftwareWebGLContext } from '../entry';

let pass = 0, fail = 0;
function check(name: string, cond: boolean, extra?: unknown) {
  if (cond) { pass++; }
  else { fail++; console.log(`FAIL: ${name}`, extra ?? ''); }
}
function expectError(gl: any, code: number, name: string) {
  const e = gl.getError();
  check(name, e === code, `got 0x${e.toString(16)} want 0x${code.toString(16)}`);
}

// --- Context creation ---
const canvas: any = { width: 300, height: 150 };
const gl: any = __createSoftwareWebGLContext(canvas, {});
check('W1 context created', !!gl);
const canvas2: any = { width: 300, height: 150 };
const gl2: any = __createSoftwareWebGLContext(canvas2, {}, 'webgl2');
check('W2 context created', !!gl2);
check('W2 instanceof W1 chain', gl2 instanceof gl.constructor);
const conflict: any = __createSoftwareWebGLContext(canvas, {}, 'webgl2');
check('canvas conflict: W2 after W1 → null', conflict === null);

// --- getContextAttributes / getParameter ---
const attrs = gl.getContextAttributes();
check('attrs alpha default', attrs.alpha === true);
check('attrs depth default', attrs.depth === true);
check('attrs stencil default', attrs.stencil === false); // spec default is false
check('attrs antialias reported', typeof attrs.antialias === 'boolean');
check('MAX_TEXTURE_SIZE', gl.getParameter(gl.MAX_TEXTURE_SIZE) === 8192, gl.getParameter(gl.MAX_TEXTURE_SIZE));
check('MAX_VERTEX_ATTRIBS', gl.getParameter(gl.MAX_VERTEX_ATTRIBS) === 16);
check('VERSION string', typeof gl.getParameter(gl.VERSION) === 'string');
check('SHADING_LANGUAGE_VERSION string', typeof gl.getParameter(gl.SHADING_LANGUAGE_VERSION) === 'string');
gl.getParameter(gl.EXTENSIONS); // spec: illegal pname for getParameter
check('getParameter(EXTENSIONS) → INVALID_ENUM', gl.getError() === gl.INVALID_ENUM);
check('EXTENSIONS via getSupportedExtensions', Array.isArray(gl.getSupportedExtensions()));
check('VIEWPORT default', gl.getParameter(gl.VIEWPORT).length === 4);
check('SCISSOR default', gl.getParameter(gl.SCISSOR_BOX).length === 4);
check('error queue empty', gl.getError() === gl.NO_ERROR);
gl.getParameter(0xDEAD);
expectError(gl, gl.INVALID_ENUM, 'bad pname → INVALID_ENUM');
expectError(gl, gl.NO_ERROR, 'queue drained');
check('drawingBufferWidth', gl.drawingBufferWidth === 300);
check('drawingBufferHeight', gl.drawingBufferHeight === 150);

// --- Buffers ---
const buf = gl.createBuffer();
check('createBuffer', !!buf);
gl.bindBuffer(gl.ARRAY_BUFFER, buf);
const data = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]);
gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
expectError(gl, gl.NO_ERROR, 'bufferData ok');
check('getBufferParameter SIZE', gl.getBufferParameter(gl.ARRAY_BUFFER, gl.BUFFER_SIZE) === data.byteLength);
gl.bindBuffer(gl.ARRAY_BUFFER, null);
check('unbind ok', gl.getError() === gl.NO_ERROR);
gl.bindBuffer(gl.ARRAY_BUFFER, buf);
gl.deleteBuffer(buf);
check('isBuffer after delete', gl.isBuffer(buf) === false);

// --- Shaders/programs (glsl stubbed → graceful failure) ---
const vs = gl.createShader(gl.VERTEX_SHADER);
check('createShader', !!vs);
gl.shaderSource(vs, 'void main(){gl_Position=vec4(0.0);}');
gl.compileShader(vs);
check('compile no-exception', true);
check('COMPILE_STATUS false (glsl stub)', gl.getShaderParameter(vs, gl.COMPILE_STATUS) === false);
check('info log non-empty', (gl.getShaderInfoLog(vs) || '').length > 0);
check('getShaderSource round-trip', gl.getShaderSource(vs) === 'void main(){gl_Position=vec4(0.0);}');
check('getShaderPrecisionFormat', (() => {
  const p = gl.getShaderPrecisionFormat(gl.VERTEX_SHADER, gl.HIGH_FLOAT);
  return p && p.rangeMin === 127 && p.rangeMax === 127 && p.precision === 23;
})());
const prog = gl.createProgram();
gl.attachShader(prog, vs);
gl.linkProgram(prog);
check('LINK_STATUS false (glsl stub)', gl.getProgramParameter(prog, gl.LINK_STATUS) === false);
gl.useProgram(prog);
expectError(gl, gl.INVALID_OPERATION, 'useProgram unlinked → INVALID_OPERATION');
gl.useProgram(null);
expectError(gl, gl.NO_ERROR, 'useProgram null ok');
gl.bindAttribLocation(prog, 0, 'a_pos');
expectError(gl, gl.NO_ERROR, 'bindAttribLocation pre-link ok');
gl.bindAttribLocation(prog, 0, 'gl_bad');
expectError(gl, gl.INVALID_OPERATION, 'gl_ prefix → INVALID_OPERATION');
check('getAttribLocation -1', gl.getAttribLocation(prog, 'a_pos') === -1);
check('getUniformLocation null on unlinked... INVALID_OPERATION first', (() => { gl.getUniformLocation(prog, 'u'); return gl.getError() === gl.INVALID_OPERATION; })());
gl.uniform1f(null, 1.0);
expectError(gl, gl.NO_ERROR, 'uniform1f(null) silent no-op');
gl.deleteShader(vs);
check('isShader after delete', gl.isShader(vs) === false);

// --- Textures (local format table path — raster stubbed) ---
const tex = gl.createTexture();
gl.bindTexture(gl.TEXTURE_2D, tex);
check('MIN_FILTER default', gl.getTexParameter(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER) === gl.NEAREST_MIPMAP_LINEAR);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
expectError(gl, gl.NO_ERROR, 'texParameteri ok');
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, 0x1234);
expectError(gl, gl.INVALID_ENUM, 'bad filter value → INVALID_ENUM');
const px = new Uint8Array(8 * 8 * 4).fill(128);
gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 8, 8, 0, gl.RGBA, gl.UNSIGNED_BYTE, px);
expectError(gl, gl.NO_ERROR, 'texImage2D RGBA/UBYTE ok');
gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 8, 8, gl.RGBA, gl.UNSIGNED_BYTE, px);
expectError(gl, gl.NO_ERROR, 'texSubImage2D ok');
gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 8, 8, 0, gl.RGBA, gl.FLOAT, new Float32Array(8 * 8 * 4));
expectError(gl, gl.INVALID_ENUM, 'W1 RGBA/FLOAT without ext → INVALID_ENUM');
gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 8, 8, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
expectError(gl, gl.NO_ERROR, 'null pixels alloc ok');
gl.generateMipmap(gl.TEXTURE_2D);
expectError(gl, gl.NO_ERROR, 'generateMipmap ok');
check('mipmap levels allocated', (tex as any)._image && (tex as any)._image.levels.length === 4); // 8x8 → 4 levels
check('NPOT upload legal', (() => { gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 100, 3, 0, gl.RGBA, gl.UNSIGNED_BYTE, null); return gl.getError() === gl.NO_ERROR; })());
gl.texStorage2D(gl.TEXTURE_2D, 4, gl.RGBA8, 16, 16);
check('W1 has no texStorage2D (W2-only method)', typeof (gl as any).texStorage2D === 'undefined');

// --- Clear / draw (raster stubbed → local fallbacks) ---
gl.clearColor(0.2, 0.4, 0.6, 1.0);
gl.clear(gl.COLOR_BUFFER_BIT);
expectError(gl, gl.NO_ERROR, 'clear COLOR ok');
gl.clear(0x4000);
expectError(gl, gl.INVALID_VALUE, 'clear bad mask → INVALID_VALUE');
gl.drawArrays(gl.TRIANGLES, 0, 3);
expectError(gl, gl.INVALID_OPERATION, 'drawArrays no program → INVALID_OPERATION');
gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array(4));
expectError(gl, gl.INVALID_OPERATION, 'readPixels no program? (should be read-FB error or ok)'); // default FB present → either ok or INVALID_OPERATION w/o crash
gl.readPixels(0, 0, 1, 1, 0x1901 /*RGB*/, gl.UNSIGNED_BYTE, new Uint8Array(3));
check('readPixels default-FB RGB → error', gl.getError() !== gl.NO_ERROR);

// --- Framebuffers (framebuffer-util now real) ---
const fbo = gl.createFramebuffer();
gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
check('empty FBO → MISSING_ATTACHMENT', gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT);
const rb = gl.createRenderbuffer();
gl.bindRenderbuffer(gl.RENDERBUFFER, rb);
gl.renderbufferStorage(gl.RENDERBUFFER, gl.RGBA4, 16, 16);
expectError(gl, gl.NO_ERROR, 'renderbufferStorage ok');
gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.RENDERBUFFER, rb);
check('FBO complete', gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE);
check('attachment param OBJECT_TYPE', gl.getFramebufferAttachmentParameter(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE) === gl.RENDERBUFFER);
gl.bindFramebuffer(gl.FRAMEBUFFER, null);

// --- Extensions ---
const ext = gl.getExtension('OES_texture_float');
check('W1 OES_texture_float exposed', ext !== null);
check('W1 WEBGL_draw_buffers exposed', gl.getExtension('WEBGL_draw_buffers') !== null);
check('W1 EXT_color_buffer_float NOT exposed', gl.getExtension('EXT_color_buffer_float') === null);
check('alias WEBKIT_ aniso', gl.getExtension('WEBKIT_EXT_texture_filter_anisotropic') !== null);
check('lose_context object', gl.getExtension('WEBGL_lose_context') !== null);
const sup = gl.getSupportedExtensions();
check('getSupportedExtensions lists getExtension objects', sup.includes('OES_texture_float') && !sup.includes('WEBGL_compressed_texture_s3tc'));

// --- WebGL2 specifics ---
check('W2 OES_texture_float null (promoted)', gl2.getExtension('OES_texture_float') === null);
check('W2 EXT_color_buffer_float exposed', gl2.getExtension('EXT_color_buffer_float') !== null);
check('W2 OES_draw_buffers_indexed', gl2.getExtension('OES_draw_buffers_indexed') !== null);
const vao = gl2.createVertexArray();
gl2.bindVertexArray(vao);
check('VAO bind ok', gl2.getError() === gl2.NO_ERROR);
gl2.bindVertexArray(null);
const sync = gl2.fenceSync(gl2.SYNC_GPU_COMMANDS_COMPLETE, 0);
check('fenceSync', !!sync);
check('clientWaitSync ALREADY_SIGNALED', gl2.clientWaitSync(sync, 0, 0) === gl2.ALREADY_SIGNALED);
check('clientWaitSync timeout>0 → INVALID_OPERATION', gl2.clientWaitSync(sync, 0, 1) === gl2.INVALID_OPERATION);
const q = gl2.createQuery();
gl2.beginQuery(gl2.ANY_SAMPLES_PASSED, q);
gl2.endQuery(gl2.ANY_SAMPLES_PASSED);
check('query result available', gl2.getQueryParameter(q, gl2.QUERY_RESULT_AVAILABLE) === true);
gl2.texStorage2D(gl2.TEXTURE_2D, 4, gl2.RGBA8, 16, 16);
expectError(gl2, gl2.NO_ERROR, 'W2 texStorage2D ok');
gl2.texStorage2D(gl2.TEXTURE_2D, 4, gl2.RGBA8, 16, 16);
expectError(gl2, gl2.INVALID_OPERATION, 'W2 texStorage2D immutable → INVALID_OPERATION');
gl2.drawArraysInstanced(gl2.TRIANGLES, 0, 3, -1);
expectError(gl2, gl2.INVALID_VALUE, 'W2 instanced negative → INVALID_VALUE');

// --- Context loss ---
const lostCanvas: any = { width: 10, height: 10 };
const gl3: any = __createSoftwareWebGLContext(lostCanvas, {});
gl3.getExtension('WEBGL_lose_context').loseContext();
check('isContextLost true', gl3.isContextLost() === true);
check('getError → CONTEXT_LOST_WEBGL', gl3.getError() === gl3.CONTEXT_LOST_WEBGL);

// --- arity / wiring sanity ---
check('drawArrays arity', gl.drawArrays.length === 3);
check('texImage2D arity', gl.texImage2D.length === 9);
check('uniform4fv arity', gl.uniform4fv.length === 2);
check('W1 has no beginQuery', typeof (gl as any).beginQuery === 'undefined');

console.log(`\nINTEGRATION: ${pass} passed, ${fail} failed`);
process.exit(fail === 0 ? 0 : 1);
