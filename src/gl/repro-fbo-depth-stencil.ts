/**
 * Node repro for the DEPTH_STENCIL_ATTACHMENT identity bug
 * (src/gl/api/framebuffers.ts): getFramebufferAttachmentParameter(..., OBJECT_NAME)
 * must return the SAME WebGLRenderbuffer/WebGLTexture instance that was passed to
 * framebufferRenderbuffer/framebufferTexture2D — not null (and not a different
 * wrapper). On WebGL2 the DEPTH_STENCIL_ATTACHMENT write path aliases the DEPTH
 * and STENCIL slots; each slot used to get its OWN record object literal, so
 * resolveAttachmentRecord saw a depth/stencil mismatch → INVALID_OPERATION + null.
 *
 * Run: npx tsx src/gl/repro-fbo-depth-stencil.ts
 * (kept out of tests/ on purpose; scratch repro per objective)
 */
import { __createSoftwareWebGLContext } from '../entry';

let failures = 0;
function check(desc: string, actual: unknown, expected: unknown): void {
  const a = JSON.stringify(actual);
  const e = JSON.stringify(expected);
  const ok = a === e;
  if (!ok) failures++;
  console.log(`${ok ? 'PASS' : 'FAIL'} ${desc}: got ${a}${ok ? '' : `, expected ${e}`}`);
}

// ---- WebGL2: renderbuffer + DEPTH_STENCIL_ATTACHMENT ----
{
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const gl: any = __createSoftwareWebGLContext(
    { width: 64, height: 64, getContext: () => null } as unknown as HTMLCanvasElement,
    {},
    'webgl2',
  );
  const fb = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
  const rb = gl.createRenderbuffer();
  gl.bindRenderbuffer(gl.RENDERBUFFER, rb);
  gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH24_STENCIL8, 16, 16);
  gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_STENCIL_ATTACHMENT, gl.RENDERBUFFER, rb);
  check('W2 rb: no error after framebufferRenderbuffer(DS)', gl.getError(), gl.NO_ERROR);
  const got = gl.getFramebufferAttachmentParameter(
    gl.FRAMEBUFFER, gl.DEPTH_STENCIL_ATTACHMENT, gl.FRAMEBUFFER_ATTACHMENT_OBJECT_NAME,
  );
  check('W2 rb: OBJECT_NAME === same instance', got === rb, true);
  check('W2 rb: no error after getFramebufferAttachmentParameter(DS)', gl.getError(), gl.NO_ERROR);
  (rb as { expando1?: string }).expando1 = 'marker';
  let expandoOk = false;
  try {
    expandoOk = got.expando1 === 'marker';
  } catch (err) {
    console.log(`  (expando read threw: ${err instanceof Error ? err.message : String(err)})`);
  }
  check('W2 rb: expando survives round-trip', expandoOk, true);
  check('W2 rb: checkFramebufferStatus complete', gl.checkFramebufferStatus(gl.FRAMEBUFFER), gl.FRAMEBUFFER_COMPLETE);
  check('W2 rb: DEPTH_ATTACHMENT OBJECT_NAME same instance',
    gl.getFramebufferAttachmentParameter(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.FRAMEBUFFER_ATTACHMENT_OBJECT_NAME) === rb, true);
  check('W2 rb: STENCIL_ATTACHMENT OBJECT_NAME same instance',
    gl.getFramebufferAttachmentParameter(gl.FRAMEBUFFER, gl.STENCIL_ATTACHMENT, gl.FRAMEBUFFER_ATTACHMENT_OBJECT_NAME) === rb, true);
}

// ---- WebGL2: texture + DEPTH_STENCIL_ATTACHMENT ----
{
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const gl: any = __createSoftwareWebGLContext(
    { width: 64, height: 64, getContext: () => null } as unknown as HTMLCanvasElement,
    {},
    'webgl2',
  );
  const fb = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.DEPTH_STENCIL_ATTACHMENT, gl.TEXTURE_2D, tex, 0);
  check('W2 tex: no error after framebufferTexture2D(DS)', gl.getError(), gl.NO_ERROR);
  const got = gl.getFramebufferAttachmentParameter(
    gl.FRAMEBUFFER, gl.DEPTH_STENCIL_ATTACHMENT, gl.FRAMEBUFFER_ATTACHMENT_OBJECT_NAME,
  );
  check('W2 tex: OBJECT_NAME === same instance', got === tex, true);
  (tex as { expando1?: string }).expando1 = 'texmarker';
  let expandoOk = false;
  try {
    expandoOk = got.expando1 === 'texmarker';
  } catch (err) {
    console.log(`  (expando read threw: ${err instanceof Error ? err.message : String(err)})`);
  }
  check('W2 tex: expando survives round-trip', expandoOk, true);
}

// ---- WebGL1 sanity: DEPTH_STENCIL_ATTACHMENT single-slot path ----
{
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const gl: any = __createSoftwareWebGLContext(
    { width: 64, height: 64, getContext: () => null } as unknown as HTMLCanvasElement,
    {},
    'webgl',
  );
  const fb = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
  const rb = gl.createRenderbuffer();
  gl.bindRenderbuffer(gl.RENDERBUFFER, rb);
  gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_STENCIL, 16, 16);
  gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_STENCIL_ATTACHMENT, gl.RENDERBUFFER, rb);
  check('W1 rb: no error after framebufferRenderbuffer(DS)', gl.getError(), gl.NO_ERROR);
  const got = gl.getFramebufferAttachmentParameter(
    gl.FRAMEBUFFER, gl.DEPTH_STENCIL_ATTACHMENT, gl.FRAMEBUFFER_ATTACHMENT_OBJECT_NAME,
  );
  check('W1 rb: OBJECT_NAME === same instance', got === rb, true);
}

console.log(failures === 0 ? '\nALL CHECKS PASSED' : `\n${failures} CHECK(S) FAILED`);
process.exit(failures === 0 ? 0 : 1);
