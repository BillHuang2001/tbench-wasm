/**
 * Probe script: verify native WebGL class structure in Playwright chromium and
 * validate the prototype-chaining approach for `instanceof` compatibility.
 * Run: node scripts/probe-native.mjs
 */
import { chromium } from 'playwright';

const browser = await chromium.launch();
const page = await browser.newPage();

const probe = await page.evaluate(() => {
  const out = {};
  const has = (n) => typeof globalThis[n] === 'function';
  out.globals = {
    WebGLRenderingContext: has('WebGLRenderingContext'),
    WebGL2RenderingContext: has('WebGL2RenderingContext'),
    WebGLBuffer: has('WebGLBuffer'),
    WebGLTexture: has('WebGLTexture'),
    WebGLProgram: has('WebGLProgram'),
    WebGLShader: has('WebGLShader'),
    WebGLFramebuffer: has('WebGLFramebuffer'),
    WebGLRenderbuffer: has('WebGLRenderbuffer'),
    WebGLUniformLocation: has('WebGLUniformLocation'),
    WebGLActiveInfo: has('WebGLActiveInfo'),
    WebGLShaderPrecisionFormat: has('WebGLShaderPrecisionFormat'),
    WebGLQuery: has('WebGLQuery'),
    WebGLSampler: has('WebGLSampler'),
    WebGLSync: has('WebGLSync'),
    WebGLTransformFeedback: has('WebGLTransformFeedback'),
    WebGLVertexArrayObject: has('WebGLVertexArrayObject'),
  };
  if (has('WebGL2RenderingContext')) {
    out.native2proto_proto =
      Object.getPrototypeOf(WebGL2RenderingContext.prototype) === WebGLRenderingContext.prototype
        ? 'WebGLRenderingContext.prototype'
        : 'other';
    out.native1proto_proto =
      Object.getPrototypeOf(WebGLRenderingContext.prototype) === Object.prototype
        ? 'Object.prototype'
        : 'other';
  }
  // Can we create a real native context (for reference)?
  try {
    const c = document.createElement('canvas');
    const gl = c.getContext('webgl2') || c.getContext('webgl');
    out.nativeCtx = gl ? {
      isGL1: gl instanceof WebGLRenderingContext,
      isGL2: gl instanceof WebGL2RenderingContext,
    } : null;
  } catch (e) {
    out.nativeCtx = 'error: ' + e.message;
  }
  // Probe: can we chain our own class under the native prototype?
  try {
    class MyContext {
      foo() { return 42; }
    }
    const proto = MyContext.prototype;
    Object.setPrototypeOf(proto, WebGLRenderingContext.prototype);
    const inst = new MyContext();
    out.chain = {
      instanceofNative1: inst instanceof WebGLRenderingContext,
      instanceofNative2: inst instanceof WebGL2RenderingContext,
      ownMethodStillWorks: inst.foo() === 42,
      direct: Object.getPrototypeOf(inst) === proto,
    };
  } catch (e) {
    out.chain = 'error: ' + e.message;
  }
  // Probe: can we define Symbol.hasInstance on the native WebGL2 constructor?
  try {
    const prev = WebGL2RenderingContext[Symbol.hasInstance];
    Object.defineProperty(WebGL2RenderingContext, Symbol.hasInstance, {
      value: function (inst) {
        if (prev.call(this, inst)) return true;
        return typeof inst === 'object' && inst !== null && inst._probeWebGL2 === true;
      },
      configurable: true,
    });
    const fake = { _probeWebGL2: true };
    out.symbolHasInstance = {
      fakeAccepted: fake instanceof WebGL2RenderingContext,
      nativeRejected: ({} instanceof WebGL2RenderingContext) === false,
    };
    Object.defineProperty(WebGL2RenderingContext, Symbol.hasInstance, { value: prev, configurable: true });
  } catch (e) {
    out.symbolHasInstance = 'error: ' + e.message;
  }
  // Probe: is native WebGLBuffer constructible / what is its prototype chain?
  if (has('WebGLBuffer')) {
    out.bufferProto = {
      protoIsObject: Object.getPrototypeOf(WebGLBuffer.prototype) === Object.prototype,
      extensible: Object.isExtensible(WebGLBuffer),
      name: WebGLBuffer.name,
    };
  }
  return out;
});

console.log(JSON.stringify(probe, null, 2));
await browser.close();
