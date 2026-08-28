/**
 * fragment-ops.ts — per-fragment operations (scissor, sample coverage,
 * stencil, depth, blend, dither, sRGB, colorMask) + the quad fragment driver
 * + clear/blit helpers for gl/.
 *
 * Order of operations per GLES 3.0 §4.1:
 *   scissor test → sample coverage → stencil test → depth test → blending →
 *   (dithering — accepted but a no-op, see below) → (sRGB re-encode) → color
 *   write.
 * The two-phase protocol (FragmentOps.test / FragmentOps.finalize, see
 * types.ts) defers stencil-zpass/depth-write until AFTER the fragment shader
 * runs, so `discard` suppresses every write and gl_FragDepth shaders get a
 * post-shader depth test.
 *
 * Blending happens in LINEAR space when the target is sRGB (destination
 * decoded before the blend, result re-encoded before the write).
 *
 * Pinned semantic decisions (documented here so gl/ and formats.ts agree):
 *  - `info.decode` returns texel values AS STORED (for sRGB formats: the
 *    sRGB-encoded 0..1 values); `info.encode` writes the values it is given.
 *    The linear↔sRGB conversion is THIS module's responsibility and is
 *    applied only in the fragment color-write path (RGB channels only — alpha
 *    is never sRGB-encoded per GLES 3.0 §4.1.8). Clear and blit pass values
 *    through unconverted (glClearColor is written as-is per spec).
 *  - Blending applies per output location. Core GLES 3.0: the global BLEND cap
 *    governs every draw buffer. With OES_draw_buffers_indexed active, gl/
 *    attaches DrawCall.blendPerDrawBuffer (entry per draw buffer; buffer 0
 *    mirrors the global cap) and each output blends with its own entry.
 *  - Dither: the DITHER state is accepted (gl.enable/getParameter) but the
 *    dithering algorithm is a NO-OP. The GL/WebGL specs leave the algorithm
 *    implementation-defined, and every shipping WebGL implementation (ANGLE,
 *    SwiftShader, native GL/Metal drivers) effectively does not dither — the
 *    CTS is calibrated to that: exact-value readbacks (e.g.
 *    EXT_texture_norm16's "should be 106" check for 0x6a35 in a u16 texture
 *    rendered to an RGBA8 target) expect the plain round-to-nearest conversion
 *    (GLES 2.0 §2.1.8: c = round(f·(2^b−1))). Any position-dependent dither
 *    (e.g. a 4×4 Bayer offset) would turn round(105.79) = 106 into 105 at
 *    some pixel and break those tests; no test verifies dither output.
 *  - Sample coverage is a deterministic per-fragment integer-lattice hash
 *    (documented approximation of the hardware coverage mask).
 *  - No depth/stencil attachment → the corresponding test always passes
 *    (GL 4.5 §17.3.5/§17.3.6 semantics; matches ANGLE).
 *  - Occlusion queries: when DrawCall carries `sampleCountRef` (attached by
 *    gl/), it is incremented exactly once per sample passing stencil+depth —
 *    in test() for non-gl_FragDepth shaders, in finalize() after the
 *    post-shader depth test for gl_FragDepth shaders. Helper invocations and
 *    rasterizerDiscard draws never reach the ops.
 */

import type {
  BlendPerDrawBufferEntry, ColorMask, DrawCall, FragmentExecCtx, FragmentOps, RasterState,
  ScissorState, StencilFaceState, Surface,
} from './types';
import { getDepthData, getStencilData } from './surface';
import { linearToSRGB, sRGBToLinear } from './formats';
import type { GLenum } from './gl-enums';
import {
  ALWAYS, CONSTANT_ALPHA, CONSTANT_COLOR, DECR, DECR_WRAP, DST_ALPHA,
  DST_COLOR, EQUAL, FUNC_REVERSE_SUBTRACT, FUNC_SUBTRACT, GEQUAL, GREATER,
  INCR, INCR_WRAP, INVERT, KEEP, LEQUAL, LESS, MAX, MIN, NEVER, NOTEQUAL,
  ONE, ONE_MINUS_CONSTANT_ALPHA, ONE_MINUS_CONSTANT_COLOR, ONE_MINUS_DST_ALPHA,
  ONE_MINUS_DST_COLOR, ONE_MINUS_SRC_ALPHA, ONE_MINUS_SRC_COLOR, REPLACE,
  SRC_ALPHA, SRC_ALPHA_SATURATE, SRC_COLOR, ZERO,
} from './gl-enums';

/** DrawCall extended with the optional occlusion-query counter (gl/ attaches
 *  `sampleCountRef` for occlusion queries; raster only increments it). */
interface DrawCallWithSampleCount extends DrawCall {
  sampleCountRef?: { value: number };
}

/* ================================================================== */
/* Small spec helpers (exported for unit tests)                        */
/* ================================================================== */

/** Result of applying a stencil op to `current` (8-bit; ref = stencil ref). */
export function applyStencilOp(op: GLenum, current: number, ref: number): number {
  switch (op) {
    case KEEP: return current;
    case ZERO: return 0;
    case REPLACE: return ref;
    case INCR: return Math.min(255, current + 1);
    case DECR: return Math.max(0, current - 1);
    case INVERT: return current ^ 0xff;
    case INCR_WRAP: return (current + 1) & 0xff;
    case DECR_WRAP: return (current - 1) & 0xff;
    default: return current;
  }
}

/** Depth/stencil compare function: does `z` pass `func` against `stored`? */
export function depthPass(func: GLenum, z: number, stored: number): boolean {
  switch (func) {
    case NEVER: return false;
    case LESS: return z < stored;
    case EQUAL: return z === stored;
    case LEQUAL: return z <= stored;
    case GREATER: return z > stored;
    case NOTEQUAL: return z !== stored;
    case GEQUAL: return z >= stored;
    case ALWAYS: return true;
    default: return true;
  }
}

/** RGB blend factor for one channel (sc/dc/cc = that channel of src/dst/const). */
function rgbFactor(f: GLenum, sc: number, dc: number, sa: number, da: number, cc: number, ca: number): number {
  switch (f) {
    case ZERO: return 0;
    case ONE: return 1;
    case SRC_COLOR: return sc;
    case ONE_MINUS_SRC_COLOR: return 1 - sc;
    case DST_COLOR: return dc;
    case ONE_MINUS_DST_COLOR: return 1 - dc;
    case SRC_ALPHA: return sa;
    case ONE_MINUS_SRC_ALPHA: return 1 - sa;
    case DST_ALPHA: return da;
    case ONE_MINUS_DST_ALPHA: return 1 - da;
    case CONSTANT_COLOR: return cc;
    case ONE_MINUS_CONSTANT_COLOR: return 1 - cc;
    case CONSTANT_ALPHA: return ca;
    case ONE_MINUS_CONSTANT_ALPHA: return 1 - ca;
    case SRC_ALPHA_SATURATE: return Math.min(sa, 1 - da);
    default: return 0;
  }
}

/** Alpha blend factor (color-ish factors use the alpha of that color). */
function alphaFactor(f: GLenum, sa: number, da: number, ca: number): number {
  switch (f) {
    case ZERO: return 0;
    case ONE: return 1;
    case SRC_COLOR:
    case SRC_ALPHA: return sa;
    case ONE_MINUS_SRC_COLOR:
    case ONE_MINUS_SRC_ALPHA: return 1 - sa;
    case DST_COLOR:
    case DST_ALPHA: return da;
    case ONE_MINUS_DST_COLOR:
    case ONE_MINUS_DST_ALPHA: return 1 - da;
    case CONSTANT_COLOR:
    case CONSTANT_ALPHA: return ca;
    case ONE_MINUS_CONSTANT_COLOR:
    case ONE_MINUS_CONSTANT_ALPHA: return 1 - ca;
    case SRC_ALPHA_SATURATE: return 1;
    default: return 0;
  }
}

/**
 * Computes one blended RGBA result (linear space) from src + dst.
 * Equations: FUNC_ADD / FUNC_SUBTRACT / FUNC_REVERSE_SUBTRACT (factors
 * applied) and MIN / MAX (factors ignored). Results are NOT clamped — callers
 * write through encode(), which clamps for normalized targets.
 */
export function blendColor(
  src: Float32Array, dst: Float32Array, out: Float32Array,
  srcRGB: number, dstRGB: number, srcAlpha: number, dstAlpha: number,
  eqRGB: number, eqAlpha: number, constColor: [number, number, number, number],
): void {
  const sr = src[0], sg = src[1], sb = src[2], sa = src[3];
  const dr = dst[0], dg = dst[1], db = dst[2], da = dst[3];
  const cr = constColor[0], cg = constColor[1], cb = constColor[2], ca = constColor[3];
  if (eqRGB === MIN || eqRGB === MAX) {
    if (eqRGB === MIN) {
      out[0] = Math.min(sr, dr); out[1] = Math.min(sg, dg); out[2] = Math.min(sb, db);
    } else {
      out[0] = Math.max(sr, dr); out[1] = Math.max(sg, dg); out[2] = Math.max(sb, db);
    }
  } else {
    const fsr = rgbFactor(srcRGB, sr, dr, sa, da, cr, ca);
    const fdr = rgbFactor(dstRGB, sr, dr, sa, da, cr, ca);
    const fsg = rgbFactor(srcRGB, sg, dg, sa, da, cg, ca);
    const fdg = rgbFactor(dstRGB, sg, dg, sa, da, cg, ca);
    const fsb = rgbFactor(srcRGB, sb, db, sa, da, cb, ca);
    const fdb = rgbFactor(dstRGB, sb, db, sa, da, cb, ca);
    if (eqRGB === FUNC_SUBTRACT) {
      out[0] = fsr * sr - fdr * dr; out[1] = fsg * sg - fdg * dg; out[2] = fsb * sb - fdb * db;
    } else if (eqRGB === FUNC_REVERSE_SUBTRACT) {
      out[0] = fdr * dr - fsr * sr; out[1] = fdg * dg - fsg * sg; out[2] = fdb * db - fsb * sb;
    } else { // FUNC_ADD (and any unknown equation)
      out[0] = fsr * sr + fdr * dr; out[1] = fsg * sg + fdg * dg; out[2] = fsb * sb + fdb * db;
    }
  }
  if (eqAlpha === MIN || eqAlpha === MAX) {
    out[3] = eqAlpha === MIN ? Math.min(sa, da) : Math.max(sa, da);
  } else {
    const fsa = alphaFactor(srcAlpha, sa, da, ca);
    const fda = alphaFactor(dstAlpha, sa, da, ca);
    if (eqAlpha === FUNC_SUBTRACT) out[3] = fsa * sa - fda * da;
    else if (eqAlpha === FUNC_REVERSE_SUBTRACT) out[3] = fda * da - fsa * sa;
    else out[3] = fsa * sa + fda * da;
  }
}

/* ================================================================== */
/* Sample coverage (per-fragment deterministic helper)                 */
/* ================================================================== */

/**
 * Deterministic per-fragment hash in [0,1) used for the sample-coverage
 * approximation. Integer-lattice mix (MurmurHash3-style finalizer over the
 * pixel coords) — better distributed than sin-based hashes for integer pixel
 * coordinates, and stable across frames. Documented approximation: the real
 * hardware coverage mask is a per-sample stochastic pattern; a deterministic
 * threshold match reproduces the pass RATE (~value fraction of fragments
 * pass) which is what the CTS coverage tests measure.
 */
function coverageHash(x: number, y: number): number {
  let h = (Math.imul(x | 0, 0x9e3779b1) + Math.imul(y | 0, 0x85ebca6b)) | 0;
  h = Math.imul(h ^ (h >>> 16), 0x7feb352d);
  h = Math.imul(h ^ (h >>> 15), 0x846ca68b);
  h ^= h >>> 13;
  return (h >>> 0) / 4294967296;
}

/* ================================================================== */
/* FragmentOpsImpl — per-draw per-fragment ops engine                  */
/* ================================================================== */

/**
 * Per-draw fragment ops engine. Constructed once per draw call by
 * rasterizer.createRasterState; methods are called per fragment.
 */
export class FragmentOpsImpl implements FragmentOps {
  private readonly dc: DrawCall;
  // Resolved surfaces (once per draw). No depth/stencil attachment → the
  // corresponding test always passes.
  private readonly depthData: Float32Array | null;
  private readonly depthWidth: number;
  private readonly stencilData: Uint8Array | null;
  private readonly stencilWidth: number;
  // Per-draw state snapshot (GL state is fixed for the duration of a draw).
  private readonly scissor: ScissorState;
  private readonly sampleCoverage: { enabled: boolean; value: number; invert: boolean };
  private readonly stencilEnabled: boolean;
  private readonly depthTestEnabled: boolean;
  private readonly depthMask: boolean;
  private readonly blend: DrawCall['blend'];
  /** Optional per-draw-buffer blend state (OES_draw_buffers_indexed). */
  private readonly blendPerDrawBuffer: readonly BlendPerDrawBufferEntry[] | undefined;
  private readonly colorMask: readonly ColorMask[];
  private readonly drawBuffers: readonly number[];
  private readonly fbColors: readonly (Surface | null)[];
  private readonly usesFragDepth: boolean;
  /** Optional occlusion-query counter; incremented once per passing sample. */
  private readonly sampleCountRef: { value: number } | null;
  // Per-fragment scratch (allocated once per draw — never per fragment).
  private readonly dstScratch = new Float32Array(4);
  private readonly linearDst = new Float32Array(4);
  private readonly blendOut = new Float32Array(4);

  constructor(dc: DrawCall) {
    this.dc = dc;
    this.depthData = dc.fb.depth ? getDepthData(dc.fb.depth) : null;
    this.depthWidth = dc.fb.depth ? dc.fb.depth.width : 0;
    this.stencilData = dc.fb.stencil ? getStencilData(dc.fb.stencil) : null;
    this.stencilWidth = dc.fb.stencil ? dc.fb.stencil.width : 0;
    this.scissor = dc.scissor;
    this.sampleCoverage = dc.sampleCoverage;
    this.stencilEnabled = dc.stencilTest.enabled;
    this.depthTestEnabled = dc.depthTest.enabled;
    this.depthMask = dc.depthMask;
    this.blend = dc.blend;
    this.blendPerDrawBuffer = dc.blendPerDrawBuffer;
    this.colorMask = dc.colorMask;
    this.drawBuffers = dc.drawBuffers;
    this.fbColors = dc.fb.color;
    this.usesFragDepth = dc.program.fragment.usesFragDepth;
    this.sampleCountRef = (dc as DrawCallWithSampleCount).sampleCountRef ?? null;
  }

  /** Scissor → coverage → stencil test → depth read. See FragmentOps. */
  test(x: number, y: number, frontFacing: boolean, depth: number): boolean {
    // 1. Scissor test (no side effects on failure).
    if (this.scissor.enabled) {
      const sx = this.scissor.x;
      const sy = this.scissor.y;
      if (x < sx || y < sy || x >= sx + this.scissor.w || y >= sy + this.scissor.h) {
        return false;
      }
    }
    // 2. Sample coverage (deterministic hash approximation; no stencil side
    //    effects on failure). Pass iff (h < value) XOR invert.
    if (this.sampleCoverage.enabled) {
      const pass = coverageHash(x, y) < this.sampleCoverage.value;
      if (pass === this.sampleCoverage.invert) return false;
    }
    // 3. Stencil test + fail op. No stencil attachment → test passes.
    let stIdx = 0;
    let stCur = 0;
    let stFace: StencilFaceState | null = null;
    if (this.stencilEnabled && this.stencilData) {
      stFace = frontFacing ? this.dc.stencilTest.front : this.dc.stencilTest.back;
      stIdx = y * this.stencilWidth + x;
      stCur = this.stencilData[stIdx];
      if (!depthPass(stFace.func, stFace.ref & stFace.valueMask, stCur & stFace.valueMask)) {
        this.applyStencilOpAt(stIdx, stCur, stFace, stFace.fail);
        return false;
      }
    }
    // 4. Depth test (READ only; deferred to finalize() for gl_FragDepth
    //    shaders, whose depth is unknown until the shader runs). No depth
    //    attachment → test passes.
    if (!this.usesFragDepth && this.depthTestEnabled && this.depthData) {
      if (!depthPass(this.dc.depthTest.func, depth, this.depthData[y * this.depthWidth + x])) {
        if (stFace) this.applyStencilOpAt(stIdx, stCur, stFace, stFace.zfail);
        return false;
      }
    }
    // Passed stencil + depth: count the sample (non-fragDepth shaders only;
    // fragDepth shaders count in finalize() after the post-shader test).
    if (this.sampleCountRef && !this.usesFragDepth) this.sampleCountRef.value++;
    return true;
  }

  /** Depth write + stencil zpass + blend + sRGB + colorMask + write. */
  finalize(
    x: number, y: number, frontFacing: boolean, depth: number,
    colors: readonly Float32Array[],
  ): void {
    // Stencil face/current value for the zpass/zfail ops (ops only apply when
    // the stencil test is enabled AND a stencil buffer exists).
    let stIdx = 0;
    let stCur = 0;
    let stFace: StencilFaceState | null = null;
    if (this.stencilEnabled && this.stencilData) {
      stFace = frontFacing ? this.dc.stencilTest.front : this.dc.stencilTest.back;
      stIdx = y * this.stencilWidth + x;
      stCur = this.stencilData[stIdx];
    }
    if (this.usesFragDepth) {
      // Post-shader depth test with the shader-computed depth.
      if (this.depthTestEnabled && this.depthData) {
        if (!depthPass(this.dc.depthTest.func, depth, this.depthData[y * this.depthWidth + x])) {
          if (stFace) this.applyStencilOpAt(stIdx, stCur, stFace, stFace.zfail);
          return;
        }
      }
      if (this.sampleCountRef) this.sampleCountRef.value++;
      if (stFace) this.applyStencilOpAt(stIdx, stCur, stFace, stFace.zpass);
      if (this.depthMask && this.depthData) this.depthData[y * this.depthWidth + x] = depth;
    } else {
      // Depth test already passed in test(): zpass op + depth write.
      if (stFace) this.applyStencilOpAt(stIdx, stCur, stFace, stFace.zpass);
      if (this.depthMask && this.depthData) this.depthData[y * this.depthWidth + x] = depth;
    }
    // Color writes (per output location; DRAW_BUFFERi = NONE skips the write).
    const n = Math.min(colors.length, this.drawBuffers.length);
    for (let L = 0; L < n; L++) this.writeColor(L, x, y, colors);
  }

  /** Stencil write with writeMask: (newVal & mask) | (current & ~mask). */
  private applyStencilOpAt(idx: number, cur: number, face: StencilFaceState, op: GLenum): void {
    const v = applyStencilOp(op, cur, face.ref);
    this.stencilData![idx] = (v & face.writeMask) | (cur & ~face.writeMask);
  }

  /**
   * Writes one fragment color output (location L) through the full pipeline:
   * blend (per-draw-buffer state when the DrawCall carries it, else the global
   * BLEND state — always in linear space for sRGB targets) → sRGB encode (RGB
   * only) → colorMask → surface write. (DITHER is a no-op — see header.)
   */
  private writeColor(L: number, x: number, y: number, colors: readonly Float32Array[]): void {
    const db = this.drawBuffers[L];
    if (db === -1) return; // DRAW_BUFFERi = NONE
    // Defensive normalization for legacy/raw draw-buffer enums: gl.drawBuffers([BACK])
    // on the default framebuffer historically stored BACK (0x0405) verbatim, so the
    // draw path's `db - COLOR_ATTACHMENT0` normalization produced a large NEGATIVE
    // index, and any raw positive enum (e.g. 0x0405 = 1029) leaks through as
    // out-of-range. Both silently dropped ALL color writes here. Treat any such
    // value as attachment 0 (the default framebuffer's single color target); the
    // semantic fix (BACK → COLOR_ATTACHMENT0 normalization) lives gl-side in
    // src/gl/api/framebuffers.ts.
    let attach = db;
    if ((attach < 0 && attach !== -1) || attach >= this.fbColors.length) attach = 0;
    const tgt = this.fbColors[attach];
    if (!tgt) return;
    const src = colors[L];
    if (!src) return;
    const info = tgt.info;
    const off = (y * tgt.width + x) * info.bytesPerPixel;
    let r = src[0], g = src[1], b = src[2], a = src[3];
    const isSRGB = info.isSRGB;
    // Per-draw-buffer blend (OES_draw_buffers_indexed): when the DrawCall
    // carries per-draw-buffer entries, output L blends with entry L (buffer 0's
    // entry always mirrors the global BLEND cap); otherwise the global blend
    // state applies to every output (core GLES 3.0: BLEND applies to all draw
    // buffers).
    const bEntry = this.blendPerDrawBuffer ? this.blendPerDrawBuffer[L] : undefined;
    const blendHere = bEntry ? bEntry.enabled : this.blend.enabled;
    if (blendHere) {
      // Decode the destination (as-stored values; dstScratch keeps them for
      // the colorMask read-modify-write below), linearize for sRGB targets,
      // blend in linear space.
      info.decode(tgt.data, off, this.dstScratch);
      let dr = this.dstScratch[0], dg = this.dstScratch[1], db = this.dstScratch[2], da = this.dstScratch[3];
      if (isSRGB) {
        dr = sRGBToLinear(dr); dg = sRGBToLinear(dg); db = sRGBToLinear(db);
        // Alpha is NOT sRGB-encoded (GLES 3.0 §4.1.8: RGB only).
      }
      this.linearDst[0] = dr; this.linearDst[1] = dg; this.linearDst[2] = db; this.linearDst[3] = da;
      blendColor(
        src, this.linearDst, this.blendOut,
        bEntry ? bEntry.srcRGB : this.blend.srcRGB,
        bEntry ? bEntry.dstRGB : this.blend.dstRGB,
        bEntry ? bEntry.srcAlpha : this.blend.srcAlpha,
        bEntry ? bEntry.dstAlpha : this.blend.dstAlpha,
        bEntry ? bEntry.eqRGB : this.blend.eqRGB,
        bEntry ? bEntry.eqAlpha : this.blend.eqAlpha,
        bEntry ? bEntry.color : this.blend.color,
      );
      r = this.blendOut[0]; g = this.blendOut[1]; b = this.blendOut[2]; a = this.blendOut[3];
    }
    // Fragment colors are linear; sRGB targets convert on write (RGB only).
    if (isSRGB) {
      r = linearToSRGB(r); g = linearToSRGB(g); b = linearToSRGB(b);
    }
    // colorMask: masked channels keep their CURRENT surface value
    // (dstScratch holds the as-stored decode from the blend path above, or a
    // fresh decode in the non-blend path).
    const mask = this.colorMask[L];
    if (mask && !(mask[0] && mask[1] && mask[2] && mask[3])) {
      if (!blendHere) info.decode(tgt.data, off, this.dstScratch);
      if (!mask[0]) r = this.dstScratch[0];
      if (!mask[1]) g = this.dstScratch[1];
      if (!mask[2]) b = this.dstScratch[2];
      if (!mask[3]) a = this.dstScratch[3];
    }
    info.encode(tgt.data, off, r, g, b, a);
  }
}

/* ================================================================== */
/* Fragment drivers (runQuad / runFragment)                            */
/* ================================================================== */

/**
 * Quad fragment driver. Runs the fragment shader for a 2×2 quad of pixels
 * with origins at (qx, qy) (pixels (qx,qy),(qx+1,qy),(qx,qy+1),(qx+1,qy+1)).
 * `rs.quadV/quadDepth/quadW/quadPointCoord` hold the 4 precomputed per-pixel
 * values (see RasterState) and `rs.frontFacing` the primitive facing.
 * `inside` is a 4-bit mask (bit p = pixel p inside the primitive).
 *
 *  - Inside pixels: ops.test → shader → (if not discarded) ops.finalize.
 *  - Outside pixels are HELPER INVOCATIONS (run whenever
 *    program.fragment.usesDerivatives so inside pixels get correct
 *    derivatives); their outputs are always discarded and they never call
 *    ops.test (no stencil/depth/occlusion side effects).
 *  - Per-invocation ctx setup: fragCoord, varyings (+ddx/ddy from the quad),
 *    pointCoord, discarded=false; helper pixels may read anything but must
 *    not affect the target.
 */
export function runQuad(rs: RasterState, qx: number, qy: number, inside: number): void {
  const dc = rs.dc;
  const prog = dc.program;
  const usesDeriv = prog.fragment.usesDerivatives;
  const ctx = rs.fragCtx;
  const outColors = ctx.out.color;
  const nOut = outColors.length;
  if (inside === 0 && !usesDeriv) return; // nothing inside, no helpers needed
  for (let p = 0; p < 4; p++) {
    const x = qx + (p & 1);
    const y = qy + (p >> 1);
    const pass = ((inside >> p) & 1) !== 0 && rs.ops.test(x, y, rs.frontFacing, rs.quadDepth[p]);
    // Helper invocations (outside pixels) run only to provide derivatives.
    if (pass || usesDeriv) {
      ctx.discarded = false;
      for (let l = 0; l < nOut; l++) {
        const c = outColors[l];
        c[0] = 0; c[1] = 0; c[2] = 0; c[3] = 1;
      }
      setupFragmentCtx(ctx, x, y, rs.quadDepth[p], rs.quadW[p], rs.quadV, rs.totalVaryComponents, p);
      ctx.pointCoord[0] = rs.quadPointCoord[2 * p];
      ctx.pointCoord[1] = rs.quadPointCoord[2 * p + 1];
      ctx.frontFacing = rs.frontFacing;
      prog.fragment.run(ctx);
    }
    if (pass && !ctx.discarded) {
      rs.ops.finalize(
        x, y, rs.frontFacing,
        prog.fragment.usesFragDepth ? ctx.out.fragDepth : rs.quadDepth[p],
        ctx.out.color,
      );
    }
  }
}

/**
 * Runs the fragment shader for a single pixel (non-quad fast path — only
 * valid when !program.fragment.usesDerivatives). `varyBase` is the float
 * offset into rs.quadV of this pixel's precomputed interpolated varyings
 * (pixel = varyBase / rs.totalVaryComponents).
 */
export function runFragment(
  rs: RasterState, x: number, y: number, depth: number, w: number,
  varyBase: number,
): void {
  if (!rs.ops.test(x, y, rs.frontFacing, depth)) return;
  const dc = rs.dc;
  const ctx = rs.fragCtx;
  const total = rs.totalVaryComponents;
  const pixel = total > 0 ? varyBase / total : 0;
  ctx.discarded = false;
  const outColors = ctx.out.color;
  for (let l = 0; l < outColors.length; l++) {
    const c = outColors[l];
    c[0] = 0; c[1] = 0; c[2] = 0; c[3] = 1;
  }
  setupFragmentCtx(ctx, x, y, depth, w, rs.quadV, total, pixel);
  ctx.pointCoord[0] = rs.quadPointCoord[2 * pixel];
  ctx.pointCoord[1] = rs.quadPointCoord[2 * pixel + 1];
  ctx.frontFacing = rs.frontFacing;
  dc.program.fragment.run(ctx);
  if (!ctx.discarded) {
    rs.ops.finalize(
      x, y, rs.frontFacing,
      dc.program.fragment.usesFragDepth ? ctx.out.fragDepth : depth,
      ctx.out.color,
    );
  }
}

/** Fills ctx for one pixel from the quad scratch (see RasterState layout). */
export function setupFragmentCtx(
  ctx: FragmentExecCtx, x: number, y: number, depth: number, w: number,
  quadV: Float32Array, quadStride: number, pixel: number,
): void {
  const fc = ctx.fragCoord;
  // gl_FragCoord.xy = window pixel center (GLSL ES 1.00 §7.1 / 3.00 §7.1):
  // (x + 0.5, y + 0.5) for the pixel at integer window coords (x, y).
  fc[0] = x + 0.5; fc[1] = y + 0.5; fc[2] = depth; fc[3] = w;
  const base = pixel * quadStride;
  const varyings = ctx.varyings;
  let offset = 0;
  for (let i = 0; i < varyings.length; i++) {
    const vi = varyings[i];
    const v = vi.v;
    const n = v.length;
    for (let j = 0; j < n; j++) v[j] = quadV[base + offset + j];
    // Derivatives = neighbor difference across the quad (x-neighbor ^1,
    // y-neighbor ^2). Flat varyings have identical quad values → zero.
    const ddx = vi.ddx;
    const ddy = vi.ddy;
    if (ddx && ddy) {
      const xb = (pixel ^ 1) * quadStride + offset;
      const yb = (pixel ^ 2) * quadStride + offset;
      for (let j = 0; j < n; j++) {
        ddx[j] = quadV[xb + j] - quadV[base + offset + j];
        ddy[j] = quadV[yb + j] - quadV[base + offset + j];
      }
    }
    offset += n;
  }
}

/** Convenience for gl/: builds the FragmentOps instance for a draw call. */
export function createFragmentOps(dc: DrawCall): FragmentOps {
  return new FragmentOpsImpl(dc);
}

/* ================================================================== */
/* Clear (glClear / glClearBuffer*) — respects scissor + masks         */
/* ================================================================== */

/** Scissored region of a surface (null or disabled scissor → full surface),
 *  clamped to surface bounds. Returns [x0, y0, x1, y1). */
function scissorRect(s: Surface, scissor: ScissorState | null): [number, number, number, number] {
  if (scissor && scissor.enabled) {
    return [
      Math.max(0, scissor.x),
      Math.max(0, scissor.y),
      Math.min(s.width, scissor.x + scissor.w),
      Math.min(s.height, scissor.y + scissor.h),
    ];
  }
  return [0, 0, s.width, s.height];
}

const _clearTmp = new Float32Array(4);

export function clearColorSurface(
  s: Surface, r: number, g: number, b: number, a: number,
  scissor: ScissorState | null, mask: ColorMask,
): void {
  const [x0, y0, x1, y1] = scissorRect(s, scissor);
  if (x0 >= x1 || y0 >= y1) return;
  const info = s.info;
  const bpp = info.bytesPerPixel;
  const w = s.width;
  if (mask[0] && mask[1] && mask[2] && mask[3]) {
    for (let y = y0; y < y1; y++) {
      let off = (y * w + x0) * bpp;
      for (let x = x0; x < x1; x++) {
        info.encode(s.data, off, r, g, b, a);
        off += bpp;
      }
    }
  } else {
    // Partial mask: read-modify-write (masked channels keep current value).
    for (let y = y0; y < y1; y++) {
      let off = (y * w + x0) * bpp;
      for (let x = x0; x < x1; x++) {
        info.decode(s.data, off, _clearTmp);
        info.encode(s.data, off,
          mask[0] ? r : _clearTmp[0],
          mask[1] ? g : _clearTmp[1],
          mask[2] ? b : _clearTmp[2],
          mask[3] ? a : _clearTmp[3]);
        off += bpp;
      }
    }
  }
}

export function clearDepthSurface(
  s: Surface, depth: number, scissor: ScissorState | null, depthMask: boolean,
): void {
  if (!depthMask) return;
  const [x0, y0, x1, y1] = scissorRect(s, scissor);
  if (x0 >= x1 || y0 >= y1) return;
  const data = getDepthData(s);
  const w = s.width;
  for (let y = y0; y < y1; y++) {
    const row = y * w;
    for (let x = x0; x < x1; x++) data[row + x] = depth;
  }
}

export function clearStencilSurface(
  s: Surface, value: number, scissor: ScissorState | null, writeMask: number,
): void {
  const [x0, y0, x1, y1] = scissorRect(s, scissor);
  if (x0 >= x1 || y0 >= y1) return;
  const data = getStencilData(s);
  const w = s.width;
  if (writeMask === 0xff) {
    const v = value & 0xff;
    for (let y = y0; y < y1; y++) {
      const row = y * w;
      for (let x = x0; x < x1; x++) data[row + x] = v;
    }
  } else {
    for (let y = y0; y < y1; y++) {
      const row = y * w;
      for (let x = x0; x < x1; x++) {
        const idx = row + x;
        data[idx] = (value & writeMask) | (data[idx] & ~writeMask);
      }
    }
  }
}

/* ================================================================== */
/* Blit (blitFramebuffer) — color with optional filtering,             */
/* depth/stencil plain copies.                                         */
/* ================================================================== */

const _blitT0 = new Float32Array(4);
const _blitT1 = new Float32Array(4);
const _blitT2 = new Float32Array(4);
const _blitT3 = new Float32Array(4);
const _blitOut = new Float32Array(4);

export function blitColorSurface(
  src: Surface, dst: Surface,
  srcX: number, srcY: number, srcW: number, srcH: number,
  dstX: number, dstY: number, dstW: number, dstH: number,
  filter: 'nearest' | 'linear',
): void {
  // Negative srcW/srcH/dstW/dstH are legal (GLES 3.0 §4.3.2: the rect spans
  // X0..X0+W, mirroring the image); only zero-size rects are degenerate.
  if (srcW === 0 || srcH === 0 || dstW === 0 || dstH === 0) return;
  const sw = src.width, sh = src.height;
  const dw = dst.width, dh = dst.height;
  // Iterate the destination rect (sign-aware: [min(X,X+W), max(X,X+W))) clipped
  // to dst bounds. Rows map directly: both surfaces have row 0 = BOTTOM.
  const x0 = Math.max(0, Math.min(dstX, dstX + dstW));
  const y0 = Math.max(0, Math.min(dstY, dstY + dstH));
  const x1 = Math.min(dw, Math.max(dstX, dstX + dstW));
  const y1 = Math.min(dh, Math.max(dstY, dstY + dstH));
  if (x0 >= x1 || y0 >= y1) return;
  const sInfo = src.info;
  const dInfo = dst.info;
  const sbpp = sInfo.bytesPerPixel;
  const dbpp = dInfo.bytesPerPixel;
  // sRGB format conversion is part of the copy (GLES 3.0 §4.3.2): sRGB sources
  // decode to linear, sRGB destinations re-encode the linear result. Alpha is
  // never sRGB-encoded (GLES 3.0 §4.1.8). Same-format sRGB→sRGB round-trips
  // through the identity.
  const decodeSRGB = sInfo.isSRGB;
  const encodeSRGB = dInfo.isSRGB;
  // GLES 3.0 §4.3.2: same-size rects blit WITHOUT filtering (as NEAREST) even
  // when LINEAR was requested — bilinear weights are exactly 0/1 there and the
  // sRGB decode→filter→encode round trip would corrupt bytes that a plain copy
  // keeps byte-exact. Integer formats never filter (gl/ validates; linear on
  // integer = nearest). The per-pixel sRGB conversion above still applies to
  // same-size copies.
  const doLinear = filter === 'linear' && !sInfo.isInteger &&
    (srcW !== dstW || srcH !== dstH);
  if (doLinear) {
    // Bilinear with CLAMP_TO_EDGE: continuous source coord
    //   u = (d − dstOrigin + 0.5)·srcSize/dstSize − 0.5
    // texels = srcOrigin + floor(u), weight = frac(u). Integer formats fall
    // through to nearest (gl/ validates; linear on integer = nearest).
    // A destination pixel is copied ONLY when its mapped CENTER lies inside
    // the source surface (the rect condition is implied: the affine map of the
    // iterated dst-rect pixels always falls inside the source rect). Pixels
    // whose center lands outside retain their previous value (GLES 3.0 §4.3.2;
    // CTS blitframebuffer-outside-readbuffer / -filter-outofbounds).
    for (let dy = y0; dy < y1; dy++) {
      const vc = srcY + (dy - dstY + 0.5) * srcH / dstH;
      if (vc < 0 || vc >= sh) continue;
      const fy = (dy - dstY + 0.5) * srcH / dstH - 0.5;
      const sy0f = Math.floor(fy);
      const wy = fy - sy0f;
      let sy0 = srcY + sy0f;
      let sy1 = sy0 + 1;
      sy0 = Math.max(0, Math.min(sh - 1, sy0));
      sy1 = Math.max(0, Math.min(sh - 1, sy1));
      let doff = (dy * dw + x0) * dbpp;
      for (let dx = x0; dx < x1; dx++) {
        const uc = srcX + (dx - dstX + 0.5) * srcW / dstW;
        if (uc < 0 || uc >= sw) { doff += dbpp; continue; }
        const fx = (dx - dstX + 0.5) * srcW / dstW - 0.5;
        const sx0f = Math.floor(fx);
        const wx = fx - sx0f;
        let sx0 = srcX + sx0f;
        let sx1 = sx0 + 1;
        sx0 = Math.max(0, Math.min(sw - 1, sx0));
        sx1 = Math.max(0, Math.min(sw - 1, sx1));
        sInfo.decode(src.data, (sy0 * sw + sx0) * sbpp, _blitT0);
        sInfo.decode(src.data, (sy0 * sw + sx1) * sbpp, _blitT1);
        sInfo.decode(src.data, (sy1 * sw + sx0) * sbpp, _blitT2);
        sInfo.decode(src.data, (sy1 * sw + sx1) * sbpp, _blitT3);
        if (decodeSRGB) {
          // sRGB sources filter in LINEAR space (RGB only — alpha is never
          // sRGB-encoded per GLES 3.0 §4.1.8).
          _blitT0[0] = sRGBToLinear(_blitT0[0]); _blitT0[1] = sRGBToLinear(_blitT0[1]); _blitT0[2] = sRGBToLinear(_blitT0[2]);
          _blitT1[0] = sRGBToLinear(_blitT1[0]); _blitT1[1] = sRGBToLinear(_blitT1[1]); _blitT1[2] = sRGBToLinear(_blitT1[2]);
          _blitT2[0] = sRGBToLinear(_blitT2[0]); _blitT2[1] = sRGBToLinear(_blitT2[1]); _blitT2[2] = sRGBToLinear(_blitT2[2]);
          _blitT3[0] = sRGBToLinear(_blitT3[0]); _blitT3[1] = sRGBToLinear(_blitT3[1]); _blitT3[2] = sRGBToLinear(_blitT3[2]);
        }
        const w00 = (1 - wx) * (1 - wy);
        const w10 = wx * (1 - wy);
        const w01 = (1 - wx) * wy;
        const w11 = wx * wy;
        _blitOut[0] = _blitT0[0] * w00 + _blitT1[0] * w10 + _blitT2[0] * w01 + _blitT3[0] * w11;
        _blitOut[1] = _blitT0[1] * w00 + _blitT1[1] * w10 + _blitT2[1] * w01 + _blitT3[1] * w11;
        _blitOut[2] = _blitT0[2] * w00 + _blitT1[2] * w10 + _blitT2[2] * w01 + _blitT3[2] * w11;
        _blitOut[3] = _blitT0[3] * w00 + _blitT1[3] * w10 + _blitT2[3] * w01 + _blitT3[3] * w11;
        if (encodeSRGB) {
          _blitOut[0] = linearToSRGB(_blitOut[0]); _blitOut[1] = linearToSRGB(_blitOut[1]); _blitOut[2] = linearToSRGB(_blitOut[2]);
        }
        dInfo.encode(dst.data, doff, _blitOut[0], _blitOut[1], _blitOut[2], _blitOut[3]);
        doff += dbpp;
      }
    }
  } else {
    // NEAREST (also the fallback for integer formats with 'linear'). Same
    // center-outside semantics as the LINEAR path: pixels whose mapped center
    // lies outside the source surface are never written.
    for (let dy = y0; dy < y1; dy++) {
      const vc = srcY + (dy - dstY + 0.5) * srcH / dstH;
      if (vc < 0 || vc >= sh) continue;
      const sy = Math.max(0, Math.min(sh - 1, srcY + Math.floor((dy - dstY + 0.5) * srcH / dstH)));
      let doff = (dy * dw + x0) * dbpp;
      for (let dx = x0; dx < x1; dx++) {
        const uc = srcX + (dx - dstX + 0.5) * srcW / dstW;
        if (uc >= 0 && uc < sw) {
          const sx = Math.max(0, Math.min(sw - 1, srcX + Math.floor((dx - dstX + 0.5) * srcW / dstW)));
          sInfo.decode(src.data, (sy * sw + sx) * sbpp, _blitOut);
          if (decodeSRGB) {
            _blitOut[0] = sRGBToLinear(_blitOut[0]); _blitOut[1] = sRGBToLinear(_blitOut[1]); _blitOut[2] = sRGBToLinear(_blitOut[2]);
          }
          if (encodeSRGB) {
            _blitOut[0] = linearToSRGB(_blitOut[0]); _blitOut[1] = linearToSRGB(_blitOut[1]); _blitOut[2] = linearToSRGB(_blitOut[2]);
          }
          dInfo.encode(dst.data, doff, _blitOut[0], _blitOut[1], _blitOut[2], _blitOut[3]);
        }
        doff += dbpp;
      }
    }
  }
}

export function blitDepthStencilSurface(
  src: Surface, dst: Surface,
  srcX: number, srcY: number, srcW: number, srcH: number,
  dstX: number, dstY: number, dstW: number, dstH: number,
): void {
  if (srcW <= 0 || srcH <= 0 || dstW <= 0 || dstH <= 0) return;
  const sw = src.width, sh = src.height;
  const dw = dst.width, dh = dst.height;
  const x0 = Math.max(0, dstX);
  const y0 = Math.max(0, dstY);
  const x1 = Math.min(dw, dstX + dstW);
  const y1 = Math.min(dh, dstY + dstH);
  if (x0 >= x1 || y0 >= y1) return;
  // Plain copies of the planes that exist on BOTH sides (nearest rect
  // mapping, no filtering). Depth-stencil surfaces copy both planes.
  const srcD = src.info.isDepth && dst.info.isDepth ? getDepthData(src) : null;
  const dstD = srcD ? getDepthData(dst) : null;
  const srcS = src.info.isStencil && dst.info.isStencil ? getStencilData(src) : null;
  const dstS = srcS ? getStencilData(dst) : null;
  if (!srcD && !srcS) return;
  for (let dy = y0; dy < y1; dy++) {
    const sy = Math.max(0, Math.min(sh - 1, srcY + Math.floor((dy - dstY + 0.5) * srcH / dstH)));
    const srow = sy * sw;
    const drow = dy * dw;
    for (let dx = x0; dx < x1; dx++) {
      const sx = Math.max(0, Math.min(sw - 1, srcX + Math.floor((dx - dstX + 0.5) * srcW / dstW)));
      const si = srow + sx;
      const di = drow + dx;
      if (srcD && dstD) dstD[di] = srcD[si];
      if (srcS && dstS) dstS[di] = srcS[si];
    }
  }
}
