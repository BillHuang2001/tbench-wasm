/**
 * rasterizer.ts — the draw driver (contract §2 entry point).
 *
 * draw(dc) flow:
 *  1. If dc.rasterizerDiscard → return (no fragments; the fragment shader is
 *     not executed at all).
 *  2. Build the per-draw RasterState (fragment ctx, FragmentOps, scratch,
 *     texture env) — allocated once per draw, never per fragment.
 *  3. Primitive assembly by dc.mode over [first, first+count):
 *       TRIANGLES        (v, v+1, v+2)   — 3i
 *       TRIANGLE_STRIP   (v+i, v+i+1, v+i+2), winding flips on odd i
 *       TRIANGLE_FAN     (v, v+i+1, v+i+2)
 *       LINES / LINE_STRIP / LINE_LOOP / POINTS
 *     Assembled records are copied into a scratch primitive buffer so that
 *     flat provoking-vertex fixup and clipping can mutate them freely.
 *  4. For each instance i: vertex base = first + i*count (records).
 *  5. Per primitive: copy provoking-vertex (LAST vertex per GLES) flat
 *     varying values into every vertex; clip (all 6 planes, clip.ts);
 *     apply the viewport transform; cull (window-space signed area vs
 *     cull.face/frontFace — strips need the winding parity); dispatch to
 *     rasterizeTriangle / rasterizeLine / rasterizePoint.
 *
 * Culling happens AFTER clipping per spec. `first` semantics and instancing
 * addressing are documented on DrawCall in types.ts.
 */

import type { DrawCall, FragmentExecCtx, RasterState, SamplerState, TextureImage, VaryingInterpolant } from './types';
import {
  POINTS, LINES, LINE_LOOP, LINE_STRIP, TRIANGLES, TRIANGLE_STRIP, TRIANGLE_FAN,
  FRONT, BACK, FRONT_AND_BACK, CW, CCW,
  NEAREST_MIPMAP_LINEAR, LINEAR, REPEAT, NONE, LEQUAL,
  UPPER_LEFT_EXT,
} from './gl-enums';
import { clipPrimitive, pointIsVisible, applyViewportTransform, MAX_CLIPPED_VERTICES } from './clip';
import { rasterizeTriangle, signedArea2 } from './triangles';
import { rasterizeLine } from './lines';
import { rasterizePoint } from './points';
import { createFragmentOps } from './fragment-ops';
import { createTextureEnv } from './sampler-env';

/**
 * Additive DrawCall fields from the raster contract (fragment uniform store,
 * uniform block stores, occlusion query counter) that are not (yet) declared
 * on the DrawCall interface in types.ts. Raster only passes them through —
 * intersect locally so this module compiles against the current types.
 */
type DrawCallExt = DrawCall & {
  uniforms: Float32Array;
  uniformBlocks?: Record<string, ArrayBufferView>;
  sampleCountRef?: { value: number };
};

/* ------------------------------------------------------------------ */
/* Module-level per-draw resources (draws are sequential; sharing is    */
/* safe and avoids per-draw GC churn in the three.js/Babylon suites).   */
/* ------------------------------------------------------------------ */

/** Minimum size of the codegen scratch buffers (grow as programs need more). */
const MIN_SCRATCH = 64;

/** Codegen float scratch for fragment execution (shared across draws). */
let scratch = new Float32Array(MIN_SCRATCH);
/** Codegen int scratch for fragment execution (shared across draws). */
let intScratch = new Int32Array(MIN_SCRATCH);

/** Growing zero-filled stores for unbound uniform blocks (read-only for codegen). */
let zeroStore = new Float32Array(0);
let zeroIntStore = new Int32Array(0);

/** Empty default-block int store fallback (programs without int uniforms). */
const EMPTY_INT_STORE = new Int32Array(0);

/**
 * Default effective sampler state for texture units with nothing bound
 * (GL spec defaults; matches gl/'s effectiveSamplerState({}, null)).
 */
const DEFAULT_STATE: SamplerState = {
  minFilter: NEAREST_MIPMAP_LINEAR, // 0x2702
  magFilter: LINEAR, // 0x2601
  wrapS: REPEAT, // 0x2901
  wrapT: REPEAT, // 0x2901
  wrapR: REPEAT, // 0x2901
  minLod: -1000,
  maxLod: 1000,
  compareMode: NONE, // 0
  compareFunc: LEQUAL, // 0x0203
  maxAnisotropy: 1,
};

/** Ensures the module-level scratch buffers can hold `n` floats / ints. */
function ensureScratch(n: number, intN: number): void {
  if (scratch.length < n) scratch = new Float32Array(Math.max(n, MIN_SCRATCH));
  if (intScratch.length < intN) intScratch = new Int32Array(Math.max(intN, MIN_SCRATCH));
}

/** Zero-filled float store of at least `n` elements (growing, shared). */
function getZeroStore(n: number): Float32Array {
  if (zeroStore.length < n) zeroStore = new Float32Array(n);
  return zeroStore;
}

/** Zero-filled int store of at least `n` elements (growing, shared). */
function getZeroIntStore(n: number): Int32Array {
  if (zeroIntStore.length < n) zeroIntStore = new Int32Array(n);
  return zeroIntStore;
}

/** Rasterizes one draw call (contract §2 `rasterizer.draw`). */
export function draw(dc: DrawCall): void {
  if (dc.rasterizerDiscard) return; // step 1: no fragments, no shader runs, nothing counted

  const rs = createRasterState(dc);
  const stride = dc.vertexStride;
  const { first, count, instanceCount, mode } = dc;

  // Per-draw scratch: one primitive buffer (3 records) + two clip buffers of
  // MAX_CLIPPED_VERTICES records each (see clip.ts). Allocated once per draw.
  const primBuf = new Float32Array(3 * stride);
  const clipA = new Float32Array(MAX_CLIPPED_VERTICES * stride);
  const clipB = new Float32Array(MAX_CLIPPED_VERTICES * stride);

  // Flat-varying component ranges as [startFloat, componentCount] pairs, with
  // startFloat an ABSOLUTE float index within the record.
  const flatRanges: [number, number][] = [];
  {
    let off = dc.varyingsOffset;
    for (const v of dc.program.varyings) {
      if (v.flat) flatRanges.push([off, v.components]);
      off += v.components;
    }
  }

  // Instance i, vertex j (0-based within the draw) → record first + i*count + j.
  for (let i = 0; i < instanceCount; i++) {
    const runBase = first + i * count;
    switch (mode) {
      case POINTS:
        for (let j = 0; j < count; j++) {
          emitPoint(dc, rs, primBuf, stride, runBase + j);
        }
        break;
      case LINES:
        for (let j = 0; j + 1 < count; j += 2) {
          emitLine(dc, rs, primBuf, clipA, clipB, stride, flatRanges, runBase + j, runBase + j + 1);
        }
        break;
      case LINE_STRIP:
        for (let j = 0; j + 1 < count; j++) {
          emitLine(dc, rs, primBuf, clipA, clipB, stride, flatRanges, runBase + j, runBase + j + 1);
        }
        break;
      case LINE_LOOP:
        for (let j = 0; j + 1 < count; j++) {
          emitLine(dc, rs, primBuf, clipA, clipB, stride, flatRanges, runBase + j, runBase + j + 1);
        }
        if (count >= 2) {
          emitLine(dc, rs, primBuf, clipA, clipB, stride, flatRanges, runBase + count - 1, runBase);
        }
        break;
      case TRIANGLES:
        for (let j = 0; j + 2 < count; j += 3) {
          emitTriangle(dc, rs, primBuf, clipA, clipB, stride, flatRanges,
            runBase + j, runBase + j + 1, runBase + j + 2);
        }
        break;
      case TRIANGLE_STRIP:
        // Odd triangles swap the first two vertices per GLES 2.0 §2.3 so the
        // winding stays consistent across the strip.
        for (let j = 0; j + 2 < count; j++) {
          emitTriangle(dc, rs, primBuf, clipA, clipB, stride, flatRanges,
            runBase + j + (j & 1), runBase + j + 1 - (j & 1), runBase + j + 2);
        }
        break;
      case TRIANGLE_FAN:
        for (let j = 1; j + 1 < count; j++) {
          emitTriangle(dc, rs, primBuf, clipA, clipB, stride, flatRanges,
            runBase, runBase + j, runBase + j + 1);
        }
        break;
      default:
        // Unknown mode: nothing to draw (gl/ validates modes before drawing).
        break;
    }
  }
}

/**
 * Builds the per-draw execution state: the full FragmentExecCtx (varyings
 * interpolants, uniform stores + scratch + texture arrays per BaseExecCtx,
 * texture env), FragmentOps, and the quad scratch buffers. Allocates once per
 * draw call.
 */
export function createRasterState(dc: DrawCall): RasterState {
  const dcx = dc as DrawCallExt;

  // Per-varying interpolant arrays (one Float32Array per varying; ddx/ddy only
  // when the fragment shader uses derivatives).
  const varyings = dc.program.varyings;
  let totalVaryComponents = 0;
  const fragVaryings: VaryingInterpolant[] = [];
  for (let i = 0; i < varyings.length; i++) {
    const c = varyings[i].components;
    totalVaryComponents += c;
    const interp: VaryingInterpolant = { v: new Float32Array(c) };
    if (dc.program.fragment.usesDerivatives) {
      interp.ddx = new Float32Array(c);
      interp.ddy = new Float32Array(c);
    }
    fragVaryings.push(interp);
  }

  // One color scratch per output location (0..maxOutputLocation); codegen may
  // index any location, so always allocate at least one.
  let maxLocation = 0;
  for (const out of dc.program.fragment.outputs) {
    if (out.location > maxLocation) maxLocation = out.location;
  }
  const colorOuts: Float32Array[] = [];
  for (let i = 0; i <= maxLocation; i++) colorOuts.push(new Float32Array(4));

  // Default-block int store: glsl Program.intStore (FLOAT-index convention —
  // samplers/ints/bools/uints share it). Empty fallback for programs without
  // int uniforms (codegen never reads it then).
  const pm = dc.program;
  const intUniforms = pm.intStore ?? EMPTY_INT_STORE;

  // Uniform-block stores per BLOCK INDEX (contract §1). The DrawCall carries
  // them keyed by block NAME; convert using the program's uniformBlocks
  // metadata (name/index/size). Unbound blocks get a zero-filled store.
  const blocks = pm.uniformBlocks ?? [];
  const blockStores: Float32Array[] = new Array(blocks.length);
  const blockIntStores: Int32Array[] = new Array(blocks.length);
  const dcBlocks = dcx.uniformBlocks;
  for (let bi = 0; bi < blocks.length; bi++) {
    const view = dcBlocks ? dcBlocks[blocks[bi].name] : undefined;
    if (view) {
      const n = view.byteLength >>> 2;
      if (view.byteOffset % 4 === 0) {
        blockStores[bi] = view as Float32Array;
        blockIntStores[bi] = new Int32Array(view.buffer, view.byteOffset, n);
      } else {
        // Defensive: unaligned view — copy into an aligned buffer so the int
        // store shares its bytes bit-exactly (gl/ views are 4-byte aligned).
        const copy = new Float32Array(n);
        copy.set(new Float32Array(view.buffer, view.byteOffset, n));
        blockStores[bi] = copy;
        blockIntStores[bi] = new Int32Array(copy.buffer);
      }
    } else {
      const n = Math.max(Math.ceil(blocks[bi].size / 4), 4);
      blockStores[bi] = getZeroStore(n);
      blockIntStores[bi] = getZeroIntStore(n);
    }
  }

  // Per-unit texture images + effective sampler state (contract §1: null
  // image / default state when nothing bound).
  const texUnits = dc.textures;
  const textures: (TextureImage | null)[] = new Array(texUnits.length);
  const samplerStates: SamplerState[] = new Array(texUnits.length);
  for (let i = 0; i < texUnits.length; i++) {
    const t = texUnits[i];
    textures[i] = t ? t.img : null;
    samplerStates[i] = t ? t.state : DEFAULT_STATE;
  }

  // Codegen scratch (module-level, grows with the program's needs).
  ensureScratch(Math.max(pm.scratchSize ?? 0, MIN_SCRATCH), Math.max(pm.intScratchSize ?? 0, MIN_SCRATCH));

  const fragCtx: FragmentExecCtx = {
    varyings: fragVaryings,
    fragCoord: new Float32Array(4),
    frontFacing: true, // driver sets it per primitive
    pointCoord: new Float32Array(2),
    uniforms: dcx.uniforms,
    intUniforms,
    blockStores,
    blockIntStores,
    textures,
    samplerStates,
    scratch,
    intScratch,
    tex: createTextureEnv(dc.textures),
    out: { color: colorOuts, fragDepth: 0 },
    discarded: false,
  };

  return {
    dc,
    fragCtx,
    ops: createFragmentOps(dc),
    totalVaryComponents,
    frontFacing: true, // driver sets it per primitive
    quadV: new Float32Array(4 * totalVaryComponents),
    quadDepth: new Float32Array(4),
    quadW: new Float32Array(4),
    quadPointCoord: new Float32Array(8),
  };
}

/** Copies the provoking vertex's flat-varying values into all vertices of a
 *  primitive (called before clipping). Provoking vertex = LAST vertex. */
export function applyFlatFixup(
  buf: Float32Array, base: number, count: number, stride: number,
  varyingsOffset: number, flatRanges: readonly (readonly [number, number])[],
): void {
  if (count <= 1) return; // single-vertex primitives have no other vertices
  const last = base + (count - 1) * stride;
  for (let r = 0; r < flatRanges.length; r++) {
    const start = flatRanges[r][0];
    const n = flatRanges[r][1];
    for (let k = 0; k < count - 1; k++) {
      const dst = base + k * stride + start;
      const src = last + start;
      for (let c = 0; c < n; c++) buf[dst + c] = buf[src + c];
    }
  }
}

/* ================================================================== */
/* Internal per-primitive dispatch helpers                             */
/* ================================================================== */

/** Copies one vertex record (stride floats) between buffers, no allocation. */
function copyRecord(
  src: Float32Array, srcBase: number,
  dst: Float32Array, dstBase: number, stride: number,
): void {
  for (let i = 0; i < stride; i++) dst[dstBase + i] = src[srcBase + i];
}

/** Assembles, flat-fixes, clips, transforms, culls and rasterizes one triangle. */
function emitTriangle(
  dc: DrawCall, rs: RasterState, primBuf: Float32Array,
  clipA: Float32Array, clipB: Float32Array, stride: number,
  flatRanges: readonly (readonly [number, number])[],
  ia: number, ib: number, ic: number,
): void {
  const src = dc.vertices;
  copyRecord(src, ia * stride, primBuf, 0, stride);
  copyRecord(src, ib * stride, primBuf, stride, stride);
  copyRecord(src, ic * stride, primBuf, 2 * stride, stride);

  applyFlatFixup(primBuf, 0, 3, stride, dc.varyingsOffset, flatRanges);

  const nv = clipPrimitive(primBuf, 0, stride, 3, clipA, clipB, 0, dc.clipDepthMode);
  if (nv === 0) return;
  applyViewportTransform(clipB, 0, stride, nv, dc.viewport, dc.depthRange, dc.clipOrigin, dc.clipDepthMode);

  // Facing + culling in window space (after clipping AND viewport transform).
  // signedArea2 expects FLOAT offsets into the packed record buffer (record k
  // lives at k*stride) — matching the rasterizeTriangle fan-loop calls below.
  // EXT_clip_control §13.7.1: with UPPER_LEFT_EXT the facing area is the
  // window-space area multiplied by -1 (the y flip already negates it, so the
  // factor cancels: front faces are unchanged by the clip origin).
  let area = signedArea2(clipB, 0, stride, 2 * stride, stride);
  if (dc.clipOrigin === UPPER_LEFT_EXT) area = -area;
  const frontFacing = (dc.cull.frontFace === CCW) ? area > 0 : area < 0;
  if (dc.cull.enabled) {
    const face = dc.cull.face;
    if (face === FRONT_AND_BACK || (face === FRONT && frontFacing) || (face === BACK && !frontFacing)) {
      return;
    }
  }
  rs.frontFacing = frontFacing;

  // Clip results are fan-able convex polygons (up to 7 vertices) — rasterize
  // as a fan from vertex 0 (reduces to (0,1,2) for unclipped triangles).
  for (let k = 1; k + 1 < nv; k++) {
    rasterizeTriangle(clipB, 0, k * stride, (k + 1) * stride, stride, rs);
  }
}

/** Assembles, flat-fixes, clips, transforms and rasterizes one line segment. */
function emitLine(
  dc: DrawCall, rs: RasterState, primBuf: Float32Array,
  clipA: Float32Array, clipB: Float32Array, stride: number,
  flatRanges: readonly (readonly [number, number])[],
  ia: number, ib: number,
): void {
  const src = dc.vertices;
  copyRecord(src, ia * stride, primBuf, 0, stride);
  copyRecord(src, ib * stride, primBuf, stride, stride);

  applyFlatFixup(primBuf, 0, 2, stride, dc.varyingsOffset, flatRanges);

  const nv = clipPrimitive(primBuf, 0, stride, 2, clipA, clipB, 0, dc.clipDepthMode);
  if (nv === 0) return;
  applyViewportTransform(clipB, 0, stride, nv, dc.viewport, dc.depthRange, dc.clipOrigin, dc.clipDepthMode);

  rs.frontFacing = false; // undefined for lines
  rasterizeLine(clipB, 0, stride, stride, rs);
}

/** Visibility-checks, transforms and rasterizes one point. */
function emitPoint(
  dc: DrawCall, rs: RasterState, primBuf: Float32Array,
  stride: number, ia: number,
): void {
  copyRecord(dc.vertices, ia * stride, primBuf, 0, stride);

  // Points are not polygon-clipped; only the 6-plane visibility test applies.
  if (!pointIsVisible(primBuf, 0, stride, dc.clipDepthMode)) return;
  applyViewportTransform(primBuf, 0, stride, 1, dc.viewport, dc.depthRange, dc.clipOrigin, dc.clipDepthMode);

  rs.frontFacing = false; // undefined for points
  rasterizePoint(primBuf, 0, stride, rs);
}
