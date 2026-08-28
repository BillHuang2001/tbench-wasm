/**
 * sampler-env.ts — TextureEnv (codegen-facing sampling environment).
 *
 * createTextureEnv builds the per-draw unit table + scratch handed to
 * generated fragment code as `ctx.tex` (see TextureEnv in types.ts). Split
 * out of sampler.ts so the sampling core stays under the line limit; imports
 * the core (sampleTextureP / sampleTextureLodP / sampleTextureShadowP /
 * texelFetch / fillPlan) from sampler.ts and nothing from sampler-raw.ts
 * directly (the import graph stays acyclic).
 *
 * Per-draw resolve hoist: the unit table is fixed for the whole draw, so every
 * unit is resolved ONCE here (binding → img/state + sampling plan; null =
 * unbound, plan null = incomplete). Per-fragment sample* calls then do a
 * single array lookup + two null checks — no per-fragment resolve() object,
 * no completeness check, no plan computation, no filter-enum re-derivation.
 */

import type { SampleCoord, SamplerState, TextureEnv, TextureImage, TextureUnitBinding } from './types';
import { fillPlan, sampleTextureLodP, sampleTextureP, sampleTextureShadowP, texelFetch } from './sampler';
import type { SamplePlan } from './sampler';

/** One resolved texture unit (precomputed once per draw). `plan` null → image incomplete (default result, integer alpha rule). */
type ResolvedUnit =
  | { img: TextureImage; state: SamplerState; plan: SamplePlan }
  | { img: TextureImage; state: SamplerState; plan: null };

/**
 * Allocates the scratch (out/outInt/outUint share one ArrayBuffer, plus the
 * per-env coordinate scratch) and binds the per-draw unit table. Called once
 * per draw by the rasterizer. All methods are allocation-free.
 */
export function createTextureEnv(units: readonly (TextureUnitBinding | null)[]): TextureEnv {
  const out = new Float32Array(4);
  const outInt = new Int32Array(out.buffer);
  const outUint = new Uint32Array(out.buffer);
  const v = new Float32Array(4);
  const dx = new Float32Array(4);
  const dy = new Float32Array(4);
  const coord: SampleCoord = { v, dx, dy };

  const resolved: (ResolvedUnit | null)[] = new Array(units.length);
  for (let i = 0; i < units.length; i++) {
    const b = units[i];
    if (!b) {
      resolved[i] = null;
    } else if (!b.img.complete) {
      resolved[i] = { img: b.img, state: b.state, plan: null };
    } else {
      resolved[i] = {
        img: b.img, state: b.state,
        plan: fillPlan(b.img, b.state, { base: 0, hi: 0, single: false, oneTexel: false, posFilter: 0, mipMin: false, u8Fast: false }),
      };
    }
  }

  /** Resolves the unit; writes the null/incomplete result into env.out and returns null. */
  function resolve(unit: number): { img: TextureImage; state: SamplerState; plan: SamplePlan } | null {
    const b = resolved[unit];
    if (!b) {
      out[0] = 0; out[1] = 0; out[2] = 0; out[3] = 1;
      return null;
    }
    if (!b.plan) {
      out[0] = 0; out[1] = 0; out[2] = 0;
      out[3] = b.img.info.isInteger ? 0 : 1;
      return null;
    }
    return b;
  }

  return {
    units,
    out,
    outInt,
    outUint,

    sample2D(unit: number, u: number, vv: number, dux: number, dvx: number, duy: number, dvy: number, bias: number): void {
      const b = resolve(unit);
      if (!b) return;
      v[0] = u; v[1] = vv; v[2] = 0;
      dx[0] = dux; dx[1] = dvx; dx[2] = 0;
      dy[0] = duy; dy[1] = dvy; dy[2] = 0;
      sampleTextureP(b.img, b.state, b.plan, coord, bias, out);
    },

    sample2DLod(unit: number, u: number, vv: number, lod: number): void {
      const b = resolve(unit);
      if (!b) return;
      v[0] = u; v[1] = vv; v[2] = 0;
      sampleTextureLodP(b.img, b.state, b.plan, coord, lod, out);
    },

    sample3D(unit: number, u: number, vv: number, w: number, dux: number, dvx: number, dwx: number, duy: number, dvy: number, dwy: number, bias: number): void {
      const b = resolve(unit);
      if (!b) return;
      v[0] = u; v[1] = vv; v[2] = w;
      dx[0] = dux; dx[1] = dvx; dx[2] = dwx;
      dy[0] = duy; dy[1] = dvy; dy[2] = dwy;
      sampleTextureP(b.img, b.state, b.plan, coord, bias, out);
    },

    sample3DLod(unit: number, u: number, vv: number, w: number, lod: number): void {
      const b = resolve(unit);
      if (!b) return;
      v[0] = u; v[1] = vv; v[2] = w;
      sampleTextureLodP(b.img, b.state, b.plan, coord, lod, out);
    },

    sampleCube(unit: number, u: number, vv: number, w: number, dux: number, dvx: number, dwx: number, duy: number, dvy: number, dwy: number, bias: number): void {
      const b = resolve(unit);
      if (!b) return;
      v[0] = u; v[1] = vv; v[2] = w;
      dx[0] = dux; dx[1] = dvx; dx[2] = dwx;
      dy[0] = duy; dy[1] = dvy; dy[2] = dwy;
      sampleTextureP(b.img, b.state, b.plan, coord, bias, out);
    },

    sampleCubeLod(unit: number, u: number, vv: number, w: number, lod: number): void {
      const b = resolve(unit);
      if (!b) return;
      v[0] = u; v[1] = vv; v[2] = w;
      sampleTextureLodP(b.img, b.state, b.plan, coord, lod, out);
    },

    sample2DArray(unit: number, u: number, vv: number, layer: number, dux: number, dvx: number, duy: number, dvy: number, bias: number): void {
      const b = resolve(unit);
      if (!b) return;
      v[0] = u; v[1] = vv; v[2] = layer;
      dx[0] = dux; dx[1] = dvx; dx[2] = 0;
      dy[0] = duy; dy[1] = dvy; dy[2] = 0;
      sampleTextureP(b.img, b.state, b.plan, coord, bias, out);
    },

    sample2DArrayLod(unit: number, u: number, vv: number, layer: number, lod: number): void {
      const b = resolve(unit);
      if (!b) return;
      v[0] = u; v[1] = vv; v[2] = layer;
      sampleTextureLodP(b.img, b.state, b.plan, coord, lod, out);
    },

    sample2DShadow(unit: number, u: number, vv: number, ref: number, dux: number, dvx: number, duy: number, dvy: number, bias: number): void {
      const b = resolve(unit);
      if (!b) return;
      v[0] = u; v[1] = vv; v[2] = 0;
      dx[0] = dux; dx[1] = dvx; dx[2] = 0;
      dy[0] = duy; dy[1] = dvy; dy[2] = 0;
      sampleTextureShadowP(b.img, b.state, b.plan, coord, ref, bias, out);
    },

    sampleCubeShadow(unit: number, u: number, vv: number, w: number, ref: number, dux: number, dvx: number, dwx: number, duy: number, dvy: number, dwy: number, bias: number): void {
      const b = resolve(unit);
      if (!b) return;
      v[0] = u; v[1] = vv; v[2] = w;
      dx[0] = dux; dx[1] = dvx; dx[2] = dwx;
      dy[0] = duy; dy[1] = dvy; dy[2] = dwy;
      sampleTextureShadowP(b.img, b.state, b.plan, coord, ref, bias, out);
    },

    sample2DArrayShadow(unit: number, u: number, vv: number, layer: number, ref: number, dux: number, dvx: number, duy: number, dvy: number, bias: number): void {
      const b = resolve(unit);
      if (!b) return;
      v[0] = u; v[1] = vv; v[2] = layer;
      dx[0] = dux; dx[1] = dvx; dx[2] = 0;
      dy[0] = duy; dy[1] = dvy; dy[2] = 0;
      sampleTextureShadowP(b.img, b.state, b.plan, coord, ref, bias, out);
    },

    texelFetch2D(unit: number, x: number, y: number, level: number): void {
      const b = resolve(unit);
      if (!b) return;
      texelFetch(b.img, x, y, 0, level, out);
    },

    texelFetch3D(unit: number, x: number, y: number, z: number, level: number): void {
      const b = resolve(unit);
      if (!b) return;
      texelFetch(b.img, x, y, z, level, out);
    },

    texelFetch2DArray(unit: number, x: number, y: number, layer: number, level: number): void {
      const b = resolve(unit);
      if (!b) return;
      texelFetch(b.img, x, y, layer, level, out);
    },
  };
}

/** Helper: resolves the (img, state) for a unit, with incomplete→null. */
export function resolveUnit(env: TextureEnv, unit: number): { img: TextureImage; state: SamplerState } | null {
  const b = env.units[unit];
  if (!b || !b.img.complete) return null;
  return { img: b.img, state: b.state };
}
