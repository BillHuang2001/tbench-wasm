/**
 * updateCompleteness (src/gl/teximage) — structural texture-completeness
 * tests, focused on the texStorage2D cube regression (commit d0a2b23):
 * a CUBE texture allocated via texStorage2D stores level records whose
 * `depth` field is a creation-site artifact (6 — the six faces live as
 * separate views in `data[]`, NOT as a depth dimension), while texImage2D /
 * generateMipmap create cube levels with depth 1. Before the fix the depth
 * comparison gated cube levels too (expected per-face depth 1), so a
 * storage-created cube never became COMPLETE even after all 6 faces were
 * uploaded via texSubImage2D — the sampler returned (0,0,0,1) and env-mapped
 * scenes rendered black.
 *
 * These tests pin:
 * - cube levels: `depth` is IGNORED for completeness (6 or 1 both fine);
 *   3D/2D_ARRAY depths are STILL enforced (the flip side of the fix);
 * - the 6 face views (none missing/undefined) at every required level;
 * - chain/level-size matching, WebGL1 NPOT rules, BASE_LEVEL handling.
 *
 * The function is purely structural: it reads texture._image.levels and
 * texture._params and mutates img.{complete, baseLevel, maxLevel}.
 * samplerForTexture is module-private, so a freshly built fake texture is
 * never in the WeakMap and its _params are used directly — no sampler
 * objects needed.
 */
import { describe, it, expect } from "vitest";
import { updateCompleteness } from "../../src/gl/teximage";
import { GL } from "./helpers";

/** WebGL spec-default texParameteri values. */
const DEFAULT_PARAMS = {
  [GL.TEXTURE_MAG_FILTER]: GL.LINEAR,
  [GL.TEXTURE_MIN_FILTER]: GL.NEAREST_MIPMAP_LINEAR,
  [GL.TEXTURE_WRAP_S]: GL.REPEAT,
  [GL.TEXTURE_WRAP_T]: GL.REPEAT,
  [GL.TEXTURE_BASE_LEVEL]: 0,
  [GL.TEXTURE_MAX_LEVEL]: 1000,
};

/** Builds `n` level/face views (tiny Uint8Arrays — size is irrelevant). */
function views(n: number): ArrayBufferView[] {
  const out: ArrayBufferView[] = [];
  for (let i = 0; i < n; i++) out.push(new Uint8Array(4));
  return out;
}

interface FakeTexture {
  _image: {
    target: number;
    baseLevel: number;
    maxLevel: number;
    complete: boolean;
    levels: Array<{ width: number; height: number; depth: number; data: ArrayBufferView[] }>;
  };
  _params: Record<number, number>;
}

/**
 * Builds a fake texture. Each level: `{width, height, depth, faces?, data?}`
 * — `data` overrides the auto-built views (use it to make a face
 * missing/undefined); `faces` (default 1) is the view count (cube = 6).
 * `params` overrides DEFAULT_PARAMS.
 */
function makeTexture(
  target: number,
  levels: Array<{
    width: number;
    height: number;
    depth?: number;
    faces?: number;
    data?: ArrayBufferView[];
  }>,
  params: Partial<Record<number, number>> = {},
): FakeTexture {
  return {
    _image: {
      target,
      baseLevel: 0,
      maxLevel: 0,
      complete: false,
      levels: levels.map((l) => ({
        width: l.width,
        height: l.height,
        depth: l.depth ?? 1,
        data: l.data ?? views(l.faces ?? 1),
      })),
    },
    _params: { ...DEFAULT_PARAMS, ...params },
  };
}

/** Runs updateCompleteness and returns the resulting img.complete. */
function complete(texture: FakeTexture, version: 1 | 2 = 2): boolean {
  updateCompleteness(texture as any, version);
  return texture._image.complete;
}

describe("updateCompleteness", () => {
  it("texStorage2D cube (level depth = 6, all 6 faces) is COMPLETE — the d0a2b23 regression", () => {
    // texStorage2D(TEXTURE_CUBE_MAP, ...) allocates level records with
    // depth 6 (creation-site artifact). depth must NOT gate cube
    // completeness; the 6 face views in data[] are what matter.
    const tex = makeTexture(
      GL.TEXTURE_CUBE_MAP,
      [
        { width: 2, height: 2, depth: 6, faces: 6 },
        { width: 1, height: 1, depth: 6, faces: 6 },
      ],
      { [GL.TEXTURE_MIN_FILTER]: GL.NEAREST },
    );
    expect(complete(tex)).toBe(true);
    // Same storage-created cube with a mipmap min filter (full chain) —
    // every level carries depth 6 and must still be complete.
    tex._params[GL.TEXTURE_MIN_FILTER] = GL.LINEAR_MIPMAP_LINEAR;
    expect(complete(tex)).toBe(true);
  });

  it("cube with a missing face (5 views) at the base level is INCOMPLETE", () => {
    const tex = makeTexture(
      GL.TEXTURE_CUBE_MAP,
      [{ width: 2, height: 2, depth: 6, data: views(5) }],
      { [GL.TEXTURE_MIN_FILTER]: GL.NEAREST },
    );
    expect(complete(tex)).toBe(false);
  });

  it("cube with an undefined face entry inside a 6-slot data[] is INCOMPLETE", () => {
    const data = views(6) as (ArrayBufferView | undefined)[];
    data[4] = undefined; // -Z face never uploaded
    const tex = makeTexture(
      GL.TEXTURE_CUBE_MAP,
      [{ width: 2, height: 2, depth: 6, data: data as ArrayBufferView[] }],
      { [GL.TEXTURE_MIN_FILTER]: GL.NEAREST },
    );
    expect(complete(tex)).toBe(false);
  });

  it("cube with a missing face in a higher mip level is INCOMPLETE (caught at the right level)", () => {
    const tex = makeTexture(
      GL.TEXTURE_CUBE_MAP,
      [
        { width: 2, height: 2, depth: 6, faces: 6 }, // level 0 complete
        { width: 1, height: 1, depth: 6, data: views(5) }, // level 1 missing a face
      ],
      { [GL.TEXTURE_MIN_FILTER]: GL.LINEAR_MIPMAP_LINEAR },
    );
    expect(complete(tex)).toBe(false);
  });

  it("cube with an incomplete mip chain is INCOMPLETE under a mipmap filter, COMPLETE with NEAREST", () => {
    const tex = makeTexture(
      GL.TEXTURE_CUBE_MAP,
      [{ width: 2, height: 2, depth: 6, faces: 6 }], // level 1 (1x1) never uploaded
      { [GL.TEXTURE_MIN_FILTER]: GL.LINEAR_MIPMAP_LINEAR },
    );
    expect(complete(tex)).toBe(false);
    tex._params[GL.TEXTURE_MIN_FILTER] = GL.NEAREST; // mips not required
    expect(complete(tex)).toBe(true);
  });

  it("texImage2D-created cube (level depth = 1, 6 face views) is COMPLETE — unchanged behavior", () => {
    const tex = makeTexture(
      GL.TEXTURE_CUBE_MAP,
      [
        { width: 2, height: 2, depth: 1, faces: 6 },
        { width: 1, height: 1, depth: 1, faces: 6 },
      ],
      { [GL.TEXTURE_MIN_FILTER]: GL.LINEAR_MIPMAP_LINEAR },
    );
    expect(complete(tex)).toBe(true);
  });

  it("storage-created TEXTURE_2D (depth = 1) full chain is COMPLETE — unchanged", () => {
    const tex = makeTexture(GL.TEXTURE_2D, [
      { width: 2, height: 2, depth: 1 },
      { width: 1, height: 1, depth: 1 },
    ]);
    expect(complete(tex)).toBe(true);
  });

  it("2D level with a mismatched size is INCOMPLETE", () => {
    const tex = makeTexture(GL.TEXTURE_2D, [
      { width: 2, height: 2, depth: 1 },
      { width: 3, height: 1, depth: 1 }, // expected 1x1
    ]);
    expect(complete(tex)).toBe(false);
  });

  it("3D: per-level depth IS enforced (halves with the mip shift)", () => {
    // 4x4x4 base: levels must be 4x4x4, 2x2x2, 1x1x1.
    const tex = makeTexture(GL.TEXTURE_3D, [
      { width: 4, height: 4, depth: 4 },
      { width: 2, height: 2, depth: 2 },
      { width: 1, height: 1, depth: 1 },
    ]);
    expect(complete(tex)).toBe(true);
  });

  it("3D depth mismatch is INCOMPLETE — the flip side of the cube fix", () => {
    const tex = makeTexture(GL.TEXTURE_3D, [
      { width: 4, height: 4, depth: 4 },
      { width: 2, height: 2, depth: 3 }, // expected depth 2
      { width: 1, height: 1, depth: 1 },
    ]);
    expect(complete(tex)).toBe(false);
  });

  it("2D_ARRAY: layer count is enforced on every level (depth does NOT halve)", () => {
    // 4x4 with 8 layers: every mip keeps depth 8.
    const tex = makeTexture(GL.TEXTURE_2D_ARRAY, [
      { width: 4, height: 4, depth: 8 },
      { width: 2, height: 2, depth: 8 },
      { width: 1, height: 1, depth: 8 },
    ]);
    expect(complete(tex)).toBe(true);
    tex._image.levels[1].depth = 4; // layer count shrank → incomplete
    expect(complete(tex)).toBe(false);
  });

  it("WebGL1 NPOT: mipmap min filter or non-CLAMP wrap is INCOMPLETE", () => {
    // 3x5 (non-pow2) with a FULL mip chain — the chain itself is fine; the
    // WebGL1 NPOT rule still rejects it under a mipmap filter.
    const tex = makeTexture(GL.TEXTURE_2D, [
      { width: 3, height: 5, depth: 1 },
      { width: 1, height: 2, depth: 1 },
      { width: 1, height: 1, depth: 1 },
    ]);
    expect(complete(tex, 1)).toBe(false); // mipmap min filter → NPOT-incomplete
    tex._params[GL.TEXTURE_MIN_FILTER] = GL.NEAREST;
    expect(complete(tex, 1)).toBe(false); // NEAREST but REPEAT wrap → still incomplete
    tex._params[GL.TEXTURE_WRAP_S] = GL.CLAMP_TO_EDGE;
    tex._params[GL.TEXTURE_WRAP_T] = GL.CLAMP_TO_EDGE;
    expect(complete(tex, 1)).toBe(true); // NEAREST + CLAMP_TO_EDGE → complete
  });

  it("BASE_LEVEL > MAX_LEVEL is INCOMPLETE and mutates img.baseLevel", () => {
    const tex = makeTexture(
      GL.TEXTURE_2D,
      [
        { width: 2, height: 2, depth: 1 },
        { width: 1, height: 1, depth: 1 },
      ],
      { [GL.TEXTURE_BASE_LEVEL]: 2, [GL.TEXTURE_MAX_LEVEL]: 1 },
    );
    expect(complete(tex)).toBe(false);
    expect(tex._image.baseLevel).toBe(2);
    expect(tex._image.maxLevel).toBe(1);
  });

  it("texture with no levels is INCOMPLETE", () => {
    const tex = makeTexture(GL.TEXTURE_2D, []);
    expect(complete(tex)).toBe(false);
  });

  it("single-level texture is COMPLETE with LINEAR/NEAREST min filter even without a mip chain", () => {
    const tex = makeTexture(GL.TEXTURE_2D, [{ width: 16, height: 16, depth: 1 }]);
    tex._params[GL.TEXTURE_MIN_FILTER] = GL.LINEAR;
    expect(complete(tex)).toBe(true);
    tex._params[GL.TEXTURE_MIN_FILTER] = GL.NEAREST;
    expect(complete(tex)).toBe(true);
    // Contrast: the default NEAREST_MIPMAP_LINEAR requires the whole chain.
    tex._params[GL.TEXTURE_MIN_FILTER] = GL.NEAREST_MIPMAP_LINEAR;
    expect(complete(tex)).toBe(false);
  });
});
