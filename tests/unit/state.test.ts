/**
 * Unit tests for the GL state container (src/gl/state.ts) — written against
 * the REAL module contract (verified against the implementation).
 *
 * Import: `../../src/gl/state` exports `createDefaultState(version: 1|2) →
 * State` (fresh state per call; 1 = WebGL1, 2 = WebGL2), plus
 * `defaultLimits()`, `defaultVertexAttrib()`, `defaultVAOState(numAttribs)`
 * and `KEEP` (0x1E00).
 *
 * State shape (NOT the DrawCall/rasterizer nesting):
 * - Capabilities are booleans in `caps` (BLEND, CULL_FACE, DEPTH_TEST,
 *   DITHER [default TRUE], POLYGON_OFFSET_FILL, SAMPLE_ALPHA_TO_COVERAGE,
 *   SAMPLE_COVERAGE, SCISSOR_TEST, STENCIL_TEST, RASTERIZER_DISCARD) — the
 *   sub-state blocks carry NO `enabled` flag.
 * - `blend{srcRGB,dstRGB,srcAlpha,dstAlpha,eqRGB,eqAlpha,color:[4]}`
 * - `depth{func,mask,range:[2]}` (mask replaces depthMask)
 * - `stencil{front,back}`, each `{func,ref,valueMask,writeMask,fail,
 *   depthFail,depthPass}`
 * - top-level `cullFace`, `frontFace`, `scissor{x,y,w,h}`,
 *   `viewport{x,y,w,h}`, `polygonOffset{factor,units}`,
 *   `sampleCoverage{value,invert}`, `colorMask[4]`, `clearColor[4]`,
 *   `clearDepth`, `clearStencil`, `lineWidth`, `activeTexture`,
 *   `currentProgram`, `version`, `limits`
 * - `pixelStore{unpack{alignment,flipY,premultiplyAlpha,...},
 *   pack{alignment,...}}`
 * All defaults asserted here are the GL spec's.
 */
import { describe, it, expect } from "vitest";
import {
  createDefaultState,
  defaultLimits,
  defaultVertexAttrib,
  defaultVAOState,
  KEEP,
} from "../../src/gl/state";
import { GL } from "./helpers";

describe("createDefaultState", () => {
  it("returns a fresh, independent state object per call", () => {
    const s1 = createDefaultState(1);
    const s2 = createDefaultState(1);
    expect(s1).not.toBe(s2);
    s1.caps.BLEND = true;
    expect(s2.caps.BLEND).toBe(false);
    s1.blend.color[0] = 1;
    expect(s2.blend.color).toEqual([0, 0, 0, 0]);
  });

  it("defaults all capabilities to disabled except DITHER", () => {
    const s = createDefaultState(1);
    expect(s.caps.BLEND).toBe(false);
    expect(s.caps.CULL_FACE).toBe(false);
    expect(s.caps.DEPTH_TEST).toBe(false);
    expect(s.caps.DITHER).toBe(true);
    expect(s.caps.POLYGON_OFFSET_FILL).toBe(false);
    expect(s.caps.SAMPLE_ALPHA_TO_COVERAGE).toBe(false);
    expect(s.caps.SAMPLE_COVERAGE).toBe(false);
    expect(s.caps.SCISSOR_TEST).toBe(false);
    expect(s.caps.STENCIL_TEST).toBe(false);
    expect(s.caps.RASTERIZER_DISCARD).toBe(false);
  });

  it("defaults clear values", () => {
    const s = createDefaultState(1);
    expect(s.clearColor).toEqual([0, 0, 0, 0]);
    expect(s.clearDepth).toBe(1);
    expect(s.clearStencil).toBe(0);
  });

  it("defaults write masks", () => {
    const s = createDefaultState(1);
    expect(s.colorMask).toEqual([true, true, true, true]);
    expect(s.depth.mask).toBe(true);
  });

  it("defaults blend state (disabled, ONE/ZERO, FUNC_ADD)", () => {
    const s = createDefaultState(1);
    expect(s.caps.BLEND).toBe(false);
    expect(s.blend.srcRGB).toBe(GL.ONE);
    expect(s.blend.dstRGB).toBe(GL.ZERO);
    expect(s.blend.srcAlpha).toBe(GL.ONE);
    expect(s.blend.dstAlpha).toBe(GL.ZERO);
    expect(s.blend.eqRGB).toBe(GL.FUNC_ADD);
    expect(s.blend.eqAlpha).toBe(GL.FUNC_ADD);
    expect(s.blend.color).toEqual([0, 0, 0, 0]);
  });

  it("defaults depth test state (disabled, LESS, mask true, range [0,1])", () => {
    const s = createDefaultState(1);
    expect(s.caps.DEPTH_TEST).toBe(false);
    expect(s.depth.func).toBe(GL.LESS);
    expect(s.depth.mask).toBe(true);
    expect(s.depth.range).toEqual([0, 1]);
  });

  it("defaults cull state (disabled, BACK, CCW)", () => {
    const s = createDefaultState(1);
    expect(s.caps.CULL_FACE).toBe(false);
    expect(s.cullFace).toBe(GL.BACK);
    expect(s.frontFace).toBe(GL.CCW);
  });

  it("defaults stencil test state (disabled, ALWAYS, ref 0, full masks, KEEP ops)", () => {
    const s = createDefaultState(1);
    expect(s.caps.STENCIL_TEST).toBe(false);
    for (const face of [s.stencil.front, s.stencil.back]) {
      expect(face.func).toBe(GL.ALWAYS);
      expect(face.ref).toBe(0);
      expect(face.valueMask).toBe(0xffffffff);
      expect(face.writeMask).toBe(0xffffffff);
      expect(face.fail).toBe(GL.KEEP);
      expect(face.depthFail).toBe(GL.KEEP);
      expect(face.depthPass).toBe(GL.KEEP);
    }
    // front and back are independent objects
    expect(s.stencil.front).not.toBe(s.stencil.back);
  });

  it("defaults scissor and viewport rects to zero", () => {
    const s = createDefaultState(1);
    expect(s.caps.SCISSOR_TEST).toBe(false);
    expect(s.scissor).toEqual({ x: 0, y: 0, w: 0, h: 0 });
    expect(s.viewport).toEqual({ x: 0, y: 0, w: 0, h: 0 });
  });

  it("defaults rasterization controls", () => {
    const s = createDefaultState(1);
    expect(s.caps.DITHER).toBe(true);
    expect(s.caps.RASTERIZER_DISCARD).toBe(false);
    expect(s.caps.POLYGON_OFFSET_FILL).toBe(false);
    expect(s.polygonOffset).toEqual({ factor: 0, units: 0 });
    expect(s.caps.SAMPLE_COVERAGE).toBe(false);
    expect(s.sampleCoverage).toEqual({ value: 1, invert: false });
    expect(s.lineWidth).toBe(1);
  });

  it("defaults bindings, active texture and pixelStore", () => {
    const s = createDefaultState(1);
    expect(s.currentProgram).toBeNull();
    // state stores the texture unit INDEX (0 = GL_TEXTURE0, reported as GL.TEXTURE0 + i)
    expect(s.activeTexture).toBe(0);
    expect(s.activeTexture + GL.TEXTURE0).toBe(GL.TEXTURE0);
    expect(s.pixelStore.pack.alignment).toBe(4);
    expect(s.pixelStore.unpack.alignment).toBe(4);
    expect(s.pixelStore.unpack.flipY).toBe(false);
    expect(s.pixelStore.unpack.premultiplyAlpha).toBe(false);
  });

  it("createDefaultState(2) reports version 2 and WebGL2-sized uniform buffer binding arrays", () => {
    const s1 = createDefaultState(1);
    const s2 = createDefaultState(2);
    expect(s1.version).toBe(1);
    expect(s2.version).toBe(2);
    expect(s1.uniformBuffers).toHaveLength(0);
    expect(s1.uniformBufferRanges).toHaveLength(0);
    expect(s2.uniformBuffers).toHaveLength(72);
    expect(s2.uniformBuffers.length).toBe(s2.limits.MAX_UNIFORM_BUFFER_BINDINGS);
    expect(s2.uniformBufferRanges).toHaveLength(72);
  });

  it("exposes spec limits (defaultLimits)", () => {
    const l = defaultLimits();
    expect(l.MAX_VERTEX_ATTRIBS).toBe(16);
    expect(l.MAX_TEXTURE_SIZE).toBe(8192);
    expect(l.MAX_UNIFORM_BUFFER_BINDINGS).toBe(72);
    expect(createDefaultState(2).limits.MAX_VERTEX_ATTRIBS).toBe(16);
  });
});

describe("defaultVertexAttrib / defaultVAOState", () => {
  it("defaultVertexAttrib has spec defaults", () => {
    const a = defaultVertexAttrib();
    expect(a.enabled).toBe(false);
    expect(a.size).toBe(4);
    expect(a.type).toBe(0x1406); // FLOAT
    expect(a.normalized).toBe(false);
    expect(a.integer).toBe(false);
    expect(a.stride).toBe(0);
    expect(a.offset).toBe(0);
    expect(a.divisor).toBe(0);
    expect(a.buffer).toBeNull();
    expect(Array.from(a.constantF)).toEqual([0, 0, 0, 1]);
    expect(Array.from(a.constantI)).toEqual([0, 0, 0, 1]);
    expect(Array.from(a.constantUI)).toEqual([0, 0, 0, 1]);
  });

  it("creates numAttribs attributes with spec defaults", () => {
    const vao = defaultVAOState(4);
    expect(vao.attribs).toHaveLength(4);
    expect(vao.elementArrayBuffer).toBeNull();
    for (const a of vao.attribs) {
      expect(a.enabled).toBe(false);
      expect(Array.from(a.constantF)).toEqual([0, 0, 0, 1]);
    }
    // per-call independence
    vao.attribs[0].enabled = true;
    expect(defaultVAOState(4).attribs[0].enabled).toBe(false);
  });

  it("a state's default VAO has MAX_VERTEX_ATTRIBS attribs", () => {
    const s = createDefaultState(2);
    expect(s.vao.attribs).toHaveLength(s.limits.MAX_VERTEX_ATTRIBS);
    expect(s.vao.elementArrayBuffer).toBeNull();
    expect(Array.from(s.vao.attribs[0].constantF)).toEqual([0, 0, 0, 1]);
  });

  it("exports KEEP (0x1E00)", () => {
    expect(KEEP).toBe(GL.KEEP);
  });
});
