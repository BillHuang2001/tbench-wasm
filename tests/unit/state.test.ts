/**
 * Unit tests for the GL state container (src/gl/state.ts) — written against
 * the FINAL contract in src/CONTEXT.md §5 ("plain mutable state container …
 * No logic, just data + defaults") and the DrawCall field names in §2.
 * Fails (module not found) until src/gl/state.ts lands; these tests are then
 * the executable spec.
 *
 * Assumed import: `../../src/gl/state` exports `createState() → fresh state`.
 * Assumed state shape follows the DrawCall contract (§2) for the fields the
 * rasterizer consumes — `blend{enabled,srcRGB,dstRGB,srcAlpha,dstAlpha,eqRGB,
 * eqAlpha,color}`, `depthTest{enabled,func}`, `depthMask`, `cull{enabled,
 * face,frontFace}`, `scissor{enabled,x,y,w,h}`, `stencilTest{enabled,front,
 * back}`, `sampleCoverage{enabled,value,invert}`, `polygonOffset{enabled,
 * factor,units}`, `colorMask[4]` — plus `clearColor/clearDepth/clearStencil`,
 * `dither`, `rasterizerDiscard`, `lineWidth`, `activeTexture`,
 * `pixelStore{packAlignment,unpackAlignment,flipY,premultiplyAlpha}`,
 * `currentProgram`. If nesting differs, update ONLY the accesses below;
 * the defaults are the GL spec's.
 */
import { describe, it, expect } from "vitest";
import { createState } from "../../src/gl/state";
import { GL } from "./helpers";

describe("createState", () => {
  it("returns a fresh, independent state object per call", () => {
    const s1 = createState();
    const s2 = createState();
    expect(s1).not.toBe(s2);
    s1.blend.enabled = true;
    expect(s2.blend.enabled).toBe(false);
  });

  it("defaults clear values", () => {
    const s = createState();
    expect(s.clearColor).toEqual([0, 0, 0, 0]);
    expect(s.clearDepth).toBe(1);
    expect(s.clearStencil).toBe(0);
  });

  it("defaults write masks", () => {
    const s = createState();
    expect(s.colorMask).toEqual([true, true, true, true]);
    expect(s.depthMask).toBe(true);
  });

  it("defaults blend state (disabled, ONE/ ZERO, FUNC_ADD)", () => {
    const s = createState();
    expect(s.blend.enabled).toBe(false);
    expect(s.blend.srcRGB).toBe(GL.ONE);
    expect(s.blend.dstRGB).toBe(GL.ZERO);
    expect(s.blend.srcAlpha).toBe(GL.ONE);
    expect(s.blend.dstAlpha).toBe(GL.ZERO);
    expect(s.blend.eqRGB).toBe(GL.FUNC_ADD);
    expect(s.blend.eqAlpha).toBe(GL.FUNC_ADD);
    expect(s.blend.color).toEqual([0, 0, 0, 0]);
  });

  it("defaults depth test state (disabled, LESS)", () => {
    const s = createState();
    expect(s.depthTest.enabled).toBe(false);
    expect(s.depthTest.func).toBe(GL.LESS);
  });

  it("defaults cull state (disabled, BACK, CCW)", () => {
    const s = createState();
    expect(s.cull.enabled).toBe(false);
    expect(s.cull.face).toBe(GL.BACK);
    expect(s.cull.frontFace).toBe(GL.CCW);
  });

  it("defaults stencil test state (disabled, ALWAYS, ref 0, KEEP)", () => {
    const s = createState();
    expect(s.stencilTest.enabled).toBe(false);
    expect(s.stencilTest.front.ref).toBe(0);
    expect(s.stencilTest.front.func).toBe(GL.ALWAYS);
    expect(s.stencilTest.back.ref).toBe(0);
    expect(s.stencilTest.back.func).toBe(GL.ALWAYS);
  });

  it("defaults scissor (disabled) and viewport/scissor rects exist", () => {
    const s = createState();
    expect(s.scissor.enabled).toBe(false);
    expect(typeof s.scissor.x).toBe("number");
    expect(typeof s.scissor.y).toBe("number");
    expect(typeof s.scissor.w).toBe("number");
    expect(typeof s.scissor.h).toBe("number");
    expect(typeof s.viewport.x).toBe("number");
    expect(typeof s.viewport.w).toBe("number");
  });

  it("defaults rasterization controls", () => {
    const s = createState();
    expect(s.dither).toBe(true);
    expect(s.rasterizerDiscard).toBe(false);
    expect(s.lineWidth).toBe(1);
    expect(s.polygonOffset).toEqual({ enabled: false, factor: 0, units: 0 });
    expect(s.sampleCoverage.enabled).toBe(false);
    expect(s.sampleCoverage.value).toBe(1);
    expect(s.sampleCoverage.invert).toBe(false);
  });

  it("defaults bindings and pixelStore", () => {
    const s = createState();
    expect(s.currentProgram).toBeNull();
    expect(s.activeTexture).toBe(GL.TEXTURE0);
    expect(s.pixelStore.packAlignment).toBe(4);
    expect(s.pixelStore.unpackAlignment).toBe(4);
    expect(s.pixelStore.flipY).toBe(false);
    expect(s.pixelStore.premultiplyAlpha).toBe(false);
  });
});
