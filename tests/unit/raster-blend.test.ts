/**
 * Direct unit tests of the dual-source blend math (WEBGL_blend_func_extended
 * SRC1_* factors) in src/raster/fragment-ops.ts `blendColor` — pinned against
 * the CTS page conformance2/extensions/webgl-blend-func-extended.html
 * (runBlendingTests) and the GLES 3.0 §4.1.7 factor tables.
 *
 * Contract points pinned here:
 * - The RGB equation reads the SECONDARY color's CHANNEL for SRC1_COLOR /
 *   ONE_MINUS_SRC1_COLOR and its ALPHA for SRC1_ALPHA / ONE_MINUS_SRC1_ALPHA.
 * - The alpha equation reads the secondary's ALPHA for EVERY SRC1_* factor
 *   (GLES 3.0 table 4.2) — SRC1_COLOR and SRC1_ALPHA are equivalent there.
 * - Absent src1 (undefined or null) → SRC1_* factors read 0 (so the ONE_MINUS
 *   variants yield 1).
 * - Results are NOT clamped (callers clamp through encode()); several
 *   expectations below are intentionally > 1 to pin that contract.
 *
 * The end-to-end draw() path (ctx.out.secondary → blend → 8-bit encode) is
 * covered by the two SRC1 pipeline tests in raster-pipeline.test.ts.
 */
import { describe, it } from "vitest";
import { blendColor } from "../../src/raster/index";
import { GL, expectArrayClose } from "./helpers";

/** One blendColor invocation with spec-default const color and FUNC_ADD. */
function blend(
  src: number[], dst: number[], src1: number[] | null | undefined,
  srcRGB: number, dstRGB: number, srcAlpha: number, dstAlpha: number,
): number[] {
  const out = new Float32Array(4);
  blendColor(
    Float32Array.from(src), Float32Array.from(dst), out,
    srcRGB, dstRGB, srcAlpha, dstAlpha,
    GL.FUNC_ADD, GL.FUNC_ADD, [0, 0, 0, 0],
    src1 ? Float32Array.from(src1) : null,
  );
  return Array.from(out);
}

describe("blendColor dual-source SRC1 factors (WEBGL_blend_func_extended)", () => {
  const WHITE = [1, 1, 1, 1];

  it("(ONE, SRC1_COLOR): src + src1·dst — CTS 'Multiply destination by SRC1 and add SRC0'", () => {
    // gl.blendFunc(ONE, SRC1_COLOR) mirrors the RGB factors into the alpha
    // slots (srcAlpha=ONE, dstAlpha=SRC1_COLOR → secondary alpha).
    const out = blend(
      [0.25, 0.375, 0.5, 0.625], WHITE, [0.125, 0.125, 0.125, 0.125],
      GL.ONE, GL.SRC1_COLOR, GL.ONE, GL.SRC1_COLOR,
    );
    expectArrayClose(out, [0.375, 0.5, 0.625, 0.75]);
  });

  it("(SRC1_COLOR, ONE_MINUS_SRC1_COLOR): src1·src + (1−src1)·dst — CTS 'Per-channel color interpolation using SRC1'", () => {
    const out = blend(
      [0.125, 0.125, 0.125, 0.125], WHITE, [0.5, 0.375, 0.25, 0.125],
      GL.SRC1_COLOR, GL.ONE_MINUS_SRC1_COLOR,
      GL.SRC1_COLOR, GL.ONE_MINUS_SRC1_COLOR,
    );
    expectArrayClose(out, [0.5625, 0.671875, 0.78125, 0.890625]);
  });

  it("exercises all four SRC1 constants in both the srcRGB and dstRGB factor slots", () => {
    // src1 RGB channels (0.25) differ from its alpha (0.375), so SRC1_COLOR
    // and SRC1_ALPHA (and their one-minus forms) produce distinct factors.
    const src = [0.25, 0.375, 0.5, 0.625];
    const src1 = [0.25, 0.25, 0.25, 0.375];
    // Alpha slots ONE/ONE → alpha = sa + da = 1.625 in every config; the
    // alpha-equation SRC1 mapping is pinned separately below. Values > 1 are
    // intentional — blendColor does NOT clamp (encode() clamps on write).
    const cases: { srcRGB: number; dstRGB: number; expected: number[] }[] = [
      { srcRGB: GL.SRC1_COLOR, dstRGB: GL.ONE, expected: [1.0625, 1.09375, 1.125, 1.625] },
      { srcRGB: GL.ONE, dstRGB: GL.SRC1_COLOR, expected: [0.5, 0.625, 0.75, 1.625] },
      { srcRGB: GL.ONE_MINUS_SRC1_COLOR, dstRGB: GL.ONE, expected: [1.1875, 1.28125, 1.375, 1.625] },
      { srcRGB: GL.ONE, dstRGB: GL.ONE_MINUS_SRC1_COLOR, expected: [1.0, 1.125, 1.25, 1.625] },
      { srcRGB: GL.SRC1_ALPHA, dstRGB: GL.ONE, expected: [1.09375, 1.140625, 1.1875, 1.625] },
      { srcRGB: GL.ONE, dstRGB: GL.SRC1_ALPHA, expected: [0.625, 0.75, 0.875, 1.625] },
      { srcRGB: GL.ONE_MINUS_SRC1_ALPHA, dstRGB: GL.ONE, expected: [1.15625, 1.234375, 1.3125, 1.625] },
      { srcRGB: GL.ONE, dstRGB: GL.ONE_MINUS_SRC1_ALPHA, expected: [0.875, 1.0, 1.125, 1.625] },
    ];
    for (const c of cases) {
      const out = blend(src, WHITE, src1, c.srcRGB, c.dstRGB, GL.ONE, GL.ONE);
      expectArrayClose(out, c.expected);
    }
  });

  it("alpha equation: every SRC1_* factor reads the secondary's ALPHA", () => {
    // s1a = 0.375: SRC1_ALPHA and SRC1_COLOR → 0.375; the ONE_MINUS forms →
    // 0.625 (GLES 3.0 table 4.2: the alpha equation has no color factors).
    const src = [0.25, 0.375, 0.5, 0.625];
    const src1 = [0.25, 0.25, 0.25, 0.375];
    // RGB ONE/ONE → out RGB = src + dst = (1.25, 1.375, 1.5) in every config.
    const cases: { srcAlpha: number; dstAlpha: number; alpha: number }[] = [
      { srcAlpha: GL.SRC1_ALPHA, dstAlpha: GL.ONE_MINUS_SRC1_ALPHA, alpha: 0.859375 },
      { srcAlpha: GL.ONE_MINUS_SRC1_ALPHA, dstAlpha: GL.SRC1_ALPHA, alpha: 0.765625 },
      { srcAlpha: GL.SRC1_COLOR, dstAlpha: GL.ONE_MINUS_SRC1_COLOR, alpha: 0.859375 },
      { srcAlpha: GL.ONE_MINUS_SRC1_COLOR, dstAlpha: GL.SRC1_COLOR, alpha: 0.765625 },
    ];
    for (const c of cases) {
      const out = blend(src, WHITE, src1, GL.ONE, GL.ONE, c.srcAlpha, c.dstAlpha);
      expectArrayClose(out, [1.25, 1.375, 1.5, c.alpha]);
    }
  });

  it("absent src1 (undefined or null): SRC1_* factors read 0, ONE_MINUS forms read 1", () => {
    const src = [0.25, 0.375, 0.5, 0.625];
    const constants = [
      GL.SRC1_COLOR, GL.ONE_MINUS_SRC1_COLOR, GL.SRC1_ALPHA, GL.ONE_MINUS_SRC1_ALPHA,
    ];
    for (const c of constants) {
      // Absent secondary → s1c = s1a = 0: SRC1_* factors are 0, ONE_MINUS_*
      // factors are 1.
      const f = (c === GL.ONE_MINUS_SRC1_COLOR || c === GL.ONE_MINUS_SRC1_ALPHA) ? 1 : 0;
      for (const missing of [undefined, null]) {
        // In the src slot: out = f·src + 1·dst.
        const inSrc = blend(src, WHITE, missing, c, GL.ONE, GL.ONE, GL.ONE);
        expectArrayClose(inSrc, [0.25 * f + 1, 0.375 * f + 1, 0.5 * f + 1, 1.625]);
        // In the dst slot: out = 1·src + f·dst.
        const inDst = blend(src, WHITE, missing, GL.ONE, c, GL.ONE, GL.ONE);
        expectArrayClose(inDst, [0.25 + f, 0.375 + f, 0.5 + f, 1.625]);
      }
    }
  });
});
