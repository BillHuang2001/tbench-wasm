/**
 * Unit tests for the PURE parts of src/present (canvas adapters) — written
 * against the FINAL contract in src/CONTEXT.md §4. Fails (module not found)
 * until src/present lands; these tests are then the executable spec.
 *
 * Assumed import: `../../src/present/index` exports
 * `createNodeSurface(width, height) → CanvasSurface` where CanvasSurface =
 * `{ width, height, getPixels(): Uint8Array (RGBA8), present(): void }`
 * ("Node adapter: pure buffer, present() no-op"). If the actual module
 * exports differ, update ONLY the call sites below. ImageSource decoding
 * tests should be added once that API settles.
 */
import { describe, it, expect } from "vitest";
import { createNodeSurface } from "../../src/present/index";

describe("createNodeSurface", () => {
  it("reports the requested dimensions", () => {
    const surface = createNodeSurface(320, 240);
    expect(surface.width).toBe(320);
    expect(surface.height).toBe(240);
  });

  it("getPixels returns an RGBA8-sized buffer (w*h*4 bytes)", () => {
    const surface = createNodeSurface(4, 3);
    const pixels = surface.getPixels();
    expect(pixels).toBeInstanceOf(Uint8Array);
    expect(pixels.length).toBe(4 * 3 * 4);
    // Deterministic contents across calls (same backing buffer).
    expect(surface.getPixels().length).toBe(pixels.length);
  });

  it("present() is a no-op in Node mode", () => {
    const surface = createNodeSurface(1, 1);
    expect(() => surface.present()).not.toThrow();
  });
});
