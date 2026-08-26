/**
 * Unit tests for the PURE parts of src/present (canvas adapters) — written
 * against the FINAL contract in src/CONTEXT.md §4 and src/present/canvas.ts.
 *
 * REAL API (src/present/index → src/present/canvas):
 * - `interface CanvasSurface { readonly width; readonly height;
 *   getPixels(): Uint8Array; present(): void; resize(w, h): void }` — RGBA8
 *   backing buffer, length = width*height*4. present() must never throw;
 *   resize() reallocates (zero-filled) and invalidates cached pixels.
 * - `class NodeCanvasSurface implements CanvasSurface` — constructor takes NO
 *   arguments; dimensions come from resize(w, h). present() is an intentional
 *   no-op (IMPLEMENTED). getPixels()/resize() are stubs that throw.
 * - `class BrowserCanvasSurface implements CanvasSurface` — constructor takes
 *   an HTMLCanvasElement; all methods are stubs (throw).
 * - `createCanvasSurface(canvas: unknown): CanvasSurface` — factory (stub,
 *   throws): object WITH a getContext function → BrowserCanvasSurface,
 *   otherwise → NodeCanvasSurface.
 *
 * RUNTIME STATUS (expected until src/present/canvas.ts implementation lands):
 * - present() no-op test PASSES (implemented).
 * - getPixels()/resize() tests FAIL (stubs throw 'not implemented').
 * - createCanvasSurface factory tests FAIL (stub throws) — the factory's
 *   structural detection (getContext function → Browser, else Node) is the
 *   spec these tests pin down.
 * These failures are the executable spec for the src/ implementer; do NOT
 * delete or skip them.
 */
import { describe, it, expect } from "vitest";
import {
  NodeCanvasSurface,
  BrowserCanvasSurface,
  createCanvasSurface,
} from "../../src/present/index";

describe("NodeCanvasSurface", () => {
  it("reports the dimensions given to resize()", () => {
    const surface = new NodeCanvasSurface();
    surface.resize(320, 240);
    expect(surface.width).toBe(320);
    expect(surface.height).toBe(240);
  });

  it("getPixels returns an RGBA8-sized buffer (w*h*4 bytes)", () => {
    const surface = new NodeCanvasSurface();
    surface.resize(4, 3);
    const pixels = surface.getPixels();
    expect(pixels).toBeInstanceOf(Uint8Array);
    expect(pixels.length).toBe(4 * 3 * 4);
    // Deterministic contents across calls (same backing buffer).
    expect(surface.getPixels().length).toBe(pixels.length);
  });

  it("present() is a no-op in Node mode", () => {
    const surface = new NodeCanvasSurface();
    surface.resize(1, 1);
    expect(() => surface.present()).not.toThrow();
  });
});

describe("createCanvasSurface factory (structural detection)", () => {
  it("returns a NodeCanvasSurface for a non-canvas object", () => {
    const surface = createCanvasSurface({});
    expect(surface).toBeInstanceOf(NodeCanvasSurface);
  });

  it("returns a BrowserCanvasSurface for an object with a getContext function", () => {
    const fakeCanvas = { getContext: () => null };
    const surface = createCanvasSurface(fakeCanvas);
    expect(surface).toBeInstanceOf(BrowserCanvasSurface);
  });
});
