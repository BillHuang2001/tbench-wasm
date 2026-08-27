/**
 * Unit tests for the pure math parts of the rasterizer (src/raster) — vertex
 * record layout, homogeneous clipping, the viewport transform, and
 * window-space triangle area. Written against the FINAL src/raster contracts
 * (src/raster/types.ts, clip.ts, triangles.ts); fails with 'not implemented'
 * until src/raster lands — these tests are then the executable spec.
 *
 * NOTE: perspective-correct varying interpolation is NOT an exported helper —
 * it is internal to rasterizeTriangle (window-space barycentrics λi,
 * value = (Σ λi·vi/wi) / (Σ λi/wi), see src/raster/triangles.ts) and is
 * covered by the conformance / visual-regression suites, not here.
 *
 * Real API (all re-exported from `../../src/raster/index`):
 * - Record layout: RECORD_OFFSET_X/Y/Z/W/POINT_SIZE, VARYINGS_OFFSET,
 *   RECORD_HEADER_FLOATS (5); computeVertexStride(varyings) → stride;
 *   writeVertexHeader(out, base, x, y, z, w, pointSize).
 * - clipPrimitive(buf, base, stride, count, scratch, out, outBase) → #verts
 *   written (0 = fully clipped). Clips ONE primitive (count 2|3) against the
 *   6 planes -w ≤ x,y,z ≤ w, interpolating EVERY record field (position +
 *   pointSize + varyings) linearly in clip space; output is fan-able; scratch
 *   must hold ≥ 2×count records and out ≥ MAX_CLIPPED_VERTICES (7).
 * - pointIsVisible(buf, base, stride) → point satisfies -w ≤ x,y,z ≤ w.
 * - applyViewportTransform(buf, base, stride, count, viewport, depthRange) —
 *   in-place divide-by-w + window mapping; clip w PRESERVED in slot 3;
 *   winZ = near + (far-near)·(z/w·0.5+0.5), clamped to [0,1].
 * - signedArea2(buf, i0, i1, i2, stride) → window-space signed area ×2
 *   (positive = CCW).
 * If the actual module exports differ, update ONLY the call sites below.
 */
import { describe, it, expect } from "vitest";
import {
  RECORD_OFFSET_X,
  RECORD_OFFSET_Y,
  RECORD_OFFSET_Z,
  RECORD_OFFSET_W,
  RECORD_OFFSET_POINT_SIZE,
  VARYINGS_OFFSET,
  RECORD_HEADER_FLOATS,
  MAX_CLIPPED_VERTICES,
  computeVertexStride,
  writeVertexHeader,
  clipPrimitive,
  pointIsVisible,
  applyViewportTransform,
  signedArea2,
  type VaryingInfo,
} from "../../src/raster/index";
import { expectArrayClose } from "./helpers";

/** Spec-fixed GLSL type enums (VaryingInfo.type is informational for raster). */
const FLOAT = 0x1406;
const FLOAT_VEC2 = 0x8b50;
const FLOAT_VEC3 = 0x8b51;
const FLOAT_VEC4 = 0x8b52;

/** Two 2-component varyings → stride = 5 + 2 + 2 = 9. */
const TWO_VEC2_VARYINGS: readonly VaryingInfo[] = [
  { name: "v_a", type: FLOAT_VEC2, components: 2, flat: false },
  { name: "v_b", type: FLOAT_VEC2, components: 2, flat: false },
];

/** Allocates a record buffer for `count` records of `stride` floats each. */
function makeRecordBuffer(count: number, stride: number): Float32Array {
  return new Float32Array(count * stride);
}

/**
 * Writes one full record (header + varyings) at `base`. `varyings` must have
 * stride - RECORD_HEADER_FLOATS entries (default: header-only record).
 */
function writeRecord(
  out: Float32Array, base: number,
  x: number, y: number, z: number, w: number, pointSize: number,
  varyings: readonly number[] = [],
): void {
  writeVertexHeader(out, base, x, y, z, w, pointSize);
  for (let i = 0; i < varyings.length; i++) {
    out[base + VARYINGS_OFFSET + i] = varyings[i];
  }
}

/** Copies the record at `base` (stride floats) into a fresh number[]. */
function readRecord(buf: Float32Array, base: number, stride: number): number[] {
  return Array.from(buf.subarray(base, base + stride));
}

/** Header [x, y, z, w, pointSize] of the record at `base`. */
function readHeader(
  buf: Float32Array, base: number,
): [number, number, number, number, number] {
  return [
    buf[base + RECORD_OFFSET_X],
    buf[base + RECORD_OFFSET_Y],
    buf[base + RECORD_OFFSET_Z],
    buf[base + RECORD_OFFSET_W],
    buf[base + RECORD_OFFSET_POINT_SIZE],
  ];
}

/**
 * Clips the `count` records at the start of `buf` and returns the output
 * buffer + vertex count.
 */
function clipRecords(
  buf: Float32Array, stride: number, count: number,
): { out: Float32Array; outCount: number } {
  const scratch = makeRecordBuffer(2 * count, stride);
  const out = makeRecordBuffer(MAX_CLIPPED_VERTICES, stride);
  const outCount = clipPrimitive(buf, 0, stride, count, scratch, out, 0);
  return { out, outCount };
}

/**
 * Asserts the output polygon's vertex SET equals `expected` (order-
 * independent: clip output is documented as fan-able, so a cyclic rotation
 * of the polygon order is legal).
 */
function expectVertices(
  out: Float32Array, outCount: number, stride: number,
  expected: ReadonlyArray<readonly number[]>,
): void {
  expect(outCount).toBe(expected.length);
  const records = Array.from(
    { length: outCount }, (_, i) => readRecord(out, i * stride, stride),
  );
  for (const want of expected) {
    const found = records.some((got) =>
      got.length === want.length &&
      got.every((v, i) => Math.abs(v - want[i]) <= 1e-5),
    );
    expect(found, `output should contain vertex [${want.join(", ")}]`).toBe(true);
  }
}

/** Every output vertex must satisfy the clip-volume constraints. */
function expectInsideClipVolume(out: Float32Array, outCount: number, stride: number): void {
  for (let i = 0; i < outCount; i++) {
    const [x, y, z, w] = readHeader(out, i * stride);
    for (const c of [x, y, z]) {
      expect(c).toBeGreaterThanOrEqual(-w - 1e-5);
      expect(c).toBeLessThanOrEqual(w + 1e-5);
    }
  }
}

describe("signedArea2", () => {
  const stride = computeVertexStride(TWO_VEC2_VARYINGS); // 9

  it("is positive for CCW winding and equals the signed area ×2", () => {
    const buf = makeRecordBuffer(3, stride);
    writeRecord(buf, 0, 0, 0, 0, 1, 1);
    writeRecord(buf, stride, 1, 0, 0, 1, 1);
    writeRecord(buf, 2 * stride, 0, 1, 0, 1, 1);
    // Triangle (0,0),(1,0),(0,1): area = 0.5 → area×2 = 1.
    expect(signedArea2(buf, 0, stride, 2 * stride, stride)).toBeGreaterThan(0);
    expect(signedArea2(buf, 0, stride, 2 * stride, stride)).toBeCloseTo(1, 10);
  });

  it("is negative for reversed (CW) winding", () => {
    const buf = makeRecordBuffer(3, stride);
    writeRecord(buf, 0, 0, 0, 0, 1, 1);
    writeRecord(buf, stride, 0, 1, 0, 1, 1);
    writeRecord(buf, 2 * stride, 1, 0, 0, 1, 1);
    expect(signedArea2(buf, 0, stride, 2 * stride, stride)).toBeCloseTo(-1, 10);
  });

  it("scales with the triangle size (area×2)", () => {
    const buf = makeRecordBuffer(3, stride);
    writeRecord(buf, 0, 0, 0, 0, 1, 1);
    writeRecord(buf, stride, 2, 0, 0, 1, 1);
    writeRecord(buf, 2 * stride, 0, 2, 0, 1, 1);
    // Triangle (0,0),(2,0),(0,2): area = 2 → area×2 = 4.
    expect(signedArea2(buf, 0, stride, 2 * stride, stride)).toBeCloseTo(4, 10);
  });

  it("is 0 for collinear (degenerate) triangles", () => {
    const buf = makeRecordBuffer(3, stride);
    writeRecord(buf, 0, 0, 0, 0, 1, 1);
    writeRecord(buf, stride, 1, 0, 0, 1, 1);
    writeRecord(buf, 2 * stride, 2, 0, 0, 1, 1);
    expect(signedArea2(buf, 0, stride, 2 * stride, stride)).toBeCloseTo(0, 10);
  });
});

describe("clipPrimitive", () => {
  it("passes a fully-inside triangle through unchanged (records copied verbatim)", () => {
    const stride = computeVertexStride(TWO_VEC2_VARYINGS); // 9
    const buf = makeRecordBuffer(3, stride);
    writeRecord(buf, 0, 0, 0, 0, 1, 1, [10, 20, 30, 40]);
    writeRecord(buf, stride, 1, 0, 0, 1, 1, [11, 21, 31, 41]);
    writeRecord(buf, 2 * stride, 0, 1, 0, 1, 1, [12, 22, 32, 42]);
    const { out, outCount } = clipRecords(buf, stride, 3);
    expect(outCount).toBe(3);
    for (let i = 0; i < outCount; i++) {
      expectArrayClose(
        readRecord(out, i * stride, stride),
        readRecord(buf, i * stride, stride),
      );
    }
  });

  it("clips a triangle with one vertex outside the x=w plane; new vertices lie on the plane, inside vertices keep their values", () => {
    const stride = RECORD_HEADER_FLOATS; // header-only records
    const buf = makeRecordBuffer(3, stride);
    // v0, v1 inside; v2 outside (x = 2 > w = 1).
    writeRecord(buf, 0, 0, 0, 0, 1, 1);
    writeRecord(buf, stride, -0.5, 0, 0, 1, 1);
    writeRecord(buf, 2 * stride, 2, 1, 0, 1, 1);
    const { out, outCount } = clipRecords(buf, stride, 3);
    // Resulting polygon = [v0, v1, A, B] with A = v0→v2 ∩ {x=w} at t=0.5 →
    // (1, 0.5), B = v1→v2 ∩ {x=w} at t=0.6 → (1, 0.6) (linear clip-space
    // interpolation: x_new = lerp(x_a, x_b, t) = w = 1).
    expectVertices(out, outCount, stride, [
      [0, 0, 0, 1, 1],
      [-0.5, 0, 0, 1, 1],
      [1, 0.5, 0, 1, 1],
      [1, 0.6, 0, 1, 1],
    ]);
    expectInsideClipVolume(out, outCount, stride);
  });

  it("interpolates varyings linearly in clip space for clipped vertices (long records)", () => {
    const stride = computeVertexStride(TWO_VEC2_VARYINGS); // 9
    const buf = makeRecordBuffer(3, stride);
    writeRecord(buf, 0, 0, 0, 0, 1, 1, [10, 20, 30, 40]);
    writeRecord(buf, stride, -0.5, 0, 0, 1, 1, [11, 21, 31, 41]);
    writeRecord(buf, 2 * stride, 2, 1, 0, 1, 1, [12, 22, 32, 42]);
    const { out, outCount } = clipRecords(buf, stride, 3);
    // A = lerp(v0, v2, 0.5) → varyings [11, 21, 31, 41];
    // B = lerp(v1, v2, 0.6) → varyings [11.6, 21.6, 31.6, 41.6].
    expectVertices(out, outCount, stride, [
      [0, 0, 0, 1, 1, 10, 20, 30, 40],
      [-0.5, 0, 0, 1, 1, 11, 21, 31, 41],
      [1, 0.5, 0, 1, 1, 11, 21, 31, 41],
      [1, 0.6, 0, 1, 1, 11.6, 21.6, 31.6, 41.6],
    ]);
    expectInsideClipVolume(out, outCount, stride);
  });

  it("returns 0 for a fully-outside triangle", () => {
    const stride = RECORD_HEADER_FLOATS;
    const buf = makeRecordBuffer(3, stride);
    writeRecord(buf, 0, 2, 0, 0, 1, 1);
    writeRecord(buf, stride, 3, 0, 0, 1, 1);
    writeRecord(buf, 2 * stride, 2, 1, 0, 1, 1);
    const { outCount } = clipRecords(buf, stride, 3);
    expect(outCount).toBe(0);
  });

  it("clips a LINE primitive (count 2) against a plane", () => {
    const stride = RECORD_HEADER_FLOATS;
    const buf = makeRecordBuffer(2, stride);
    // Both endpoints on the same side (x > w) → segment never enters the
    // volume → fully clipped.
    writeRecord(buf, 0, 2, 0, 0, 1, 1);
    writeRecord(buf, stride, 3, 0, 0, 1, 1);
    const { outCount } = clipRecords(buf, stride, 2);
    expect(outCount).toBe(0);
    // One endpoint inside: crosses the x=w plane at t=0.5 → (1, 0, 0, 1).
    writeRecord(buf, 0, 0, 0, 0, 1, 1);
    writeRecord(buf, stride, 2, 0, 0, 1, 1);
    const { out, outCount: n } = clipRecords(buf, stride, 2);
    expect(n).toBe(2);
    expectVertices(out, n, stride, [
      [0, 0, 0, 1, 1],
      [1, 0, 0, 1, 1],
    ]);
  });
});

describe("applyViewportTransform", () => {
  const stride = RECORD_HEADER_FLOATS;

  it("maps clip (0,0,0,1) to the viewport center and depth midpoint", () => {
    const buf = makeRecordBuffer(1, stride);
    writeRecord(buf, 0, 0, 0, 0, 1, 1);
    applyViewportTransform(
      buf, 0, stride, 1, { x: 0, y: 0, w: 100, h: 50 }, { near: 0, far: 1 },
    );
    // winX = 0 + 100·(0·0.5+0.5) = 50; winY = 0 + 50·0.5 = 25; winZ = 0.5.
    expectArrayClose(readHeader(buf, 0), [50, 25, 0.5, 1, 1]);
  });

  it("divides by clip w and preserves clip w in slot 3", () => {
    const buf = makeRecordBuffer(1, stride);
    writeRecord(buf, 0, 20, 10, 0, 2, 1);
    applyViewportTransform(
      buf, 0, stride, 1, { x: 0, y: 0, w: 100, h: 50 }, { near: 0, far: 1 },
    );
    // NDC (10, 5, 0): winX = 100·(10·0.5+0.5) = 550; winY = 50·(5·0.5+0.5) = 150.
    expectArrayClose(readHeader(buf, 0), [550, 150, 0.5, 2, 1]);
  });

  it("maps depth through the depth range (near→near, far→far)", () => {
    const buf = makeRecordBuffer(3, stride);
    writeRecord(buf, 0, 0, 0, -1, 1, 1);
    writeRecord(buf, stride, 0, 0, 0, 1, 1);
    writeRecord(buf, 2 * stride, 0, 0, 1, 1, 1);
    applyViewportTransform(
      buf, 0, stride, 3, { x: 0, y: 0, w: 1, h: 1 }, { near: 0, far: 1 },
    );
    expect(readHeader(buf, 0)[RECORD_OFFSET_Z]).toBeCloseTo(0, 10);
    expect(readHeader(buf, stride)[RECORD_OFFSET_Z]).toBeCloseTo(0.5, 10);
    expect(readHeader(buf, 2 * stride)[RECORD_OFFSET_Z]).toBeCloseTo(1, 10);

    // Custom range near=0.25, far=0.75 (fresh buffer — transform is in-place).
    const buf2 = makeRecordBuffer(3, stride);
    writeRecord(buf2, 0, 0, 0, -1, 1, 1);
    writeRecord(buf2, stride, 0, 0, 0, 1, 1);
    writeRecord(buf2, 2 * stride, 0, 0, 1, 1, 1);
    applyViewportTransform(
      buf2, 0, stride, 3, { x: 0, y: 0, w: 1, h: 1 }, { near: 0.25, far: 0.75 },
    );
    expect(readHeader(buf2, 0)[RECORD_OFFSET_Z]).toBeCloseTo(0.25, 10);
    expect(readHeader(buf2, stride)[RECORD_OFFSET_Z]).toBeCloseTo(0.5, 10);
    expect(readHeader(buf2, 2 * stride)[RECORD_OFFSET_Z]).toBeCloseTo(0.75, 10);
  });

  it("clamps the mapped depth to [0,1]", () => {
    const buf = makeRecordBuffer(2, stride);
    writeRecord(buf, 0, 0, 0, 2, 1, 1);  // z/w = 2 → 1.5 → clamped to 1
    writeRecord(buf, stride, 0, 0, -2, 1, 1); // z/w = -2 → -0.5 → clamped to 0
    applyViewportTransform(
      buf, 0, stride, 2, { x: 0, y: 0, w: 1, h: 1 }, { near: 0, far: 1 },
    );
    expect(readHeader(buf, 0)[RECORD_OFFSET_Z]).toBeCloseTo(1, 10);
    expect(readHeader(buf, stride)[RECORD_OFFSET_Z]).toBeCloseTo(0, 10);
  });
});

describe("pointIsVisible", () => {
  const stride = RECORD_HEADER_FLOATS;

  it("accepts points inside -w ≤ x,y,z ≤ w (boundaries inclusive)", () => {
    const buf = makeRecordBuffer(1, stride);
    for (const [x, y, z, w] of [
      [0, 0, 0, 1],
      [1, 0, 0, 1],   // x = +w boundary
      [-1, 0, 0, 1],  // x = -w boundary
      [0, 1, 0, 1],   // y = +w boundary
      [0, 0, 1, 1],   // z = +w boundary
      [2, 1, -3, 4],
    ]) {
      writeRecord(buf, 0, x, y, z, w, 1);
      expect(pointIsVisible(buf, 0, stride)).toBe(true);
    }
  });

  it("rejects points outside any clip plane", () => {
    const buf = makeRecordBuffer(1, stride);
    for (const [x, y, z, w] of [
      [2, 0, 0, 1],   // x > w
      [-2, 0, 0, 1],  // x < -w
      [0, 2, 0, 1],   // y > w
      [0, 0, 2, 1],   // z > w
      [0, 0, 0, -1],  // w < 0: -w ≤ 0 ≤ w is empty
    ]) {
      writeRecord(buf, 0, x, y, z, w, 1);
      expect(pointIsVisible(buf, 0, stride)).toBe(false);
    }
  });
});

describe("record layout (computeVertexStride / writeVertexHeader)", () => {
  it("exposes the documented record-layout constants", () => {
    expect(RECORD_OFFSET_X).toBe(0);
    expect(RECORD_OFFSET_Y).toBe(1);
    expect(RECORD_OFFSET_Z).toBe(2);
    expect(RECORD_OFFSET_W).toBe(3);
    expect(RECORD_OFFSET_POINT_SIZE).toBe(4);
    expect(VARYINGS_OFFSET).toBe(5);
    expect(RECORD_HEADER_FLOATS).toBe(5);
    expect(MAX_CLIPPED_VERTICES).toBeGreaterThanOrEqual(7);
  });

  it("computeVertexStride sums the header and varying components", () => {
    expect(computeVertexStride([])).toBe(RECORD_HEADER_FLOATS);
    expect(computeVertexStride(TWO_VEC2_VARYINGS)).toBe(9);
    expect(computeVertexStride([
      { name: "v0", type: FLOAT_VEC3, components: 3, flat: false },
      { name: "v1", type: FLOAT_VEC4, components: 4, flat: true },
      { name: "v2", type: FLOAT, components: 1, flat: false },
    ])).toBe(13);
  });

  it("writeVertexHeader places fields at the documented offsets (base 0 and non-zero base)", () => {
    const stride = computeVertexStride(TWO_VEC2_VARYINGS); // 9
    const buf = makeRecordBuffer(2, stride);
    writeVertexHeader(buf, 0, 1.5, -2.5, 3.5, 4.5, 8);
    writeVertexHeader(buf, stride, -1, 2, -3, 4, 16);
    expectArrayClose(readHeader(buf, 0), [1.5, -2.5, 3.5, 4.5, 8]);
    expectArrayClose(readHeader(buf, stride), [-1, 2, -3, 4, 16]);
    // Varying slots are untouched by the header write.
    expect(buf[VARYINGS_OFFSET]).toBe(0);
    expect(buf[stride + VARYINGS_OFFSET]).toBe(0);
    expect(buf[stride + VARYINGS_OFFSET + 1]).toBe(0);
  });
});
