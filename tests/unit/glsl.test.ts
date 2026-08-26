/**
 * Unit tests for the GLSL ES compiler/linker (src/glsl) — written against the
 * FINAL contract in src/CONTEXT.md §1 (compileShader / linkProgram / Program
 * model). Fails (module not found) until src/glsl lands; these tests are then
 * the executable spec.
 *
 * Assumed import: `../../src/glsl/index` exports `compileShader` and
 * `linkProgram` with the exact signatures from src/CONTEXT.md §1.
 * If the glsl module layout differs, update ONLY this import line.
 */
import { describe, it, expect } from "vitest";
import { compileShader, linkProgram } from "../../src/glsl/index";
import { expectArrayClose } from "./helpers";

type CompileOpts = { type: "VERTEX" | "FRAGMENT"; version: 100 | 300 };
type Shader = Parameters<typeof linkProgram>[0];

/** Compiles and returns the shader on success; throws a descriptive error otherwise. */
function compileOk(src: string, opts: CompileOpts): Shader {
  const res = compileShader(src, opts);
  if (!res.ok) {
    throw new Error(`expected compile to succeed, got errors: ${JSON.stringify(res.errors)}`);
  }
  return res.shader;
}

function compileFail(src: string, opts: CompileOpts) {
  const res = compileShader(src, opts);
  if (res.ok) {
    throw new Error("expected compile to FAIL, but it succeeded");
  }
  return res.errors;
}

function linkOk(vs: Shader, fs: Shader) {
  const res = linkProgram(vs, fs);
  if (!res.ok) {
    throw new Error(`expected link to succeed: ${res.log}`);
  }
  return res.program;
}

const VERT_SIMPLE = `attribute vec4 a_position;
void main() { gl_Position = a_position; }`;

const FRAG_SIMPLE = `precision mediump float;
void main() { gl_FragColor = vec4(1.0, 0.5, 0.0, 1.0); }`;

/** Minimal vertex-execution context per src/CONTEXT.md §1. */
function vertexCtx(attribs: (ArrayLike<number> | number | undefined)[], uniformCount: number) {
  return {
    attribs,
    uniforms: new Float32Array(Math.max(16, uniformCount)),
    vertexId: 0,
    instanceId: 0,
    out: { position: [0, 0, 0, 0], pointSize: 1, varyings: new Float32Array(0) },
  };
}

describe("compileShader", () => {
  it("compiles a trivial vertex shader (GLSL ES 1.00)", () => {
    const shader = compileOk(VERT_SIMPLE, { type: "VERTEX", version: 100 });
    expect(shader).toBeTruthy();
  });

  it("compiles a trivial fragment shader (GLSL ES 1.00)", () => {
    const shader = compileOk(FRAG_SIMPLE, { type: "FRAGMENT", version: 100 });
    expect(shader).toBeTruthy();
  });

  it("reports compile errors with 1-based line numbers", () => {
    const src = `attribute vec4 a_position;
void main() {
  gl_Position = a_position + missing_thing;
}`;
    const errors = compileFail(src, { type: "VERTEX", version: 100 });
    expect(errors.length).toBeGreaterThan(0);
    expect(errors[0].line).toBe(3);
    expect(typeof errors[0].message).toBe("string");
  });

  it("rejects GLSL ES 3.00 `in` keyword under version 100 (WebGL1 strictness)", () => {
    const errors = compileFail(
      "in vec4 a_position;\nvoid main() { gl_Position = a_position; }",
      { type: "VERTEX", version: 100 },
    );
    expect(errors.length).toBeGreaterThan(0);
  });

  it("accepts GLSL ES 3.00 in/out under version 300", () => {
    const vs = compileOk(
      `#version 300 es
in vec4 a_position;
out vec4 v_color;
void main() { v_color = a_position; gl_Position = a_position; }`,
      { type: "VERTEX", version: 300 },
    );
    const fs = compileOk(
      `#version 300 es
precision mediump float;
out vec4 outColor;
void main() { outColor = vec4(1.0); }`,
      { type: "FRAGMENT", version: 300 },
    );
    expect(vs).toBeTruthy();
    expect(fs).toBeTruthy();
  });
});

describe("linkProgram", () => {
  it("links a trivial program and exposes attribute metadata", () => {
    const program = linkOk(
      compileOk(VERT_SIMPLE, { type: "VERTEX", version: 100 }),
      compileOk(FRAG_SIMPLE, { type: "FRAGMENT", version: 100 }),
    );
    expect(program.attributes.length).toBe(1);
    expect(program.attributes[0].name).toBe("a_position");
    expect(program.attributes[0].location).toBe(0);
    expect(program.attributes[0].components).toBe(4);
    expect(program.uniforms).toEqual([]);
    expect(program.varyings).toEqual([]);
  });

  it("reports a link log on failure (mismatched varying)", () => {
    const vs = compileOk(
      `attribute vec4 a_position;
varying vec2 v_uv;
void main() { v_uv = a_position.xy; gl_Position = a_position; }`,
      { type: "VERTEX", version: 100 },
    );
    // Fragment declares no varying at all → linker error.
    const fs = compileOk(FRAG_SIMPLE, { type: "FRAGMENT", version: 100 });
    const res = linkProgram(vs, fs);
    expect(res.ok).toBe(false);
    if (!res.ok) {
      expect(typeof res.log).toBe("string");
      expect(res.log.length).toBeGreaterThan(0);
    }
  });
});

describe("vertex program execution", () => {
  it("writes clip-space position from attributes", () => {
    const program = linkOk(
      compileOk(VERT_SIMPLE, { type: "VERTEX", version: 100 }),
      compileOk(FRAG_SIMPLE, { type: "FRAGMENT", version: 100 }),
    );
    const ctx = vertexCtx([new Float32Array([1, 2, 3, 1])], 0);
    program.vertex.run(ctx);
    expectArrayClose(ctx.out.position, [1, 2, 3, 1]);
  });

  it("passes varyings through to vertex output in declaration order", () => {
    const vs = compileOk(
      `attribute vec4 a_position;
attribute vec2 a_uv;
varying vec2 v_uv;
void main() { v_uv = a_uv; gl_Position = a_position; }`,
      { type: "VERTEX", version: 100 },
    );
    const fs = compileOk(
      `precision mediump float;
varying vec2 v_uv;
void main() { gl_FragColor = vec4(v_uv, 0.0, 1.0); }`,
      { type: "FRAGMENT", version: 100 },
    );
    const program = linkOk(vs, fs);

    expect(program.varyings.length).toBe(1);
    expect(program.varyings[0].name).toBe("v_uv");
    expect(program.varyings[0].components).toBe(2);

    const ctx = vertexCtx(
      [new Float32Array([0, 0, 0, 1]), new Float32Array([0.25, 0.75])],
      0,
    );
    ctx.out.varyings = new Float32Array(2);
    program.vertex.run(ctx);
    expectArrayClose(ctx.out.varyings, [0.25, 0.75]);
  });

  it("reads uniforms through the per-program uniform store by location", () => {
    const vs = compileOk(
      `uniform vec4 u_color;
attribute vec4 a_position;
void main() { gl_Position = a_position * u_color; }`,
      { type: "VERTEX", version: 100 },
    );
    const program = linkOk(vs, compileOk(FRAG_SIMPLE, { type: "FRAGMENT", version: 100 }));

    const uniform = program.uniforms.find((u) => u.name === "u_color");
    expect(uniform).toBeTruthy();
    if (!uniform) return;
    expect(uniform.components).toBe(4);

    const ctx = vertexCtx([new Float32Array([1, 1, 1, 1])], 4);
    ctx.uniforms[uniform.location + 0] = 0.5;
    ctx.uniforms[uniform.location + 1] = 0.25;
    ctx.uniforms[uniform.location + 2] = 0.125;
    ctx.uniforms[uniform.location + 3] = 1.0;
    program.vertex.run(ctx);
    expectArrayClose(ctx.out.position, [0.5, 0.25, 0.125, 1.0]);
  });

  it("exposes usesPointSize and writes gl_PointSize", () => {
    const vs = compileOk(
      `attribute vec4 a_position;
void main() { gl_PointSize = 3.0; gl_Position = a_position; }`,
      { type: "VERTEX", version: 100 },
    );
    const program = linkOk(vs, compileOk(FRAG_SIMPLE, { type: "FRAGMENT", version: 100 }));
    expect(program.usesPointSize).toBe(true);

    const ctx = vertexCtx([new Float32Array([0, 0, 0, 1])], 0);
    program.vertex.run(ctx);
    expect(ctx.out.pointSize).toBeCloseTo(3.0, 5);
  });

  it("evaluates built-in functions (sin/mix/clamp/normalize/dot)", () => {
    const vs = compileOk(
      `attribute vec3 a;
void main() {
  float s = sin(a.x);
  float m = mix(0.0, 10.0, 0.5);
  float c = clamp(a.y, 0.0, 1.0);
  vec3 n = normalize(vec3(3.0, 4.0, 0.0));
  float d = dot(n, vec3(1.0, 0.0, 0.0));
  gl_Position = vec4(s, m, c, d);
}`,
      { type: "VERTEX", version: 100 },
    );
    const program = linkOk(vs, compileOk(FRAG_SIMPLE, { type: "FRAGMENT", version: 100 }));
    const ctx = vertexCtx([new Float32Array([0, 2.0, 0])], 0);
    program.vertex.run(ctx);
    // sin(0)=0, mix=5, clamp(2,0,1)=1, dot(normalize(3,4,0),(1,0,0))=0.6
    expectArrayClose(ctx.out.position, [0, 5, 1, 0.6], 1e-4);
  });

  it("exposes gl_VertexID / gl_InstanceID (GLSL ES 3.00)", () => {
    const vs = compileOk(
      `#version 300 es
void main() {
  gl_Position = vec4(float(gl_VertexID), float(gl_InstanceID), 0.0, 1.0);
}`,
      { type: "VERTEX", version: 300 },
    );
    const fs = compileOk(
      `#version 300 es
precision mediump float;
out vec4 outColor;
void main() { outColor = vec4(1.0); }`,
      { type: "FRAGMENT", version: 300 },
    );
    const program = linkOk(vs, fs);
    const ctx = vertexCtx([], 0);
    ctx.vertexId = 7;
    ctx.instanceId = 3;
    program.vertex.run(ctx);
    expectArrayClose(ctx.out.position, [7, 3, 0, 1]);
  });
});

describe("fragment program execution", () => {
  it("writes gl_FragColor to output location 0", () => {
    const program = linkOk(
      compileOk(VERT_SIMPLE, { type: "VERTEX", version: 100 }),
      compileOk(FRAG_SIMPLE, { type: "FRAGMENT", version: 100 }),
    );
    const ctx = {
      varyings: [],
      fragCoord: [0, 0, 0, 1],
      frontFacing: true,
      pointCoord: [0, 0],
      uniforms: new Float32Array(0),
      out: { color: [[0, 0, 0, 0]] },
    };
    program.fragment.run(ctx);
    expectArrayClose(ctx.out.color[0], [1, 0.5, 0, 1], 1e-4);
  });
});
