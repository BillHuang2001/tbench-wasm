/**
 * Unit tests for the GLSL ES compiler/linker (src/glsl) — written against the
 * REAL contract in src/glsl/compiler.ts + src/glsl/program.ts (both re-exported
 * by `../../src/glsl/index`). Fails (module not found / stub 'not implemented')
 * until src/glsl lands; these tests are then the executable spec.
 *
 * Execution contexts (src/glsl/program.ts):
 * - BaseExecCtx: { uniforms: Float32Array (program.floatStore),
 *   intUniforms: Int32Array (program.intStore), blockStores: Float32Array[],
 *   blockIntStores: Int32Array[], textures: (TextureImage|null)[],
 *   samplerStates: SamplerState[], scratch: Float32Array (>= scratchSize),
 *   intScratch: Int32Array (>= intScratchSize) }
 * - VertexExecCtx = BaseExecCtx + { attribs: AttribSource[],
 *   attribIndices: Int32Array, vertexId: number, instanceId: number,
 *   out: { position: Float32Array(4), pointSize: number,
 *   varyings: Float32Array (packed per Program.varyings order) } }
 * - FragmentExecCtx = BaseExecCtx + { varyings: VaryingValues[]
 *   ({ v: Float32Array, ddx?, ddy? }), fragCoord: Float32Array(4),
 *   frontFacing: boolean, pointCoord: Float32Array(2), discarded: boolean,
 *   out: { color: Float32Array[], fragDepth: number } }
 *
 * Helpers `makeVertexCtx` / `makeFragmentCtx` (below) build full structural
 * contexts: uniforms/intUniforms ARE the program's own stores (so uniform
 * writes via UniformInfo.location hit the real data), scratch is sized from
 * program.scratchSize/intScratchSize, and out arrays are preallocated to the
 * program's layout (varyings packed per Program.varyings order; one color
 * Float32Array(4) per fragment.outputs entry).
 */
import { describe, it, expect } from "vitest";
import {
  compileShader,
  linkProgram,
  type Program,
  type VertexExecCtx,
  type FragmentExecCtx,
} from "../../src/glsl/index";
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

/** Total packed varying components, in Program.varyings order. */
function totalVaryingComponents(program: Program): number {
  return program.varyings.reduce((n, v) => n + v.components, 0);
}

/**
 * Full structural VertexExecCtx per src/glsl/program.ts. `attribs` is indexed
 * by attribute location (Float32Array per attribute, or a scalar constant).
 */
function makeVertexCtx(
  program: Program,
  attribs: (Float32Array | number)[],
  opts?: { vertexId?: number; instanceId?: number },
): VertexExecCtx {
  return {
    attribs,
    attribIndices: new Int32Array(attribs.length),
    uniforms: program.floatStore,
    intUniforms: program.intStore,
    blockStores: [],
    blockIntStores: [],
    textures: [],
    samplerStates: [],
    scratch: new Float32Array(Math.max(program.scratchSize, 16)),
    intScratch: new Int32Array(Math.max(program.intScratchSize, 16)),
    vertexId: opts?.vertexId ?? 0,
    instanceId: opts?.instanceId ?? 0,
    out: {
      position: new Float32Array(4),
      pointSize: 1,
      varyings: new Float32Array(totalVaryingComponents(program)),
    },
  };
}

/** Full structural FragmentExecCtx per src/glsl/program.ts. */
function makeFragmentCtx(
  program: Program,
  opts?: {
    fragCoord?: Float32Array;
    frontFacing?: boolean;
    pointCoord?: Float32Array;
  },
): FragmentExecCtx {
  return {
    uniforms: program.floatStore,
    intUniforms: program.intStore,
    blockStores: [],
    blockIntStores: [],
    textures: [],
    samplerStates: [],
    scratch: new Float32Array(Math.max(program.scratchSize, 16)),
    intScratch: new Int32Array(Math.max(program.intScratchSize, 16)),
    varyings: program.varyings.map((v) => ({ v: new Float32Array(v.components) })),
    fragCoord: opts?.fragCoord ?? new Float32Array([0, 0, 0, 1]),
    frontFacing: opts?.frontFacing ?? true,
    pointCoord: opts?.pointCoord ?? new Float32Array(2),
    discarded: false,
    out: {
      color: program.fragment.outputs.map(() => new Float32Array(4)),
      fragDepth: 0,
    },
  };
}

const VERT_SIMPLE = `attribute vec4 a_position;
void main() { gl_Position = a_position; }`;

const FRAG_SIMPLE = `precision mediump float;
void main() { gl_FragColor = vec4(1.0, 0.5, 0.0, 1.0); }`;

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

  it("reports a link log on failure (fragment varying not matched)", () => {
    // Native-verified failing direction (probe case B): the FS reads a varying
    // the VS never declares → link fails in both WebGL1 and WebGL2.
    const vs = compileOk(VERT_SIMPLE, { type: "VERTEX", version: 100 });
    const fs = compileOk(
      `precision mediump float;
varying vec2 v_uv;
void main() { gl_FragColor = vec4(v_uv, 0.0, 1.0); }`,
      { type: "FRAGMENT", version: 100 },
    );
    const res = linkProgram(vs, fs);
    expect(res.ok).toBe(false);
    if (!res.ok) {
      expect(typeof res.log).toBe("string");
      expect(res.log).toContain("not matched");
    }
  });

  it("links when the VS writes an extra varying the FS ignores (v100)", () => {
    const vs = compileOk(
      `attribute vec4 a_position;
varying vec2 v_uv;
void main() { v_uv = a_position.xy; gl_Position = a_position; }`,
      { type: "VERTEX", version: 100 },
    );
    const fs = compileOk(FRAG_SIMPLE, { type: "FRAGMENT", version: 100 });
    expect(linkOk(vs, fs)).toBeTruthy();
  });

  it("links when the FS declares an unused varying the VS doesn't have (v100)", () => {
    const vs = compileOk(VERT_SIMPLE, { type: "VERTEX", version: 100 });
    const fs = compileOk(
      `precision mediump float;
varying vec2 v_uv;
void main() { gl_FragColor = vec4(1.0); }`,
      { type: "FRAGMENT", version: 100 },
    );
    expect(linkOk(vs, fs)).toBeTruthy();
  });

  it("links when the VS writes an `out` the FS doesn't declare (v300)", () => {
    const vs = compileOk(
      `#version 300 es
in vec4 a_position;
out vec2 v_uv;
void main() { v_uv = a_position.xy; gl_Position = a_position; }`,
      { type: "VERTEX", version: 300 },
    );
    const fs = compileOk(
      `#version 300 es
precision mediump float;
out vec4 outColor;
void main() { outColor = vec4(1.0, 0.5, 0.0, 1.0); }`,
      { type: "FRAGMENT", version: 300 },
    );
    expect(linkOk(vs, fs)).toBeTruthy();
  });

  it("links when the FS declares an unused `in` the VS doesn't write (v300)", () => {
    const vs = compileOk(
      `#version 300 es
in vec4 a_position;
void main() { gl_Position = a_position; }`,
      { type: "VERTEX", version: 300 },
    );
    const fs = compileOk(
      `#version 300 es
precision mediump float;
in vec2 v_uv;
out vec4 outColor;
void main() { outColor = vec4(1.0); }`,
      { type: "FRAGMENT", version: 300 },
    );
    expect(linkOk(vs, fs)).toBeTruthy();
  });
});

describe("vertex program execution", () => {
  it("writes clip-space position from attributes", () => {
    const program = linkOk(
      compileOk(VERT_SIMPLE, { type: "VERTEX", version: 100 }),
      compileOk(FRAG_SIMPLE, { type: "FRAGMENT", version: 100 }),
    );
    const ctx = makeVertexCtx(program, [new Float32Array([1, 2, 3, 1])]);
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

    // out.varyings is preallocated by makeVertexCtx to the packed layout (2).
    const ctx = makeVertexCtx(program, [
      new Float32Array([0, 0, 0, 1]),
      new Float32Array([0.25, 0.75]),
    ]);
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

    // ctx.uniforms IS program.floatStore: writes by location hit the real store.
    const ctx = makeVertexCtx(program, [new Float32Array([1, 1, 1, 1])]);
    program.floatStore[uniform.location + 0] = 0.5;
    program.floatStore[uniform.location + 1] = 0.25;
    program.floatStore[uniform.location + 2] = 0.125;
    program.floatStore[uniform.location + 3] = 1.0;
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

    const ctx = makeVertexCtx(program, [new Float32Array([0, 0, 0, 1])]);
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
    const ctx = makeVertexCtx(program, [new Float32Array([0, 2.0, 0])]);
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
    const ctx = makeVertexCtx(program, [], { vertexId: 7, instanceId: 3 });
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
    // GL_FLOAT_VEC4 = 0x8b52; WebGL1 programs have exactly one output.
    expect(program.fragment.outputs).toEqual([{ location: 0, type: 0x8b52 }]);

    const ctx = makeFragmentCtx(program);
    program.fragment.run(ctx);
    expectArrayClose(ctx.out.color[0], [1, 0.5, 0, 1], 1e-4);
  });
});
