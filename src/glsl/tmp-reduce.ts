// TMP REDUCE (delete before finishing): feed captured shaders + minimal repros
// to compileShader directly.
import { compileShader, linkProgram } from "./compiler.ts";
import { readFileSync } from "node:fs";

function tryCompile(label: string, source: string, type: "VERTEX" | "FRAGMENT" = "VERTEX") {
  const res = compileShader(source, { type, version: 300, extensions: new Set<string>() });
  if (!res.ok) {
    console.log(`[${label}] COMPILE FAIL:`);
    for (const e of res.errors) console.log(`  ${e.line}: ${e.message}`);
    return null;
  }
  console.log(`[${label}] compile OK`);
  return res.shader;
}

// 1. minimal repro
const minimal = `#version 300 es
layout(std140,column_major) uniform;
void main() { gl_Position = vec4(0.0); }
`;
tryCompile("minimal-layout-uniform", minimal);

// 2. with a block after it
const withBlock = `#version 300 es
layout(std140,column_major) uniform;
uniform Material { vec4 color; };
layout(std140,column_major) uniform;
uniform Scene { mat4 view; };
void main() { gl_Position = vec4(0.0); }
`;
tryCompile("layout+blocks", withBlock);

// 3. layout with just std140 (no column_major)
tryCompile("layout-std140-only", `#version 300 es
layout(std140) uniform;
void main() { gl_Position = vec4(0.0); }
`);

// 4. layout in; (standalone in layout — per grammar type_qualifier SEMICOLON)
tryCompile("layout-in", `#version 300 es
layout(location=0) in;
void main() { gl_Position = vec4(0.0); }
`);

// 5. the full captured vertex shader
const diag = JSON.parse(readFileSync("/tmp/babylon-diag.json", "utf8"));
if (diag.failed && diag.failed[0]) {
  tryCompile("captured-vs", diag.failed[0].source);
}
