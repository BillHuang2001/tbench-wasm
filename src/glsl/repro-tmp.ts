// temp repro script — compile shaders and print results
import { compileShader, linkProgram } from './index.js';

function tryCompile(name: string, source: string, version: 100 | 300, type: 'VERTEX' | 'FRAGMENT') {
  const res = compileShader(source, { type, version });
  if (!res.ok) {
    console.log(`=== ${name}: COMPILE FAIL ===`);
    for (const e of res.errors) console.log(`  ${e.line}: ${e.message}`);
    return;
  }
  // try link (needs a trivial vs)
  const vsSrc = version === 300
    ? '#version 300 es\nvoid main() {}\n'
    : 'void main() { gl_Position = vec4(0.0); }\n';
  const vs = compileShader(vsSrc, { type: 'VERTEX', version });
  if (!vs.ok) { console.log(`=== ${name}: vs compile fail`); return; }
  const link = linkProgram(vs.shader, res.shader);
  console.log(`=== ${name}: COMPILE OK, link=${link.ok}${link.ok ? '' : ' log=' + link.log}`);
}

// 1. switch-case nested case label
tryCompile('switch-nested-case', `#version 300 es
precision highp float;
out vec4 my_FragColor;
uniform int u_zero;
void main()
{
    my_FragColor = vec4(1, 0, 0, 1);
    switch(u_zero)
    {
        case 1:
        {
            case 0:
                my_FragColor = vec4(1, 0, 0, 1);
        }
    }
    my_FragColor = vec4(0, 1, 0, 1);
}
`, 300, 'FRAGMENT');

// 2. sampler array indexing with loop var (ESSL 300 must FAIL)
tryCompile('sampler-array-loop-index', `#version 300 es
precision mediump float;
uniform sampler2D u_tex[2];
void main()
{
  for (int i = 0; i < 2; i++) {
    texture(u_tex[i], vec2(0));
  }
}
`, 300, 'FRAGMENT');

// 2b. sampler array indexing with literal (must PASS)
tryCompile('sampler-array-const-index', `#version 300 es
precision mediump float;
uniform sampler2D u_tex[2];
void main()
{
  texture(u_tex[0], vec2(0));
  texture(u_tex[1], vec2(0));
}
`, 300, 'FRAGMENT');

// 3. sampler3D with no precision (ESSL 300 must FAIL)
tryCompile('sampler3D-no-precision', `#version 300 es
precision mediump float;
uniform sampler3D u_sampler;
void main() { gl_Position = vec4(0.0); }
`, 300, 'VERTEX');

// 3b. sampler3D with explicit precision (must PASS)
tryCompile('sampler3D-with-precision', `#version 300 es
precision mediump float;
precision mediump sampler3D;
uniform sampler3D u_sampler;
void main() { gl_Position = vec4(0.0); }
`, 300, 'VERTEX');

// 3c. sampler2D with no precision (must PASS — predeclared lowp)
tryCompile('sampler2D-no-precision', `#version 300 es
precision mediump float;
uniform sampler2D u_sampler;
void main() { gl_Position = vec4(0.0); }
`, 300, 'VERTEX');

// 4. tricky loop conditions
tryCompile('tricky-V[func()][0]++', `#version 300 es
precision mediump float;
out vec4 color;
int sideEffectCounter = 0;
int func() {
  sideEffectCounter++;
  return sideEffectCounter > 1 ? 1 : 0;
}
void main() {
  vec4[2] V;
  V[0] = vec4(1.0);
  V[1] = vec4(3.0);
  for (int i = 0; true; bool(V[func()][0]++))
  {
    ++i;
    if (i > 1) {
      if (i > 3) { break; }
    }
  }
  color = vec4(0, 1.0, 0, 1.0);
}
`, 300, 'FRAGMENT');

// 4b. dynamic index version
tryCompile('tricky-V[func()][u_zero+1]++', `#version 300 es
precision mediump float;
out vec4 color;
int sideEffectCounter = 0;
uniform int u_zero;
int func() {
  sideEffectCounter++;
  return sideEffectCounter > 1 ? 1 : 0;
}
void main() {
  vec4[2] V;
  V[0] = vec4(1.0);
  V[1] = vec4(3.0);
  bool b = bool(V[func()][u_zero + 1]++);
  color = b ? vec4(0, 1.0, 0, 1.0) : vec4(1.0, 0, 0, 1.0);
}
`, 300, 'FRAGMENT');
