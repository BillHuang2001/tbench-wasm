/**
 * index.ts — public API of the raster module.
 *
 * This is the compile-time contract for gl/ (draw pipeline + formats +
 * surfaces) and glsl/ (fragment exec ctx + texture env). Exports:
 *
 *  - draw(dc: DrawCall) — the rasterizer entry (contract §2).
 *  - All shared types (types.ts): DrawCall, RasterState, FragmentExecCtx,
 *    TextureEnv, SampleCoord, SamplerState, TextureImage, Surface,
 *    FramebufferTarget, record-layout constants + helpers.
 *  - The pixel-format registry (formats.ts) — single shared format source
 *    (contract §3).
 *  - The texture sampler (sampler.ts) — sampleTexture + env entry points.
 *  - Surface helpers (surface.ts), fragment ops + clear/blit
 *    (fragment-ops.ts), clipping (clip.ts), primitive rasterizers
 *    (triangles/lines/points.ts), and the GL constants raster interprets
 *    (gl-enums.ts).
 *
 * glsl/ codegen: fragment shaders are compiled to `function(ctx)` bodies
 * that read ctx.varyings[i].v / .ddx / .ddy, ctx.fragCoord, ctx.frontFacing,
 * ctx.pointCoord, ctx.uniforms, ctx.out.color[loc], ctx.out.fragDepth,
 * ctx.discarded, and sample textures via ctx.tex.* (see TextureEnv).
 */

// Draw driver
export { draw, createRasterState, applyFlatFixup } from './rasterizer';

// Shared types, record layout, impl limits
export * from './types';

// Pixel-format registry (contract §3)
export * from './formats';

// Texture sampling (contract §3)
export * from './sampler';

// Surfaces
export * from './surface';

// Fragment ops, quad driver, clear/blit
export * from './fragment-ops';

// Clipping + viewport transform
export * from './clip';

// Primitive rasterizers
export * from './triangles';
export * from './lines';
export * from './points';

// GL constants interpreted by raster
export * from './gl-enums';
