import fs from 'node:fs';
import path from 'node:path';

export const DEFAULT_PAGES: string[] = [
  'webgl_animation_keyframes', 'webgl_animation_skinning_blending', 'webgl_animation_skinning_morph',
  'webgl_buffergeometry', 'webgl_buffergeometry_instancing', 'webgl_geometry_shapes',
  'webgl_geometry_spline_editor', 'webgl_geometry_terrain', 'webgl_geometry_text',
  'webgl_instancing_dynamic', 'webgl_instancing_raycast', 'webgl_interactive_raycasting_points',
  'webgl_lights_hemisphere', 'webgl_lines_fat', 'webgl_materials_cubemap_dynamic',
  'webgl_materials_displacementmap', 'webgl_materials_envmaps', 'webgl_materials_normalmap',
  'webgl_materials_physical_clearcoat', 'webgl_materials_physical_transmission',
  'webgl_materials_texture_anisotropy', 'webgl_materials_video', 'webgl_mirror',
  'webgl_morphtargets', 'webgl_morphtargets_horse', 'webgl_multisampled_renderbuffers',
  'webgl_points_billboards', 'webgl_points_dynamic', 'webgl_points_sprites',
  'webgl_points_waves', 'webgl_postprocessing_afterimage', 'webgl_postprocessing_dof',
  'webgl_postprocessing_fxaa', 'webgl_postprocessing_ssao', 'webgl_postprocessing_unreal_bloom',
  'webgl_shadowmap', 'webgl_shadowmap_vsm', 'webgl_tonemapping'
]; // EXACTLY 38 entries — do not add/remove/reorder

export const NETWORK_EXCLUDED: string[] = ['physics_ammo_cloth'];

export function repoPath(): string {
  const p = process.env.THREEJS_REPO ?? '/testsuites/three.js';
  if (!fs.existsSync(p)) {
    throw new Error(`three.js repository not found at "${p}" (set env THREEJS_REPO to override)`);
  }
  return p;
}

export function pageHtmlPath(repo: string, name: string): string {
  return `${repo}/examples/${name}.html`;
}

export function goldenPath(repo: string, name: string): string {
  return `${repo}/examples/screenshots/${name}.jpg`;
}

export function hasGolden(repo: string, name: string): boolean {
  return fs.existsSync(goldenPath(repo, name));
}

export function selectPages(repo: string, opts: { full?: boolean; filter?: string }): string[] {
  let names: string[];
  if (opts.full) {
    const entries = fs.readdirSync(path.join(repo, 'examples'));
    names = entries
      .filter((f) => f.endsWith('.html'))
      .map((f) => f.slice(0, -'.html'.length))
      .filter((name) => name !== 'index.html')
      .filter((name) => !name.startsWith('webgpu_'))
      .filter((name) => !name.startsWith('webxr_'))
      .filter((name) => !name.startsWith('css2d_'))
      .filter((name) => !name.startsWith('css3d_'))
      .filter((name) => !NETWORK_EXCLUDED.includes(name))
      .filter((name) => hasGolden(repo, name))
      .sort();
  } else {
    names = [...DEFAULT_PAGES];
  }

  if (opts.filter) {
    const parts = opts.filter.split(',').map((s) => s.trim()).filter((s) => s.length > 0);
    if (parts.length > 0) {
      names = names.filter((name) => parts.some((part) => name.includes(part)));
    }
  }

  return [...new Set(names)];
}
