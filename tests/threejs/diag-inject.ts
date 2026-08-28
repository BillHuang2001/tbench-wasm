/**
 * TEMPORARY diagnostic injection v7 (do not commit).
 * v7 adds to v6: s.drawBuffers hex logging at every draw/clear/bind-after
 * (tests the BACK-vs-COLOR_ATTACHMENT0 corruption hypothesis), post-clear
 * center sampling, and a first-run uniform-store sample.
 */
export function buildDiagScript(): string {
  return `(() => {
  if (window.__diag7) return;
  window.__diag7 = true;
  const log = (s) => { try { console.error('DIAG7 ' + s); } catch (_) {} };
  const objIds = new WeakMap();
  let objCounter = 1;
  const oid = (o) => {
    if (o === null || o === undefined) return String(o);
    let id = objIds.get(o);
    if (!id) { id = 'o' + (objCounter++); objIds.set(o, id); }
    return id;
  };
  const pxStr = (arr, w, x, y) => {
    try {
      if (!arr || arr.length < (y * w + x) * 4 + 4) return 'oob';
      const i = (y * w + x) * 4;
      return arr[i] + ',' + arr[i + 1] + ',' + arr[i + 2] + ',' + arr[i + 3];
    } catch (e) { return 'err'; }
  };
  const dbHex = (st) => {
    try {
      if (!st || !st.drawBuffers) return '?';
      const hex = st.drawBuffers.map((v) => '0x' + (v >>> 0).toString(16)).join(',');
      const back = Array.isArray(st.drawBuffers) && st.drawBuffers[0] === 0x0405;
      return hex + (back ? ' <<<BACK!!!' : '');
    } catch (e) { return 'db-err'; }
  };
  const dfbCenter = (gl) => {
    try {
      const dfb = gl._defaultFB;
      if (!dfb || !dfb.color || !(dfb.color.data instanceof Uint8Array)) return '?';
      return pxStr(dfb.color.data, dfb.color.width, dfb.color.width >> 1, dfb.color.height >> 1);
    } catch (e) { return 'err'; }
  };
  // ---- present hook (installed on context creation) ----
  function hookPresent(gl) {
    try {
      const s = gl._presentSurface;
      if (!s || s.__diagHooked) return;
      s.__diagHooked = true;
      const origPresent = s.present;
      s.present = function (...args) {
        let info = '';
        try {
          const px = s._pixels;
          const dfb = gl._defaultFB;
          const dfbPx = dfb && dfb.color ? dfb.color.data : null;
          const same = px === dfbPx;
          const c = gl.canvas;
          info = 'canvasAttr=' + c.width + 'x' + c.height +
            ' surf=' + s.width + 'x' + s.height +
            ' pxIsDfb=' + same +
            ' ctx2d=' + (s._ctx2d ? 'Y' : (s._ctxFailed ? 'FAILED' : 'lazy')) +
            ' blitCenter=' + pxStr(px, s.width, s.width >> 1, s.height >> 1) +
            ' dfbCenter=' + pxStr(dfbPx, dfb ? dfb.width : 0, dfb ? (dfb.width >> 1) : 0, dfb ? (dfb.height >> 1) : 0);
        } catch (e) { info = 'hook-err:' + String(e); }
        log('PRESENT ' + info);
        return origPresent.apply(this, args);
      };
    } catch (e) { log('present-hook-err ' + String(e)); }
  }
  // ---- program model wrapping ----
  const wrappedModels = new WeakSet();
  const stageCounts = { vertex: 0, fragment: 0 };
  function wrapModel(pm, name) {
    if (!pm || wrappedModels.has(pm)) return;
    wrappedModels.add(pm);
    try { if (!pm.__diagName && name) pm.__diagName = name; } catch (_) {}
    for (const stage of ['vertex', 'fragment']) {
      const obj = pm[stage];
      if (!obj || typeof obj.run !== 'function') continue;
      const fn = obj.run;
      let first = true;
      let first2 = 0;
      obj.run = function (...args) {
        const n = ++stageCounts[stage];
        if (first || n <= 4) {
          first = false;
          let outStr = '';
          let unifStr = '';
          try {
            const out = args[0] && args[0].out;
            if (out) {
              const arr = out.position || out.color;
              if (arr) outStr = '[' + Array.from(arr.slice(0, 4)).map((v) => Number.isFinite(v) ? (+v).toFixed(2) : String(v)).join(',') + ']';
              else outStr = '[no-arr]';
            } else outStr = '[no-out]';
            if (stage === 'vertex' && args[0] && args[0].uniforms) {
              const u = args[0].uniforms;
              unifStr = ' u[0..11]=[' + Array.from(u.slice(0, 12)).map((v) => Number.isFinite(v) ? (+v).toFixed(3) : String(v)).join(',') + ']';
            }
          } catch (e) { outStr = 'out-err'; }
          log('RUN-' + stage.toUpperCase() + '#' + n + ' pm=' + (pm.__diagName || 'unnamed') + ' hasTex=' + !!(args[0] && args[0].tex) + ' out=' + outStr + unifStr);
        }
        let res;
        try {
          res = fn.apply(this, args);
        } catch (e) {
          log(stage.toUpperCase() + '-EXC pm=' + (pm.__diagName || 'unnamed') + ' ' + (e && e.stack ? e.stack.split('\\n').slice(0, 4).join(' | ') : String(e)));
          throw e;
        }
        if (first2 < 2) {
          first2++;
          let outStr = '';
          try {
            const out = args[0] && args[0].out;
            if (out) {
              const arr = stage === 'vertex' ? out.position : out.color;
              if (arr) outStr = '[' + Array.from(arr.slice(0, 4)).map((v) => Number.isFinite(v) ? (+v).toFixed(2) : String(v)).join(',') + ']';
            }
          } catch (e) { outStr = 'err'; }
          log('RUN-' + stage.toUpperCase() + '-POST#' + n + ' pm=' + (pm.__diagName || 'unnamed') + ' ' + (stage === 'vertex' ? 'pos=' : 'color=') + outStr);
        }
        return res;
      };
    }
  }
  const trappedProgs = new WeakSet();
  let progCounter = 1;
  function trapProg(wp) {
    if (!wp || trappedProgs.has(wp)) return;
    trappedProgs.add(wp);
    let name = wp.__diagName;
    if (!name) { name = 'p' + (progCounter++); wp.__diagName = name; }
    let model;
    try { model = wp._program; } catch (_) {}
    try {
      Object.defineProperty(wp, '_program', {
        configurable: true,
        get: () => model,
        set: (v) => { model = v; if (v) wrapModel(v, name); },
      });
    } catch (e) { return; }
    if (model) wrapModel(model, name);
  }
  const DRAW_METHODS = ['drawArrays', 'drawElements', 'drawArraysInstanced', 'drawElementsInstanced'];
  function wrapContext(gl) {
    if (!gl || gl.__diag7Wrapped) return gl;
    gl.__diag7Wrapped = true;
    hookPresent(gl);
    log('HOOK state=' + !!gl._state + ' methods=' + DRAW_METHODS.filter((m) => typeof gl[m] === 'function').join(','));
    for (const m of DRAW_METHODS) {
      const orig = gl[m];
      if (typeof orig !== 'function') continue;
      gl[m] = function (...args) {
        let extra = '';
        let vBefore = 0, fBefore = 0;
        let tgtWasNull = false;
        try {
          const st = gl._state;
          const wp = st ? st.currentProgram : null;
          if (wp) trapProg(wp);
          const prog = wp ? wp._program : null;
          vBefore = stageCounts.vertex; fBefore = stageCounts.fragment;
          tgtWasNull = st ? st.drawFramebuffer === null : false;
          extra = 'prog=' + (wp ? (wp.__diagName || '?') : 'null') + ' model=' + (prog ? 'Y' : 'N') +
            ' db=' + dbHex(st) +
            ' depth=' + (st ? (st.caps.DEPTH_TEST ? 'on' : 'off') + '/' + (st.depth.mask ? 'W' : 'w') + '/' + st.depth.func.toString(16) + '/' + st.clearDepth : '?') +
            ' blend=' + (st ? (st.caps.BLEND ? 'on/' + st.blend.srcRGB.toString(16) + ',' + st.blend.dstRGB.toString(16) + '/' + st.blend.eqRGB.toString(16) : 'off') : '?') +
            ' colorMask=' + (st ? st.colorMask.join('') : '?');
        } catch (e) { extra = 'state-err=' + String(e); }
        log(m.toUpperCase() + ' ' + JSON.stringify(args) + ' | ' + extra);
        const res = orig.apply(this, args);
        let readback = '';
        if (tgtWasNull) {
          try {
            const dfb = gl._defaultFB;
            if (dfb && dfb.color && dfb.color.data instanceof Uint8Array) {
              const w = dfb.color.width, h = dfb.color.height;
              readback = ' fbCenter=' + pxStr(dfb.color.data, w, w >> 1, h >> 1) +
                ' fbQ1=' + pxStr(dfb.color.data, w, w >> 2, (h * 3) >> 2) +
                ' fbQ3=' + pxStr(dfb.color.data, w, (w * 3) >> 2, h >> 2);
            }
          } catch (e) { readback = ' rb-err'; }
        }
        log(m.toUpperCase() + '-done dV=' + (stageCounts.vertex - vBefore) + ' dF=' + (stageCounts.fragment - fBefore) + readback);
        return res;
      };
    }
    for (const m of ['clear', 'bindFramebuffer', 'viewport', 'scissor']) {
      const orig = gl[m];
      if (typeof orig !== 'function') continue;
      gl[m] = function (...args) {
        if (m === 'bindFramebuffer') {
          log('BINDFB ' + JSON.stringify(args[0]) + ' ' + (args[1] === null || args[1] === undefined ? 'null' : oid(args[1])));
          const res = orig.apply(this, args);
          let after = '?';
          try { after = tgtStr(gl._state) + ' db=' + dbHex(gl._state) + ' canvasAttr=' + gl.canvas.width + 'x' + gl.canvas.height; } catch (e) { after = 'err'; }
          log('BINDFB-after drawFB=' + after);
          return res;
        }
        if (m === 'clear') {
          let extra = '';
          try {
            const st = gl._state;
            const cc = st ? st.clearColor : null;
            extra = 'tgt=' + tgtStr(st) + ' db=' + dbHex(st) + ' canvasAttr=' + gl.canvas.width + 'x' + gl.canvas.height +
              (cc ? ' cc=[' + cc.map((v) => (+v).toFixed(2)).join(',') + ']' : '') +
              ' clearDepth=' + (st ? st.clearDepth : '?') +
              ' fbCenterBefore=' + dfbCenter(gl);
          } catch (e) { extra = 'err'; }
          log('CLEAR ' + JSON.stringify(args) + ' | ' + extra);
          const res = orig.apply(this, args);
          log('CLEAR-after fbCenter=' + dfbCenter(gl));
          return res;
        }
        log(m.toUpperCase() + ' ' + JSON.stringify(args));
        return orig.apply(this, args);
      };
    }
    return gl;
  }
  function tgtStr(st) {
    if (!st) return 'NO_STATE';
    const f = st.drawFramebuffer;
    return f === undefined ? 'UNDEF' : oid(f);
  }
  const origGetContext = HTMLCanvasElement.prototype.getContext;
  HTMLCanvasElement.prototype.getContext = function (...args) {
    const gl = origGetContext.apply(this, args);
    if (typeof args[0] === 'string' && args[0].indexOf('webgl') === 0) {
      log('GETCTX ' + args[0] + ' -> ' + (gl ? 'software-ctx' : 'null'));
      return wrapContext(gl);
    }
    return gl;
  };
})();`;
}
