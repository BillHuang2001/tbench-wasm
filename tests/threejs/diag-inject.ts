/**
 * TEMPORARY diagnostic injection v2 (do not commit) — probes whether draws are
 * issued, whether vertex/fragment runs execute, and whether they throw.
 * Logs via console.error (captured by the driver into report consoleErrors).
 */
export function buildDiagScript(): string {
  return `(() => {
  if (window.__diagInstalled) return;
  window.__diagInstalled = true;
  const log = (s) => { try { console.error('DIAG ' + s); } catch (_) {} };
  const wrappedProgs = new WeakSet();
  const stageCounters = { vertex: 0, fragment: 0 };
  function wrapProg(prog) {
    if (!prog || wrappedProgs.has(prog)) return;
    wrappedProgs.add(prog);
    for (const stage of ['vertex', 'fragment']) {
      const obj = prog[stage];
      if (!obj || typeof obj.run !== 'function') continue;
      const fn = obj.run;
      let firstRun = true;
      obj.run = function (...args) {
        if (firstRun) { firstRun = false; log('RUN-' + stage.toUpperCase() + '-FIRST prog=' + (prog.name || 'unnamed') + ' hasTex=' + !!(args[0] && args[0].tex)); }
        stageCounters[stage]++;
        if (stageCounters[stage] % 500 === 0) log('RUN-' + stage.toUpperCase() + ' n=' + stageCounters[stage]);
        try {
          return fn.apply(this, args);
        } catch (e) {
          log(stage.toUpperCase() + '-EXC prog=' + (prog.name || 'unnamed') + ' ' + (e && e.stack ? e.stack.split('\\n').slice(0, 5).join(' | ') : String(e)));
          throw e;
        }
      };
    }
  }
  const DRAW_METHODS = ['drawArrays', 'drawElements', 'drawArraysInstanced', 'drawElementsInstanced'];
  function wrapContext(gl) {
    if (!gl || gl.__diagWrapped) return gl;
    gl.__diagWrapped = true;
    log('HOOK has _state=' + !!gl._state + ' methods=' + DRAW_METHODS.filter((m) => typeof gl[m] === 'function').join(','));
    const logTarget = (m, args) => {
      let extra = '';
      try {
        const st = this._state;
        const prog = st && st.currentProgram && st.currentProgram._program;
        if (prog) wrapProg(prog);
        const fbo = st && st.boundFramebuffer;
        extra = ' fbo=' + (fbo ? (fbo._id || fbo.id || 'obj') : 'default');
      } catch (e) { extra = ' state-err=' + String(e); }
      log(m.toUpperCase() + ' ' + JSON.stringify(args) + extra);
    };
    for (const m of DRAW_METHODS) {
      const orig = gl[m];
      if (typeof orig !== 'function') continue;
      gl[m] = function (...args) { logTarget.call(this, m, args); return orig.apply(this, args); };
    }
    for (const m of ['clear', 'bindFramebuffer', 'viewport', 'scissor']) {
      const orig = gl[m];
      if (typeof orig !== 'function') continue;
      gl[m] = function (...args) { log(m.toUpperCase() + ' ' + JSON.stringify(args)); return orig.apply(this, args); };
    }
    return gl;
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
