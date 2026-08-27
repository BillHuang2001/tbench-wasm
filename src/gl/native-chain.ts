/**
 * src/gl/native-chain.ts — browser instanceof compatibility via prototype chaining.
 *
 * The CTS harness (`sdk/tests/js/webgl-test-utils.js` `isWebGLContext`) checks
 * `ctx instanceof WebGLRenderingContext` / `WebGL2RenderingContext` against the
 * NATIVE browser classes. Our classes are pure JS (no WebIDL), so instances
 * fail those checks. Fix: re-chain each of our class prototypes UNDER the
 * native prototype of the same name (browser only; no-op in Node where the
 * native globals don't exist).
 *
 * PROTOTYPE CHAINING ONLY. NEVER `Object.setPrototypeOf(OurClass, Native)` —
 * that redirects the constructor's [[Prototype]] (static inheritance) and makes
 * `super()` resolution inside our constructors call the native constructor,
 * which throws "Illegal constructor". Constructor chains stay untouched:
 * instances are still `new OurClass(...)`, so `instance instanceof OurClass`
 * keeps working (our prototype is still the instance's immediate prototype),
 * while `instance instanceof NativeClass` now resolves through the re-chained
 * prototype.
 */

/** Re-chain a class prototype under the native browser class of the same name (browser only; no-op in Node). */
export function chainToNative(ourProto: object, nativeGlobalName: string): void {
  const Native = (globalThis as any)[nativeGlobalName];
  if (typeof Native === 'function' && Native.prototype != null) {
    Object.setPrototypeOf(ourProto, Native.prototype);
  }
}
