/**
 * src/gl/types.ts — WebGL type aliases (WebIDL typedefs) and attribute types.
 *
 * Public API contract for the WebGL context classes. DOM lib types are used where
 * the WebGL spec references DOM types (canvas, TexImageSource); core paths must
 * still run headless (Node) — DOM types are compile-time only here.
 */

// ---- WebIDL typedefs (GL types) ----
export type GLenum = number;
export type GLboolean = boolean;
export type GLbitfield = number;
export type GLbyte = number;
export type GLshort = number;
export type GLint = number;
export type GLsizei = number;
export type GLintptr = number;
export type GLsizeiptr = number;
export type GLuint = number;
export type GLuint64 = number;
export type GLfloat = number;
export type GLclampf = number;

/** WebIDL Float32List — uniform/vertexAttrib vector arguments. */
export type Float32List = Float32Array | number[];
/** WebIDL Int32List */
export type Int32List = Int32Array | number[];
/** WebIDL Uint32List */
export type Uint32List = Uint32Array | number[];
/** WebIDL Uint8List */
export type Uint8List = Uint8Array | number[];

/** Any typed array / DataView accepted by bufferData / texImage2D. */
export type BufferDataSource = ArrayBufferView | ArrayBuffer;

/** Context creation type string accepted by `__createSoftwareWebGLContext`. */
export type ContextType = 'webgl' | 'webgl2' | 'experimental-webgl';

/** The canvas object a context is attached to (real DOM canvas or headless mock). */
export type CanvasLike = HTMLCanvasElement | OffscreenCanvas;

/** WebIDL TexImageSource (WebGL 1.0.2+ / 2.0 overloads). */
export type TexImageSource =
  | ImageData
  | HTMLImageElement
  | HTMLCanvasElement
  | HTMLVideoElement
  | ImageBitmap
  | OffscreenCanvas;

/** WebIDL WebGLPowerPreference. */
export type WebGLPowerPreference = 'default' | 'high-performance' | 'low-power';

/** WebIDL WebGLContextAttributes (getContextAttributes return / creation attrs). */
export interface WebGLContextAttributes {
  alpha: boolean;
  antialias: boolean;
  depth: boolean;
  stencil: boolean;
  premultipliedAlpha: boolean;
  preserveDrawingBuffer: boolean;
  powerPreference: WebGLPowerPreference;
  failIfMajorPerformanceCaveat: boolean;
}

/** Context creation attrs as passed by the caller (getContext 2nd arg) — all optional. */
export type WebGLContextAttributesInit = Partial<WebGLContextAttributes> & {
  /** Non-standard but used by tests/three.js: desynchronized hint — accepted, ignored. */
  desynchronized?: boolean;
};

/**
 * Default context attributes per WebGL spec §"Default Context Attributes"
 * (alpha=true, antialias=true, depth=true, stencil=false, premultipliedAlpha=true,
 * preserveDrawingBuffer=false, powerPreference='default', failIfMajorPerformanceCaveat=false).
 */
export const DEFAULT_CONTEXT_ATTRIBUTES: WebGLContextAttributes = {
  alpha: true,
  antialias: true,
  depth: true,
  stencil: false,
  premultipliedAlpha: true,
  preserveDrawingBuffer: false,
  powerPreference: 'default',
  failIfMajorPerformanceCaveat: false,
};
