export interface PixelmatchOptions { threshold?: number; includeAA?: boolean; alpha?: number; aaColor?: number[]; diffColor?: number[]; diffColorAlt?: number[]; }
declare function pixelmatch(img1: Uint8Array | Buffer, img2: Uint8Array | Buffer, output: Uint8Array | Buffer, width: number, height: number, options?: PixelmatchOptions): number;
export = pixelmatch;
