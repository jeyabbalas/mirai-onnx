import fs from "node:fs";

export type NpyDtype = "<u2" | "<i4" | "<f4" | "<f8";

export interface NpyArray {
  dtype: NpyDtype;
  shape: number[];
  fortranOrder: boolean;
  data: Uint16Array | Int32Array | Float32Array | Float64Array;
}

const MAGIC = Uint8Array.of(0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59);

function assertMagic(buf: Buffer): void {
  if (buf.length < MAGIC.length) throw new Error("npy: file too short");
  for (let i = 0; i < MAGIC.length; i++) {
    if (buf[i] !== MAGIC[i]) throw new Error("npy: bad magic");
  }
}

function parseHeaderDict(dict: string): { dtype: NpyDtype; shape: number[]; fortranOrder: boolean } {
  const descrMatch = dict.match(/'descr':\s*'([^']+)'/);
  if (!descrMatch) throw new Error(`npy: no descr in header: ${dict}`);
  const dtype = descrMatch[1] as NpyDtype;
  if (!["<u2", "<i4", "<f4", "<f8"].includes(dtype)) {
    throw new Error(`npy: unsupported dtype '${dtype}'`);
  }

  const foMatch = dict.match(/'fortran_order':\s*(True|False)/);
  if (!foMatch) throw new Error(`npy: no fortran_order in header`);
  const fortranOrder = foMatch[1] === "True";

  const shapeMatch = dict.match(/'shape':\s*\(([^)]*)\)/);
  if (!shapeMatch) throw new Error(`npy: no shape in header`);
  const shape = shapeMatch[1]
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s.length > 0)
    .map((s) => Number.parseInt(s, 10));

  return { dtype, shape, fortranOrder };
}

export function parseNpy(buf: Buffer): NpyArray {
  assertMagic(buf);
  const major = buf[6];
  const minor = buf[7];
  let headerLen: number;
  let dataStart: number;
  if (major === 1) {
    headerLen = buf.readUInt16LE(8);
    dataStart = 10 + headerLen;
  } else if (major === 2 || major === 3) {
    headerLen = buf.readUInt32LE(8);
    dataStart = 12 + headerLen;
  } else {
    throw new Error(`npy: unsupported version ${major}.${minor}`);
  }

  const headerBytes = buf.subarray(dataStart - headerLen, dataStart);
  const dict = headerBytes.toString("utf8").replace(/\s+$/, "");
  const { dtype, shape, fortranOrder } = parseHeaderDict(dict);

  const total = shape.reduce((a, b) => a * b, 1);
  const dataBuf = buf.subarray(dataStart);
  const elemBytes: Record<NpyDtype, number> = { "<u2": 2, "<i4": 4, "<f4": 4, "<f8": 8 };
  const expectedBytes = total * elemBytes[dtype];
  if (dataBuf.length < expectedBytes) {
    throw new Error(
      `npy: data underrun — expected ${expectedBytes} bytes, got ${dataBuf.length}`,
    );
  }

  // Slice out exactly the bytes we need, then re-view on a detached buffer.
  // Buffer.subarray shares storage with the parent Buffer (which is in a
  // larger pool); typed arrays require alignment, so we copy into a fresh
  // ArrayBuffer instead of attempting to view-in-place.
  const ab = new ArrayBuffer(expectedBytes);
  new Uint8Array(ab).set(dataBuf.subarray(0, expectedBytes));

  let data: NpyArray["data"];
  switch (dtype) {
    case "<u2":
      data = new Uint16Array(ab);
      break;
    case "<i4":
      data = new Int32Array(ab);
      break;
    case "<f4":
      data = new Float32Array(ab);
      break;
    case "<f8":
      data = new Float64Array(ab);
      break;
  }

  return { dtype, shape, fortranOrder, data };
}

export function loadNpy(path: string): NpyArray {
  return parseNpy(fs.readFileSync(path));
}
