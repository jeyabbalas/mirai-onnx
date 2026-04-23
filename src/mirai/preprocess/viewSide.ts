import type { DataSet } from "dicom-parser";
import type { View, Side } from "../types.js";

const VALID_VIEW = new Set(["CC", "MLO", "ML"]);
const VALID_SIDE = new Set(["R", "L"]);

export function readViewSide(ds: DataSet): { view: View; side: Side; viewStr: string; sideStr: string } {
  const viewRaw = ds.string("x00185101");
  if (!viewRaw) throw new Error("ViewPosition (0018,5101) missing");

  let sideStr = ds.string("x00200062");
  let viewStr = viewRaw.toUpperCase();
  if (!sideStr) {
    if (viewStr.includes("RIGHT")) sideStr = "R";
    else if (viewStr.includes("LEFT")) sideStr = "L";
    else throw new Error("ImageLaterality (0020,0062) missing and cannot infer from ViewPosition");
  }
  viewStr = viewStr.replace(/RIGHT/g, "").replace(/LEFT/g, "").trim();

  if (!VALID_VIEW.has(viewStr)) throw new Error(`Invalid ViewPosition: ${viewStr}`);
  if (!VALID_SIDE.has(sideStr)) throw new Error(`Invalid ImageLaterality: ${sideStr}`);

  const view: View = viewStr === "CC" ? 0 : 1;
  const side: Side = sideStr === "R" ? 0 : 1;
  return { view, side, viewStr, sideStr };
}
