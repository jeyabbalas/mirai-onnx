import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export const REPO_ROOT = path.resolve(__dirname, "..", "..");
export const FIXTURES_DIR = path.join(REPO_ROOT, "tests", "reference", "fixtures");
export const DEMO_DATA_DIR = path.join(REPO_ROOT, "mirai_demo_data");

export const DEMO_DICOMS: ReadonlyArray<{
  file: string;
  viewStr: "CC" | "MLO";
  sideStr: "L" | "R";
  view: 0 | 1;
  side: 0 | 1;
  label: "CC_L" | "CC_R" | "MLO_L" | "MLO_R";
}> = [
  { file: "ccl1.dcm", viewStr: "CC", sideStr: "L", view: 0, side: 1, label: "CC_L" },
  { file: "ccr1.dcm", viewStr: "CC", sideStr: "R", view: 0, side: 0, label: "CC_R" },
  { file: "mlol2.dcm", viewStr: "MLO", sideStr: "L", view: 1, side: 1, label: "MLO_L" },
  { file: "mlor2.dcm", viewStr: "MLO", sideStr: "R", view: 1, side: 0, label: "MLO_R" },
];
