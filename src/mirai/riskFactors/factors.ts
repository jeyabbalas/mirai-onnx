import type {
  ExamFactors,
  MiraiRiskFactors,
  PatientFactors,
  Relative,
  RelativeCode,
} from "./types.js";
import { coerceInt } from "./missing.js";

const RELATIVE_CODES: readonly RelativeCode[] = ["M", "MA", "PA", "MG", "PG", "S"];

export function buildInternalFactors(input: MiraiRiskFactors): {
  patient: PatientFactors;
  exam: ExamFactors;
} {
  const exam: ExamFactors = {
    age: coerceInt(input.age),
    density: coerceInt(input.density),
    prior_hist: coerceInt(input.priorHist),
    weight: coerceInt(input.weight),
    height: coerceInt(input.height),
  };

  const hrt = input.hrt ?? {};
  const combined = hrt.type === "combined" ? 1 : 0;
  const estrogen = hrt.type === "estrogen" ? 1 : 0;
  const unknown = hrt.type === "unknown" ? 1 : 0;
  const firstAge = coerceInt(hrt.firstAge);
  const lastAge = coerceInt(hrt.lastAge);
  const duration = coerceInt(hrt.duration);

  const rels: Record<RelativeCode, Relative[]> = {} as Record<RelativeCode, Relative[]>;
  const userRelatives = input.relatives ?? {};
  for (const code of RELATIVE_CODES) {
    const list = userRelatives[code] ?? [];
    rels[code] = list.map((r) => ({
      breastCancer: r.breastCancer === true,
      ovarianCancer: r.ovarianCancer === true,
    }));
  }

  const patient: PatientFactors = {
    race: coerceInt(input.race),
    menarche_age: coerceInt(input.menarcheAge),
    menopause_age: coerceInt(input.menopauseAge),
    first_pregnancy_age: coerceInt(input.firstPregnancyAge),
    num_births: coerceInt(input.numBirths),
    ashkenazi: coerceInt(input.ashkenazi),
    brca1: coerceInt(input.brca1),
    brca2: coerceInt(input.brca2),
    biopsy_hyperplasia: coerceInt(input.biopsyHyperplasia),
    biopsy_hyperplasia_age: coerceInt(input.biopsyHyperplasiaAge),
    biopsy_LCIS: coerceInt(input.biopsyLCIS),
    biopsy_LCIS_age: coerceInt(input.biopsyLCISAge),
    biopsy_atypical_hyperplasia: coerceInt(input.biopsyAtypicalHyperplasia),
    biopsy_atypical_hyperplasia_age: coerceInt(input.biopsyAtypicalHyperplasiaAge),
    ovarian_cancer: coerceInt(input.ovarianCancer),
    ovarian_cancer_age: coerceInt(input.ovarianCancerAge),
    combined_hrt: combined,
    estrogen_hrt: estrogen,
    unknown_hrt: unknown,
    combined_hrt_first_age: combined === 1 ? firstAge : -1,
    combined_hrt_last_age: combined === 1 ? lastAge : -1,
    combined_hrt_duration: combined === 1 ? duration : -1,
    estrogen_hrt_first_age: estrogen === 1 ? firstAge : -1,
    estrogen_hrt_last_age: estrogen === 1 ? lastAge : -1,
    estrogen_hrt_duration: estrogen === 1 ? duration : -1,
    unknown_hrt_first_age: unknown === 1 ? firstAge : -1,
    unknown_hrt_last_age: unknown === 1 ? lastAge : -1,
    unknown_hrt_duration: unknown === 1 ? duration : -1,
    relatives: rels,
  };

  return { patient, exam };
}

export function deriveKnownByKey(input: MiraiRiskFactors): Record<string, boolean> {
  const present = (v: unknown): boolean => v !== undefined && v !== null;
  const relatives = input.relatives ?? {};
  const perRelative = (code: RelativeCode) =>
    code in relatives && relatives[code] !== undefined;
  const hrt = input.hrt ?? {};
  const hrtKnown = present(hrt.type);

  const anyRelativeKey = Object.keys(relatives).some(
    (k) => relatives[k as RelativeCode] !== undefined,
  );

  return {
    density: present(input.density),
    binary_family_history: anyRelativeKey,
    binary_biopsy_benign: present(input.biopsyHyperplasia),
    binary_biopsy_LCIS: present(input.biopsyLCIS),
    binary_biopsy_atypical_hyperplasia: present(input.biopsyAtypicalHyperplasia),
    age: present(input.age),
    menarche_age: present(input.menarcheAge),
    menopause_age: present(input.menopauseAge),
    first_pregnancy_age: present(input.firstPregnancyAge),
    prior_hist: present(input.priorHist),
    race: present(input.race),
    parous: present(input.numBirths) || present(input.firstPregnancyAge),
    menopausal_status: present(input.menopauseAge),
    weight: present(input.weight),
    height: present(input.height),
    ovarian_cancer: present(input.ovarianCancer),
    ovarian_cancer_age: present(input.ovarianCancerAge),
    ashkenazi: present(input.ashkenazi),
    brca: present(input.brca1) || present(input.brca2),
    mom_bc_cancer_history: perRelative("M"),
    m_aunt_bc_cancer_history: perRelative("MA"),
    p_aunt_bc_cancer_history: perRelative("PA"),
    m_grandmother_bc_cancer_history: perRelative("MG"),
    p_grantmother_bc_cancer_history: perRelative("PG"),
    sister_bc_cancer_history: perRelative("S"),
    mom_oc_cancer_history: perRelative("M"),
    m_aunt_oc_cancer_history: perRelative("MA"),
    p_aunt_oc_cancer_history: perRelative("PA"),
    m_grandmother_oc_cancer_history: perRelative("MG"),
    p_grantmother_oc_cancer_history: perRelative("PG"),
    sister_oc_cancer_history: perRelative("S"),
    hrt_type: hrtKnown,
    hrt_duration: hrtKnown,
    hrt_years_ago_stopped: hrtKnown,
  };
}
