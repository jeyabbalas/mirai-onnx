import type { MiraiRiskFactors, VectorizerResult } from "./types.js";
import { RISK_FACTOR_KEYS, RF_KEY_TO_NUM_CLASS, RF_KEY_TO_OFFSET, RF_DIM } from "./keys.js";
import { buildInternalFactors, deriveKnownByKey } from "./factors.js";

import { densityInto, densityFeatureNames } from "./transformers/imageBiomarker.js";
import {
  ageInto,
  weightInto,
  heightInto,
  ageFeatureNames,
  weightFeatureNames,
  heightFeatureNames,
} from "./transformers/examOneHot.js";
import {
  menarcheAgeInto,
  menopauseAgeInto,
  firstPregnancyAgeInto,
  ovarianCancerAgeInto,
  menarcheFeatureNames,
  menopauseAgeFeatureNames,
  firstPregnancyFeatureNames,
  ovarianCancerAgeFeatureNames,
} from "./transformers/ageBased.js";
import {
  biopsyBenignInto,
  biopsyLCISInto,
  biopsyAtypicalHyperplasiaInto,
  ovarianCancerInto,
} from "./transformers/binaryOccurrence.js";
import { priorHistInto, ashkenaziInto } from "./transformers/binary.js";
import { raceInto, raceFeatureNames } from "./transformers/race.js";
import { parousInto } from "./transformers/parous.js";
import { menopausalStatusInto, menopausalStatusFeatureNames } from "./transformers/menopausalStatus.js";
import { brcaInto, brcaFeatureNames } from "./transformers/brca.js";
import { binaryFamilyHistoryInto } from "./transformers/familyHistory.js";
import { relativeCancerHistoryInto } from "./transformers/relative.js";
import {
  deriveHrtState,
  hrtTypeInto,
  hrtDurationInto,
  hrtYearsAgoStoppedInto,
  hrtTypeFeatureNames,
  hrtDurationFeatureNames,
  hrtYearsAgoStoppedFeatureNames,
} from "./transformers/hrt.js";

// Feature names in the exact order Python's get_feature_names() produces.
function buildFeatureNames(): string[] {
  const names: string[] = [];
  names.push(...densityFeatureNames());
  names.push("binary_family_history");
  names.push("binary_biopsy_hyperplasia");
  names.push("binary_biopsy_LCIS");
  names.push("binary_biopsy_atypical_hyperplasia");
  names.push(...ageFeatureNames());
  names.push(...menarcheFeatureNames());
  names.push(...menopauseAgeFeatureNames());
  names.push(...firstPregnancyFeatureNames());
  names.push("binary_prior_hist");
  names.push(...raceFeatureNames());
  names.push("parous");
  names.push(...menopausalStatusFeatureNames());
  names.push(...weightFeatureNames());
  names.push(...heightFeatureNames());
  names.push("binary_ovarian_cancer");
  names.push(...ovarianCancerAgeFeatureNames());
  names.push("binary_ashkenazi");
  names.push(...brcaFeatureNames());
  names.push("M_breast_cancer_hist");
  names.push("MA_breast_cancer_hist");
  names.push("PA_breast_cancer_hist");
  names.push("MG_breast_cancer_hist");
  names.push("PG_breast_cancer_hist");
  names.push("S_breast_cancer_hist");
  names.push("M_ovarian_cancer_hist");
  names.push("MA_ovarian_cancer_hist");
  names.push("PA_ovarian_cancer_hist");
  names.push("MG_ovarian_cancer_hist");
  names.push("PG_ovarian_cancer_hist");
  names.push("S_ovarian_cancer_hist");
  names.push(...hrtTypeFeatureNames());
  names.push(...hrtDurationFeatureNames());
  names.push(...hrtYearsAgoStoppedFeatureNames());
  return names;
}

export const FEATURE_NAMES: readonly string[] = Object.freeze(buildFeatureNames());

export function vectorizeRiskFactors(input: MiraiRiskFactors = {}): VectorizerResult {
  const { patient, exam } = buildInternalFactors(input);
  const known = deriveKnownByKey(input);

  const vector = new Float32Array(RF_DIM);
  const knownMask = new Float32Array(RF_DIM);

  const hrtState = known.hrt_type ? deriveHrtState(patient, exam) : null;

  for (const key of RISK_FACTOR_KEYS) {
    if (!known[key]) continue;
    const off = RF_KEY_TO_OFFSET[key];
    const width = RF_KEY_TO_NUM_CLASS[key];

    switch (key) {
      case "density":
        densityInto(exam, vector, off);
        break;
      case "binary_family_history":
        binaryFamilyHistoryInto(patient, vector, off);
        break;
      case "binary_biopsy_benign":
        biopsyBenignInto(patient, exam, vector, off);
        break;
      case "binary_biopsy_LCIS":
        biopsyLCISInto(patient, exam, vector, off);
        break;
      case "binary_biopsy_atypical_hyperplasia":
        biopsyAtypicalHyperplasiaInto(patient, exam, vector, off);
        break;
      case "age":
        ageInto(exam, vector, off);
        break;
      case "menarche_age":
        menarcheAgeInto(patient, exam, vector, off);
        break;
      case "menopause_age":
        menopauseAgeInto(patient, exam, vector, off);
        break;
      case "first_pregnancy_age":
        firstPregnancyAgeInto(patient, exam, vector, off);
        break;
      case "prior_hist":
        priorHistInto(patient, exam, vector, off);
        break;
      case "race":
        raceInto(patient, vector, off);
        break;
      case "parous":
        parousInto(patient, exam, vector, off);
        break;
      case "menopausal_status":
        menopausalStatusInto(patient, exam, vector, off);
        break;
      case "weight":
        weightInto(exam, vector, off);
        break;
      case "height":
        heightInto(exam, vector, off);
        break;
      case "ovarian_cancer":
        ovarianCancerInto(patient, exam, vector, off);
        break;
      case "ovarian_cancer_age":
        ovarianCancerAgeInto(patient, exam, vector, off);
        break;
      case "ashkenazi":
        ashkenaziInto(patient, exam, vector, off);
        break;
      case "brca":
        brcaInto(patient, vector, off);
        break;
      case "mom_bc_cancer_history":
        relativeCancerHistoryInto(patient, "M", "breastCancer", vector, off);
        break;
      case "m_aunt_bc_cancer_history":
        relativeCancerHistoryInto(patient, "MA", "breastCancer", vector, off);
        break;
      case "p_aunt_bc_cancer_history":
        relativeCancerHistoryInto(patient, "PA", "breastCancer", vector, off);
        break;
      case "m_grandmother_bc_cancer_history":
        relativeCancerHistoryInto(patient, "MG", "breastCancer", vector, off);
        break;
      case "p_grantmother_bc_cancer_history":
        relativeCancerHistoryInto(patient, "PG", "breastCancer", vector, off);
        break;
      case "sister_bc_cancer_history":
        relativeCancerHistoryInto(patient, "S", "breastCancer", vector, off);
        break;
      case "mom_oc_cancer_history":
        relativeCancerHistoryInto(patient, "M", "ovarianCancer", vector, off);
        break;
      case "m_aunt_oc_cancer_history":
        relativeCancerHistoryInto(patient, "MA", "ovarianCancer", vector, off);
        break;
      case "p_aunt_oc_cancer_history":
        relativeCancerHistoryInto(patient, "PA", "ovarianCancer", vector, off);
        break;
      case "m_grandmother_oc_cancer_history":
        relativeCancerHistoryInto(patient, "MG", "ovarianCancer", vector, off);
        break;
      case "p_grantmother_oc_cancer_history":
        relativeCancerHistoryInto(patient, "PG", "ovarianCancer", vector, off);
        break;
      case "sister_oc_cancer_history":
        relativeCancerHistoryInto(patient, "S", "ovarianCancer", vector, off);
        break;
      case "hrt_type":
        if (hrtState) hrtTypeInto(hrtState, vector, off);
        break;
      case "hrt_duration":
        if (hrtState) hrtDurationInto(hrtState, vector, off);
        break;
      case "hrt_years_ago_stopped":
        if (hrtState) hrtYearsAgoStoppedInto(hrtState, vector, off);
        break;
    }

    for (let i = off; i < off + width; i++) knownMask[i] = 1;
  }

  return { vector, knownMask, featureNames: FEATURE_NAMES };
}
