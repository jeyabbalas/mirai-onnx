export type RaceCode = 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13;
export type DensityCode = 1 | 2 | 3 | 4;
export type HrtType = "combined" | "estrogen" | "unknown";
export type RelativeCode = "M" | "MA" | "PA" | "MG" | "PG" | "S";

export interface Relative {
  breastCancer?: boolean;
  ovarianCancer?: boolean;
}

export interface HrtInfo {
  type?: HrtType;
  firstAge?: number;
  lastAge?: number;
  duration?: number;
}

export interface MiraiRiskFactors {
  age?: number;
  density?: DensityCode;
  race?: RaceCode;
  priorHist?: boolean;
  weight?: number;
  height?: number;
  menarcheAge?: number;
  menopauseAge?: number;
  firstPregnancyAge?: number;
  numBirths?: number;
  ashkenazi?: boolean;
  brca1?: boolean;
  brca2?: boolean;
  biopsyHyperplasia?: boolean;
  biopsyHyperplasiaAge?: number;
  biopsyLCIS?: boolean;
  biopsyLCISAge?: number;
  biopsyAtypicalHyperplasia?: boolean;
  biopsyAtypicalHyperplasiaAge?: number;
  ovarianCancer?: boolean;
  ovarianCancerAge?: number;
  relatives?: Partial<Record<RelativeCode, Relative[]>>;
  hrt?: HrtInfo;
}

export interface VectorizerResult {
  vector: Float32Array;
  knownMask: Float32Array;
  featureNames: readonly string[];
}

export interface PatientFactors {
  race: number;
  menarche_age: number;
  menopause_age: number;
  first_pregnancy_age: number;
  num_births: number;
  ashkenazi: number;
  brca1: number;
  brca2: number;
  biopsy_hyperplasia: number;
  biopsy_hyperplasia_age: number;
  biopsy_LCIS: number;
  biopsy_LCIS_age: number;
  biopsy_atypical_hyperplasia: number;
  biopsy_atypical_hyperplasia_age: number;
  ovarian_cancer: number;
  ovarian_cancer_age: number;
  combined_hrt: number;
  estrogen_hrt: number;
  unknown_hrt: number;
  combined_hrt_first_age: number;
  combined_hrt_last_age: number;
  combined_hrt_duration: number;
  estrogen_hrt_first_age: number;
  estrogen_hrt_last_age: number;
  estrogen_hrt_duration: number;
  unknown_hrt_first_age: number;
  unknown_hrt_last_age: number;
  unknown_hrt_duration: number;
  relatives: Record<RelativeCode, Relative[]>;
}

export interface ExamFactors {
  age: number;
  density: number;
  prior_hist: number;
  weight: number;
  height: number;
}
