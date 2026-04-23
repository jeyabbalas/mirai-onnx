export const RISK_FACTOR_KEYS = [
  "density",
  "binary_family_history",
  "binary_biopsy_benign",
  "binary_biopsy_LCIS",
  "binary_biopsy_atypical_hyperplasia",
  "age",
  "menarche_age",
  "menopause_age",
  "first_pregnancy_age",
  "prior_hist",
  "race",
  "parous",
  "menopausal_status",
  "weight",
  "height",
  "ovarian_cancer",
  "ovarian_cancer_age",
  "ashkenazi",
  "brca",
  "mom_bc_cancer_history",
  "m_aunt_bc_cancer_history",
  "p_aunt_bc_cancer_history",
  "m_grandmother_bc_cancer_history",
  "p_grantmother_bc_cancer_history",
  "sister_bc_cancer_history",
  "mom_oc_cancer_history",
  "m_aunt_oc_cancer_history",
  "p_aunt_oc_cancer_history",
  "m_grandmother_oc_cancer_history",
  "p_grantmother_oc_cancer_history",
  "sister_oc_cancer_history",
  "hrt_type",
  "hrt_duration",
  "hrt_years_ago_stopped",
] as const;

export type RiskFactorKey = (typeof RISK_FACTOR_KEYS)[number];

export const RF_KEY_TO_NUM_CLASS: Record<RiskFactorKey, number> = {
  density: 4,
  binary_family_history: 1,
  binary_biopsy_benign: 1,
  binary_biopsy_LCIS: 1,
  binary_biopsy_atypical_hyperplasia: 1,
  age: 6,
  menarche_age: 5,
  menopause_age: 5,
  first_pregnancy_age: 6,
  prior_hist: 1,
  race: 13,
  parous: 1,
  menopausal_status: 4,
  weight: 7,
  height: 7,
  ovarian_cancer: 1,
  ovarian_cancer_age: 6,
  ashkenazi: 1,
  brca: 4,
  mom_bc_cancer_history: 1,
  m_aunt_bc_cancer_history: 1,
  p_aunt_bc_cancer_history: 1,
  m_grandmother_bc_cancer_history: 1,
  p_grantmother_bc_cancer_history: 1,
  sister_bc_cancer_history: 1,
  mom_oc_cancer_history: 1,
  m_aunt_oc_cancer_history: 1,
  p_aunt_oc_cancer_history: 1,
  m_grandmother_oc_cancer_history: 1,
  p_grantmother_oc_cancer_history: 1,
  sister_oc_cancer_history: 1,
  hrt_type: 3,
  hrt_duration: 5,
  hrt_years_ago_stopped: 5,
};

export const RF_DIM = 100;

export const RF_KEY_TO_OFFSET: Record<RiskFactorKey, number> = (() => {
  const out = {} as Record<RiskFactorKey, number>;
  let off = 0;
  for (const k of RISK_FACTOR_KEYS) {
    out[k] = off;
    off += RF_KEY_TO_NUM_CLASS[k];
  }
  return out;
})();
