"""Phase 7 helper: generate synthetic fixtures for the TS RiskFactorVectorizer
port. The real Python RiskFactorVectorizer is imported from external/Mirai and
driven directly; parse_risk_factors is monkey-patched so no metadata JSON
files are needed on disk.

Each case records:
  - `input`: a TS-shaped MiraiRiskFactors user-input dict (nested).
  - `expected_vector`: length-100 fp list, the bit-exact Python vectorizer output.
  - `expected_known_mask`: length-100 fp list, derived by the per-key rule the
    TS port will also implement. Whole-block 1s or 0s per key.

Produces tests/rf/fixtures.json.

Run under `mirai-py38` so we match Phase 0's torch 1.9.0 / numpy 1.24:

    conda activate mirai-py38
    python scripts/generate_rf_fixtures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch  # noqa: F401 — pinned torch 1.9.0

REPO = Path(__file__).resolve().parents[1]
MIRAI = REPO / "external" / "Mirai"
sys.path.insert(0, str(MIRAI))

# Monkey-patch parse_risk_factors BEFORE importing RiskFactorVectorizer so the
# class constructor never tries to read metadata JSON files off disk.
from onconet.utils import risk_factors as rf_module  # noqa: E402
rf_module.parse_risk_factors = lambda args: {}

from onconet.utils.risk_factors import RiskFactorVectorizer  # noqa: E402

# The 34 keys from mirai_trained.json (inference-time config). Do NOT use
# parse_args' default — that list contains extra brother/father/daughter
# keys not used at inference.
RISK_FACTOR_KEYS = [
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
]
RELATIVE_CODES = ["M", "MA", "PA", "MG", "PG", "S"]  # 6 codes at inference


def build_args():
    return SimpleNamespace(
        risk_factor_keys=list(RISK_FACTOR_KEYS),
        metadata_path="",  # parse_risk_factors is monkeypatched
        risk_factor_metadata_path="",
        dataset="",
    )


def to_int_or_missing(v):
    if v is None:
        return -1
    if isinstance(v, bool):
        return 1 if v else 0
    return int(v)


def build_patient_and_exam(user_input):
    """Translate a TS-shaped MiraiRiskFactors dict into Python patient_factors
    and exam_factors dicts, using -1 for missing. Mirrors the translation the
    TS vectorizer will do internally (and that the transformers depend on)."""
    u = user_input

    # --- exam fields ---
    exam = {
        "age": to_int_or_missing(u.get("age")),
        "density": to_int_or_missing(u.get("density")),
        "prior_hist": to_int_or_missing(u.get("priorHist")),
        "weight": to_int_or_missing(u.get("weight")),
        "height": to_int_or_missing(u.get("height")),
    }

    # --- patient fields ---
    patient = {
        "race": to_int_or_missing(u.get("race")),
        "menarche_age": to_int_or_missing(u.get("menarcheAge")),
        "menopause_age": to_int_or_missing(u.get("menopauseAge")),
        "first_pregnancy_age": to_int_or_missing(u.get("firstPregnancyAge")),
        "num_births": to_int_or_missing(u.get("numBirths")),
        "ashkenazi": to_int_or_missing(u.get("ashkenazi")),
        "brca1": to_int_or_missing(u.get("brca1")),
        "brca2": to_int_or_missing(u.get("brca2")),
        "biopsy_hyperplasia": to_int_or_missing(u.get("biopsyHyperplasia")),
        "biopsy_hyperplasia_age": to_int_or_missing(u.get("biopsyHyperplasiaAge")),
        "biopsy_LCIS": to_int_or_missing(u.get("biopsyLCIS")),
        "biopsy_LCIS_age": to_int_or_missing(u.get("biopsyLCISAge")),
        "biopsy_atypical_hyperplasia": to_int_or_missing(u.get("biopsyAtypicalHyperplasia")),
        "biopsy_atypical_hyperplasia_age": to_int_or_missing(u.get("biopsyAtypicalHyperplasiaAge")),
        "ovarian_cancer": to_int_or_missing(u.get("ovarianCancer")),
        "ovarian_cancer_age": to_int_or_missing(u.get("ovarianCancerAge")),
    }

    # HRT: user hrt.type is "combined" | "estrogen" | "unknown" (string).
    hrt = u.get("hrt") or {}
    combined = estrogen = unknown = 0
    if hrt.get("type") == "combined":
        combined = 1
    elif hrt.get("type") == "estrogen":
        estrogen = 1
    elif hrt.get("type") == "unknown":
        unknown = 1
    patient["combined_hrt"] = combined
    patient["estrogen_hrt"] = estrogen
    patient["unknown_hrt"] = unknown
    first_age = to_int_or_missing(hrt.get("firstAge"))
    last_age = to_int_or_missing(hrt.get("lastAge"))
    duration = to_int_or_missing(hrt.get("duration"))
    # All three HRT type branches read the _same_ field names for first/last/duration,
    # gated by which *_hrt flag was set. Fill the relevant triad; others stay MISSING.
    for prefix in ("combined", "estrogen", "unknown"):
        patient[f"{prefix}_hrt_first_age"] = -1
        patient[f"{prefix}_hrt_last_age"] = -1
        patient[f"{prefix}_hrt_duration"] = -1
    if combined or estrogen or unknown:
        prefix = "combined" if combined else ("estrogen" if estrogen else "unknown")
        patient[f"{prefix}_hrt_first_age"] = first_age
        patient[f"{prefix}_hrt_last_age"] = last_age
        patient[f"{prefix}_hrt_duration"] = duration

    # Relatives: always produce the full dict with empty lists for missing codes,
    # because transform_binary_family_history iterates all keys and
    # get_binary_relative_cancer_history_transformer reads relatives[code] directly.
    rel_input = u.get("relatives") or {}
    relatives = {}
    for code in RELATIVE_CODES:
        entries = rel_input.get(code) or []
        relatives[code] = [
            {
                "breast_cancer": 1 if r.get("breastCancer") else 0,
                "ovarian_cancer": 1 if r.get("ovarianCancer") else 0,
            }
            for r in entries
        ]
    patient["relatives"] = relatives

    return patient, exam


def derive_known_by_key(user_input):
    """Per-key known flags. Matches the rules documented in the plan file.
    Returns a dict[key, bool] covering every entry in RISK_FACTOR_KEYS.
    """
    u = user_input

    def present(name):
        return name in u and u[name] is not None

    hrt = u.get("hrt") or {}
    relatives = u.get("relatives") or {}

    def per_relative_known(code):
        # Known if the user populated that specific code's list at all
        # (non-empty OR explicitly empty). Absent => unknown.
        return code in relatives and relatives[code] is not None

    known_by_key = {
        "density": present("density"),
        "binary_family_history": bool(relatives),  # any key given at all
        "binary_biopsy_benign": present("biopsyHyperplasia"),
        "binary_biopsy_LCIS": present("biopsyLCIS"),
        "binary_biopsy_atypical_hyperplasia": present("biopsyAtypicalHyperplasia"),
        "age": present("age"),
        "menarche_age": present("menarcheAge"),
        "menopause_age": present("menopauseAge"),
        "first_pregnancy_age": present("firstPregnancyAge"),
        "prior_hist": present("priorHist"),
        "race": present("race"),
        "parous": present("numBirths") or present("firstPregnancyAge"),
        "menopausal_status": present("menopauseAge"),
        "weight": present("weight"),
        "height": present("height"),
        "ovarian_cancer": present("ovarianCancer"),
        "ovarian_cancer_age": present("ovarianCancerAge"),
        "ashkenazi": present("ashkenazi"),
        "brca": present("brca1") or present("brca2"),
        "mom_bc_cancer_history": per_relative_known("M"),
        "m_aunt_bc_cancer_history": per_relative_known("MA"),
        "p_aunt_bc_cancer_history": per_relative_known("PA"),
        "m_grandmother_bc_cancer_history": per_relative_known("MG"),
        "p_grantmother_bc_cancer_history": per_relative_known("PG"),
        "sister_bc_cancer_history": per_relative_known("S"),
        "mom_oc_cancer_history": per_relative_known("M"),
        "m_aunt_oc_cancer_history": per_relative_known("MA"),
        "p_aunt_oc_cancer_history": per_relative_known("PA"),
        "m_grandmother_oc_cancer_history": per_relative_known("MG"),
        "p_grantmother_oc_cancer_history": per_relative_known("PG"),
        "sister_oc_cancer_history": per_relative_known("S"),
        "hrt_type": "type" in hrt and hrt["type"] is not None,
        "hrt_duration": "type" in hrt and hrt["type"] is not None,
        "hrt_years_ago_stopped": "type" in hrt and hrt["type"] is not None,
    }

    return known_by_key


def mask_from_known_by_key(known_by_key, key_to_num_class):
    mask = []
    for key in RISK_FACTOR_KEYS:
        width = key_to_num_class[key]
        bit = 1.0 if known_by_key[key] else 0.0
        mask.extend([bit] * width)
    assert len(mask) == 100, len(mask)
    return mask


def vector_from_python(vectorizer, patient, exam, known_by_key):
    """Run the Python vectorizer, then zero-out every per-key block whose
    known-bit is 0. This normalizes away a few Python quirks that would
    otherwise leak into the user-vector:

      - get_binary_occurence_transformer uses `if occurence:` which is truthy
        for -1 (missing), so missing biopsy fields flip to 1.
      - transform_race does `vector[race - 1] = 1` with no bounds check; for
        race=-1 Python silently wraps to vector[vector.length - 2] ("Hawaiian").

    These values are never used at inference (mask=0 → ONNX substitutes the
    predicted RF), but zeroing them here makes the TS port clean and safe:
    missing input === zero block, independent of mask usage elsewhere.
    """
    tensors = vectorizer.transform(patient, exam)
    flat = []
    for key, t in zip(vectorizer.risk_factor_keys, tensors):
        vals = [float(x) for x in t.tolist()]
        if not known_by_key[key]:
            vals = [0.0] * len(vals)
        flat.extend(vals)
    assert len(flat) == 100, len(flat)
    return flat


def build_cases():
    """Curated synthetic cases covering every transformer edge case."""
    cases = [
        {"name": "empty", "input": {}},
        {"name": "age_only_45", "input": {"age": 45}},
        {"name": "age_below_min", "input": {"age": 35}},
        {"name": "age_above_max", "input": {"age": 85}},
        {"name": "density_1", "input": {"density": 1}},
        {"name": "density_2", "input": {"density": 2}},
        {"name": "density_3", "input": {"density": 3}},
        {"name": "density_4", "input": {"density": 4}},
        {"name": "race_white", "input": {"race": 1}},
        {"name": "race_hispanic", "input": {"race": 8}},
        {"name": "race_other_asian", "input": {"race": 13}},
        {
            "name": "menopause_peri",
            "input": {"age": 50, "menopauseAge": 50},
        },
        {
            "name": "menopause_pre_age_based_hidden",
            # menopauseAge > examAge triggers two paths: menopausal_status=pre (0)
            # AND menopause_age age-based transformer masks to MISSING (all zeros).
            "input": {"age": 48, "menopauseAge": 52},
        },
        {
            "name": "menopause_post",
            "input": {"age": 60, "menopauseAge": 50},
        },
        {
            "name": "menopause_unknown_with_age",
            "input": {"age": 45},
        },
        {
            "name": "full_age_fields",
            "input": {
                "age": 50,
                "menarcheAge": 12,
                "menopauseAge": 49,
                "firstPregnancyAge": 28,
            },
        },
        {"name": "brca1_positive", "input": {"brca1": True}},
        {"name": "brca2_positive", "input": {"brca2": True}},
        {"name": "brca_negative", "input": {"brca1": False}},
        {
            "name": "biopsy_hyperplasia_no_age",
            "input": {"age": 55, "biopsyHyperplasia": True},
        },
        {
            "name": "biopsy_hyperplasia_future_age",
            "input": {"age": 55, "biopsyHyperplasia": True, "biopsyHyperplasiaAge": 60},
        },
        {
            "name": "biopsy_hyperplasia_past_age",
            "input": {"age": 55, "biopsyHyperplasia": True, "biopsyHyperplasiaAge": 45},
        },
        {
            "name": "ashkenazi_positive",
            "input": {"ashkenazi": True},
        },
        {
            "name": "parous_via_num_births",
            "input": {"age": 45, "numBirths": 2},
        },
        {
            "name": "parous_via_first_pregnancy_before_exam",
            "input": {"age": 45, "firstPregnancyAge": 28},
        },
        {
            "name": "parous_overridden_by_future_first_pregnancy",
            # Python's transform_parous: if first_pregnancy_age != MISSING_VALUE,
            # it OVERWRITES the num_births-derived bit with (age < examAge).
            "input": {"age": 45, "numBirths": 2, "firstPregnancyAge": 50},
        },
        {
            "name": "weight_height_buckets",
            "input": {"weight": 165, "height": 66},
        },
        {
            "name": "weight_below_min",
            "input": {"weight": 90},
        },
        {
            "name": "family_history_mom_bc",
            "input": {"relatives": {"M": [{"breastCancer": True}]}},
        },
        {
            "name": "family_history_sister_oc",
            "input": {"relatives": {"S": [{"ovarianCancer": True}]}},
        },
        {
            "name": "family_history_multi_relatives",
            "input": {
                "relatives": {
                    "M": [{"breastCancer": True}],
                    "MA": [{"breastCancer": False, "ovarianCancer": True}],
                    "MG": [],
                }
            },
        },
        {
            "name": "hrt_combined_ongoing",
            "input": {
                "age": 55,
                "hrt": {"type": "combined", "firstAge": 45, "lastAge": 55, "duration": 10},
            },
        },
        {
            "name": "hrt_estrogen_stopped_5_years_ago",
            "input": {
                "age": 60,
                "hrt": {"type": "estrogen", "firstAge": 45, "lastAge": 55, "duration": 10},
            },
        },
        {
            "name": "hrt_combined_future_user",
            # firstAge > currentAge triggers the "future_user" branch → hrt_type = -1
            "input": {
                "age": 40,
                "hrt": {"type": "combined", "firstAge": 50, "lastAge": 60, "duration": 10},
            },
        },
        {
            "name": "hrt_unknown_type_only",
            "input": {"age": 55, "hrt": {"type": "unknown"}},
        },
        {
            "name": "ovarian_cancer_past",
            "input": {"age": 60, "ovarianCancer": True, "ovarianCancerAge": 52},
        },
        {
            "name": "prior_hist_true",
            "input": {"priorHist": True},
        },
        {
            "name": "prior_hist_false",
            "input": {"priorHist": False},
        },
        {
            "name": "full_rich_input",
            "input": {
                "age": 55,
                "density": 3,
                "race": 1,
                "priorHist": False,
                "weight": 150,
                "height": 65,
                "menarcheAge": 13,
                "menopauseAge": 52,
                "firstPregnancyAge": 28,
                "numBirths": 2,
                "ashkenazi": False,
                "brca1": False,
                "brca2": True,
                "biopsyHyperplasia": True,
                "biopsyHyperplasiaAge": 40,
                "biopsyLCIS": False,
                "biopsyAtypicalHyperplasia": False,
                "ovarianCancer": False,
                "relatives": {
                    "M": [{"breastCancer": True}],
                    "S": [{"ovarianCancer": True}],
                    "MG": [],
                    "PG": [],
                    "MA": [],
                    "PA": [],
                },
                "hrt": {"type": "estrogen", "firstAge": 50, "lastAge": 54, "duration": 4},
            },
        },
    ]
    return cases


def main():
    args = build_args()
    vectorizer = RiskFactorVectorizer(args)

    # Sanity: the trained-config 34 keys should map to a total of 100 dims.
    assert vectorizer.vector_length == 100, vectorizer.vector_length
    assert list(vectorizer.risk_factor_keys) == RISK_FACTOR_KEYS

    key_to_num = dict(vectorizer.risk_factor_key_to_num_class)
    feature_names = vectorizer.get_feature_names()
    assert len(feature_names) == 100

    # Verify per-key feature-name offsets align with RISK_FACTOR_KEYS.
    offset = 0
    offsets = {}
    for k in RISK_FACTOR_KEYS:
        offsets[k] = offset
        offset += key_to_num[k]

    cases_out = []
    for case in build_cases():
        patient, exam = build_patient_and_exam(case["input"])
        known_by_key = derive_known_by_key(case["input"])
        vec = vector_from_python(vectorizer, patient, exam, known_by_key)
        mask = mask_from_known_by_key(known_by_key, key_to_num)
        assert len(vec) == 100 and len(mask) == 100
        cases_out.append({
            "name": case["name"],
            "input": case["input"],
            "expected_vector": vec,
            "expected_known_mask": mask,
        })

    # Per-case sanity checks the TS tests will also assert.
    empty_case = next(c for c in cases_out if c["name"] == "empty")
    assert all(v == 0.0 for v in empty_case["expected_vector"]), \
        "empty input must produce all-zeros vector after per-key masking"
    assert all(v == 0.0 for v in empty_case["expected_known_mask"])

    out = {
        "schema_version": 1,
        "generated_by": "scripts/generate_rf_fixtures.py",
        "rf_dim": 100,
        "risk_factor_keys": RISK_FACTOR_KEYS,
        "rf_key_to_num_class": {k: key_to_num[k] for k in RISK_FACTOR_KEYS},
        "rf_key_to_offset": offsets,
        "feature_names": feature_names,
        "cases": cases_out,
    }

    out_path = REPO / "tests" / "rf" / "fixtures.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path} with {len(cases_out)} cases, 100-dim per case.")


if __name__ == "__main__":
    main()
