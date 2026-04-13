"""
Generate 4 sample CSV files for testing Aequitas Prime UI.

All files use the Adult Income column schema so they work with ADULT_CONFIG
and the existing preprocessor without any changes.

Files produced in data/samples/:
  biased_gender.csv    -- strong sex bias    (female DI ~0.30, clearly illegal)
  biased_race.csv      -- strong race bias   (non-white DI ~0.35, clearly illegal)
  unbiased_gender.csv  -- balanced by sex    (DI ~0.92+, passes all thresholds)
  unbiased_race.csv    -- balanced by race   (DI ~0.94+, passes all thresholds)

Each file: ~3,000 rows, realistic feature distributions.
"""

import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path("data/samples")
OUT.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)

WORKCLASSES  = ["Private","Self-emp-not-inc","Self-emp-inc","Federal-gov","Local-gov","State-gov"]
EDUCATIONS   = ["Bachelors","Some-college","11th","HS-grad","Prof-school","Assoc-acdm",
                "Assoc-voc","9th","7th-8th","12th","Masters","1st-4th","10th","Doctorate"]
EDU_NUM_MAP  = {"Bachelors":13,"Some-college":10,"11th":7,"HS-grad":9,"Prof-school":15,
                "Assoc-acdm":12,"Assoc-voc":11,"9th":5,"7th-8th":4,"12th":8,
                "Masters":14,"1st-4th":2,"10th":6,"Doctorate":16}
MARITALS     = ["Married-civ-spouse","Divorced","Never-married","Separated","Widowed",
                "Married-spouse-absent"]
OCCUPATIONS  = ["Tech-support","Craft-repair","Other-service","Sales","Exec-managerial",
                "Prof-specialty","Handlers-cleaners","Machine-op-inspct","Adm-clerical",
                "Farming-fishing","Transport-moving","Priv-house-serv","Protective-serv"]
RELATIONSHIPS= ["Wife","Own-child","Husband","Not-in-family","Other-relative","Unmarried"]
COUNTRIES    = ["United-States","Cuba","Jamaica","India","Mexico","South","Japan",
                "China","Philippines","Germany","Vietnam","Taiwan","Iran","England"]
RACES        = ["White","Black","Asian-Pac-Islander","Amer-Indian-Eskimo","Other"]

COLS_ORDER   = ["age","workclass","fnlwgt","education","education-num","marital-status",
                "occupation","relationship","race","sex","capital-gain","capital-loss",
                "hours-per-week","native-country","income"]


def base_features(n):
    age       = RNG.integers(18, 75, n)
    workclass = RNG.choice(WORKCLASSES,  n, p=[0.70,0.07,0.05,0.04,0.07,0.07])
    edu_p = np.array([0.16,0.21,0.04,0.16,0.02,0.06,0.06,0.02,0.02,0.02,0.08,0.01,0.04,0.08])
    edu_p /= edu_p.sum()
    education = RNG.choice(EDUCATIONS, n, p=edu_p)
    edu_num   = np.array([EDU_NUM_MAP[e] for e in education])
    marital   = RNG.choice(MARITALS,     n, p=[0.45,0.14,0.25,0.06,0.06,0.04])
    occ       = RNG.choice(OCCUPATIONS,  n)
    relation  = RNG.choice(RELATIONSHIPS,n, p=[0.10,0.18,0.37,0.20,0.06,0.09])
    fnlwgt    = RNG.integers(12285, 1490400, n)
    cap_gain  = np.where(RNG.random(n) < 0.08, RNG.integers(1000, 99999, n), 0)
    cap_loss  = np.where(RNG.random(n) < 0.05, RNG.integers(200,  4356,  n), 0)
    hours     = np.clip(RNG.normal(40, 12, n).astype(int), 1, 99)
    country   = RNG.choice(COUNTRIES, n, p=[0.905,0.01,0.01,0.01,0.01,
                                             0.01,0.01,0.005,0.005,0.005,
                                             0.005,0.005,0.005,0.005])
    return {
        "age": age, "workclass": workclass, "fnlwgt": fnlwgt,
        "education": education, "education-num": edu_num,
        "marital-status": marital, "occupation": occ, "relationship": relation,
        "capital-gain": cap_gain, "capital-loss": cap_loss,
        "hours-per-week": hours, "native-country": country,
    }


def to_logit(p):
    return np.log(np.clip(p, 1e-6, 1-1e-6) / (1 - np.clip(p, 1e-6, 1-1e-6)))


def base_prob(feat, boost=0.0):
    """Income >50K probability driven purely by legitimate features."""
    edu   = np.array([EDU_NUM_MAP[e] for e in feat["education"]])
    age   = feat["age"]
    hours = feat["hours-per-week"]
    cgain = feat["capital-gain"]
    logit = -6.0 + 0.04*(age-18) + 0.25*(edu-9) + 0.02*(hours-40) + 0.00008*cgain + boost
    return 1 / (1 + np.exp(-logit))


def make_df(feat, sex, race, income):
    df = pd.DataFrame(feat)
    df["sex"]    = sex
    df["race"]   = race
    df["income"] = income
    return df[COLS_ORDER]


# ── 1. BIASED — Gender ────────────────────────────────────────────────────────
def gen_biased_gender(n=3000):
    """Males get strong positive logit boost; females get penalty. DI(sex) ~0.28"""
    feat = base_features(n)
    sex  = RNG.choice(["Male","Female"], n, p=[0.67, 0.33])
    race = RNG.choice(RACES, n, p=[0.86, 0.09, 0.03, 0.01, 0.01])

    p0   = base_prob(feat)
    shift = np.where(sex == "Male", 1.8, -0.8)
    p    = 1 / (1 + np.exp(-(to_logit(p0) + shift)))
    inc  = np.where(RNG.random(n) < p, ">50K", "<=50K")
    return make_df(feat, sex, race, inc)


# ── 2. BIASED — Race ──────────────────────────────────────────────────────────
def gen_biased_race(n=3000):
    """White people get large positive boost; others get penalty. DI(race) ~0.32"""
    feat = base_features(n)
    sex  = RNG.choice(["Male","Female"], n, p=[0.60, 0.40])
    race = RNG.choice(RACES, n, p=[0.55, 0.22, 0.12, 0.06, 0.05])

    p0   = base_prob(feat)
    shift = np.where(race == "White", 2.0, -0.7)
    p    = 1 / (1 + np.exp(-(to_logit(p0) + shift)))
    inc  = np.where(RNG.random(n) < p, ">50K", "<=50K")
    return make_df(feat, sex, race, inc)


# ── 3. UNBIASED — Gender ──────────────────────────────────────────────────────
def gen_unbiased_gender(n=5000):
    """Sex has zero influence on outcome. Higher n + boost ensures DI(sex) stays >0.88."""
    feat = base_features(n)
    sex  = RNG.choice(["Male","Female"], n, p=[0.50, 0.50])
    race = RNG.choice(RACES, n, p=[0.55, 0.20, 0.15, 0.05, 0.05])

    # Higher boost raises overall positive rate (~22%), which dramatically
    # reduces sampling variance so DI stays reliably above 0.88.
    p   = base_prob(feat, boost=1.2)
    inc = np.where(RNG.random(n) < p, ">50K", "<=50K")
    return make_df(feat, sex, race, inc)


# ── 4. UNBIASED — Race ────────────────────────────────────────────────────────
def gen_unbiased_race(n=5000):
    """Race has zero influence on outcome. DI(race) stays reliably >0.90."""
    feat = base_features(n)
    sex  = RNG.choice(["Male","Female"], n, p=[0.55, 0.45])
    race = RNG.choice(RACES, n, p=[0.40, 0.25, 0.20, 0.08, 0.07])

    p   = base_prob(feat, boost=1.2)
    inc = np.where(RNG.random(n) < p, ">50K", "<=50K")
    return make_df(feat, sex, race, inc)


if __name__ == "__main__":
    generators = {
        "biased_gender.csv":   gen_biased_gender,
        "biased_race.csv":     gen_biased_race,
        "unbiased_gender.csv": gen_unbiased_gender,
        "unbiased_race.csv":   gen_unbiased_race,
    }

    print("Generating sample CSV files...\n")
    for fname, fn in generators.items():
        df   = fn()
        path = OUT / fname
        df.to_csv(path, index=False)

        male_pos  = (df[df["sex"]=="Male"]["income"]==">50K").mean()
        fem_pos   = (df[df["sex"]=="Female"]["income"]==">50K").mean()
        white_pos = (df[df["race"]=="White"]["income"]==">50K").mean()
        nonw_pos  = (df[df["race"]!="White"]["income"]==">50K").mean()
        di_sex    = round(fem_pos  / male_pos  if male_pos  > 0 else 0, 3)
        di_race   = round(nonw_pos / white_pos if white_pos > 0 else 0, 3)

        tag_sex  = "BIASED" if di_sex  < 0.80 else "FAIR"
        tag_race = "BIASED" if di_race < 0.80 else "FAIR"

        print(f"  {fname}  ({len(df):,} rows)")
        print(f"    sex  DI = {di_sex:.3f}  [{tag_sex}]   Male: {male_pos:.1%}  Female: {fem_pos:.1%}")
        print(f"    race DI = {di_race:.3f}  [{tag_race}]  White: {white_pos:.1%}  Non-white: {nonw_pos:.1%}")
        print(f"    Saved -> {path}\n")

    print("Done. Upload any of these in the Bias X-Ray tab.")
