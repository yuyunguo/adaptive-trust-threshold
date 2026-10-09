#!/usr/bin/env python3
"""
Extract MIMIC-IV features aligned with MIMIC-III feature schema
for cross-dataset validation of the trust framework.
Produces models/mimic4_features.csv with identical column names.
"""

import sys
import os
import time
import logging
import psycopg2
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

DB_CONFIG = {
    'host': os.getenv('DB_HOST', '192.168.1.249'),
    'port': int(os.getenv('DB_PORT', 5432)),
    'database': os.getenv('DB_NAME', 'mimiciv'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'postgres'),
}

SCHEMA_CORE = 'mimic_core'
SCHEMA_HOSP = 'mimic_hosp'
SCHEMA_ICU = 'mimic_icu'


def extract_mimic4_cohort(n_samples=5000):
    conn = psycopg2.connect(**DB_CONFIG)
    logger.info(f"Connected to MIMIC-IV: {DB_CONFIG['host']}/{DB_CONFIG['database']}")

    query = f"""
    WITH patient_base AS (
        SELECT
            p.subject_id,
            (p.anchor_age + EXTRACT(YEAR FROM a.admittime) - p.anchor_year) AS age,
            CASE WHEN p.gender = 'M' THEN 1 ELSE 0 END AS gender,
            CASE WHEN a.hospital_expire_flag = 1 THEN 1 ELSE 0 END AS mortality,
            i.stay_id,
            a.hadm_id,
            i.intime,
            CASE WHEN a.admission_type IN ('EW EMER.', 'URGENT', 'EMERGENCY') THEN 1 ELSE 0 END AS emergency_admit
        FROM {SCHEMA_CORE}.patients p
        INNER JOIN {SCHEMA_ICU}.icustays i ON p.subject_id = i.subject_id
        INNER JOIN {SCHEMA_CORE}.admissions a ON i.hadm_id = a.hadm_id
        WHERE (p.anchor_age + EXTRACT(YEAR FROM a.admittime) - p.anchor_year) BETWEEN 18 AND 89
        AND EXTRACT(EPOCH FROM (i.outtime - i.intime)) / 3600 > 0.5
        ORDER BY RANDOM()
        LIMIT {n_samples}
    ),

    vital_signs AS (
        SELECT
            pb.stay_id,
            AVG(CASE WHEN ce.itemid = 220045 THEN ce.valuenum END) AS hr_mean,
            MIN(CASE WHEN ce.itemid = 220045 THEN ce.valuenum END) AS hr_min,
            MAX(CASE WHEN ce.itemid = 220045 THEN ce.valuenum END) AS hr_max,
            STDDEV(CASE WHEN ce.itemid = 220045 THEN ce.valuenum END) AS hr_std,
            AVG(CASE WHEN ce.itemid IN (220050, 220179) THEN ce.valuenum END) AS sbp_mean,
            MIN(CASE WHEN ce.itemid IN (220050, 220179) THEN ce.valuenum END) AS sbp_min,
            MAX(CASE WHEN ce.itemid IN (220050, 220179) THEN ce.valuenum END) AS sbp_max,
            AVG(CASE WHEN ce.itemid IN (220051, 220180) THEN ce.valuenum END) AS dbp_mean,
            AVG(CASE WHEN ce.itemid IN (220210, 224690) THEN ce.valuenum END) AS resp_mean,
            AVG(CASE WHEN ce.itemid = 223761 THEN ce.valuenum
                     WHEN ce.itemid = 223762 THEN (ce.valuenum * 9.0/5.0) + 32 END) AS temp_mean,
            AVG(CASE WHEN ce.itemid = 220277 THEN ce.valuenum END) AS spo2_mean
        FROM patient_base pb
        LEFT JOIN {SCHEMA_ICU}.chartevents ce ON pb.stay_id = ce.stay_id
            AND ce.charttime BETWEEN pb.intime AND pb.intime + INTERVAL '24 hours'
            AND ce.valuenum IS NOT NULL
            AND ce.valuenum > 0
        GROUP BY pb.stay_id
    ),

    lab_values AS (
        SELECT
            pb.stay_id,
            AVG(CASE WHEN l.itemid = 50912 THEN l.valuenum END) AS creatinine_mean,
            AVG(CASE WHEN l.itemid = 50971 THEN l.valuenum END) AS potassium_mean,
            AVG(CASE WHEN l.itemid = 50983 THEN l.valuenum END) AS sodium_mean,
            AVG(CASE WHEN l.itemid = 51006 THEN l.valuenum END) AS bun_mean,
            AVG(CASE WHEN l.itemid = 50885 THEN l.valuenum END) AS bilirubin_mean,
            AVG(CASE WHEN l.itemid = 50820 THEN l.valuenum END) AS ph_mean,
            AVG(CASE WHEN l.itemid = 50902 THEN l.valuenum END) AS chloride_mean,
            AVG(CASE WHEN l.itemid = 50931 THEN l.valuenum END) AS glucose_mean,
            AVG(CASE WHEN l.itemid = 51222 THEN l.valuenum END) AS hemoglobin_mean,
            AVG(CASE WHEN l.itemid = 51301 THEN l.valuenum END) AS wbc_mean,
            COUNT(DISTINCT l.itemid) AS unique_lab_tests,
            COUNT(CASE WHEN l.flag = 'abnormal' THEN 1 END) AS abnormal_labs_count
        FROM patient_base pb
        LEFT JOIN {SCHEMA_HOSP}.labevents l ON pb.hadm_id = l.hadm_id
            AND l.charttime BETWEEN pb.intime AND pb.intime + INTERVAL '24 hours'
            AND l.valuenum IS NOT NULL
        GROUP BY pb.stay_id
    ),

    comorbidities AS (
        SELECT
            pb.stay_id,
            COUNT(CASE WHEN d.icd_version = 10 AND d.icd_code LIKE 'E08%' THEN 1 END)
              + COUNT(CASE WHEN d.icd_version = 10 AND d.icd_code LIKE 'E09%' THEN 1 END)
              + COUNT(CASE WHEN d.icd_version = 10 AND d.icd_code LIKE 'E10%' THEN 1 END)
              + COUNT(CASE WHEN d.icd_version = 10 AND d.icd_code LIKE 'E11%' THEN 1 END)
              + COUNT(CASE WHEN d.icd_version = 10 AND d.icd_code LIKE 'E13%' THEN 1 END)
              + COUNT(CASE WHEN d.icd_version = 9 AND d.icd_code LIKE '250%' THEN 1 END)
              AS diabetes,
            COUNT(CASE WHEN (d.icd_version = 10 AND d.icd_code LIKE 'I10%') THEN 1 END)
              + COUNT(CASE WHEN (d.icd_version = 10 AND d.icd_code LIKE 'I11%') THEN 1 END)
              + COUNT(CASE WHEN (d.icd_version = 10 AND d.icd_code LIKE 'I12%') THEN 1 END)
              + COUNT(CASE WHEN (d.icd_version = 10 AND d.icd_code LIKE 'I13%') THEN 1 END)
              + COUNT(CASE WHEN (d.icd_version = 10 AND d.icd_code LIKE 'I15%') THEN 1 END)
              + COUNT(CASE WHEN (d.icd_version = 9 AND (d.icd_code LIKE '401%' OR d.icd_code LIKE '402%')) THEN 1 END)
              AS hypertension,
            COUNT(CASE WHEN (d.icd_version = 10 AND (d.icd_code LIKE 'I21%' OR d.icd_code LIKE 'I22%')) THEN 1 END)
              + COUNT(CASE WHEN (d.icd_version = 10 AND d.icd_code LIKE 'I24%') THEN 1 END)
              + COUNT(CASE WHEN (d.icd_version = 9 AND (d.icd_code LIKE '410%' OR d.icd_code LIKE '411%')) THEN 1 END)
              AS mi,
            COUNT(CASE WHEN (d.icd_version = 10 AND d.icd_code LIKE 'I50%') THEN 1 END)
              + COUNT(CASE WHEN (d.icd_version = 9 AND d.icd_code LIKE '428%') THEN 1 END)
              AS heart_failure,
            COUNT(CASE WHEN (d.icd_version = 10 AND (d.icd_code LIKE 'J40%' OR d.icd_code LIKE 'J41%'
                  OR d.icd_code LIKE 'J42%' OR d.icd_code LIKE 'J43%' OR d.icd_code LIKE 'J44%'
                  OR d.icd_code LIKE 'J45%' OR d.icd_code LIKE 'J46%' OR d.icd_code LIKE 'J47%'
                  OR d.icd_code LIKE 'J96%')) THEN 1 END)
              + COUNT(CASE WHEN (d.icd_version = 9 AND (d.icd_code LIKE '490%' OR d.icd_code LIKE '491%'
                  OR d.icd_code LIKE '492%' OR d.icd_code LIKE '493%' OR d.icd_code LIKE '494%'
                  OR d.icd_code LIKE '495%' OR d.icd_code LIKE '496%'
                  OR d.icd_code LIKE '518%')) THEN 1 END)
              AS respiratory_disease,
            COUNT(DISTINCT d.icd_code) AS unique_diagnoses
        FROM patient_base pb
        LEFT JOIN {SCHEMA_HOSP}.diagnoses_icd d ON pb.hadm_id = d.hadm_id
        GROUP BY pb.stay_id
    ),

    medications AS (
        SELECT
            pb.stay_id,
            COUNT(CASE WHEN LOWER(pr.drug) LIKE '%%insulin%%' THEN 1 END) AS insulin_orders,
            COUNT(CASE WHEN LOWER(pr.drug) LIKE '%%heparin%%' THEN 1 END) AS heparin_orders,
            COUNT(CASE WHEN LOWER(pr.drug) LIKE '%%dopamine%%'
                        OR LOWER(pr.drug) LIKE '%%norepinephrine%%'
                        OR LOWER(pr.drug) LIKE '%%epinephrine%%'
                        OR LOWER(pr.drug) LIKE '%%vasopressin%%' THEN 1 END) AS vasopressor_orders,
            COUNT(DISTINCT pr.drug) AS unique_medications
        FROM patient_base pb
        LEFT JOIN {SCHEMA_HOSP}.prescriptions pr ON pb.hadm_id = pr.hadm_id
        GROUP BY pb.stay_id
    )

    SELECT
        pb.subject_id,
        pb.age,
        pb.gender,
        pb.mortality,
        pb.emergency_admit,
        vs.hr_mean, vs.hr_min, vs.hr_max, vs.hr_std,
        vs.sbp_mean, vs.sbp_min, vs.sbp_max,
        vs.dbp_mean,
        vs.resp_mean,
        vs.temp_mean,
        vs.spo2_mean,
        lv.creatinine_mean,
        lv.potassium_mean,
        lv.sodium_mean,
        lv.bun_mean,
        lv.bilirubin_mean,
        lv.ph_mean,
        lv.chloride_mean,
        lv.glucose_mean,
        lv.hemoglobin_mean,
        lv.wbc_mean,
        lv.unique_lab_tests,
        lv.abnormal_labs_count,
        com.diabetes,
        com.hypertension,
        com.mi,
        com.heart_failure,
        com.respiratory_disease,
        com.unique_diagnoses,
        med.insulin_orders,
        med.heparin_orders,
        med.vasopressor_orders,
        med.unique_medications
    FROM patient_base pb
    LEFT JOIN vital_signs vs ON pb.stay_id = vs.stay_id
    LEFT JOIN lab_values lv ON pb.stay_id = lv.stay_id
    LEFT JOIN comorbidities com ON pb.stay_id = com.stay_id
    LEFT JOIN medications med ON pb.stay_id = med.stay_id
    """

    logger.info(f"Extracting {n_samples} patients from MIMIC-IV...")
    start = time.time()
    df = pd.read_sql(query, conn)
    conn.close()
    elapsed = time.time() - start
    logger.info(f"Raw extraction: {len(df)} patients in {elapsed:.1f}s")

    vital_bounds = {
        'hr_mean': (30, 200), 'hr_min': (20, 200), 'hr_max': (30, 250),
        'sbp_mean': (50, 250), 'sbp_min': (30, 200), 'sbp_max': (50, 300),
        'dbp_mean': (30, 150), 'resp_mean': (5, 50),
        'temp_mean': (90, 110), 'spo2_mean': (70, 100),
    }
    for col, (lo, hi) in vital_bounds.items():
        if col in df.columns:
            mask = (df[col] < lo) | (df[col] > hi)
            n_out = mask.sum()
            if n_out > 0:
                logger.info(f"  Removing {n_out} outliers from {col}")
                df = df[~mask]

    before = len(df)
    df = df.drop_duplicates(subset=['subject_id'])
    if len(df) < before:
        logger.info(f"  Dedup: {before} -> {len(df)}")

    high_missing = (df.isnull().sum() / len(df))
    drop_cols = high_missing[high_missing > 0.95].index.tolist()
    if drop_cols:
        logger.info(f"  Dropping high-missing cols: {drop_cols}")
        df = df.drop(columns=drop_cols)

    logger.info(f"Final cohort: {len(df)} patients, {len(df.columns)} features")
    logger.info(f"Mortality rate: {df['mortality'].mean():.3f}")
    logger.info(f"Age: {df['age'].mean():.1f} +/- {df['age'].std():.1f}")
    logger.info(f"Male %: {df['gender'].mean()*100:.1f}%")

    return df


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=5000)
    parser.add_argument('--output', type=str, default='models/mimic4_features')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("MIMIC-IV FEATURE EXTRACTION FOR CROSS-DATASET VALIDATION")
    logger.info("=" * 60)

    df = extract_mimic4_cohort(n_samples=args.n)

    csv_path = f"{args.output}.csv"
    parquet_path = f"{args.output}.parquet"

    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)

    logger.info(f"Saved to {csv_path} and {parquet_path}")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Missing %:")
    miss = (df.isnull().sum() / len(df) * 100)
    for col in miss[miss > 0].index:
        logger.info(f"  {col}: {miss[col]:.1f}%")


if __name__ == '__main__':
    main()
