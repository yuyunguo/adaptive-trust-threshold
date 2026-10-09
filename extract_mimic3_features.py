#!/usr/bin/env python3
"""
Extract MIMIC-III features ALIGNED with MIMIC-IV extraction schema.
Fixes applied:
  1. Mortality: hospital_expire_flag (not dod IS NOT NULL) — matches MIMIC-IV
  2. WBC: itemid 51301 (not 51300) — matches MIMIC-IV
  3. Respiratory: ICD-9 490-496 + 518 — matches MIMIC-IV ICD mapping
  4. No admission_type filter — matches MIMIC-IV all-ICU cohort
  5. Temperature: Fahrenheit (itemid 223761, 678) — same unit as MIMIC-IV
  6. Chartevents: valuenum > 0 filter — matches MIMIC-IV
  7. Age cap 18-89 — matches MIMIC-IV
  8. ICU LOS > 0.5h filter — matches MIMIC-IV
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
    'host': os.getenv('MIMIC3_DB_HOST', '192.168.1.207'),
    'port': int(os.getenv('MIMIC3_DB_PORT', 5432)),
    'database': os.getenv('MIMIC3_DB_NAME', 'postgres'),
    'user': os.getenv('MIMIC3_DB_USER', 'postgres'),
    'password': os.getenv('MIMIC3_DB_PASSWORD', 'postgres'),
}


def extract_mimic3_cohort(n_samples=5000):
    conn = psycopg2.connect(**DB_CONFIG)
    logger.info(f"Connected to MIMIC-III: {DB_CONFIG['host']}/{DB_CONFIG['database']}")
    cursor = conn.cursor()

    try:
        cursor.execute("DROP TABLE IF EXISTS _m3_pb")

        logger.info(f"Creating patient base temp table (n={n_samples})...")
        cursor.execute(f"""
        SELECT
            p.subject_id,
            EXTRACT(YEAR FROM AGE(i.intime, p.dob)) AS age,
            CASE WHEN p.gender = 'M' THEN 1 ELSE 0 END AS gender,
            CASE WHEN a.hospital_expire_flag = 1 THEN 1 ELSE 0 END AS mortality,
            i.icustay_id AS stay_id,
            a.hadm_id,
            i.intime,
            CASE WHEN a.admission_type = 'EMERGENCY' THEN 1 ELSE 0 END AS emergency_admit
        INTO TEMP TABLE _m3_pb
        FROM mimiciii.patients p
        INNER JOIN mimiciii.icustays i ON p.subject_id = i.subject_id
        INNER JOIN mimiciii.admissions a ON i.hadm_id = a.hadm_id
        WHERE p.dob IS NOT NULL
        AND EXTRACT(YEAR FROM AGE(i.intime, p.dob)) BETWEEN 18 AND 89
        AND EXTRACT(EPOCH FROM (i.outtime - i.intime)) / 3600 > 0.5
        ORDER BY RANDOM()
        LIMIT {n_samples}
        """)
        logger.info(f"Patient base: {cursor.rowcount} rows")

        logger.info("Extracting vital signs...")
        cursor.execute("""
        SELECT
            pb.stay_id,
            AVG(CASE WHEN ce.itemid IN (211, 220045) THEN ce.valuenum END) AS hr_mean,
            MIN(CASE WHEN ce.itemid IN (211, 220045) THEN ce.valuenum END) AS hr_min,
            MAX(CASE WHEN ce.itemid IN (211, 220045) THEN ce.valuenum END) AS hr_max,
            STDDEV(CASE WHEN ce.itemid IN (211, 220045) THEN ce.valuenum END) AS hr_std,
            AVG(CASE WHEN ce.itemid IN (51, 442, 455, 6701, 220050, 220179) THEN ce.valuenum END) AS sbp_mean,
            MIN(CASE WHEN ce.itemid IN (51, 442, 455, 6701, 220050, 220179) THEN ce.valuenum END) AS sbp_min,
            MAX(CASE WHEN ce.itemid IN (51, 442, 455, 6701, 220050, 220179) THEN ce.valuenum END) AS sbp_max,
            AVG(CASE WHEN ce.itemid IN (8368, 8440, 8441, 8555, 220051, 220180) THEN ce.valuenum END) AS dbp_mean,
            AVG(CASE WHEN ce.itemid IN (618, 615, 220210, 224690) THEN ce.valuenum END) AS resp_mean,
            AVG(CASE WHEN ce.itemid IN (223761, 678) THEN ce.valuenum END) AS temp_mean,
            AVG(CASE WHEN ce.itemid IN (646, 220277) THEN ce.valuenum END) AS spo2_mean
        FROM _m3_pb pb
        LEFT JOIN mimiciii.chartevents ce ON pb.stay_id = ce.icustay_id
            AND ce.charttime BETWEEN pb.intime AND pb.intime + INTERVAL '24 hours'
            AND ce.valuenum IS NOT NULL
            AND ce.valuenum > 0
        GROUP BY pb.stay_id
        """)
        vs_cols = [d[0] for d in cursor.description]
        df_vs = pd.DataFrame(cursor.fetchall(), columns=vs_cols)
        logger.info(f"Vital signs: {len(df_vs)} rows")

        logger.info("Extracting lab values...")
        cursor.execute("""
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
        FROM _m3_pb pb
        LEFT JOIN mimiciii.labevents l ON pb.hadm_id = l.hadm_id
            AND l.charttime BETWEEN pb.intime AND pb.intime + INTERVAL '24 hours'
            AND l.valuenum IS NOT NULL
        GROUP BY pb.stay_id
        """)
        lv_cols = [d[0] for d in cursor.description]
        df_lv = pd.DataFrame(cursor.fetchall(), columns=lv_cols)
        logger.info(f"Lab values: {len(df_lv)} rows")

        logger.info("Extracting comorbidities...")
        cursor.execute("""
        SELECT
            pb.stay_id,
            COUNT(CASE WHEN d.icd9_code LIKE '250%%' THEN 1 END) AS diabetes,
            COUNT(CASE WHEN d.icd9_code LIKE '401%%' OR d.icd9_code LIKE '402%%' THEN 1 END) AS hypertension,
            COUNT(CASE WHEN d.icd9_code LIKE '410%%' OR d.icd9_code LIKE '411%%' THEN 1 END) AS mi,
            COUNT(CASE WHEN d.icd9_code LIKE '428%%' THEN 1 END) AS heart_failure,
            COUNT(CASE WHEN d.icd9_code LIKE '490%%' OR d.icd9_code LIKE '491%%'
                       OR d.icd9_code LIKE '492%%' OR d.icd9_code LIKE '493%%'
                       OR d.icd9_code LIKE '494%%' OR d.icd9_code LIKE '495%%'
                       OR d.icd9_code LIKE '496%%'
                       OR d.icd9_code LIKE '518%%' THEN 1 END) AS respiratory_disease,
            COUNT(DISTINCT d.icd9_code) AS unique_diagnoses
        FROM _m3_pb pb
        LEFT JOIN mimiciii.diagnoses_icd d ON pb.hadm_id = d.hadm_id
        GROUP BY pb.stay_id
        """)
        com_cols = [d[0] for d in cursor.description]
        df_com = pd.DataFrame(cursor.fetchall(), columns=com_cols)
        logger.info(f"Comorbidities: {len(df_com)} rows")

        logger.info("Extracting medications...")
        cursor.execute("""
        SELECT
            pb.stay_id,
            COUNT(CASE WHEN LOWER(pr.drug) LIKE '%%insulin%%' THEN 1 END) AS insulin_orders,
            COUNT(CASE WHEN LOWER(pr.drug) LIKE '%%heparin%%' THEN 1 END) AS heparin_orders,
            COUNT(CASE WHEN LOWER(pr.drug) LIKE '%%dopamine%%'
                        OR LOWER(pr.drug) LIKE '%%norepinephrine%%'
                        OR LOWER(pr.drug) LIKE '%%epinephrine%%'
                        OR LOWER(pr.drug) LIKE '%%vasopressin%%' THEN 1 END) AS vasopressor_orders,
            COUNT(DISTINCT pr.drug) AS unique_medications
        FROM _m3_pb pb
        LEFT JOIN mimiciii.prescriptions pr ON pb.hadm_id = pr.hadm_id
        GROUP BY pb.stay_id
        """)
        med_cols = [d[0] for d in cursor.description]
        df_med = pd.DataFrame(cursor.fetchall(), columns=med_cols)
        logger.info(f"Medications: {len(df_med)} rows")

        logger.info("Extracting patient base features...")
        cursor.execute("""
        SELECT subject_id, age, gender, mortality, emergency_admit, stay_id, hadm_id
        FROM _m3_pb ORDER BY subject_id
        """)
        pb_cols = [d[0] for d in cursor.description]
        df_pb = pd.DataFrame(cursor.fetchall(), columns=pb_cols)
        logger.info(f"Patient base: {len(df_pb)} rows")

        cursor.execute("DROP TABLE IF EXISTS _m3_pb")
        cursor.close()

        logger.info("Merging features...")
        df = df_pb
        for sub in [df_vs, df_lv, df_com, df_med]:
            if not sub.empty:
                df = df.merge(sub, on='stay_id', how='left')
        df = df.drop(columns=['hadm_id'], errors='ignore')

        logger.info(f"Merged: {len(df)} rows, {len(df.columns)} cols")
        for col in df.select_dtypes(include=['object']).columns:
            try:
                converted = pd.to_numeric(df[col], errors='coerce')
                if converted.notna().sum() > 0:
                    df[col] = converted
            except (ValueError, TypeError):
                pass

        return df

    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        try:
            cursor.execute("DROP TABLE IF EXISTS _m3_pb")
        except Exception:
            pass
        cursor.close()
        raise

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
    parser.add_argument('--output', type=str, default='models/comprehensive_features')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("MIMIC-III FEATURE EXTRACTION (ALIGNED WITH MIMIC-IV)")
    logger.info("=" * 60)

    df = extract_mimic3_cohort(n_samples=args.n)

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
