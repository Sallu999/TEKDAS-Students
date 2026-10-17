"""Prepare the Student Academic Success and Dropout dataset.

Flow for students:
    dataset_StudentDOA.xlsx -> validation -> cleaning/normalization -> train/test split -> CSV outputs

Run:
    python data_prep.py
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset_StudentDOA.xlsx"

TARGET = "Target"
ID_COLUMN = "student_id"

CATEGORICAL_COLUMNS = [
    "Marital status",
    "Application mode",
    "Application order",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
]

NUMERIC_COLUMNS = [
    "Age at enrollment",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
]

FEATURE_COLUMNS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS
EXPECTED_COLUMNS = set(FEATURE_COLUMNS + [TARGET])

COURSE_MAP = {
    1: "Biofuel Production Technologies",
    2: "Animation and Multimedia Design",
    3: "Social Service (Evening)",
    4: "Agronomy",
    5: "Communication Design",
    6: "Veterinary Nursing",
    7: "Informatics Engineering",
    8: "Equiniculture",
    9: "Management",
    10: "Social Service",
    11: "Tourism",
    12: "Nursing",
    13: "Oral Hygiene",
    14: "Advertising and Marketing",
    15: "Journalism and Communication",
    16: "Basic Education",
    17: "Management (Evening)",
}


def load_source_data(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Memuat data mentah dari dataset_StudentDOA.xlsx."""
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {data_path}")

    excel_file = pd.ExcelFile(data_path, engine="openpyxl")
    candidate_sheets = ["dataset (1)", "dataset", "Sheet1"]
    sheet = next((s for s in candidate_sheets if s in excel_file.sheet_names), excel_file.sheet_names[0])
    df = pd.read_excel(excel_file, sheet_name=sheet)

    missing = sorted(EXPECTED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan pada file Excel: {missing}")

    return df


def clean_and_prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Melakukan sanitasi format desimal dan membuat ID mahasiswa."""
    data = df.dropna(subset=[TARGET]).copy()
    data[TARGET] = data[TARGET].astype(str).str.strip()

    # Tambahkan student_id unik jika belum ada
    if ID_COLUMN not in data.columns:
        data.insert(0, ID_COLUMN, [f"MHS-{i+1:04d}" for i in range(len(data))])

    # Konversi kolom numerik
    for col in NUMERIC_COLUMNS:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # Normalisasi nilai desimal akibat ekspor Excel
    for grade_col in ["Curricular units 1st sem (grade)", "Curricular units 2nd sem (grade)"]:
        def fix_grade(v):
            if pd.isna(v) or v <= 20:
                return v
            s = str(v).replace(".", "")
            if len(s) >= 2 and 10 <= int(s[:2]) <= 20:
                return float(s[:2] + "." + s[2:])
            return v / 10.0
        data[grade_col] = data[grade_col].apply(fix_grade)

    if (data["Unemployment rate"] > 30).any():
        data["Unemployment rate"] = data["Unemployment rate"] / 10.0

    if (data["Inflation rate"].abs() > 20).any():
        data["Inflation rate"] = data["Inflation rate"] / 10.0

    if (data["GDP"].abs() > 10).any():
        data["GDP"] = data["GDP"] / 100.0

    data["course_name"] = data["Course"].map(lambda c: COURSE_MAP.get(int(c), f"Jurusan {c}"))
    return data[[ID_COLUMN, "course_name"] + FEATURE_COLUMNS + [TARGET]]


def build_data_quality_report(data: pd.DataFrame) -> dict:
    """Menghasilkan laporan kualitas data untuk audit pembelajaran."""
    target_counts = data[TARGET].value_counts().to_dict()
    target_rates = (data[TARGET].value_counts(normalize=True) * 100).round(2).to_dict()

    return {
        "total_rows": int(len(data)),
        "total_columns": int(len(data.columns)),
        "target_distribution_counts": target_counts,
        "target_distribution_pct": target_rates,
        "missing_values_total": int(data.isna().sum().sum()),
        "average_age": round(float(data["Age at enrollment"].mean()), 2),
        "tuition_debtor_rate_pct": round(float((data["Tuition fees up to date"] == 0).mean() * 100), 2),
        "scholarship_rate_pct": round(float((data["Scholarship holder"] == 1).mean() * 100), 2),
        "data_quality_notes": (
            "File Excel asli kehilangan titik desimal pada kolom nilai mata kuliah dan makroekonomi. "
            "data_prep.py telah menormalkannya kembali ke skala 0-20 (nilai akademik) dan persentase."
        ),
    }


def main():
    print(f"Loading {DATA_PATH.name}...")
    raw_df = load_source_data()
    cleaned_df = clean_and_prepare_features(raw_df)

    train_df, test_df = train_test_split(
        cleaned_df,
        test_size=0.20,
        random_state=42,
        stratify=cleaned_df[TARGET],
    )

    cleaned_df.to_csv(BASE_DIR / "student_ml_dataset.csv", index=False)
    train_df.to_csv(BASE_DIR / "student_train.csv", index=False)
    test_df.to_csv(BASE_DIR / "student_test.csv", index=False)

    report = build_data_quality_report(cleaned_df)
    report["train_rows"] = int(len(train_df))
    report["test_rows"] = int(len(test_df))

    with open(BASE_DIR / "data_quality_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Data preparation complete.")
    print(f"Total baris   : {len(cleaned_df):,}")
    print(f"Train / Test  : {len(train_df):,} / {len(test_df):,}")
    print(f"Target ratios : {report['target_distribution_pct']}")
    print("Files created: student_ml_dataset.csv, student_train.csv, student_test.csv, data_quality_report.json")


if __name__ == "__main__":
    main()
