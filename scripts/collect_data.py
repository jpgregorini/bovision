#!/usr/bin/env python3
"""
Download cattle body measurement + weight dataset from Kaggle and
map columns to BoviSight format, then save to data/weighings/dataset.csv.

This script replaces the original collect_data.py placeholder and acts as
the regression data downloader (equivalent to download_regression_data.py).

Usage:
    python scripts/collect_data.py [--skip-download]
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw" / "regression"
DATA_WEIGHINGS = ROOT / "data" / "weighings"
OUTPUT_CSV = DATA_WEIGHINGS / "dataset.csv"

KAGGLE_DATASETS = [
    "ujsinghania/cattle-weight-live-weight-dataset",
    "gfrfranco/cattle-weight-estimation",
]

# Maps from common Kaggle column names to BoviSight canonical names.
# Lower-case keys are matched after lowercasing the actual column names.
COLUMN_MAP = {
    # body length variants
    "body_length": "comprimento_m",
    "body length": "comprimento_m",
    "length": "comprimento_m",
    "comprimento": "comprimento_m",
    # height variants
    "height": "altura_m",
    "hip_height": "altura_m",
    "wither_height": "altura_m",
    "altura": "altura_m",
    # width variants
    "width": "largura_m",
    "hip_width": "largura_m",
    "body_width": "largura_m",
    "largura": "largura_m",
    # heart girth / perimeter variants
    "heart_girth": "perimetro_m",
    "girth": "perimetro_m",
    "chest_girth": "perimetro_m",
    "perimetro": "perimetro_m",
    "perimeter": "perimetro_m",
    # area variants
    "area": "area_m2",
    "body_area": "area_m2",
    "area_m2": "area_m2",
    # weight target (kg)
    "weight": "peso_kg",
    "live_weight": "peso_kg",
    "liveweight": "peso_kg",
    "body_weight": "peso_kg",
    "peso": "peso_kg",
    "weight_kg": "peso_kg",
}

WEIGHT_MIN_KG = 50.0
WEIGHT_MAX_KG = 1500.0


# ── Kaggle helpers ─────────────────────────────────────────────────────────────

def check_kaggle_credentials() -> bool:
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        return True
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return False


def download_from_kaggle(dataset: str, dest: Path) -> bool:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApiExtended
        api = KaggleApiExtended()
        api.authenticate()
        dest.mkdir(parents=True, exist_ok=True)
        print(f"[download] Baixando '{dataset}' ...")
        api.dataset_download_files(dataset, path=str(dest), unzip=True, quiet=False)
        return True
    except Exception as exc:
        print(f"[aviso] Falha ao baixar '{dataset}': {exc}")
        return False


def print_manual_instructions():
    print("\n" + "=" * 70)
    print("DOWNLOAD MANUAL NECESSARIO")
    print("=" * 70)
    print("Baixe um dos datasets abaixo e extraia o conteudo em:")
    print(f"  {DATA_RAW}\n")
    for ds in KAGGLE_DATASETS:
        print(f"  https://www.kaggle.com/datasets/{ds}")
    print("\nOu configure a API do Kaggle:")
    print("  1. Acesse https://www.kaggle.com/account e gere um token API")
    print("  2. Salve kaggle.json em ~/.kaggle/kaggle.json")
    print("  3. Execute novamente este script")
    print("=" * 70 + "\n")


# ── Column normalization ───────────────────────────────────────────────────────

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns using COLUMN_MAP (case-insensitive)."""
    rename = {}
    for col in df.columns:
        mapped = COLUMN_MAP.get(col.strip().lower())
        if mapped and col != mapped:
            rename[col] = mapped
    df = df.rename(columns=rename)
    return df


def derive_missing_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive area_m2 if missing but length and height are present."""
    if "area_m2" not in df.columns:
        if "comprimento_m" in df.columns and "altura_m" in df.columns:
            df["area_m2"] = df["comprimento_m"] * df["altura_m"]
            print("[derive] area_m2 calculado como comprimento_m * altura_m")
    return df


def convert_units(df: pd.DataFrame) -> pd.DataFrame:
    """
    If measurements look like they are in cm (typical values > 5 for body dims),
    convert to meters. Weight stays in kg.
    """
    measurement_cols = ["comprimento_m", "altura_m", "largura_m", "perimetro_m"]
    for col in measurement_cols:
        if col in df.columns:
            median_val = df[col].median()
            if median_val > 5:  # almost certainly in cm
                df[col] = df[col] / 100.0
                print(f"[convert] {col}: convertido de cm para m (mediana original={median_val:.1f})")
    return df


# ── Cleaning ───────────────────────────────────────────────────────────────────

def clean_outliers(df: pd.DataFrame) -> pd.DataFrame:
    n_before = len(df)

    # Weight range
    if "peso_kg" in df.columns:
        df = df[(df["peso_kg"] >= WEIGHT_MIN_KG) & (df["peso_kg"] <= WEIGHT_MAX_KG)]

    # No negative values in numeric measurement columns
    measurement_cols = ["comprimento_m", "altura_m", "largura_m", "area_m2", "perimetro_m"]
    for col in measurement_cols:
        if col in df.columns:
            df = df[df[col] >= 0]

    # Drop rows where all measurement features are NaN
    feature_cols = [c for c in measurement_cols if c in df.columns]
    if feature_cols:
        df = df.dropna(subset=feature_cols, how="all")

    # Drop rows without target weight
    if "peso_kg" in df.columns:
        df = df.dropna(subset=["peso_kg"])

    n_removed = n_before - len(df)
    if n_removed > 0:
        print(f"[clean] {n_removed} linhas removidas por outliers/valores inválidos")

    return df.reset_index(drop=True)


# ── CSV loading ────────────────────────────────────────────────────────────────

def load_csv_files(raw_dir: Path) -> pd.DataFrame:
    """Load and concatenate all CSV files found under raw_dir."""
    csv_files = list(raw_dir.rglob("*.csv"))
    if not csv_files:
        print(f"[erro] Nenhum arquivo CSV encontrado em {raw_dir}")
        sys.exit(1)

    frames = []
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
            print(f"[load] {csv_path.name}: {len(df)} linhas, colunas={list(df.columns)}")
            frames.append(df)
        except Exception as exc:
            print(f"[aviso] Erro ao ler {csv_path}: {exc}")

    if not frames:
        print("[erro] Nenhum CSV pôde ser carregado.")
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)
    print(f"[load] Total combinado: {len(combined)} linhas")
    return combined


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download e prepara dataset de medidas corporais de bovinos para regressão de peso"
    )
    parser.add_argument("--raw-dir", type=Path, default=DATA_RAW,
                        help="Diretório onde o dataset bruto será/está salvo")
    parser.add_argument("--out", type=Path, default=OUTPUT_CSV,
                        help="Caminho de saída do CSV processado")
    parser.add_argument("--skip-download", action="store_true",
                        help="Pula o download e usa dados existentes em --raw-dir")
    args = parser.parse_args()

    raw_dir: Path = args.raw_dir
    output_csv: Path = args.out

    # ── 1. Download ────────────────────────────────────────────────────────────
    if not args.skip_download:
        if not check_kaggle_credentials():
            print("[aviso] Credenciais do Kaggle não encontradas.")
            print_manual_instructions()
            if not raw_dir.exists() or not any(raw_dir.rglob("*.csv")):
                sys.exit(1)
            print(f"[info] Usando dados existentes em {raw_dir}")
        else:
            downloaded = False
            for dataset in KAGGLE_DATASETS:
                if download_from_kaggle(dataset, raw_dir):
                    downloaded = True
                    break
            if not downloaded:
                print_manual_instructions()
                if not raw_dir.exists() or not any(raw_dir.rglob("*.csv")):
                    sys.exit(1)
    else:
        print(f"[info] Usando dados existentes em {raw_dir}")

    # ── 2. Load ────────────────────────────────────────────────────────────────
    df = load_csv_files(raw_dir)

    # ── 3. Normalize & map columns ─────────────────────────────────────────────
    df = normalize_columns(df)
    print(f"[normalize] Colunas após mapeamento: {list(df.columns)}")

    # ── 4. Derive missing features ─────────────────────────────────────────────
    df = derive_missing_features(df)

    # ── 5. Unit conversion ─────────────────────────────────────────────────────
    df = convert_units(df)

    # ── 6. Clean outliers ─────────────────────────────────────────────────────
    df = clean_outliers(df)

    # ── 7. Keep only relevant columns ─────────────────────────────────────────
    keep = ["comprimento_m", "altura_m", "largura_m", "area_m2", "perimetro_m", "peso_kg"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep]

    # ── 8. Summary stats ───────────────────────────────────────────────────────
    print("\n[stats] Estatísticas do dataset processado:")
    print(df.describe().to_string())
    print(f"\n[info] Colunas presentes: {list(df.columns)}")
    print(f"[info] Total de amostras: {len(df)}")

    # ── 9. Save ────────────────────────────────────────────────────────────────
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\n[OK] Dataset salvo em {output_csv}")


if __name__ == "__main__":
    main()
