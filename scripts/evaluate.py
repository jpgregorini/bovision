#!/usr/bin/env python3
"""
Evaluate the trained XGBoost weight-estimation model.

Loads model, scaler, and metadata from models/regression/, evaluates on
the full dataset, and prints a pass/fail verdict:

  APROVADO  — MAPE < 5% AND within_20kg > 90%
  ACEITAVEL — MAPE < 10%
  REPROVADO — otherwise

Usage:
    python scripts/evaluate.py [--data path/to/dataset.csv]
                               [--model-dir path/to/models/regression]
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = ROOT / "data" / "weighings" / "dataset.csv"
MODELS_DIR = ROOT / "models" / "regression"

TARGET = "peso_kg"


# ── Dependency checks ──────────────────────────────────────────────────────────

def check_dependencies():
    missing = []
    for pkg in ("sklearn", "xgboost"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        pip_names = {"sklearn": "scikit-learn"}
        installs = " ".join(pip_names.get(p, p) for p in missing)
        print(f"[erro] Pacotes ausentes: {missing}")
        print(f"       Execute: pip install {installs}")
        sys.exit(1)


# ── Load artifacts ─────────────────────────────────────────────────────────────

def load_artifacts(model_dir: Path):
    model_path = model_dir / "weight_model.pkl"
    scaler_path = model_dir / "scaler.pkl"
    metadata_path = model_dir / "metadata.json"

    for p in (model_path, scaler_path, metadata_path):
        if not p.exists():
            print(f"[erro] Artefato não encontrado: {p}")
            print("       Execute primeiro: python scripts/train_regression.py")
            sys.exit(1)

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    with open(metadata_path) as f:
        metadata = json.load(f)

    feature_names = metadata.get("feature_names")
    if not feature_names:
        print("[erro] 'feature_names' não encontrado em metadata.json")
        sys.exit(1)

    return model, scaler, metadata, feature_names


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-6, None))) * 100)
    within_10 = float(np.mean(np.abs(y_true - y_pred) <= 10) * 100)
    within_20 = float(np.mean(np.abs(y_true - y_pred) <= 20) * 100)

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape_pct": mape,
        "within_10kg_pct": within_10,
        "within_20kg_pct": within_20,
    }


def print_metrics(metrics: dict, n_samples: int, title: str = "METRICAS DE AVALIACAO"):
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)
    print(f"  Amostras avaliadas    : {n_samples}")
    print(f"  MAE                   : {metrics['mae']:.2f} kg")
    print(f"  RMSE                  : {metrics['rmse']:.2f} kg")
    print(f"  R²                    : {metrics['r2']:.4f}")
    print(f"  MAPE                  : {metrics['mape_pct']:.2f} %")
    print(f"  Dentro de ±10 kg      : {metrics['within_10kg_pct']:.1f} %")
    print(f"  Dentro de ±20 kg      : {metrics['within_20kg_pct']:.1f} %")
    print("=" * 60)


# ── Verdict ────────────────────────────────────────────────────────────────────

def get_verdict(metrics: dict) -> str:
    mape = metrics["mape_pct"]
    within_20 = metrics["within_20kg_pct"]

    if mape < 5.0 and within_20 > 90.0:
        return "APROVADO"
    elif mape < 10.0:
        return "ACEITAVEL"
    else:
        return "REPROVADO"


def print_verdict(verdict: str, metrics: dict):
    symbols = {
        "APROVADO": "✓",
        "ACEITAVEL": "~",
        "REPROVADO": "✗",
    }
    criteria = {
        "APROVADO": "MAPE < 5% E dentro_20kg > 90%",
        "ACEITAVEL": "MAPE < 10% (mas nao atende criterio APROVADO)",
        "REPROVADO": "MAPE >= 10%",
    }

    sym = symbols.get(verdict, "?")
    crit = criteria.get(verdict, "")

    print(f"\n{'=' * 60}")
    print(f"  VEREDICTO: [{sym}] {verdict}")
    print(f"  Criterio : {crit}")
    print(f"  MAPE     : {metrics['mape_pct']:.2f}% | dentro±20kg: {metrics['within_20kg_pct']:.1f}%")
    print("=" * 60 + "\n")


# ── Per-sample analysis ────────────────────────────────────────────────────────

def print_error_distribution(y_true: np.ndarray, y_pred: np.ndarray):
    errors = np.abs(y_true - y_pred)
    thresholds = [5, 10, 20, 30, 50]
    print("\n[distribuicao de erros absolutos]")
    for t in thresholds:
        pct = np.mean(errors <= t) * 100
        print(f"  <= {t:3d} kg : {pct:5.1f}%")
    print(f"  > 50 kg   : {np.mean(errors > 50) * 100:5.1f}%")
    print(f"  Max erro  : {errors.max():.1f} kg")
    print(f"  P95 erro  : {np.percentile(errors, 95):.1f} kg")
    print(f"  Mediana   : {np.median(errors):.1f} kg")


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Avalia modelo de estimativa de peso bovino")
    parser.add_argument("--data", type=Path, default=DATA_CSV,
                        help="CSV com features e target peso_kg")
    parser.add_argument("--model-dir", type=Path, default=MODELS_DIR,
                        help="Diretório contendo weight_model.pkl, scaler.pkl, metadata.json")
    parser.add_argument("--verbose", action="store_true",
                        help="Exibe distribuicao detalhada de erros")
    return parser.parse_args()


def main():
    args = parse_args()
    check_dependencies()

    # ── Load model artifacts ───────────────────────────────────────────────────
    print(f"[load] Carregando artefatos de {args.model_dir} ...")
    model, scaler, metadata, feature_names = load_artifacts(args.model_dir)
    print(f"[load] Features: {feature_names}")
    print(f"[load] Modelo treinado com {metadata.get('n_samples', '?')} amostras")

    # ── Load evaluation dataset ────────────────────────────────────────────────
    if not args.data.exists():
        print(f"[erro] Dataset não encontrado: {args.data}")
        print("       Execute primeiro: python scripts/collect_data.py")
        sys.exit(1)

    df = pd.read_csv(args.data)
    print(f"[load] {len(df)} amostras carregadas de {args.data}")

    # Check that all required features are present
    missing_features = [f for f in feature_names if f not in df.columns]
    if missing_features:
        print(f"[erro] Features ausentes no dataset: {missing_features}")
        print(f"       Features disponíveis: {list(df.columns)}")
        sys.exit(1)

    if TARGET not in df.columns:
        print(f"[erro] Coluna target '{TARGET}' nao encontrada no dataset.")
        sys.exit(1)

    # Drop rows with NaN in required columns
    eval_cols = feature_names + [TARGET]
    df_clean = df[eval_cols].dropna()
    n_dropped = len(df) - len(df_clean)
    if n_dropped > 0:
        print(f"[clean] {n_dropped} linhas com NaN ignoradas")

    if len(df_clean) == 0:
        print("[erro] Nenhuma amostra válida para avaliação.")
        sys.exit(1)

    X = df_clean[feature_names].values
    y = df_clean[TARGET].values

    # ── Predict ────────────────────────────────────────────────────────────────
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

    # ── Compute & display metrics ──────────────────────────────────────────────
    metrics = compute_metrics(y, y_pred)
    print_metrics(metrics, n_samples=len(y))

    if args.verbose:
        print_error_distribution(y, y_pred)

    # ── Print CV metrics from training for comparison ──────────────────────────
    cv_avg = metadata.get("cv_metrics_avg")
    if cv_avg:
        print("\n[referencia] Metricas CV do treinamento (para comparacao):")
        print(f"  MAE   : {cv_avg.get('mae', '?'):.2f} kg")
        print(f"  RMSE  : {cv_avg.get('rmse', '?'):.2f} kg")
        print(f"  R²    : {cv_avg.get('r2', '?'):.4f}")
        print(f"  MAPE  : {cv_avg.get('mape_pct', '?'):.2f} %")

    # ── Verdict ────────────────────────────────────────────────────────────────
    verdict = get_verdict(metrics)
    print_verdict(verdict, metrics)

    # Exit code: 0 for APROVADO/ACEITAVEL, 1 for REPROVADO
    sys.exit(0 if verdict in ("APROVADO", "ACEITAVEL") else 1)


if __name__ == "__main__":
    main()
