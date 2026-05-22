#!/usr/bin/env python3
"""
Train XGBoost regressor for cattle weight estimation.

Reads data/weighings/dataset.csv, trains with 5-fold CV, and saves:
  models/regression/weight_model.pkl
  models/regression/scaler.pkl
  models/regression/metadata.json

Usage:
    python scripts/train_regression.py [--data path/to/dataset.csv]
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

FEATURE_CANDIDATES = [
    "comprimento_m",
    "altura_m",
    "largura_m",
    "area_m2",
    "perimetro_m",
]
TARGET = "peso_kg"

RANDOM_SEED = 42


# ── Dependency checks ──────────────────────────────────────────────────────────

def check_dependencies():
    missing = []
    for pkg in ("xgboost", "sklearn"):
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


# ── Metric helpers ─────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-6, None))) * 100)
    within_10 = float(np.mean(np.abs(y_true - y_pred) <= 10) * 100)
    within_20 = float(np.mean(np.abs(y_true - y_pred) <= 20) * 100)

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "mape_pct": float(mape),
        "within_10kg_pct": float(within_10),
        "within_20kg_pct": float(within_20),
    }


def print_metrics(metrics: dict, title: str = "METRICAS"):
    print(f"\n{'=' * 50}")
    print(title)
    print("=" * 50)
    print(f"  MAE              : {metrics['mae']:.2f} kg")
    print(f"  RMSE             : {metrics['rmse']:.2f} kg")
    print(f"  R²               : {metrics['r2']:.4f}")
    print(f"  MAPE             : {metrics['mape_pct']:.2f} %")
    print(f"  Dentro de ±10 kg : {metrics['within_10kg_pct']:.1f} %")
    print(f"  Dentro de ±20 kg : {metrics['within_20kg_pct']:.1f} %")
    print("=" * 50)


# ── Cross-validation ───────────────────────────────────────────────────────────

def cross_validate(X: np.ndarray, y: np.ndarray, model_cls, model_params: dict, n_folds: int = 5):
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_val_sc = scaler.transform(X_val)

        model = model_cls(**model_params)
        model.fit(X_tr_sc, y_tr, eval_set=[(X_val_sc, y_val)], verbose=False)

        preds = model.predict(X_val_sc)
        m = compute_metrics(y_val, preds)
        fold_metrics.append(m)
        print(f"  Fold {fold_idx}/{n_folds}: MAE={m['mae']:.2f} RMSE={m['rmse']:.2f} "
              f"R²={m['r2']:.4f} MAPE={m['mape_pct']:.2f}%")

    avg = {k: float(np.mean([f[k] for f in fold_metrics])) for k in fold_metrics[0]}
    std = {k: float(np.std([f[k] for f in fold_metrics])) for k in fold_metrics[0]}

    print(f"\n  Media 5-fold: MAE={avg['mae']:.2f}±{std['mae']:.2f} "
          f"RMSE={avg['rmse']:.2f}±{std['rmse']:.2f} "
          f"R²={avg['r2']:.4f}±{std['r2']:.4f}")
    return avg, std


# ── Feature importance ─────────────────────────────────────────────────────────

def print_feature_importance(model, feature_names: list):
    importances = model.feature_importances_
    pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    print("\n[importancia] Feature importances (gain):")
    for name, imp in pairs:
        bar = "#" * int(imp * 40)
        print(f"  {name:20s}: {imp:.4f} {bar}")


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Treina modelo XGBoost para estimativa de peso bovino")
    parser.add_argument("--data", type=Path, default=DATA_CSV,
                        help="CSV com features e target peso_kg")
    parser.add_argument("--out-dir", type=Path, default=MODELS_DIR,
                        help="Diretório de saída para artefatos do modelo")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--folds", type=int, default=5, help="Número de folds para CV")
    return parser.parse_args()


def main():
    args = parse_args()
    check_dependencies()

    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBRegressor

    # ── Load data ──────────────────────────────────────────────────────────────
    if not args.data.exists():
        print(f"[erro] Dataset não encontrado: {args.data}")
        print("       Execute primeiro: python scripts/collect_data.py")
        sys.exit(1)

    df = pd.read_csv(args.data)
    print(f"[load] {len(df)} amostras carregadas de {args.data}")
    print(f"       Colunas: {list(df.columns)}")

    # ── Select available features ──────────────────────────────────────────────
    feature_names = [f for f in FEATURE_CANDIDATES if f in df.columns]
    if not feature_names:
        print("[erro] Nenhuma feature esperada encontrada no dataset.")
        print(f"       Esperadas: {FEATURE_CANDIDATES}")
        print(f"       Presentes: {list(df.columns)}")
        sys.exit(1)
    if TARGET not in df.columns:
        print(f"[erro] Coluna target '{TARGET}' não encontrada no dataset.")
        sys.exit(1)

    print(f"[features] Usando: {feature_names}")

    # Drop rows with NaN in selected features or target
    df_clean = df[feature_names + [TARGET]].dropna()
    n_dropped = len(df) - len(df_clean)
    if n_dropped > 0:
        print(f"[clean] {n_dropped} linhas com NaN removidas")

    X = df_clean[feature_names].values
    y = df_clean[TARGET].values
    print(f"[info] Amostras para treinamento: {len(X)}")

    # ── Cross-validation ───────────────────────────────────────────────────────
    model_params = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "early_stopping_rounds": 20,
    }

    print(f"\n[cv] Iniciando validacao cruzada {args.folds}-fold ...")
    cv_avg, cv_std = cross_validate(X, y, XGBRegressor, model_params, n_folds=args.folds)
    print_metrics(cv_avg, title=f"MEDIA {args.folds}-FOLD CV")

    # ── Final model on full dataset ────────────────────────────────────────────
    print("\n[train] Treinando modelo final em todo o dataset ...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Remove early stopping for final model (no validation set)
    final_params = {k: v for k, v in model_params.items() if k != "early_stopping_rounds"}
    model = XGBRegressor(**final_params)
    model.fit(X_scaled, y)

    # Evaluate on training data (informational)
    preds = model.predict(X_scaled)
    train_metrics = compute_metrics(y, preds)
    print_metrics(train_metrics, title="METRICAS FINAIS (treino completo)")

    # Feature importance
    print_feature_importance(model, feature_names)

    # ── Save artifacts ─────────────────────────────────────────────────────────
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model_path = args.out_dir / "weight_model.pkl"
    scaler_path = args.out_dir / "scaler.pkl"
    metadata_path = args.out_dir / "metadata.json"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    metadata = {
        "feature_names": feature_names,
        "target": TARGET,
        "n_samples": int(len(X)),
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "cv_folds": args.folds,
        "cv_metrics_avg": cv_avg,
        "cv_metrics_std": cv_std,
        "train_metrics": train_metrics,
        "random_seed": RANDOM_SEED,
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[OK] Artefatos salvos em {args.out_dir}")
    print(f"     Modelo  : {model_path}")
    print(f"     Scaler  : {scaler_path}")
    print(f"     Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
