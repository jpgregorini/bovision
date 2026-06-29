"""
Fase 2 — Keypoints da vista lateral (B3/Side)
==============================================
Baixa COCO_Side.json, extrai features dos 9 keypoints anotados e avalia
o ganho sobre o feature set v2 via ablation study.

Keypoints (índices 0-8 no COCO flat list, cada um = [x, y, visibility]):
    0  1_wither
    1  2_pinbone
    2  3_shoulderbone
    3  4_front_girth_top
    4  5_front_girth_bottom
    5  6_rear_girth_top
    6  7_rear_girth_bottom
    7  8_Height_top
    8  9_Height_bottom

Features geradas (todas normalizadas pelo sticker_h do features.csv v2):
    kp_front_girth_n   — distância vertical entre front_girth_top e bottom
    kp_rear_girth_n    — distância vertical entre rear_girth_top e bottom
    kp_body_height_n   — distância vertical Height_top → bottom
    kp_back_len_n      — distância horizontal wither → pinbone (comprimento dorsal)
    kp_girth_ratio     — front_girth / rear_girth (forma do tronco)
    kp_height_ratio    — body_height / back_len (índice de porte)
    kp_vol_proxy_n     — front_girth² × back_len / s³  (proxy volumétrico keypoint)
    kp_wither_y_n      — altura absoluta do wither desde o topo (posição dorsal)
    kp_front_x_n       — posição horizontal da girth dianteira (= ponto de medição)

Usage:
    cd research/
    python fase2_keypoints.py
"""

import json
import logging
import math
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── Caminhos ────────────────────────────────────────────────────────────────
KAGGLE_DATASET  = "sadhliroomyprime/cattle-weight-detection-model-dataset-12k"
COCO_REMOTE     = "www.acmeai.tech Dataset - BMGF-LivestockWeight-CV/Vector/B3/Side/data/COCO_Side.json"
COCO_LOCAL      = Path("data/coco_side.json")
FEATURES_V2     = Path("features.csv")           # gerado pelo script principal (v2)
FEATURES_KP_OUT = Path("features_kp.csv")        # v2 + keypoints

# XGBoost padrão (mesmos hiperparâmetros da Fase 1)
XGB_PARAMS = dict(
    n_estimators=1200, max_depth=3, learning_rate=0.01,
    subsample=0.6, colsample_bytree=0.6, min_child_weight=1,
    reg_alpha=1.0, reg_lambda=5.0,
    objective="reg:squarederror", random_state=42, n_jobs=-1,
)

KP_NAMES = [
    "wither", "pinbone", "shoulderbone",
    "front_girth_top", "front_girth_bottom",
    "rear_girth_top", "rear_girth_bottom",
    "height_top", "height_bottom",
]


# ── Download do COCO_Side.json ───────────────────────────────────────────────

def download_coco_json():
    if COCO_LOCAL.exists():
        log.info("COCO_Side.json já presente: %s", COCO_LOCAL)
        return

    repo_cfg    = Path(__file__).parent / "kaggle.json"
    default_cfg = Path.home() / ".kaggle" / "kaggle.json"
    if repo_cfg.exists():
        os.environ["KAGGLE_CONFIG_DIR"] = str(repo_cfg.parent)
    elif not default_cfg.exists():
        log.error("kaggle.json não encontrado. Coloque em ~/.kaggle/kaggle.json")
        sys.exit(1)

    try:
        import kaggle
        api = kaggle.KaggleApi()
        api.authenticate()
    except Exception as e:
        log.error("Autenticação Kaggle falhou: %s", e)
        sys.exit(1)

    COCO_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    log.info("Baixando %s …", COCO_REMOTE)
    while True:
        try:
            api.dataset_download_file(
                KAGGLE_DATASET, COCO_REMOTE,
                path=str(COCO_LOCAL.parent), quiet=False,
            )
            break
        except Exception as e:
            if "429" in str(e):
                log.warning("Rate-limited — aguardando 60 s …")
                time.sleep(60)
            else:
                log.error("Download falhou: %s", e)
                sys.exit(1)

    # Kaggle entrega arquivos individuais como .zip; descompactar se necessário
    zip_path = COCO_LOCAL.parent / (Path(COCO_REMOTE).name + ".zip")
    if zip_path.exists():
        log.info("Descompactando %s …", zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            names_in_zip = zf.namelist()
            json_name = next((n for n in names_in_zip if n.endswith(".json")), None)
            if json_name:
                with zf.open(json_name) as src, open(COCO_LOCAL, "wb") as dst:
                    dst.write(src.read())
            else:
                log.error("Nenhum .json encontrado no zip: %s", names_in_zip)
                sys.exit(1)
        zip_path.unlink()
    else:
        # Já foi salvo sem zip (nome original)
        saved = COCO_LOCAL.parent / Path(COCO_REMOTE).name
        if saved.exists() and saved != COCO_LOCAL:
            saved.rename(COCO_LOCAL)

    if not COCO_LOCAL.exists():
        log.error("COCO_Side.json não encontrado após download.")
        sys.exit(1)

    log.info("Salvo em %s (%.1f KB)", COCO_LOCAL, COCO_LOCAL.stat().st_size / 1024)


# ── Parse dos keypoints ──────────────────────────────────────────────────────

def load_keypoints() -> dict[str, dict]:
    """
    Retorna {file_stem: {kp_name: (x, y)}} para keypoints visíveis (v > 0).
    Stems seguem a convenção do dataset: "100_s_109_M".
    """
    with open(COCO_LOCAL, encoding="utf-8") as f:
        coco = json.load(f)

    # image_id → stem do arquivo
    id_to_stem = {}
    for img in coco["images"]:
        stem = Path(img["file_name"]).stem
        id_to_stem[img["id"]] = stem

    result = {}
    for ann in coco["annotations"]:
        stem = id_to_stem.get(ann["image_id"])
        if stem is None:
            continue
        flat = ann.get("keypoints", [])
        if len(flat) < len(KP_NAMES) * 3:
            continue
        kps = {}
        for i, name in enumerate(KP_NAMES):
            x, y, v = flat[3 * i], flat[3 * i + 1], flat[3 * i + 2]
            if v > 0:
                kps[name] = (float(x), float(y))
        result[stem] = kps

    log.info("Keypoints carregados: %d anotações", len(result))
    return result


def _dist_y(kps: dict, a: str, b: str) -> float | None:
    """Distância vertical (pixels) entre dois keypoints, ou None se ausentes."""
    if a not in kps or b not in kps:
        return None
    return abs(kps[a][1] - kps[b][1])


def _dist_x(kps: dict, a: str, b: str) -> float | None:
    if a not in kps or b not in kps:
        return None
    return abs(kps[a][0] - kps[b][0])


def compute_kp_features(kps: dict, sticker_h: float) -> dict | None:
    """
    Transforma keypoints em features normalizadas pelo adesivo.
    Retorna None se os keypoints críticos estiverem ausentes.
    """
    s = sticker_h
    if s <= 5:
        return None

    front_girth = _dist_y(kps, "front_girth_top", "front_girth_bottom")
    rear_girth  = _dist_y(kps, "rear_girth_top",  "rear_girth_bottom")
    body_height = _dist_y(kps, "height_top",       "height_bottom")
    back_len    = _dist_x(kps, "wither",           "pinbone")

    # Requer ao menos girth dianteira e traseira para ser útil
    if front_girth is None or rear_girth is None:
        return None

    feats = {
        "kp_front_girth_n": front_girth / s,
        "kp_rear_girth_n":  rear_girth / s,
    }

    if body_height is not None:
        feats["kp_body_height_n"] = body_height / s
    else:
        feats["kp_body_height_n"] = float("nan")

    if back_len is not None and back_len > 0:
        feats["kp_back_len_n"]   = back_len / s
        feats["kp_height_ratio"] = (body_height / back_len
                                    if body_height is not None else float("nan"))
        feats["kp_vol_proxy_n"]  = (front_girth ** 2 * back_len) / s ** 3
    else:
        feats["kp_back_len_n"]   = float("nan")
        feats["kp_height_ratio"] = float("nan")
        feats["kp_vol_proxy_n"]  = float("nan")

    feats["kp_girth_ratio"] = (front_girth / rear_girth
                               if rear_girth > 0 else float("nan"))

    # Posição absoluta do wither (distância desde topo da imagem → forma de corcunda)
    if "wither" in kps:
        feats["kp_wither_y_n"] = kps["wither"][1] / s
    else:
        feats["kp_wither_y_n"] = float("nan")

    # Posição horizontal da girth dianteira (ponto de medição no corpo)
    if "front_girth_top" in kps:
        feats["kp_front_x_n"] = kps["front_girth_top"][0] / s
    else:
        feats["kp_front_x_n"] = float("nan")

    return feats


# ── Merge com features v2 ────────────────────────────────────────────────────

def _clean_stem(mask_stem: str) -> str:
    """
    Converte o stem da máscara para o stem da imagem original.
    Ex.: "100_s_109_M.jpg___fuse" → "100_s_109_M"
    """
    # Remove sufixo ___fuse e qualquer extensão residual (.jpg, .png, etc.)
    s = mask_stem.split("___")[0]          # "100_s_109_M.jpg"
    return Path(s).stem                    # "100_s_109_M"


def build_combined_features(kp_map: dict[str, dict]) -> tuple:
    """
    Carrega features.csv (v2), faz join com keypoints e retorna:
        X_v2, X_kp, y, feature_names_v2, feature_names_kp
    X_kp contém apenas as amostras que têm keypoints válidos.
    """
    if not FEATURES_V2.exists():
        log.error("features.csv não encontrado. Rode o script principal primeiro.")
        sys.exit(1)

    df = pd.read_csv(FEATURES_V2)
    log.info("features.csv carregado: %d amostras × %d colunas", *df.shape)

    # Recupera stems na mesma ordem em que features.csv foi gerado (sorted glob)
    image_dir = Path("data/images")
    import re
    pattern = re.compile(r"^\d+(?:\.\d+)?_[sr]_(\d+(?:\.\d+)?)_[MF]", re.I)

    all_paths = sorted(p for p in image_dir.glob("*")
                       if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    valid_paths = [p for p in all_paths if pattern.match(_clean_stem(p.stem))]

    # features.csv foi gerado pulando imagens sem sticker (skipped=3)
    # → mapeamos apenas os que não foram pulados (sticker_h > 0 na v2)
    clean_stems = [_clean_stem(p.stem) for p in valid_paths]

    if len(clean_stems) < len(df):
        log.warning("Stems (%d) < linhas CSV (%d)", len(clean_stems), len(df))

    df["_stem"] = clean_stems[:len(df)]
    log.info("Amostra de stems: %s …", clean_stems[:3])

    feat_cols = [c for c in df.columns if c not in ("weight_kg", "_stem")]
    y_all = df["weight_kg"].values

    kp_rows, kp_indices = [], []
    for i, row in df.iterrows():
        stem = row["_stem"]
        kps  = kp_map.get(stem, {})
        kp_feats = compute_kp_features(kps, row["sticker_h"])
        if kp_feats is not None:
            kp_rows.append(kp_feats)
            kp_indices.append(i)

    log.info("Amostras com keypoints válidos: %d / %d",
             len(kp_indices), len(df))

    df_kp = df.loc[kp_indices].reset_index(drop=True)
    kp_df = pd.DataFrame(kp_rows)

    X_v2  = df_kp[feat_cols].values.astype(np.float32)
    X_kp  = kp_df.values.astype(np.float32)
    y_sub = df_kp["weight_kg"].values.astype(np.float32)

    return X_v2, X_kp, y_sub, feat_cols, list(kp_df.columns)


# ── Ablation study ───────────────────────────────────────────────────────────

def cv_r2(X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    kf   = KFold(n_splits=5, shuffle=True, random_state=42)
    # Remover NaNs por coluna antes do CV
    mask = ~np.isnan(X).any(axis=1)
    scores = cross_val_score(
        xgb.XGBRegressor(**XGB_PARAMS),
        X[mask], y[mask], cv=kf, scoring="r2",
    )
    return float(scores.mean()), float(scores.std())


def run_ablation(X_v2, X_kp, y, feat_names_v2, feat_names_kp):
    log.info("=" * 55)
    log.info("ABLATION STUDY — impacto dos keypoints (5-fold CV)")
    log.info("=" * 55)

    # Baseline v2 no subconjunto com keypoints
    r2_base, std_base = cv_r2(X_v2, y)
    log.info("v2 base (subconjunto c/ kp)  :  R² = %.4f ± %.4f", r2_base, std_base)

    # v2 + keypoints
    X_combined = np.hstack([X_v2, X_kp])
    r2_comb, std_comb = cv_r2(X_combined, y)
    log.info("v2 + keypoints               :  R² = %.4f ± %.4f", r2_comb, std_comb)

    delta = r2_comb - r2_base
    log.info("Delta R²                     :  %+.4f", delta)

    if delta >= 0.02:
        log.info("✅ Ganho ≥ 0.02 — keypoints aprovados para integrar no pipeline.")
    elif delta >= 0.005:
        log.info("⚠  Ganho marginal (%.4f) — avaliar custo/benefício.", delta)
    else:
        log.info("❌ Sem ganho significativo (%.4f) — keypoints não ajudam.", delta)

    # Feature importances do modelo combinado
    mask = ~np.isnan(X_combined).any(axis=1)
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_combined[mask], y[mask], verbose=False)
    all_names = list(feat_names_v2) + list(feat_names_kp)
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:15]
    log.info("\nTop 15 features (modelo combinado):")
    for rank, i in enumerate(top_idx, 1):
        log.info("  %2d. %-22s %.4f", rank, all_names[i], importances[i])

    # Salvar features combinadas para referência
    df_v2  = pd.read_csv(FEATURES_V2)
    # Só as linhas que tinham keypoints válidos (mesma lógica de build_combined_features)
    n_kp = X_kp.shape[0]
    df_out = df_v2.iloc[:n_kp].copy().reset_index(drop=True)
    for j, col in enumerate(feat_names_kp):
        df_out[col] = X_kp[:, j]
    df_out.to_csv(FEATURES_KP_OUT, index=False)
    log.info("\nFeatures combinadas salvas em %s", FEATURES_KP_OUT)

    return {"r2_base": r2_base, "r2_kp": r2_comb, "delta": delta}


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("=" * 55)
    log.info("Fase 2 — Keypoints da vista lateral (B3/Side)")
    log.info("=" * 55)

    download_coco_json()

    kp_map = load_keypoints()

    X_v2, X_kp, y, feat_names_v2, feat_names_kp = build_combined_features(kp_map)
    log.info("Feature set v2: %d features | Keypoints: %d features",
             X_v2.shape[1], X_kp.shape[1])

    results = run_ablation(X_v2, X_kp, y, feat_names_v2, feat_names_kp)

    log.info("=" * 55)
    log.info("Fase 2 concluída.")
    log.info("  R² base (v2)     = %.4f", results["r2_base"])
    log.info("  R² v2+keypoints  = %.4f", results["r2_kp"])
    log.info("  Delta            = %+.4f", results["delta"])
    log.info("  Critério fase 2  = ΔR² ≥ 0.02")
    log.info("=" * 55)
