"""
Fase 3 — Vista traseira (B4/Side + B4/Rear)
============================================
Baixa as 1943 máscaras laterais e 1943 traseiras do batch B4, casa os
pares por animal (id_subbatch_peso_sexo), extrai features de cada vista
e avalia o ganho de R² da combinação sobre a lateral sozinha.

Convenção de nome B4:
    <id>_b4-<subbatch>_<s|r>_<weight>_<sex>.jpg___fuse.png
    ex.: 100_b4-1_s_124_F.jpg___fuse.png  ←→  100_b4-1_r_124_F.jpg___fuse.png

Features traseiras extraídas (todas normalizadas pelo adesivo):
    rear_area_n        — área da silhueta traseira / s²
    rear_width_n       — largura máxima (left→right) / s
    rear_height_n      — altura máxima (top→bottom) / s
    rear_aspect_ratio  — largura / altura
    rear_solidity      — área / área do hull convexo
    rear_extent        — área / (bbox_w × bbox_h)
    rear_max_chord_n   — maior corda horizontal / s  (profundidade lateral vista de trás)
    rear_mean_chord_n  — média das cordas horizontais / s
    rear_vol_proxy_n   — width² × height / s³  (proxy volumétrico)
    rear_sticker_h     — altura do adesivo traseiro em pixels (confere escala)

Usage:
    cd research/
    python fase3_rear_view.py
"""

import csv
import logging
import math
import os
import re
import sys
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── Caminhos ─────────────────────────────────────────────────────────────────
KAGGLE_DATASET   = "sadhliroomyprime/cattle-weight-detection-model-dataset-12k"
FILE_LIST        = Path("data/file_list.txt")
SIDE_DIR         = Path("data/b4_side")
REAR_DIR         = Path("data/b4_rear")
FEATURES_B4_OUT  = Path("features_b4.csv")

B4_SIDE_SUBPATH  = "Pixel/B4/Side/annotations"
B4_REAR_SUBPATH  = "Pixel/B4/Rear/annotations"
DATASET_PREFIX   = "www.acmeai.tech Dataset - BMGF-LivestockWeight-CV/"

# B3/B4 class colours (RGB)
CATTLE_RGB  = (255, 30, 249)
STICKER_RGB = (0, 117, 255)
COLOR_TOL   = 12
N_CHORDS    = 8   # cordas horizontais para o perfil da vista traseira

XGB_PARAMS = dict(
    n_estimators=1200, max_depth=3, learning_rate=0.01,
    subsample=0.6, colsample_bytree=0.6, min_child_weight=1,
    reg_alpha=1.0, reg_lambda=5.0,
    objective="reg:squarederror", random_state=42, n_jobs=-1,
)

# Regex: extrai (id, subbatch, side/rear, weight, sex) do stem B4
_B4_PAT = re.compile(
    r"^(\d+)_b4-(\d+)_([sr])_(\d+(?:\.\d+)?)_([MF])", re.I
)


# ── Download ──────────────────────────────────────────────────────────────────

def _kaggle_api():
    repo_cfg    = Path(__file__).parent / "kaggle.json"
    default_cfg = Path.home() / ".kaggle" / "kaggle.json"
    if repo_cfg.exists():
        os.environ["KAGGLE_CONFIG_DIR"] = str(repo_cfg.parent)
    elif not default_cfg.exists():
        log.error("kaggle.json não encontrado.")
        sys.exit(1)
    try:
        import kaggle
        api = kaggle.KaggleApi()
        api.authenticate()
        return api
    except Exception as e:
        log.error("Autenticação Kaggle falhou: %s", e)
        sys.exit(1)


def _download_subpath(subpath: str, dest_dir: Path):
    """Baixa todos os arquivos de um subpath para dest_dir (offline-aware)."""
    dest_dir.mkdir(parents=True, exist_ok=True)

    names = FILE_LIST.read_text(encoding="utf-8").splitlines()
    targets = [n for n in names if subpath in n and
               n.lower().endswith((".png", ".jpg", ".jpeg"))]
    log.info("Encontrados %d arquivos em '%s'", len(targets), subpath)

    missing = [n for n in targets if not (dest_dir / Path(n).name).exists()]
    if not missing:
        log.info("Todos já presentes em %s — pulando download.", dest_dir)
        return len(targets)

    log.info("Baixando %d arquivos → %s …", len(missing), dest_dir)
    api = _kaggle_api()
    failed = 0
    for i, name in enumerate(missing, 1):
        while True:
            try:
                api.dataset_download_file(
                    KAGGLE_DATASET, name, path=str(dest_dir), quiet=True
                )
                # descompactar .zip se necessário
                zip_p = dest_dir / (Path(name).name + ".zip")
                if zip_p.exists():
                    with zipfile.ZipFile(zip_p) as zf:
                        inner = next(
                            (x for x in zf.namelist()
                             if x.lower().endswith((".png", ".jpg"))), None
                        )
                        if inner:
                            with zf.open(inner) as src:
                                (dest_dir / Path(inner).name).write_bytes(src.read())
                    zip_p.unlink()
                break
            except Exception as e:
                if "429" in str(e):
                    log.warning("Rate-limited — aguardando 60 s …")
                    time.sleep(60)
                else:
                    log.error("Falhou: %s (%s)", name, e)
                    failed += 1
                    break
        if i % 200 == 0:
            log.info("  %d / %d baixados", i, len(missing))

    if failed:
        log.warning("%d arquivos falharam — re-execute para tentar novamente.", failed)
    return len(targets)


def download_b4():
    log.info("=== Download B4/Side ===")
    n_side = _download_subpath(B4_SIDE_SUBPATH, SIDE_DIR)
    log.info("=== Download B4/Rear ===")
    n_rear = _download_subpath(B4_REAR_SUBPATH, REAR_DIR)
    return n_side, n_rear


# ── Feature extraction ────────────────────────────────────────────────────────

def _class_mask(img_bgr, rgb, tol=COLOR_TOL):
    bgr   = np.array(rgb[::-1], dtype=np.int16)
    lower = np.clip(bgr - tol, 0, 255).astype(np.uint8)
    upper = np.clip(bgr + tol, 0, 255).astype(np.uint8)
    return cv2.inRange(img_bgr, lower, upper)


def _largest_contour(binary, min_area=500):
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    return c if cv2.contourArea(c) >= min_area else None


def _sticker_h(img_bgr) -> float:
    c = _largest_contour(_class_mask(img_bgr, STICKER_RGB), min_area=20)
    if c is None:
        return 0.0
    return float(cv2.boundingRect(c)[3])


def extract_lateral(img_bgr) -> dict | None:
    """Features laterais v2 (subset compacto para combinar com rear)."""
    cnt = _largest_contour(_class_mask(img_bgr, CATTLE_RGB))
    if cnt is None:
        return None
    s = _sticker_h(img_bgr)
    if s <= 5:
        return None

    body = np.zeros(img_bgr.shape[:2], np.uint8)
    cv2.drawContours(body, [cnt], -1, 255, cv2.FILLED)
    x, y, w, h = cv2.boundingRect(cnt)

    # Leg removal
    k = max(3, int(0.06 * w) | 1)
    torso = cv2.morphologyEx(
        body, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    )

    # Canonical orientation
    col_h = (body[:, x:x + w] > 0).sum(0)
    third = max(1, w // 3)
    if col_h[:third].mean() > col_h[-third:].mean():
        body, torso = cv2.flip(body, 1), cv2.flip(torso, 1)

    ys, xs = np.where(body > 0)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    w, h = x1 - x0 + 1, y1 - y0 + 1

    chords = np.zeros(12)
    for i in range(12):
        cx  = x0 + int((i + 0.5) / 12 * w)
        col = torso[:, max(x0, cx - 1): cx + 2]
        rows = np.where(np.any(col > 0, axis=1))[0]
        chords[i] = (rows[-1] - rows[0]) if len(rows) >= 2 else 0.0

    area       = float((body > 0).sum())
    torso_area = float((torso > 0).sum())
    perimeter  = cv2.arcLength(cnt, True)
    hull_area  = cv2.contourArea(cv2.convexHull(np.column_stack([xs, ys])))
    max_chord  = float(chords.max())

    feats = {
        "lat_area_n":        area / s ** 2,
        "lat_torso_area_n":  torso_area / s ** 2,
        "lat_length_n":      w / s,
        "lat_height_n":      h / s,
        "lat_perimeter_n":   perimeter / s,
        "lat_aspect_ratio":  w / h if h else 0.0,
        "lat_solidity":      area / hull_area if hull_area else 0.0,
        "lat_extent":        area / (w * h) if w * h else 0.0,
        "lat_max_chord_n":   max_chord / s,
        "lat_mean_chord_n":  float(chords.mean()) / s,
        "lat_vol_proxy_n":   (max_chord ** 2 * w) / s ** 3,
        "lat_sticker_h":     s,
    }
    for i, c in enumerate(chords):
        feats[f"lat_chord_{i}_n"] = c / s

    hu = cv2.HuMoments(cv2.moments(body)).flatten()
    for i, v in enumerate(hu):
        feats[f"lat_hu_{i}"] = -math.copysign(1, v) * math.log10(abs(v) + 1e-10)

    return feats


def extract_rear(img_bgr, sticker_h_lateral: float = 0.0) -> dict | None:
    """
    Features da vista traseira: largura, cordas horizontais, proporções.
    O adesivo não aparece na vista traseira — usa o sticker_h da lateral
    (mesmo animal, mesma sessão) como âncora de escala.
    """
    cnt = _largest_contour(_class_mask(img_bgr, CATTLE_RGB))
    if cnt is None:
        return None
    s = sticker_h_lateral   # escala vem da vista lateral pareada
    if s <= 5:
        return None

    body = np.zeros(img_bgr.shape[:2], np.uint8)
    cv2.drawContours(body, [cnt], -1, 255, cv2.FILLED)

    ys, xs = np.where(body > 0)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    w, h = x1 - x0 + 1, y1 - y0 + 1

    area     = float((body > 0).sum())
    hull     = cv2.convexHull(np.column_stack([xs, ys]))
    hull_a   = cv2.contourArea(hull)
    perim    = cv2.arcLength(cnt, True)

    # Cordas horizontais ao longo da altura (equivalente aos chords laterais)
    chords = np.zeros(N_CHORDS)
    for i in range(N_CHORDS):
        ry  = y0 + int((i + 0.5) / N_CHORDS * h)
        row = body[max(y0, ry - 1): ry + 2, :]
        cols = np.where(np.any(row > 0, axis=0))[0]
        chords[i] = (cols[-1] - cols[0]) if len(cols) >= 2 else 0.0

    max_chord = float(chords.max())

    feats = {
        "rear_area_n":       area / s ** 2,
        "rear_width_n":      w / s,
        "rear_height_n":     h / s,
        "rear_aspect_ratio": w / h if h else 0.0,
        "rear_solidity":     area / hull_a if hull_a else 0.0,
        "rear_extent":       area / (w * h) if w * h else 0.0,
        "rear_max_chord_n":  max_chord / s,
        "rear_mean_chord_n": float(chords.mean()) / s,
        "rear_vol_proxy_n":  (max_chord ** 2 * h) / s ** 3,
        "rear_sticker_h":    s,
    }
    for i, c in enumerate(chords):
        feats[f"rear_chord_{i}_n"] = c / s

    hu = cv2.HuMoments(cv2.moments(body)).flatten()
    for i, v in enumerate(hu):
        feats[f"rear_hu_{i}"] = -math.copysign(1, v) * math.log10(abs(v) + 1e-10)

    return feats


# ── Build feature matrix ──────────────────────────────────────────────────────

def _parse_b4_stem(stem: str):
    """Retorna (animal_key, weight_kg) ou None."""
    # stem pode ter ___fuse; limpar
    clean = stem.split("___")[0]
    clean = Path(clean).stem
    m = _B4_PAT.match(clean)
    if not m:
        return None, None
    aid, sub, view, wt, sex = m.groups()
    key = f"{aid}_b4-{sub}_{wt}_{sex.upper()}"
    return key, float(wt)


def build_feature_matrix():
    side_imgs = {p.stem: p for p in SIDE_DIR.glob("*")
                 if p.suffix.lower() in (".png", ".jpg")}
    rear_imgs = {p.stem: p for p in REAR_DIR.glob("*")
                 if p.suffix.lower() in (".png", ".jpg")}

    log.info("Imagens laterais B4: %d | traseiras: %d",
             len(side_imgs), len(rear_imgs))

    # Indexar por animal_key
    side_by_key = {}
    for stem, path in side_imgs.items():
        key, wt = _parse_b4_stem(stem)
        if key:
            side_by_key[key] = (path, wt)

    rear_by_key = {}
    for stem, path in rear_imgs.items():
        key, wt = _parse_b4_stem(stem)
        if key:
            rear_by_key[key] = (path, wt)

    common_keys = sorted(set(side_by_key) & set(rear_by_key))
    log.info("Pares laterais+traseiros casados: %d", len(common_keys))

    rows, targets, skipped = [], [], 0
    for key in common_keys:
        side_path, wt = side_by_key[key]
        rear_path, _  = rear_by_key[key]

        img_side = cv2.imread(str(side_path), cv2.IMREAD_COLOR)
        img_rear = cv2.imread(str(rear_path), cv2.IMREAD_COLOR)
        if img_side is None or img_rear is None:
            skipped += 1
            continue

        lat_feats  = extract_lateral(img_side)
        if lat_feats is None:
            skipped += 1
            continue
        rear_feats = extract_rear(img_rear, sticker_h_lateral=lat_feats["lat_sticker_h"])
        if lat_feats is None or rear_feats is None:
            skipped += 1
            continue

        rows.append({**lat_feats, **rear_feats})
        targets.append(wt)

    log.info("Amostras válidas: %d | puladas: %d", len(rows), skipped)
    if not rows:
        log.error("Nenhuma amostra. Verifique as imagens.")
        sys.exit(1)

    feat_names = list(rows[0].keys())
    X = np.array([[r[k] for k in feat_names] for r in rows], dtype=np.float32)
    y = np.array(targets, dtype=np.float32)

    with open(FEATURES_B4_OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(feat_names + ["weight_kg"])
        for r, wt in zip(rows, targets):
            w.writerow([r[k] for k in feat_names] + [wt])
    log.info("Features salvas em %s", FEATURES_B4_OUT)

    return X, y, feat_names


# ── Ablation ─────────────────────────────────────────────────────────────────

def ablation(X: np.ndarray, y: np.ndarray, feat_names: list[str]):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Índices das colunas por vista
    lat_idx  = [i for i, n in enumerate(feat_names) if n.startswith("lat_")]
    rear_idx = [i for i, n in enumerate(feat_names) if n.startswith("rear_")]

    def cv(idx):
        scores = cross_val_score(
            xgb.XGBRegressor(**XGB_PARAMS), X[:, idx], y, cv=kf, scoring="r2"
        )
        return float(scores.mean()), float(scores.std())

    r2_lat,  s_lat  = cv(lat_idx)
    r2_rear, s_rear = cv(rear_idx)
    r2_both, s_both = cv(lat_idx + rear_idx)

    log.info("=" * 55)
    log.info("ABLATION — impacto da vista traseira (5-fold CV)")
    log.info("=" * 55)
    log.info("Lateral apenas  (%2d feat):  R² = %.4f ± %.4f",
             len(lat_idx),  r2_lat,  s_lat)
    log.info("Traseira apenas (%2d feat):  R² = %.4f ± %.4f",
             len(rear_idx), r2_rear, s_rear)
    log.info("Lateral+Traseira(%2d feat):  R² = %.4f ± %.4f",
             len(lat_idx) + len(rear_idx), r2_both, s_both)
    log.info("Delta (ambas vs lateral): %+.4f", r2_both - r2_lat)

    # Feature importances — modelo combinado
    model = xgb.XGBRegressor(**XGB_PARAMS)
    all_idx = lat_idx + rear_idx
    model.fit(X[:, all_idx], y, verbose=False)
    imp = model.feature_importances_
    names_used = [feat_names[i] for i in all_idx]
    top = np.argsort(imp)[::-1][:15]
    log.info("\nTop 15 features (lateral + traseira):")
    for rank, i in enumerate(top, 1):
        log.info("  %2d. %-26s %.4f", rank, names_used[i], imp[i])

    return {"r2_lat": r2_lat, "r2_rear": r2_rear,
            "r2_both": r2_both, "delta": r2_both - r2_lat}


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("=" * 55)
    log.info("Fase 3 — Vista traseira B4 (lateral + rear)")
    log.info("=" * 55)

    n_side, n_rear = download_b4()
    log.info("B4/Side: %d | B4/Rear: %d arquivos", n_side, n_rear)

    X, y, feat_names = build_feature_matrix()
    log.info("Feature matrix: %s | Peso: %.1f–%.1f kg", X.shape, y.min(), y.max())

    results = ablation(X, y, feat_names)

    log.info("=" * 55)
    log.info("Fase 3 concluída.")
    log.info("  R² lateral apenas  = %.4f", results["r2_lat"])
    log.info("  R² traseira apenas = %.4f", results["r2_rear"])
    log.info("  R² lateral+traseira= %.4f", results["r2_both"])
    log.info("  Delta              = %+.4f", results["delta"])
    log.info("  Meta               = R² CV ≥ 0.65")
    log.info("=" * 55)
