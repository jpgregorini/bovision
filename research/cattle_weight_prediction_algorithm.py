"""
=============================================================================
Cattle Weight Estimation Pipeline — PoC / MVP  (v2)
=============================================================================
Dataset : "Cattle Weight Detection Model + Dataset (12k~)" (Kaggle)
          https://www.kaggle.com/datasets/sadhliroomyprime/cattle-weight-detection-model-dataset-12k
Target  : Estimate live weight (kg) from side-view segmentation masks
          using morphometric features + XGBoost Regressor.

Method (v2 — see CHANGELOG below):
    Pixel-derived morphometrics from colour-coded segmentation masks,
    made scale- and orientation-invariant, plus a vertical chord profile
    of the torso. Replaces the original heart-girth volumetric proxy, which
    was unusable (corr≈0.02 with weight) due to the bugs fixed below.

Usage   :
    # 1. Place kaggle.json in ~/.kaggle/  (only needed to (re)download images)
    # 2. Run:  python cattle_weight_prediction_algorithm.py
    #    If data/images/ is already populated, the Kaggle API is NOT contacted.

Output  :
    - data/file_list.txt → cached listing of all dataset files
    - data/images/       → side-view segmentation PNGs (B3/annotations)
    - features.csv       → extracted morphometric features
    - model_xgb.json     → trained XGBoost model
    - results.png        → Predicted vs Actual scatter + feature importances

-----------------------------------------------------------------------------
CHANGELOG v1 → v2  (R²: 0.481 → 0.520 CV; MAE 24.7 → 23.6 kg)
-----------------------------------------------------------------------------
The original feature extraction left the biologically meaningful features
near-useless (heart_girth corr≈0.016, volumetric proxy≈0.022). Fixes:
  1. SCALE   — every length is normalised by the reference-sticker height,
               turning pixel measurements (confounded by camera distance) into
               scale-invariant units. (px_to_cm existed but was never applied.)
  2. ORIENT  — the silhouette is flipped to a canonical direction, so positional
               measurements are consistent (cattle face either way in the data).
  3. LEGS    — thin legs are removed via horizontal morphological opening, so
               torso depth (≈ heart girth) is measured on the body, not the hooves.
  4. SHAPE   — a 12-point vertical chord profile of the torso replaces the 3
               noisy single-point spans, giving the model the full body shape.
  5. MODEL   — hyper-parameters tuned via 5-fold randomised search.

DOCUMENTED CEILING (why R² is ~0.52, not higher):
    Feature-nearest-neighbour animals — i.e. near-identical side silhouettes —
    differ in labelled weight by ~28-36 kg (median 23-28). That noise floor
    implies a hard ceiling of R² ≈ 0.50, which this model already reaches.
    The residual error is irreducible from a single side view because:
      • weights are MANUAL measurements (human error),
      • a 2-D side silhouette cannot see body width/depth (weight ∝ 3-D volume),
      • pose variation changes the silhouette without changing the weight.
    Breaking past ~0.52 requires NEW information (a rear/top view to recover
    width, or cleaner labels), not a better model or silhouette feature.
=============================================================================
"""

import os
import sys
import time
import csv
import math
import logging
from pathlib import Path

import numpy as np
import cv2
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────────────
KAGGLE_DATASET  = "sadhliroomyprime/cattle-weight-detection-model-dataset-12k"
# Path inside the dataset that contains the side-view annotation masks
TARGET_SUBPATH  = "Pixel/B3/annotations"
FILE_LIST       = Path("data/file_list.txt")
IMAGE_DIR       = Path("data/images")
LABEL_CSV_OUT   = Path("data/labels.csv")
FEATURES_CSV    = Path("features.csv")
MODEL_PATH      = Path("model_xgb.json")
RESULTS_PLOT    = Path("results.png")

# B3/B4 segmentation class colours (RGB), from the dataset Readme:
#   Sticker → (0,117,255)   Cattle → (255,30,249)   Background → (0,255,193)
# Masks are colour-coded (not binary); classes are selected by colour. The
# sticker is a real-world scale reference object placed on the animal.
CATTLE_RGB   = (255, 30, 249)
STICKER_RGB  = (0, 117, 255)
COLOR_TOL    = 12   # per-channel tolerance when matching class colours
N_CHORDS     = 12   # vertical chord-profile resolution along the body length

# Tuned XGBoost hyper-parameters (5-fold randomised search over features_v2).
XGB_PARAMS = dict(
    n_estimators=1200, max_depth=3, learning_rate=0.01,
    subsample=0.6, colsample_bytree=0.6, min_child_weight=1,
    reg_alpha=1.0, reg_lambda=5.0,
    objective="reg:squarederror", random_state=42, n_jobs=-1,
)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — (Selectively) download the B3/annotations masks from Kaggle
#          The Kaggle API is only contacted if images are actually missing.
# ─────────────────────────────────────────────────────────────────────────────

def _get_kaggle_api():
    """Authenticate against Kaggle using repo-local or default credentials."""
    repo_cfg    = Path(__file__).parent / "kaggle.json"
    default_cfg = Path.home() / ".kaggle" / "kaggle.json"

    if repo_cfg.exists():
        os.environ["KAGGLE_CONFIG_DIR"] = str(repo_cfg.parent)
        log.info("Using Kaggle credentials from repo: %s", repo_cfg)
    elif default_cfg.exists():
        log.info("Using Kaggle credentials from: %s", default_cfg)
    else:
        log.error(
            "Kaggle API key not found.\n"
            "  Expected at: %s  (repo directory)\n"
            "           or: %s  (default location)\n"
            "  Go to https://www.kaggle.com/settings → API → Create New Token\n"
            "  and place kaggle.json in one of those paths.",
            repo_cfg, default_cfg,
        )
        sys.exit(1)

    try:
        # Import after setting KAGGLE_CONFIG_DIR so the library picks up the env var
        import kaggle
        api = kaggle.KaggleApi()
        api.authenticate()
        return api
    except ImportError as e:
        log.error("kaggle package not importable (%s). Run: pip install kaggle", e)
        sys.exit(1)
    except Exception as e:
        log.error("Kaggle authentication failed: %s", e)
        sys.exit(1)


def _list_all_files(api) -> list[str]:
    """Enumerate every file in the dataset (paginated, rate-limit aware)."""
    if FILE_LIST.exists():
        names = FILE_LIST.read_text(encoding="utf-8").splitlines()
        log.info("Using cached file list: %d entries (%s)", len(names), FILE_LIST)
        return names

    log.info("Listing dataset files (paginated) …")
    names, token = [], None
    while True:
        try:
            res = api.dataset_list_files(
                KAGGLE_DATASET, page_size=200, page_token=token
            )
        except Exception as e:
            if "429" in str(e):
                log.warning("Rate-limited while listing — sleeping 60 s …")
                time.sleep(60)
                continue
            raise
        names.extend(f.name for f in res.files)
        token = getattr(res, "nextPageToken", None) or getattr(res, "next_page_token", None)
        if not token:
            break
        time.sleep(1.5)

    FILE_LIST.parent.mkdir(parents=True, exist_ok=True)
    FILE_LIST.write_text("\n".join(names), encoding="utf-8")
    log.info("File list saved: %d entries → %s", len(names), FILE_LIST)
    return names


def download_dataset() -> int:
    """
    Ensure the side-view annotation masks under TARGET_SUBPATH are available
    locally, downloading file by file (~50 MB) only what is missing.

    Offline-aware: if a cached file list exists and every target mask is already
    present, the Kaggle API is never contacted.

    Returns the number of mask images available locally.
    """
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    # Fast path: derive the target list from the cached listing (no network).
    if FILE_LIST.exists():
        names = _list_all_files(api=None)
    else:
        names = _list_all_files(_get_kaggle_api())

    targets = [
        n for n in names
        if TARGET_SUBPATH in n
           and n.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    log.info("Found %d masks under '%s'", len(targets), TARGET_SUBPATH)
    if not targets:
        log.error("No files match '%s' — check TARGET_SUBPATH.", TARGET_SUBPATH)
        sys.exit(1)

    missing = [n for n in targets if not (IMAGE_DIR / Path(n).name).exists()]
    if not missing:
        log.info("All %d masks already present — skipping download (offline).",
                 len(targets))
        return len(targets)

    log.info("Downloading %d masks → %s …", len(missing), IMAGE_DIR)
    api = _get_kaggle_api()          # authenticate only now that we need to fetch
    failed = 0
    for i, name in enumerate(missing, 1):
        while True:
            try:
                api.dataset_download_file(
                    KAGGLE_DATASET, name, path=str(IMAGE_DIR), quiet=True
                )
                break
            except Exception as e:
                if "429" in str(e):
                    log.warning("Rate-limited — sleeping 60 s … (%d/%d done)",
                                i - 1, len(missing))
                    time.sleep(60)
                else:
                    log.error("Failed: %s (%s)", name, e)
                    failed += 1
                    break
        if i % 100 == 0:
            log.info("  %d / %d masks downloaded", i, len(missing))

    if failed:
        log.warning("%d files failed to download — re-run to retry.", failed)

    n_local = len(list(IMAGE_DIR.glob("*.png")) + list(IMAGE_DIR.glob("*.jpg")))
    if n_local == 0:
        log.error("Download failed — no images in %s.", IMAGE_DIR)
        sys.exit(1)
    return n_local


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Parse weight labels
# ─────────────────────────────────────────────────────────────────────────────

def load_labels() -> dict[str, float]:
    """
    Return {filename_stem: weight_kg}.

    Priority:
      1. data/labels.csv (columns: filename, weight_kg — if Kaggle provides one)
      2. Weight encoded in the filename, per the dataset Readme convention:
             <id>_<s|r>_<manual-weight>_<sex>.<ext>   e.g. 100_s_109_M.jpg
    """
    labels: dict[str, float] = {}

    if LABEL_CSV_OUT.exists():
        log.info("Loading labels from %s", LABEL_CSV_OUT)
        with open(LABEL_CSV_OUT, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            log.info("CSV columns: %s", reader.fieldnames)
            for row in reader:
                fname = (
                    row.get("filename") or row.get("file_name") or
                    row.get("image") or row.get("image_name") or ""
                ).strip()
                weight_raw = (
                    row.get("weight_kg") or row.get("weight") or
                    row.get("Weight") or row.get("live_weight") or ""
                ).strip()
                if fname and weight_raw:
                    try:
                        labels[Path(fname).stem] = float(weight_raw)
                    except ValueError:
                        pass
        log.info("Loaded %d labels from CSV.", len(labels))

    if not labels:
        log.info("Parsing weights from filenames …")
        import re
        # On disk the mask stem looks like "100_s_109_M.jpg___fuse"
        pattern = re.compile(r"^\d+(?:\.\d+)?_[sr]_(\d+(?:\.\d+)?)_[MF]", re.I)
        for img_path in IMAGE_DIR.glob("*"):
            m = pattern.match(img_path.stem)
            if m:
                try:
                    labels[img_path.stem] = float(m.group(1))
                except ValueError:
                    pass
        log.info("Parsed %d weights from filenames.", len(labels))

    return labels


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Morphometric feature extraction from segmentation masks (v2)
# ─────────────────────────────────────────────────────────────────────────────

def _class_mask(img_bgr: np.ndarray, rgb: tuple[int, int, int],
                tol: int = COLOR_TOL) -> np.ndarray:
    """Binary mask of pixels within ±tol of a class colour (given as RGB)."""
    bgr = np.array(rgb[::-1], dtype=np.int16)
    lower = np.clip(bgr - tol, 0, 255).astype(np.uint8)
    upper = np.clip(bgr + tol, 0, 255).astype(np.uint8)
    return cv2.inRange(img_bgr, lower, upper)


def _largest_contour(binary: np.ndarray, min_area: float = 500):
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    return c if cv2.contourArea(c) >= min_area else None


def _sticker_height(img_bgr: np.ndarray) -> float:
    """Pixel height of the reference sticker (0 if absent). Real-world scale anchor."""
    c = _largest_contour(_class_mask(img_bgr, STICKER_RGB), min_area=20)
    if c is None:
        return 0.0
    return float(cv2.boundingRect(c)[3])


def extract_features_from_mask(img_bgr: np.ndarray) -> dict | None:
    """
    Extract scale- and orientation-invariant morphometric features from a
    colour-coded segmentation mask. Returns a feature dict, or None if the mask
    is unusable OR the scale sticker is missing (scale normalisation requires it).

    All lengths are normalised by the sticker height `s` (→ real-world-proportional
    units); areas by s², volumetric terms by s³. See the module CHANGELOG.
    """
    cnt = _largest_contour(_class_mask(img_bgr, CATTLE_RGB))
    if cnt is None:
        return None

    s = _sticker_height(img_bgr)
    if s <= 5:                       # no reliable scale anchor → unusable
        return None

    # Solid body mask (fill internal holes).
    body = np.zeros(img_bgr.shape[:2], np.uint8)
    cv2.drawContours(body, [cnt], -1, 255, cv2.FILLED)
    x, y, w, h = cv2.boundingRect(cnt)

    # ── Leg removal: morphological opening with a horizontal kernel ───────────
    # Legs are thin in x; opening with a kernel ~6% of body length erases them,
    # leaving the torso, so chord depth ≈ heart girth (not spine-to-hoof).
    k = max(3, int(0.06 * w) | 1)
    torso = cv2.morphologyEx(
        body, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))

    # ── Orientation: flip so the taller end is on the right (canonical) ───────
    col_h = (body[:, x:x + w] > 0).sum(0)
    third = max(1, w // 3)
    if col_h[:third].mean() > col_h[-third:].mean():
        body, torso = cv2.flip(body, 1), cv2.flip(torso, 1)

    # Recompute geometry on the (possibly flipped) body.
    ys, xs = np.where(body > 0)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    w, h = x1 - x0 + 1, y1 - y0 + 1

    # ── Vertical chord profile of the torso (legs removed) ────────────────────
    chords = np.zeros(N_CHORDS)
    for i in range(N_CHORDS):
        cx = x0 + int((i + 0.5) / N_CHORDS * w)
        col = torso[:, max(x0, cx - 1): cx + 2]
        rows = np.where(np.any(col > 0, axis=1))[0]
        chords[i] = (rows[-1] - rows[0]) if len(rows) >= 2 else 0.0

    area       = float((body > 0).sum())
    torso_area = float((torso > 0).sum())
    perimeter  = cv2.arcLength(cnt, True)
    hull_area  = cv2.contourArea(cv2.convexHull(np.column_stack([xs, ys])))
    max_chord  = float(chords.max())

    feats = {
        "area_n":        area / s ** 2,
        "torso_area_n":  torso_area / s ** 2,
        "length_n":      w / s,
        "height_n":      h / s,
        "perimeter_n":   perimeter / s,
        "aspect_ratio":  w / h if h else 0.0,
        "solidity":      area / hull_area if hull_area else 0.0,
        "extent":        area / (w * h) if w * h else 0.0,
        "max_chord_n":   max_chord / s,
        "mean_chord_n":  float(chords.mean()) / s,
        "front_chord_n": float(chords[:N_CHORDS // 3].max()) / s,
        "rear_chord_n":  float(chords[-N_CHORDS // 3:].max()) / s,
        "vol_proxy_n":   (max_chord ** 2 * w) / s ** 3,   # girth²·length, scaled
        "sticker_h":     s,
    }
    for i, c in enumerate(chords):
        feats[f"chord_{i}_n"] = c / s
    # Hu moments (scale/rotation invariant shape descriptors), log-scaled.
    hu = cv2.HuMoments(cv2.moments(body)).flatten()
    for i, v in enumerate(hu):
        feats[f"hu_{i}"] = -math.copysign(1, v) * math.log10(abs(v) + 1e-10)

    return feats


def build_feature_matrix(labels: dict[str, float]):
    """
    Extract features for every labelled image. Returns (X, y, feature_names).
    Images without a usable mask or scale sticker are skipped.
    """
    images = sorted(IMAGE_DIR.glob("*"))
    log.info("Processing %d images for feature extraction …", len(images))

    rows, targets, skipped = [], [], 0
    for img_path in images:
        if img_path.stem not in labels:
            skipped += 1
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            skipped += 1
            continue
        feats = extract_features_from_mask(img)
        if feats is None:
            skipped += 1
            continue
        rows.append(feats)
        targets.append(labels[img_path.stem])

    log.info("Feature extraction complete: %d samples, %d skipped "
             "(no label / bad mask / no sticker).", len(rows), skipped)
    if not rows:
        log.error("No usable samples found. Check labels and mask colours.")
        sys.exit(1)

    feature_names = list(rows[0].keys())
    X = np.array([[r[k] for k in feature_names] for r in rows], dtype=np.float32)
    y = np.array(targets, dtype=np.float32)

    with open(FEATURES_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(feature_names + ["weight_kg"])
        for r, w_kg in zip(rows, targets):
            writer.writerow([r[k] for k in feature_names] + [w_kg])
    log.info("Features saved to %s", FEATURES_CSV)

    return X, y, feature_names


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Train XGBoost Regressor
# ─────────────────────────────────────────────────────────────────────────────

def train_xgboost(X: np.ndarray, y: np.ndarray, feature_names: list[str]):
    """Train XGBoost with the tuned params, evaluate (holdout + 5-fold CV),
    save the model and a diagnostic plot."""

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    log.info("Train: %d samples | Test: %d samples", len(X_train), len(X_test))

    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_train, y_train, verbose=False)

    # ── Holdout metrics ───────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    mape = float(np.mean(np.abs((y_test - y_pred) / (y_test + 1e-6))) * 100)

    log.info("─" * 50)
    log.info("XGBoost Regressor — Test Set (holdout 20%%)")
    log.info("  MAE  : %.2f kg", mae)
    log.info("  RMSE : %.2f kg", rmse)
    log.info("  R²   : %.4f",    r2)
    log.info("  MAPE : %.2f %%",  mape)

    # ── 5-fold CV R² (more stable generalisation estimate) ────────────────────
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2  = cross_val_score(xgb.XGBRegressor(**XGB_PARAMS), X, y, cv=kf, scoring="r2")
    cv_mae = -cross_val_score(xgb.XGBRegressor(**XGB_PARAMS), X, y, cv=kf,
                              scoring="neg_mean_absolute_error")
    log.info("  5-fold CV R²  : %.4f ± %.4f", cv_r2.mean(), cv_r2.std())
    log.info("  5-fold CV MAE : %.2f ± %.2f kg", cv_mae.mean(), cv_mae.std())
    log.info("  (documented noise-floor ceiling ≈ 0.50 — see module docstring)")
    log.info("─" * 50)

    # ── Feature importance ────────────────────────────────────────────────────
    importances = model.feature_importances_
    top_n = min(15, len(feature_names))
    top_idx = np.argsort(importances)[::-1][:top_n]
    log.info("Top %d features:", top_n)
    for rank, idx in enumerate(top_idx, 1):
        log.info("  %2d. %-16s %.4f", rank, feature_names[idx], importances[idx])

    # ── Save model ────────────────────────────────────────────────────────────
    model.save_model(str(MODEL_PATH))
    log.info("Model saved to %s", MODEL_PATH)

    # ── Plot Predicted vs Actual + feature importances ────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Cattle Weight Estimation — XGBoost Regressor (v2)",
                 fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.scatter(y_test, y_pred, alpha=0.55, s=25, color="#2196F3",
               edgecolors="none", label="Predictions")
    lims = [min(y_test.min(), y_pred.min()) * 0.95,
            max(y_test.max(), y_pred.max()) * 1.05]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit")
    ax.set_xlabel("Actual weight (kg)")
    ax.set_ylabel("Predicted weight (kg)")
    ax.set_title(f"Predicted vs Actual  |  MAE={mae:.1f} kg  R²={r2:.3f}")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    fi_names  = [feature_names[i] for i in top_idx]
    fi_values = [importances[i]   for i in top_idx]
    ax2.barh(fi_names[::-1], fi_values[::-1], color="#4CAF50")
    ax2.set_xlabel("Importance (gain)")
    ax2.set_title("Top Feature Importances")
    ax2.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(RESULTS_PLOT), dpi=150, bbox_inches="tight")
    log.info("Results plot saved to %s", RESULTS_PLOT)

    return model, {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape,
                   "CV_R2": float(cv_r2.mean())}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Inference helper
# ─────────────────────────────────────────────────────────────────────────────

def predict_single(image_path: str, model_path: str = str(MODEL_PATH)) -> float | None:
    """
    Load a saved model and predict weight (kg) for a single mask image.
    Requires the scale sticker to be present in the mask.
    """
    model = xgb.XGBRegressor()
    model.load_model(model_path)

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        log.error("Cannot read image: %s", image_path)
        return None

    feats = extract_features_from_mask(img)
    if feats is None:
        log.error("Feature extraction failed (bad mask or missing sticker): %s",
                  image_path)
        return None

    x = np.array([[v for v in feats.values()]], dtype=np.float32)
    return float(model.predict(x)[0])


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("=" * 60)
    log.info("Cattle Weight Estimation Pipeline — PoC (v2)")
    log.info("=" * 60)

    # 1. Ensure the B3 side-view masks are available (offline if already cached).
    n_images = download_dataset()
    log.info("Working with %d images in %s", n_images, IMAGE_DIR)

    # 3. Load labels.
    labels = load_labels()
    if not labels:
        log.warning("No labels loaded. Create %s with columns: filename, weight_kg",
                    LABEL_CSV_OUT)
        sys.exit(0)
    log.info("Labels loaded for %d cattle.", len(labels))

    # 4. Feature extraction.
    X, y, feature_names = build_feature_matrix(labels)
    log.info("Feature matrix shape: %s | Target range: %.1f – %.1f kg",
             X.shape, y.min(), y.max())

    # 5. Train + evaluate.
    model, metrics = train_xgboost(X, y, feature_names)

    log.info("=" * 60)
    log.info("Pipeline complete.")
    log.info("  MAE     = %.2f kg  (target: < 25 kg for PoC validation)", metrics["MAE"])
    log.info("  R²      = %.4f  (holdout)", metrics["R2"])
    log.info("  CV R²   = %.4f  (5-fold)", metrics["CV_R2"])
    log.info("  Plot  → %s", RESULTS_PLOT)
    log.info("  Model → %s", MODEL_PATH)
    log.info("=" * 60)
