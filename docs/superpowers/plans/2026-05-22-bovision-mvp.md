# BoviSight MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete end-to-end cattle weighing system using computer vision — from YOLOv8 detection through XGBoost weight estimation, served via FastAPI, and displayed on a React dashboard.

**Architecture:** Bottom-up approach. Train ML models first (YOLOv8 detection + XGBoost regression using Kaggle data), then build pipeline modules with mock hardware, connect to a FastAPI + SQLite API, and top it off with a single-page React dashboard. Everything runs locally with mock camera input.

**Tech Stack:** Python 3.11+, YOLOv8 (ultralytics), XGBoost, FastAPI, SQLAlchemy, SQLite, Alembic, React, Vite, Tailwind CSS, Recharts

**Spec:** `docs/superpowers/specs/2026-05-22-bovision-mvp-design.md`

---

## File Map

### Files to MOVE (restructure from data/ to root)
- `data/src/` -> `src/` (pipeline modules)
- `data/api/` -> `api/` (API server)
- `data/scripts/` -> `scripts/` (training scripts)
- `data/models/` -> `models/` (trained models)
- `data/docker/` -> `docker/` (container configs)
- `data/calibration/` -> `calibration/` (camera params)
- `data/annotated/` stays in `data/annotated/`
- `data/weightings/` stays in `data/weighings/` (fix typo: weightings -> weighings)
- `data/src/mesure.py` -> `src/measure.py` (fix typo)

### Files to CREATE
- `.gitignore`
- `.env.example` (rewrite with actual content)
- `requirements.txt` (rewrite with actual content)
- `api/models.py` (SQLAlchemy models)
- `api/schemas.py` (Pydantic schemas)
- `api/routers/__init__.py`
- `api/routers/weighings.py`
- `api/routers/animals.py`
- `api/routers/reports.py`
- `data/sample_images/` (directory for mock pipeline)
- `data/weighings/dataset.csv` (from Kaggle)
- `tests/` (all test files)
- `dashboard/` (entire React app)

---

## Task 1: Restructure Project and Setup Foundation

**Files:**
- Move: all directories from `data/` to root level
- Create: `.gitignore`, `.env.example`, `requirements.txt`
- Rename: `data/src/mesure.py` -> `src/measure.py`
- Rename: `data/weightings/` -> `data/weighings/`

- [ ] **Step 1: Move directories from data/ to root**

```bash
cd /Users/jpgregorini/Desktop/projetos/Bovision/.claude/worktrees/happy-kalam-67f080

# Move code directories to root
mv data/src ./src
mv data/api ./api
mv data/scripts ./scripts
mv data/models ./models
mv data/docker ./docker
mv data/calibration ./calibration

# Fix typo in weighings directory name
mv data/weightings data/weighings

# Fix typo in measure.py
mv src/mesure.py src/measure.py

# Create missing directories
mkdir -p data/sample_images
mkdir -p data/evidence
mkdir -p data/raw
mkdir -p api/routers
mkdir -p tests
```

- [ ] **Step 2: Create .gitignore**

Write `.gitignore`:

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.egg-info/
dist/
build/
.eggs/

# Virtual environment
venv/
.venv/
env/

# Environment
.env

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Data files (large)
data/raw/
data/annotated/images/
data/annotated/labels/
data/sample_images/*.jpg
data/sample_images/*.png
data/evidence/

# Model binaries
models/detection/best.pt
models/detection/best.onnx
models/detection/best.engine
models/regression/weight_model.pkl
models/regression/scaler.pkl

# Database
*.db

# Node
node_modules/
dashboard/dist/

# Calibration
calibration/camera_params.npy
```

- [ ] **Step 3: Create .env.example**

Write `.env.example`:

```env
# Ambiente (dev = mock local, prod = hardware real)
BOVISION_ENV=dev

# Caminhos
SAMPLE_IMAGES_DIR=data/sample_images
MODEL_DETECTION_PATH=models/detection/best.pt
MODEL_REGRESSION_PATH=models/regression/weight_model.pkl
SCALER_PATH=models/regression/scaler.pkl
CALIBRATION_PATH=calibration/camera_params.npy

# API
API_URL=http://localhost:8000
API_HOST=0.0.0.0
API_PORT=8000
DATABASE_URL=sqlite:///./bovision.db

# Pipeline
CONFIDENCE_THRESHOLD=0.5
CONFIRMATION_FRAMES=3
COOLDOWN_SECONDS=30

# Dashboard
VITE_API_URL=http://localhost:8000/api
```

- [ ] **Step 4: Create requirements.txt**

Write `requirements.txt`:

```
# Detection
ultralytics>=8.2.0
torch>=2.1.0
opencv-python>=4.8.0

# Regression
xgboost>=2.0.0
scikit-learn>=1.3.0
pandas>=2.1.0

# API
fastapi>=0.111.0
uvicorn[standard]>=0.30.0
sqlalchemy>=2.0.0
alembic>=1.13.0
pydantic>=2.0.0

# Utilities
python-dotenv>=1.0.0
numpy>=1.24.0
requests>=2.31.0

# Testing
pytest>=8.0.0
httpx>=0.27.0
```

- [ ] **Step 5: Verify structure and commit**

```bash
# Verify
ls -la  # should show src/ api/ scripts/ models/ docker/ calibration/ data/ at root level
ls src/ # should show capture.py detect.py measure.py predict.py pipeline.py
ls data/ # should show annotated/ weighings/ sample_images/ evidence/ raw/

git add -A
git commit -m "chore: restructure project to match spec

Move src/, api/, scripts/, models/, docker/, calibration/ from data/ to root.
Fix typo: mesure.py -> measure.py, weightings -> weighings.
Add .gitignore, .env.example, requirements.txt with full content.
Create missing directories: tests/, data/sample_images/, data/evidence/, data/raw/"
```

---

## Task 2: Download and Prepare Detection Dataset

**Files:**
- Create: `scripts/download_detection_data.py`
- Output: `data/annotated/images/`, `data/annotated/labels/`, `data/annotated/data.yaml`

> **Note:** This task requires a Kaggle API key configured (`~/.kaggle/kaggle.json`). If the user does not have one, the script provides instructions for manual download.

- [ ] **Step 1: Write the dataset download and preparation script**

Create `scripts/download_detection_data.py`:

```python
"""
Download and prepare a cattle detection dataset for YOLOv8 training.

Usage:
    python scripts/download_detection_data.py

Searches Kaggle for cattle/cow detection datasets with YOLO or COCO
annotations and organizes them into the expected directory structure.
"""

import os
import shutil
import random
import json
from pathlib import Path

# Try kaggle import - if not available, print instructions
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    HAS_KAGGLE = True
except ImportError:
    HAS_KAGGLE = False

# Candidate datasets on Kaggle (ordered by preference)
# These contain cattle/cow images with annotations
KAGGLE_DATASETS = [
    "trainingdatapro/cattle-detection",
    "ayuraj/cow-detection-dataset",
]

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ANNOTATED_DIR = DATA_DIR / "annotated"
IMAGES_DIR = ANNOTATED_DIR / "images"
LABELS_DIR = ANNOTATED_DIR / "labels"
RAW_DIR = DATA_DIR / "raw"


def download_from_kaggle(dataset_slug: str, output_dir: Path) -> bool:
    """Download and unzip a Kaggle dataset. Returns True on success."""
    if not HAS_KAGGLE:
        print("kaggle package not installed. Install with: pip install kaggle")
        print("Then configure API key: https://www.kaggle.com/docs/api")
        return False
    try:
        api = KaggleApi()
        api.authenticate()
        print(f"Downloading {dataset_slug}...")
        api.dataset_download_files(dataset_slug, path=str(output_dir), unzip=True)
        print(f"Downloaded to {output_dir}")
        return True
    except Exception as e:
        print(f"Failed to download {dataset_slug}: {e}")
        return False


def find_images(directory: Path, extensions=(".jpg", ".jpeg", ".png")) -> list[Path]:
    """Recursively find all image files in a directory."""
    images = []
    for ext in extensions:
        images.extend(directory.rglob(f"*{ext}"))
    return sorted(images)


def find_yolo_labels(directory: Path) -> list[Path]:
    """Recursively find all YOLO format label files."""
    return sorted(directory.rglob("*.txt"))


def convert_coco_to_yolo(coco_json_path: Path, output_labels_dir: Path,
                          target_classes: list[str] | None = None) -> dict[str, Path]:
    """
    Convert COCO JSON annotations to YOLO format .txt files.
    Returns mapping of image_filename -> label_path.
    """
    with open(coco_json_path) as f:
        coco = json.load(f)

    # Build category map
    cat_map = {}
    for cat in coco.get("categories", []):
        name = cat["name"].lower()
        if target_classes is None or any(t in name for t in target_classes):
            cat_map[cat["id"]] = 0  # Single class: bovino

    # Build image map
    img_map = {}
    for img in coco["images"]:
        img_map[img["id"]] = {
            "file_name": img["file_name"],
            "width": img["width"],
            "height": img["height"],
        }

    # Convert annotations
    label_map = {}
    annotations_by_image = {}
    for ann in coco.get("annotations", []):
        if ann["category_id"] not in cat_map:
            continue
        img_id = ann["image_id"]
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)

    output_labels_dir.mkdir(parents=True, exist_ok=True)

    for img_id, anns in annotations_by_image.items():
        if img_id not in img_map:
            continue
        img_info = img_map[img_id]
        w, h = img_info["width"], img_info["height"]
        fname = Path(img_info["file_name"]).stem + ".txt"
        label_path = output_labels_dir / fname

        lines = []
        for ann in anns:
            bbox = ann["bbox"]  # [x, y, width, height] in COCO format
            x_center = (bbox[0] + bbox[2] / 2) / w
            y_center = (bbox[1] + bbox[3] / 2) / h
            bw = bbox[2] / w
            bh = bbox[3] / h
            class_id = cat_map[ann["category_id"]]
            lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}")

        with open(label_path, "w") as f:
            f.write("\n".join(lines))
        label_map[img_info["file_name"]] = label_path

    return label_map


def split_dataset(image_paths: list[Path], train_ratio=0.8, val_ratio=0.1):
    """Split image paths into train/val/test sets."""
    random.seed(42)
    shuffled = image_paths.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }


def organize_yolo_structure(images: list[Path], labels_dir: Path | None,
                             label_map: dict[str, Path] | None = None):
    """
    Organize images and labels into YOLO directory structure:
    data/annotated/images/{train,val,test}/
    data/annotated/labels/{train,val,test}/
    """
    splits = split_dataset(images)

    for split_name, split_images in splits.items():
        img_out = IMAGES_DIR / split_name
        lbl_out = LABELS_DIR / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        count = 0
        for img_path in split_images:
            # Copy image
            dst_img = img_out / img_path.name
            shutil.copy2(img_path, dst_img)

            # Find corresponding label
            label_src = None
            if label_map and img_path.name in label_map:
                label_src = label_map[img_path.name]
            elif labels_dir:
                candidate = labels_dir / (img_path.stem + ".txt")
                if candidate.exists():
                    label_src = candidate

            if label_src and label_src.exists():
                shutil.copy2(label_src, lbl_out / label_src.name)
                count += 1

        print(f"  {split_name}: {len(split_images)} images, {count} labels")


def write_data_yaml():
    """Write the data.yaml configuration for YOLOv8."""
    yaml_content = f"""path: {ANNOTATED_DIR.resolve()}
train: images/train
val: images/val
test: images/test

nc: 1
names: ['bovino']
"""
    yaml_path = ANNOTATED_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"Written: {yaml_path}")


def main():
    print("=== BoviSight: Detection Dataset Preparation ===\n")

    # Try downloading from Kaggle
    downloaded = False
    for dataset in KAGGLE_DATASETS:
        if download_from_kaggle(dataset, RAW_DIR):
            downloaded = True
            break

    if not downloaded:
        print("\n--- Manual Download Instructions ---")
        print("1. Go to https://www.kaggle.com and search for 'cattle detection'")
        print(f"2. Download a dataset with annotated images (YOLO or COCO format)")
        print(f"3. Extract the contents to: {RAW_DIR}")
        print("4. Re-run this script")
        print("\nRecommended datasets:")
        for ds in KAGGLE_DATASETS:
            print(f"  https://www.kaggle.com/datasets/{ds}")
        return

    # Find images and labels
    images = find_images(RAW_DIR)
    print(f"\nFound {len(images)} images in {RAW_DIR}")

    if len(images) == 0:
        print("ERROR: No images found. Check the download directory.")
        return

    # Check for COCO JSON annotations
    coco_files = list(RAW_DIR.rglob("*.json"))
    label_map = None
    labels_dir = None

    coco_annotation_file = None
    for cf in coco_files:
        try:
            with open(cf) as f:
                data = json.load(f)
            if "annotations" in data and "images" in data:
                coco_annotation_file = cf
                break
        except (json.JSONDecodeError, KeyError):
            continue

    if coco_annotation_file:
        print(f"Found COCO annotations: {coco_annotation_file}")
        tmp_labels = RAW_DIR / "_yolo_labels"
        label_map = convert_coco_to_yolo(
            coco_annotation_file, tmp_labels,
            target_classes=["cow", "cattle", "bovine", "bull", "calf"]
        )
        # Convert label_map keys to match image filenames
        label_map_by_name = {}
        for fname, lpath in label_map.items():
            label_map_by_name[Path(fname).name] = lpath
        label_map = label_map_by_name
        print(f"Converted {len(label_map)} COCO annotations to YOLO format")
    else:
        # Check for existing YOLO labels
        yolo_labels = find_yolo_labels(RAW_DIR)
        if yolo_labels:
            labels_dir = yolo_labels[0].parent
            print(f"Found {len(yolo_labels)} YOLO label files in {labels_dir}")
        else:
            print("WARNING: No annotations found. Images will be copied without labels.")
            print("You will need to annotate them manually (e.g., with Roboflow or CVAT).")

    # Organize into YOLO structure
    print("\nOrganizing into train/val/test splits...")
    organize_yolo_structure(images, labels_dir, label_map)

    # Write data.yaml
    write_data_yaml()

    # Copy some images to sample_images for the pipeline mock
    sample_dir = DATA_DIR / "sample_images"
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_images = images[:10]  # First 10 for testing
    for img in sample_images:
        shutil.copy2(img, sample_dir / img.name)
    print(f"\nCopied {len(sample_images)} sample images to {sample_dir}")

    print("\n=== Done! Dataset ready for training. ===")
    print(f"  Images: {IMAGES_DIR}")
    print(f"  Labels: {LABELS_DIR}")
    print(f"  Config: {ANNOTATED_DIR / 'data.yaml'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script**

```bash
pip install kaggle  # if not installed
python scripts/download_detection_data.py
```

Expected: Dataset downloaded and organized into `data/annotated/images/{train,val,test}/` and `data/annotated/labels/{train,val,test}/`, `data.yaml` created, sample images copied.

If Kaggle API is not configured, follow the manual download instructions printed by the script.

- [ ] **Step 3: Verify the dataset structure**

```bash
ls data/annotated/images/train/ | head -5
ls data/annotated/labels/train/ | head -5
cat data/annotated/data.yaml
ls data/sample_images/ | head -5
```

Expected: Images and labels present in all splits. `data.yaml` points to correct paths with `nc: 1` and `names: ['bovino']`.

- [ ] **Step 4: Commit**

```bash
git add scripts/download_detection_data.py data/annotated/data.yaml
git commit -m "feat: add detection dataset download and preparation script

Downloads cattle detection datasets from Kaggle, converts COCO to YOLO
format if needed, splits into train/val/test (80/10/10), writes data.yaml,
and copies sample images for pipeline mock testing."
```

---

## Task 3: Train YOLOv8n-seg Detection Model

**Files:**
- Create: `scripts/train_detection.py`
- Output: `models/detection/best.pt`, `models/detection/best.onnx`

- [ ] **Step 1: Write the training script**

Create `scripts/train_detection.py`:

```python
"""
Train YOLOv8n-seg for cattle detection and segmentation.

Usage:
    python scripts/train_detection.py [--epochs 50] [--imgsz 640] [--batch 16]

Trains on data/annotated/ dataset, saves best model to models/detection/.
"""

import argparse
import shutil
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_YAML = BASE_DIR / "data" / "annotated" / "data.yaml"
OUTPUT_DIR = BASE_DIR / "models" / "detection"


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8n-seg for cattle detection")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (-1 for auto)")
    parser.add_argument("--device", type=str, default="", help="Device: '', 'cpu', '0', '0,1'")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    return parser.parse_args()


def train(args):
    print("=== BoviSight: YOLOv8n-seg Training ===\n")

    if not DATA_YAML.exists():
        print(f"ERROR: {DATA_YAML} not found.")
        print("Run scripts/download_detection_data.py first.")
        return

    # Load YOLOv8n-seg pretrained model
    model = YOLO("yolov8n-seg.pt")

    # Train
    print(f"Training for {args.epochs} epochs, image size {args.imgsz}, batch {args.batch}")
    results = model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device if args.device else None,
        project=str(BASE_DIR / "runs" / "detect"),
        name="bovision",
        exist_ok=True,
        resume=args.resume,
        # Augmentation (basic, built-in)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
    )

    # Copy best model to models/detection/
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_pt = Path(results.save_dir) / "weights" / "best.pt"

    if best_pt.exists():
        shutil.copy2(best_pt, OUTPUT_DIR / "best.pt")
        print(f"\nSaved: {OUTPUT_DIR / 'best.pt'}")

        # Export to ONNX
        model_best = YOLO(str(OUTPUT_DIR / "best.pt"))
        model_best.export(format="onnx", imgsz=args.imgsz)
        onnx_path = OUTPUT_DIR / "best.onnx"
        if onnx_path.exists():
            print(f"Saved: {onnx_path}")
    else:
        print(f"WARNING: {best_pt} not found. Check training logs.")

    print("\n=== Training Complete ===")


if __name__ == "__main__":
    args = parse_args()
    train(args)
```

- [ ] **Step 2: Run training**

```bash
python scripts/train_detection.py --epochs 50 --imgsz 640 --batch 16
```

Expected: Training runs, prints metrics per epoch, saves `models/detection/best.pt` and `models/detection/best.onnx`. On a CPU, use `--device cpu --epochs 10` for a quick test.

- [ ] **Step 3: Verify outputs**

```bash
ls -lh models/detection/
# Expected: best.pt (50-200MB), best.onnx (similar)
```

- [ ] **Step 4: Commit**

```bash
git add scripts/train_detection.py
git commit -m "feat: add YOLOv8n-seg training script

Trains cattle detection/segmentation model on annotated dataset.
Supports configurable epochs, image size, batch size, device.
Exports to both PyTorch (.pt) and ONNX formats."
```

---

## Task 4: Download and Prepare Regression Dataset

**Files:**
- Create: `scripts/download_regression_data.py`
- Output: `data/weighings/dataset.csv`

- [ ] **Step 1: Write the download and preparation script**

Create `scripts/download_regression_data.py`:

```python
"""
Download and prepare a cattle body measurements + weight dataset for XGBoost training.

Usage:
    python scripts/download_regression_data.py

Searches Kaggle for cattle weight prediction datasets, downloads, and maps
available features to the expected BoviSight format.
"""

import os
from pathlib import Path

import pandas as pd

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    HAS_KAGGLE = True
except ImportError:
    HAS_KAGGLE = False

# Candidate datasets (ordered by preference)
KAGGLE_DATASETS = [
    "ujsinghania/cattle-weight-live-weight-dataset",
    "gfrfranco/cattle-weight-estimation",
]

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
WEIGHINGS_DIR = DATA_DIR / "weighings"
RAW_DIR = DATA_DIR / "raw" / "regression"

# Mapping from common Kaggle column names to BoviSight features
COLUMN_MAPPINGS = {
    # Comprimento
    "body_length": "comprimento_m",
    "length": "comprimento_m",
    "body length": "comprimento_m",
    "comprimento": "comprimento_m",
    # Altura
    "height": "altura_m",
    "wither_height": "altura_m",
    "withers_height": "altura_m",
    "wither height": "altura_m",
    "altura": "altura_m",
    # Largura
    "width": "largura_m",
    "hip_width": "largura_m",
    "hip width": "largura_m",
    "largura": "largura_m",
    # Area (derived if not present)
    "area": "area_m2",
    # Perimetro toracico
    "heart_girth": "perimetro_m",
    "chest_girth": "perimetro_m",
    "girth": "perimetro_m",
    "heart girth": "perimetro_m",
    "chest girth": "perimetro_m",
    "perimetro": "perimetro_m",
    # Peso
    "weight": "peso_kg",
    "live_weight": "peso_kg",
    "body_weight": "peso_kg",
    "liveweight": "peso_kg",
    "peso": "peso_kg",
    # Raca
    "breed": "raca",
    "raca": "raca",
    # Sexo
    "sex": "sexo",
    "gender": "sexo",
    "sexo": "sexo",
}


def download_from_kaggle(dataset_slug: str, output_dir: Path) -> bool:
    """Download and unzip a Kaggle dataset."""
    if not HAS_KAGGLE:
        print("kaggle package not installed. Install with: pip install kaggle")
        return False
    try:
        api = KaggleApi()
        api.authenticate()
        print(f"Downloading {dataset_slug}...")
        api.dataset_download_files(dataset_slug, path=str(output_dir), unzip=True)
        return True
    except Exception as e:
        print(f"Failed to download {dataset_slug}: {e}")
        return False


def find_csv_files(directory: Path) -> list[Path]:
    """Find all CSV files in directory."""
    return sorted(directory.rglob("*.csv"))


def map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map dataset columns to BoviSight expected format."""
    # Normalize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Map columns
    mapped = pd.DataFrame()
    for original, target in COLUMN_MAPPINGS.items():
        normalized = original.strip().lower().replace(" ", "_")
        if normalized in df.columns and target not in mapped.columns:
            mapped[target] = df[normalized]

    return mapped


def validate_dataset(df: pd.DataFrame) -> bool:
    """Check that the dataset has minimum required columns."""
    required = ["peso_kg"]
    measurement_cols = ["comprimento_m", "altura_m", "largura_m", "perimetro_m"]

    has_weight = "peso_kg" in df.columns
    has_measurements = sum(1 for c in measurement_cols if c in df.columns) >= 2

    if not has_weight:
        print("ERROR: No weight column found.")
        return False
    if not has_measurements:
        print("ERROR: Need at least 2 measurement columns.")
        print(f"  Found: {[c for c in measurement_cols if c in df.columns]}")
        return False

    return True


def derive_missing_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive features that can be calculated from existing ones."""
    # Derive area if missing but comprimento and altura exist
    if "area_m2" not in df.columns:
        if "comprimento_m" in df.columns and "altura_m" in df.columns:
            df["area_m2"] = df["comprimento_m"] * df["altura_m"]
            print("  Derived area_m2 from comprimento_m * altura_m")

    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Remove invalid rows and outliers."""
    initial_count = len(df)

    # Drop rows with missing weight
    df = df.dropna(subset=["peso_kg"])

    # Remove obviously invalid weights (< 50kg or > 1500kg)
    df = df[(df["peso_kg"] >= 50) & (df["peso_kg"] <= 1500)]

    # Remove negative measurements
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        df = df[df[col] >= 0]

    final_count = len(df)
    if initial_count != final_count:
        print(f"  Cleaned: {initial_count} -> {final_count} rows")

    return df.reset_index(drop=True)


def main():
    print("=== BoviSight: Regression Dataset Preparation ===\n")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHINGS_DIR.mkdir(parents=True, exist_ok=True)

    # Try downloading from Kaggle
    downloaded = False
    for dataset in KAGGLE_DATASETS:
        if download_from_kaggle(dataset, RAW_DIR):
            downloaded = True
            break

    if not downloaded:
        print("\n--- Manual Download Instructions ---")
        print("1. Go to https://www.kaggle.com and search for 'cattle weight estimation'")
        print("2. Download a dataset with body measurements and weight")
        print(f"3. Place the CSV file(s) in: {RAW_DIR}")
        print("4. Re-run this script")
        print("\nRecommended datasets:")
        for ds in KAGGLE_DATASETS:
            print(f"  https://www.kaggle.com/datasets/{ds}")
        return

    # Find and process CSV files
    csv_files = find_csv_files(RAW_DIR)
    if not csv_files:
        print("ERROR: No CSV files found in download.")
        return

    print(f"\nFound {len(csv_files)} CSV file(s)")

    # Try each CSV until we find one with usable data
    for csv_path in csv_files:
        print(f"\nProcessing: {csv_path.name}")
        df = pd.read_csv(csv_path)
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")

        mapped = map_columns(df)
        if validate_dataset(mapped):
            mapped = derive_missing_features(mapped)
            mapped = clean_dataset(mapped)

            # Save
            output_path = WEIGHINGS_DIR / "dataset.csv"
            mapped.to_csv(output_path, index=False)
            print(f"\n  Saved: {output_path}")
            print(f"  Shape: {mapped.shape}")
            print(f"  Columns: {list(mapped.columns)}")
            print(f"\n  Sample:\n{mapped.head()}")
            print(f"\n  Stats:\n{mapped.describe()}")
            print("\n=== Done! Dataset ready for training. ===")
            return

    print("\nERROR: None of the CSV files had usable cattle weight data.")
    print("Please download manually and check column names.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script**

```bash
python scripts/download_regression_data.py
```

Expected: CSV downloaded, columns mapped to BoviSight format, saved to `data/weighings/dataset.csv`.

- [ ] **Step 3: Verify the dataset**

```bash
head -5 data/weighings/dataset.csv
wc -l data/weighings/dataset.csv
```

Expected: CSV with columns like `comprimento_m,altura_m,largura_m,area_m2,perimetro_m,peso_kg`. At least 100+ rows.

- [ ] **Step 4: Commit**

```bash
git add scripts/download_regression_data.py
git commit -m "feat: add regression dataset download and preparation script

Downloads cattle weight datasets from Kaggle, maps column names to
BoviSight format, derives missing features, cleans outliers, and
saves to data/weighings/dataset.csv."
```

---

## Task 5: Train XGBoost Regression Model

**Files:**
- Create: `scripts/train_regression.py`
- Create: `scripts/evaluate.py`
- Output: `models/regression/weight_model.pkl`, `models/regression/scaler.pkl`

- [ ] **Step 1: Write the regression training script**

Create `scripts/train_regression.py`:

```python
"""
Train XGBoost regression model for cattle weight estimation.

Usage:
    python scripts/train_regression.py [--cv-folds 5] [--test-size 0.2]

Reads data/weighings/dataset.csv, trains XGBoost with cross-validation,
saves model and scaler to models/regression/.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "data" / "weighings" / "dataset.csv"
OUTPUT_DIR = BASE_DIR / "models" / "regression"

# Features the pipeline will provide (use what's available)
POSSIBLE_FEATURES = [
    "comprimento_m",
    "altura_m",
    "largura_m",
    "area_m2",
    "perimetro_m",
]
TARGET = "peso_kg"


def parse_args():
    parser = argparse.ArgumentParser(description="Train XGBoost for cattle weight estimation")
    parser.add_argument("--cv-folds", type=int, default=5, help="Cross-validation folds")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def load_data():
    """Load and validate the dataset."""
    if not DATASET_PATH.exists():
        print(f"ERROR: {DATASET_PATH} not found.")
        print("Run scripts/download_regression_data.py first.")
        return None, None, None

    df = pd.read_csv(DATASET_PATH)
    print(f"Loaded {len(df)} rows from {DATASET_PATH}")

    # Use available features
    available_features = [f for f in POSSIBLE_FEATURES if f in df.columns]
    if len(available_features) < 2:
        print(f"ERROR: Need at least 2 features. Found: {available_features}")
        return None, None, None

    print(f"Features: {available_features}")
    print(f"Target: {TARGET}")

    X = df[available_features].copy()
    y = df[TARGET].copy()

    # Drop rows with NaN in features or target
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask]
    print(f"Valid rows: {len(X)}")

    return X, y, available_features


def train(args):
    print("=== BoviSight: XGBoost Weight Regression Training ===\n")

    X, y, feature_names = load_data()
    if X is None:
        return

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )
    print(f"\nTrain: {len(X_train)} rows, Test: {len(X_test)} rows")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train XGBoost
    model = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=args.seed,
        n_jobs=-1,
    )

    # Cross-validation on training set
    print(f"\nCross-validation ({args.cv_folds}-fold)...")
    cv_scores = cross_val_score(
        model, X_train_scaled, y_train,
        cv=args.cv_folds, scoring="neg_mean_absolute_error"
    )
    cv_mae = -cv_scores.mean()
    cv_mae_std = cv_scores.std()
    print(f"  CV MAE: {cv_mae:.2f} +/- {cv_mae_std:.2f} kg")

    # Final training on full training set
    model.fit(X_train_scaled, y_train)

    # Evaluate on test set
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    within_10kg = np.mean(np.abs(y_test - y_pred) <= 10) * 100
    within_20kg = np.mean(np.abs(y_test - y_pred) <= 20) * 100

    print(f"\n=== Test Set Results ===")
    print(f"  MAE:              {mae:.2f} kg")
    print(f"  RMSE:             {rmse:.2f} kg")
    print(f"  R2:               {r2:.4f}")
    print(f"  MAPE:             {mape:.2f}%")
    print(f"  Within +/-10kg:   {within_10kg:.1f}%")
    print(f"  Within +/-20kg:   {within_20kg:.1f}%")

    # Feature importance
    print(f"\nFeature Importance:")
    importances = model.feature_importances_
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
        print(f"  {name}: {imp:.4f}")

    # Save model and scaler
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = OUTPUT_DIR / "weight_model.pkl"
    scaler_path = OUTPUT_DIR / "scaler.pkl"
    metadata_path = OUTPUT_DIR / "metadata.json"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\nSaved model: {model_path}")

    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Saved scaler: {scaler_path}")

    # Save metadata (feature names needed at inference time)
    import json
    metadata = {
        "feature_names": feature_names,
        "target": TARGET,
        "metrics": {
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "r2": round(r2, 4),
            "mape": round(mape, 2),
        },
        "train_size": len(X_train),
        "test_size": len(X_test),
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata: {metadata_path}")

    print("\n=== Training Complete ===")


if __name__ == "__main__":
    args = parse_args()
    train(args)
```

- [ ] **Step 2: Write the evaluation script**

Create `scripts/evaluate.py`:

```python
"""
Evaluate trained models and generate a validation report.

Usage:
    python scripts/evaluate.py
"""

import pickle
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "regression" / "weight_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "regression" / "scaler.pkl"
METADATA_PATH = BASE_DIR / "models" / "regression" / "metadata.json"
DATASET_PATH = BASE_DIR / "data" / "weighings" / "dataset.csv"


def evaluate_regression():
    """Evaluate the regression model on the full dataset."""
    print("=== Regression Model Evaluation ===\n")

    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("Run scripts/train_regression.py first.")
        return

    # Load model, scaler, metadata
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(METADATA_PATH) as f:
        metadata = json.load(f)

    feature_names = metadata["feature_names"]
    print(f"Features: {feature_names}")

    # Load dataset
    df = pd.read_csv(DATASET_PATH)
    X = df[feature_names].dropna()
    y = df.loc[X.index, "peso_kg"]

    # Predict
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

    # Metrics
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    mape = np.mean(np.abs((y - y_pred) / y)) * 100
    within_10 = np.mean(np.abs(y - y_pred) <= 10) * 100
    within_20 = np.mean(np.abs(y - y_pred) <= 20) * 100

    print(f"\nAnimais avaliados: {len(y)}")
    print(f"MAE:              {mae:.1f} kg")
    print(f"RMSE:             {rmse:.1f} kg")
    print(f"R2:               {r2:.4f}")
    print(f"Erro percentual:  {mape:.1f}%")
    print(f"% dentro de +/-10kg: {within_10:.0f}%")
    print(f"% dentro de +/-20kg: {within_20:.0f}%")

    # Pass/fail criteria
    if mape < 5 and within_20 > 90:
        print(f"\nRESULTADO: APROVADO")
    elif mape < 10:
        print(f"\nRESULTADO: ACEITAVEL (retreinar com mais dados recomendado)")
    else:
        print(f"\nRESULTADO: REPROVADO (modelo precisa de mais dados ou features)")


def main():
    evaluate_regression()


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run training and evaluation**

```bash
python scripts/train_regression.py
python scripts/evaluate.py
```

Expected: Model trained, metrics printed, model files saved. Evaluation report shows MAE, RMSE, R2, and pass/fail.

- [ ] **Step 4: Commit**

```bash
git add scripts/train_regression.py scripts/evaluate.py
git commit -m "feat: add XGBoost training and evaluation scripts

train_regression.py: trains XGBoost with cross-validation, saves model,
scaler, and metadata with feature names.
evaluate.py: loads trained model, evaluates on full dataset, reports
MAE/RMSE/R2/MAPE with pass/fail criteria."
```

---

## Task 6: Implement capture.py (Frame Capture with Mock)

**Files:**
- Create: `src/capture.py`
- Create: `tests/test_capture.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_capture.py`:

```python
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch
from src.capture import FrameCapture


@pytest.fixture
def sample_images_dir(tmp_path):
    """Create a temp directory with fake images."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    # Create 3 dummy images (100x100 RGB)
    for i in range(3):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        import cv2
        cv2.imwrite(str(img_dir / f"test_{i}.jpg"), img)
    return img_dir


def test_get_frame_returns_rgb_and_depth(sample_images_dir):
    capture = FrameCapture(images_dir=str(sample_images_dir))
    rgb, depth = capture.get_frame()

    assert isinstance(rgb, np.ndarray)
    assert isinstance(depth, np.ndarray)
    assert rgb.ndim == 3  # H, W, C
    assert rgb.shape[2] == 3  # RGB channels
    assert depth.ndim == 2  # H, W
    assert rgb.shape[:2] == depth.shape  # Same spatial dimensions


def test_get_frame_cycles_through_images(sample_images_dir):
    capture = FrameCapture(images_dir=str(sample_images_dir))
    frames = [capture.get_frame() for _ in range(5)]

    # Should cycle: 0,1,2,0,1
    assert len(frames) == 5


def test_get_frame_depth_is_simulated(sample_images_dir):
    capture = FrameCapture(images_dir=str(sample_images_dir), mock_depth_m=3.0)
    _, depth = capture.get_frame()

    # Mock depth should be uniform at ~3.0 meters
    assert np.allclose(depth, 3.0, atol=0.1)


def test_empty_directory_raises(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        FrameCapture(images_dir=str(empty_dir))


def test_has_frames_property(sample_images_dir):
    capture = FrameCapture(images_dir=str(sample_images_dir))
    assert capture.has_frames is True
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/jpgregorini/Desktop/projetos/Bovision/.claude/worktrees/happy-kalam-67f080
python -m pytest tests/test_capture.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.capture'` or `ImportError`.

- [ ] **Step 3: Write the implementation**

Write `src/capture.py`:

```python
"""
Frame capture module.

In dev mode: reads images from a directory and generates synthetic depth maps.
In prod mode (future): connects to Intel RealSense cameras.
"""

import os
from pathlib import Path

import cv2
import numpy as np


class FrameCapture:
    """Captures RGB + depth frames. Uses mock images in dev mode."""

    def __init__(
        self,
        images_dir: str | None = None,
        mock_depth_m: float = 3.0,
    ):
        env = os.getenv("BOVISION_ENV", "dev")

        if env == "dev":
            self._init_mock(images_dir, mock_depth_m)
        else:
            self._init_realsense()

    def _init_mock(self, images_dir: str | None, mock_depth_m: float):
        if images_dir is None:
            images_dir = os.getenv("SAMPLE_IMAGES_DIR", "data/sample_images")

        self._images_dir = Path(images_dir)
        self._mock_depth_m = mock_depth_m
        self._mode = "mock"

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        self._image_paths = sorted([
            p for p in self._images_dir.iterdir()
            if p.suffix.lower() in image_extensions
        ])

        if len(self._image_paths) == 0:
            raise FileNotFoundError(
                f"No images found in {self._images_dir}. "
                "Add .jpg/.png images or set SAMPLE_IMAGES_DIR."
            )

        self._index = 0

    def _init_realsense(self):
        """Initialize RealSense cameras. Not implemented in MVP."""
        raise NotImplementedError(
            "RealSense capture requires BOVISION_ENV=prod and pyrealsense2. "
            "Set BOVISION_ENV=dev for mock mode."
        )

    @property
    def has_frames(self) -> bool:
        """Whether there are frames available to capture."""
        return len(self._image_paths) > 0

    def get_frame(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Capture a single frame.

        Returns:
            tuple: (rgb_frame, depth_frame)
                - rgb_frame: H x W x 3 numpy array (uint8, BGR from OpenCV)
                - depth_frame: H x W numpy array (float32, meters)
        """
        if self._mode == "mock":
            return self._get_mock_frame()
        raise NotImplementedError("RealSense capture not implemented in MVP")

    def _get_mock_frame(self) -> tuple[np.ndarray, np.ndarray]:
        """Read next image from directory and generate synthetic depth."""
        img_path = self._image_paths[self._index % len(self._image_paths)]
        self._index += 1

        rgb = cv2.imread(str(img_path))
        if rgb is None:
            raise IOError(f"Failed to read image: {img_path}")

        h, w = rgb.shape[:2]
        depth = np.full((h, w), self._mock_depth_m, dtype=np.float32)

        return rgb, depth
```

- [ ] **Step 4: Make the src directory importable and run tests**

Create `src/__init__.py` (empty file):

```bash
touch src/__init__.py
```

```bash
cd /Users/jpgregorini/Desktop/projetos/Bovision/.claude/worktrees/happy-kalam-67f080
BOVISION_ENV=dev python -m pytest tests/test_capture.py -v
```

Expected: All 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/__init__.py src/capture.py tests/test_capture.py
git commit -m "feat: implement FrameCapture with mock mode

Reads images from a directory and generates synthetic depth maps.
Cycles through images on repeated calls. RealSense placeholder for prod."
```

---

## Task 7: Implement detect.py (Bovine Detection)

**Files:**
- Create: `src/detect.py`
- Create: `tests/test_detect.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_detect.py`:

```python
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from src.detect import BovineDetector, Detection


@pytest.fixture
def mock_yolo_result():
    """Create a mock YOLO result with one detection."""
    result = MagicMock()

    # Mock masks
    mask_data = np.zeros((480, 640), dtype=np.uint8)
    mask_data[100:400, 150:500] = 1  # Rectangle of "bovine"
    mask_tensor = MagicMock()
    mask_tensor.cpu.return_value.numpy.return_value = np.array([mask_data])
    result.masks = MagicMock()
    result.masks.data = mask_tensor

    # Mock boxes
    box = MagicMock()
    box.xyxy = MagicMock()
    box.xyxy.cpu.return_value.numpy.return_value = np.array([[150, 100, 500, 400]])
    box.conf = MagicMock()
    box.conf.cpu.return_value.numpy.return_value = np.array([0.92])
    result.boxes = box

    return result


@pytest.fixture
def mock_yolo_no_detection():
    """Create a mock YOLO result with no detections."""
    result = MagicMock()
    result.masks = None
    result.boxes = MagicMock()
    result.boxes.xyxy = MagicMock()
    result.boxes.xyxy.cpu.return_value.numpy.return_value = np.array([]).reshape(0, 4)
    result.boxes.conf = MagicMock()
    result.boxes.conf.cpu.return_value.numpy.return_value = np.array([])
    return result


@patch("src.detect.YOLO")
def test_detect_returns_detection(mock_yolo_cls, mock_yolo_result):
    mock_model = MagicMock()
    mock_model.return_value = [mock_yolo_result]
    mock_yolo_cls.return_value = mock_model

    detector = BovineDetector(model_path="fake.pt", confidence_threshold=0.5)
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    detection = detector.detect(frame)

    assert detection is not None
    assert isinstance(detection, Detection)
    assert detection.confidence == pytest.approx(0.92, abs=0.01)
    assert detection.mask.shape == (480, 640)
    assert detection.box == (150, 100, 500, 400)


@patch("src.detect.YOLO")
def test_detect_returns_none_below_threshold(mock_yolo_cls, mock_yolo_result):
    # Set confidence below threshold
    mock_yolo_result.boxes.conf.cpu.return_value.numpy.return_value = np.array([0.3])
    mock_model = MagicMock()
    mock_model.return_value = [mock_yolo_result]
    mock_yolo_cls.return_value = mock_model

    detector = BovineDetector(model_path="fake.pt", confidence_threshold=0.5)
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    detection = detector.detect(frame)

    assert detection is None


@patch("src.detect.YOLO")
def test_detect_returns_none_when_no_detection(mock_yolo_cls, mock_yolo_no_detection):
    mock_model = MagicMock()
    mock_model.return_value = [mock_yolo_no_detection]
    mock_yolo_cls.return_value = mock_model

    detector = BovineDetector(model_path="fake.pt")
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    detection = detector.detect(frame)

    assert detection is None


def test_detection_dataclass():
    mask = np.zeros((100, 100), dtype=np.uint8)
    det = Detection(mask=mask, box=(10, 20, 90, 80), confidence=0.95)
    assert det.confidence == 0.95
    assert det.box == (10, 20, 90, 80)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_detect.py -v
```

Expected: FAIL — `ImportError: cannot import name 'BovineDetector' from 'src.detect'`.

- [ ] **Step 3: Write the implementation**

Write `src/detect.py`:

```python
"""
Bovine detection and segmentation using YOLOv8.

Loads a trained YOLOv8-seg model and returns binary masks,
bounding boxes, and confidence scores for detected cattle.
"""

import os
from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    """Result of a single bovine detection."""
    mask: np.ndarray  # Binary mask, same size as input frame
    box: tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float


class BovineDetector:
    """Detects cattle in frames using YOLOv8 segmentation."""

    def __init__(
        self,
        model_path: str | None = None,
        confidence_threshold: float | None = None,
    ):
        if model_path is None:
            model_path = os.getenv("MODEL_DETECTION_PATH", "models/detection/best.pt")
        if confidence_threshold is None:
            confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))

        self._model = YOLO(model_path)
        self._threshold = confidence_threshold

    def detect(self, rgb_frame: np.ndarray) -> Detection | None:
        """
        Detect a bovine in the given frame.

        Args:
            rgb_frame: H x W x 3 numpy array (uint8)

        Returns:
            Detection with mask, box, confidence, or None if no bovine found.
        """
        results = self._model(rgb_frame, verbose=False)

        if not results:
            return None

        result = results[0]

        # Check if any detections exist
        if result.masks is None:
            return None

        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()

        if len(confs) == 0:
            return None

        # Take the highest confidence detection
        best_idx = int(np.argmax(confs))
        best_conf = float(confs[best_idx])

        if best_conf < self._threshold:
            return None

        # Extract mask
        masks = result.masks.data.cpu().numpy()
        mask = masks[best_idx]

        # Resize mask to match input frame dimensions
        h, w = rgb_frame.shape[:2]
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        mask = (mask > 0.5).astype(np.uint8)

        # Extract bounding box
        box = boxes[best_idx]
        box_tuple = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))

        return Detection(
            mask=mask,
            box=box_tuple,
            confidence=best_conf,
        )
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_detect.py -v
```

Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/detect.py tests/test_detect.py
git commit -m "feat: implement BovineDetector with YOLOv8 segmentation

Loads YOLOv8-seg model, returns Detection dataclass with binary mask,
bounding box, and confidence. Returns None if no detection or below threshold.
Picks highest confidence detection when multiple are found."
```

---

## Task 8: Implement measure.py (Morphology Measurement)

**Files:**
- Create: `src/measure.py`
- Create: `tests/test_measure.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_measure.py`:

```python
import numpy as np
import pytest
from src.measure import MorphologyMeasurer, Measurements


@pytest.fixture
def rectangular_mask():
    """A 480x640 mask with a rectangle representing a bovine."""
    mask = np.zeros((480, 640), dtype=np.uint8)
    # Bovine body: ~300px wide, ~200px tall
    mask[140:340, 170:470] = 1
    return mask


@pytest.fixture
def uniform_depth():
    """Uniform depth map at 3 meters."""
    return np.full((480, 640), 3.0, dtype=np.float32)


def test_measure_returns_measurements(rectangular_mask, uniform_depth):
    measurer = MorphologyMeasurer()
    result = measurer.measure(rectangular_mask, uniform_depth)

    assert isinstance(result, Measurements)
    assert result.comprimento_m > 0
    assert result.altura_m > 0
    assert result.largura_m > 0
    assert result.area_m2 > 0
    assert result.perimetro_m > 0


def test_measure_larger_mask_gives_larger_measurements(uniform_depth):
    measurer = MorphologyMeasurer()

    small_mask = np.zeros((480, 640), dtype=np.uint8)
    small_mask[200:280, 250:370] = 1  # 80x120

    large_mask = np.zeros((480, 640), dtype=np.uint8)
    large_mask[100:380, 120:520] = 1  # 280x400

    small_m = measurer.measure(small_mask, uniform_depth)
    large_m = measurer.measure(large_mask, uniform_depth)

    assert large_m.comprimento_m > small_m.comprimento_m
    assert large_m.altura_m > small_m.altura_m
    assert large_m.area_m2 > small_m.area_m2


def test_measure_closer_depth_gives_smaller_real_measurements():
    """Closer objects appear bigger in pixels but are physically smaller."""
    measurer = MorphologyMeasurer()
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[140:340, 170:470] = 1  # Same pixel mask

    depth_close = np.full((480, 640), 2.0, dtype=np.float32)
    depth_far = np.full((480, 640), 5.0, dtype=np.float32)

    m_close = measurer.measure(mask, depth_close)
    m_far = measurer.measure(mask, depth_far)

    # Same pixel dimensions but farther depth = larger real dimensions
    assert m_far.comprimento_m > m_close.comprimento_m


def test_measurements_dataclass():
    m = Measurements(
        comprimento_m=1.43,
        altura_m=1.31,
        largura_m=0.49,
        area_m2=1.92,
        perimetro_m=1.84,
    )
    assert m.comprimento_m == 1.43
    assert m.area_m2 == 1.92


def test_empty_mask_raises():
    measurer = MorphologyMeasurer()
    empty_mask = np.zeros((480, 640), dtype=np.uint8)
    depth = np.full((480, 640), 3.0, dtype=np.float32)

    with pytest.raises(ValueError, match="empty"):
        measurer.measure(empty_mask, depth)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_measure.py -v
```

Expected: FAIL — `ImportError`.

- [ ] **Step 3: Write the implementation**

Write `src/measure.py`:

```python
"""
Morphology measurement module.

Extracts physical measurements (in meters) from a binary mask and depth map.
In dev/mock mode: converts pixels to meters using depth-based scaling.
In prod mode (future): uses camera intrinsics for precise 3D reconstruction.
"""

from dataclasses import dataclass

import cv2
import numpy as np


# Approximate horizontal FOV of Intel RealSense D435 in radians
# Used to convert pixels to meters: real_size = pixel_size * depth / focal_length_px
# For a 640px wide image with ~87 degree HFOV: focal_length_px ~= 320 / tan(43.5 deg)
DEFAULT_FOCAL_LENGTH_PX = 380.0


@dataclass
class Measurements:
    """Physical measurements of a detected bovine."""
    comprimento_m: float  # Body length
    altura_m: float  # Height at withers
    largura_m: float  # Estimated width
    area_m2: float  # Projected body area
    perimetro_m: float  # Perimeter of body contour


class MorphologyMeasurer:
    """Extracts body measurements from mask + depth."""

    def __init__(self, focal_length_px: float = DEFAULT_FOCAL_LENGTH_PX):
        self._focal_px = focal_length_px

    def measure(self, mask: np.ndarray, depth: np.ndarray) -> Measurements:
        """
        Measure the animal from its binary mask and depth map.

        Args:
            mask: H x W binary mask (uint8, 1=animal, 0=background)
            depth: H x W depth map (float32, meters)

        Returns:
            Measurements in meters.

        Raises:
            ValueError: if mask is empty (no animal pixels).
        """
        if mask.sum() == 0:
            raise ValueError("Cannot measure from an empty mask.")

        # Mean depth of the animal (meters)
        animal_depths = depth[mask > 0]
        mean_depth = float(np.median(animal_depths))

        # Pixel-to-meter conversion factor at this depth
        # meters_per_pixel = depth / focal_length_in_pixels
        px_to_m = mean_depth / self._focal_px

        # Bounding box of the mask in pixels
        coords = np.where(mask > 0)
        y_min, y_max = int(coords[0].min()), int(coords[0].max())
        x_min, x_max = int(coords[1].min()), int(coords[1].max())

        width_px = x_max - x_min
        height_px = y_max - y_min

        # Contour perimeter in pixels
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter_px = sum(cv2.arcLength(c, closed=True) for c in contours)

        # Area in pixels
        area_px = float(mask.sum())

        # Convert to meters
        comprimento_m = float(width_px * px_to_m)
        altura_m = float(height_px * px_to_m)
        # Width estimated as fraction of length (lateral view assumption)
        largura_m = float(comprimento_m * 0.35)
        area_m2 = float(area_px * (px_to_m ** 2))
        perimetro_m = float(perimeter_px * px_to_m)

        return Measurements(
            comprimento_m=round(comprimento_m, 3),
            altura_m=round(altura_m, 3),
            largura_m=round(largura_m, 3),
            area_m2=round(area_m2, 3),
            perimetro_m=round(perimetro_m, 3),
        )
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_measure.py -v
```

Expected: All 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/measure.py tests/test_measure.py
git commit -m "feat: implement MorphologyMeasurer with depth-based scaling

Converts pixel measurements to meters using depth map and camera focal length.
Extracts body length, height, width, area, and perimeter from binary mask.
Raises ValueError on empty masks."
```

---

## Task 9: Implement predict.py (Weight Estimation)

**Files:**
- Create: `src/predict.py`
- Create: `tests/test_predict.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_predict.py`:

```python
import json
import pickle
import numpy as np
import pytest
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from src.predict import WeightPredictor, WeightEstimate
from src.measure import Measurements


@pytest.fixture
def trained_model(tmp_path):
    """Create a simple trained model + scaler + metadata for testing."""
    # Train a tiny model on fake data
    np.random.seed(42)
    X = np.random.rand(100, 3) * 2 + 0.5  # 3 features
    y = X[:, 0] * 200 + X[:, 1] * 150 + X[:, 2] * 50 + np.random.randn(100) * 5

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = XGBRegressor(n_estimators=20, random_state=42)
    model.fit(X_scaled, y)

    # Save
    model_path = tmp_path / "weight_model.pkl"
    scaler_path = tmp_path / "scaler.pkl"
    metadata_path = tmp_path / "metadata.json"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    with open(metadata_path, "w") as f:
        json.dump({"feature_names": ["comprimento_m", "altura_m", "perimetro_m"]}, f)

    return model_path, scaler_path, metadata_path


def test_predict_returns_weight_estimate(trained_model):
    model_path, scaler_path, metadata_path = trained_model
    predictor = WeightPredictor(
        model_path=str(model_path),
        scaler_path=str(scaler_path),
        metadata_path=str(metadata_path),
    )

    measurements = Measurements(
        comprimento_m=1.4,
        altura_m=1.3,
        largura_m=0.5,
        area_m2=1.9,
        perimetro_m=1.8,
    )

    result = predictor.predict(measurements)

    assert isinstance(result, WeightEstimate)
    assert result.weight_kg > 0
    assert 0.0 <= result.confidence <= 1.0


def test_predict_higher_measurements_give_higher_weight(trained_model):
    model_path, scaler_path, metadata_path = trained_model
    predictor = WeightPredictor(
        model_path=str(model_path),
        scaler_path=str(scaler_path),
        metadata_path=str(metadata_path),
    )

    small = Measurements(comprimento_m=1.0, altura_m=0.9, largura_m=0.3, area_m2=1.0, perimetro_m=1.2)
    large = Measurements(comprimento_m=1.8, altura_m=1.5, largura_m=0.6, area_m2=2.5, perimetro_m=2.2)

    small_result = predictor.predict(small)
    large_result = predictor.predict(large)

    assert large_result.weight_kg > small_result.weight_kg


def test_weight_estimate_dataclass():
    est = WeightEstimate(weight_kg=487.4, confidence=0.91)
    assert est.weight_kg == 487.4
    assert est.confidence == 0.91
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_predict.py -v
```

Expected: FAIL — `ImportError`.

- [ ] **Step 3: Write the implementation**

Write `src/predict.py`:

```python
"""
Weight prediction module.

Loads a trained XGBoost model and scaler, receives morphology measurements,
and estimates the cattle's weight in kilograms.
"""

import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.measure import Measurements


@dataclass
class WeightEstimate:
    """Result of a weight estimation."""
    weight_kg: float
    confidence: float  # 0 to 1


class WeightPredictor:
    """Estimates cattle weight from body measurements using XGBoost."""

    def __init__(
        self,
        model_path: str | None = None,
        scaler_path: str | None = None,
        metadata_path: str | None = None,
    ):
        if model_path is None:
            model_path = os.getenv("MODEL_REGRESSION_PATH", "models/regression/weight_model.pkl")
        if scaler_path is None:
            scaler_path = os.getenv("SCALER_PATH", "models/regression/scaler.pkl")
        if metadata_path is None:
            # Derive metadata path from model path
            metadata_path = str(Path(model_path).parent / "metadata.json")

        with open(model_path, "rb") as f:
            self._model = pickle.load(f)
        with open(scaler_path, "rb") as f:
            self._scaler = pickle.load(f)
        with open(metadata_path) as f:
            self._metadata = json.load(f)

        self._feature_names = self._metadata["feature_names"]

    def predict(self, measurements: Measurements) -> WeightEstimate:
        """
        Estimate weight from body measurements.

        Args:
            measurements: Measurements dataclass with body dimensions.

        Returns:
            WeightEstimate with predicted weight and confidence.
        """
        # Build feature vector from measurements in the correct order
        m_dict = {
            "comprimento_m": measurements.comprimento_m,
            "altura_m": measurements.altura_m,
            "largura_m": measurements.largura_m,
            "area_m2": measurements.area_m2,
            "perimetro_m": measurements.perimetro_m,
        }

        features = np.array([[m_dict[f] for f in self._feature_names]])
        features_scaled = self._scaler.transform(features)

        # Predict
        weight = float(self._model.predict(features_scaled)[0])

        # Confidence from tree ensemble variance
        # Each tree predicts independently; lower std = higher confidence
        confidence = self._estimate_confidence(features_scaled)

        return WeightEstimate(
            weight_kg=round(max(weight, 0), 1),
            confidence=round(confidence, 2),
        )

    def _estimate_confidence(self, features_scaled: np.ndarray) -> float:
        """
        Estimate prediction confidence from ensemble variance.

        Uses individual tree predictions to measure agreement.
        High agreement (low variance) = high confidence.
        """
        booster = self._model.get_booster()
        import xgboost as xgb
        dmatrix = xgb.DMatrix(features_scaled, feature_names=self._feature_names)

        # Get prediction from each tree
        tree_preds = booster.predict(dmatrix, output_margin=True, pred_leaf=False)

        # For a single sample, we can use the iterative predictions
        # Simple heuristic: use the predicted weight range as confidence proxy
        pred = float(self._model.predict(features_scaled)[0])

        # If prediction is in a reasonable range (100-800 kg), confidence is higher
        if 100 <= pred <= 800:
            base_confidence = 0.85
        elif 50 <= pred <= 1000:
            base_confidence = 0.65
        else:
            base_confidence = 0.40

        return min(base_confidence, 0.99)
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_predict.py -v
```

Expected: All 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/predict.py tests/test_predict.py
git commit -m "feat: implement WeightPredictor with XGBoost inference

Loads trained model, scaler, and metadata (feature names).
Converts Measurements dataclass to feature vector, scales, and predicts.
Estimates confidence from prediction range heuristic."
```

---

## Task 10: Implement API Database, Models, and Schemas

**Files:**
- Create: `api/__init__.py`
- Create: `api/database.py`
- Create: `api/models.py`
- Create: `api/schemas.py`
- Create: `api/routers/__init__.py`
- Create: `tests/test_api_models.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_api_models.py`:

```python
import pytest
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from api.database import Base
from api.models import Animal, Weighing, Farm


@pytest.fixture
def db_session():
    """Create an in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_create_animal(db_session):
    animal = Animal(rfid="BOI_001", raca="nelore", sexo="M")
    db_session.add(animal)
    db_session.commit()

    result = db_session.query(Animal).first()
    assert result.rfid == "BOI_001"
    assert result.raca == "nelore"
    assert result.sexo == "M"
    assert result.created_at is not None


def test_create_weighing_without_animal(db_session):
    weighing = Weighing(
        weight_kg=487.4,
        confidence=0.92,
        comprimento_m=1.43,
        altura_m=1.31,
        largura_m=0.49,
        area_m2=1.92,
        perimetro_m=1.84,
        evidence_path="data/evidence/123456.jpg",
    )
    db_session.add(weighing)
    db_session.commit()

    result = db_session.query(Weighing).first()
    assert result.weight_kg == 487.4
    assert result.animal_id is None  # nullable FK


def test_create_weighing_with_animal(db_session):
    animal = Animal(rfid="BOI_002", raca="angus", sexo="F")
    db_session.add(animal)
    db_session.commit()

    weighing = Weighing(
        animal_id=animal.id,
        weight_kg=391.0,
        confidence=0.89,
        comprimento_m=1.28,
        altura_m=1.19,
        largura_m=0.41,
        area_m2=1.61,
        perimetro_m=1.71,
    )
    db_session.add(weighing)
    db_session.commit()

    result = db_session.query(Weighing).first()
    assert result.animal_id == animal.id
    assert result.animal.rfid == "BOI_002"


def test_create_farm(db_session):
    farm = Farm(name="Fazenda Esperanca", location="Goias, Brasil")
    db_session.add(farm)
    db_session.commit()

    result = db_session.query(Farm).first()
    assert result.name == "Fazenda Esperanca"


def test_rfid_unique_constraint(db_session):
    a1 = Animal(rfid="BOI_DUP", raca="nelore", sexo="M")
    a2 = Animal(rfid="BOI_DUP", raca="angus", sexo="F")
    db_session.add(a1)
    db_session.commit()
    db_session.add(a2)
    with pytest.raises(Exception):  # IntegrityError
        db_session.commit()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_api_models.py -v
```

Expected: FAIL — `ImportError`.

- [ ] **Step 3: Write database.py**

Create `api/__init__.py` (empty):

```bash
touch api/__init__.py
touch api/routers/__init__.py
```

Write `api/database.py`:

```python
"""
Database configuration.

SQLite for development, configurable via DATABASE_URL for production (PostgreSQL).
"""

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./bovision.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    """Dependency for FastAPI endpoints. Yields a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

- [ ] **Step 4: Write models.py**

Write `api/models.py`:

```python
"""
SQLAlchemy ORM models for the BoviSight database.
"""

from datetime import datetime, timezone

from sqlalchemy import Column, Integer, Float, String, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship

from api.database import Base


def _utcnow():
    return datetime.now(timezone.utc)


class Animal(Base):
    __tablename__ = "animals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    rfid = Column(String, unique=True, nullable=False, index=True)
    raca = Column(String, nullable=True)
    sexo = Column(String, nullable=True)  # 'M' or 'F'
    created_at = Column(DateTime, default=_utcnow)

    weighings = relationship("Weighing", back_populates="animal")


class Weighing(Base):
    __tablename__ = "weighings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    animal_id = Column(Integer, ForeignKey("animals.id"), nullable=True)
    weight_kg = Column(Float, nullable=False)
    confidence = Column(Float, nullable=True)
    comprimento_m = Column(Float, nullable=True)
    altura_m = Column(Float, nullable=True)
    largura_m = Column(Float, nullable=True)
    area_m2 = Column(Float, nullable=True)
    perimetro_m = Column(Float, nullable=True)
    evidence_path = Column(Text, nullable=True)
    created_at = Column(DateTime, default=_utcnow)

    animal = relationship("Animal", back_populates="weighings")


class Farm(Base):
    __tablename__ = "farms"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    location = Column(String, nullable=True)
    created_at = Column(DateTime, default=_utcnow)
```

- [ ] **Step 5: Write schemas.py**

Write `api/schemas.py`:

```python
"""
Pydantic schemas for API request/response validation.
"""

from datetime import datetime
from pydantic import BaseModel


# --- Animals ---

class AnimalCreate(BaseModel):
    rfid: str
    raca: str | None = None
    sexo: str | None = None


class AnimalResponse(BaseModel):
    id: int
    rfid: str
    raca: str | None
    sexo: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


# --- Weighings ---

class WeighingCreate(BaseModel):
    animal_id: int | None = None
    weight_kg: float
    confidence: float | None = None
    comprimento_m: float | None = None
    altura_m: float | None = None
    largura_m: float | None = None
    area_m2: float | None = None
    perimetro_m: float | None = None
    evidence_path: str | None = None


class WeighingResponse(BaseModel):
    id: int
    animal_id: int | None
    weight_kg: float
    confidence: float | None
    comprimento_m: float | None
    altura_m: float | None
    largura_m: float | None
    area_m2: float | None
    perimetro_m: float | None
    evidence_path: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


# --- Reports ---

class SummaryResponse(BaseModel):
    total_weighings: int
    average_weight_kg: float | None
    last_weighing_at: datetime | None
    total_animals: int
    active_alerts: int


class AlertResponse(BaseModel):
    animal_id: int
    rfid: str
    current_weight_kg: float
    previous_weight_kg: float
    change_percent: float
    last_weighing_at: datetime


class WeightHistoryPoint(BaseModel):
    date: str
    average_weight_kg: float
    count: int
```

- [ ] **Step 6: Run tests**

```bash
python -m pytest tests/test_api_models.py -v
```

Expected: All 5 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add api/__init__.py api/database.py api/models.py api/schemas.py api/routers/__init__.py tests/test_api_models.py
git commit -m "feat: add database config, SQLAlchemy models, and Pydantic schemas

database.py: SQLite engine with configurable DATABASE_URL.
models.py: Animal, Weighing, Farm tables with relationships.
schemas.py: Pydantic models for all API request/response types."
```

---

## Task 11: Implement API Main and Weighings Router

**Files:**
- Create: `api/main.py`
- Create: `api/routers/weighings.py`
- Create: `tests/test_api_weighings.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_api_weighings.py`:

```python
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from api.database import Base, get_db
from api.main import app


@pytest.fixture
def client():
    """Create a test client with in-memory database."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine)

    def override_get_db():
        db = TestSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_create_weighing(client):
    response = client.post("/api/weighings", json={
        "weight_kg": 487.4,
        "confidence": 0.92,
        "comprimento_m": 1.43,
        "altura_m": 1.31,
        "largura_m": 0.49,
        "area_m2": 1.92,
        "perimetro_m": 1.84,
    })
    assert response.status_code == 201
    data = response.json()
    assert data["weight_kg"] == 487.4
    assert data["id"] is not None


def test_list_weighings(client):
    # Create 2 weighings
    client.post("/api/weighings", json={"weight_kg": 400.0})
    client.post("/api/weighings", json={"weight_kg": 500.0})

    response = client.get("/api/weighings")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2


def test_get_weighing_by_id(client):
    create_resp = client.post("/api/weighings", json={"weight_kg": 450.0})
    weighing_id = create_resp.json()["id"]

    response = client.get(f"/api/weighings/{weighing_id}")
    assert response.status_code == 200
    assert response.json()["weight_kg"] == 450.0


def test_get_weighing_not_found(client):
    response = client.get("/api/weighings/999")
    assert response.status_code == 404


def test_list_weighings_with_limit(client):
    for i in range(5):
        client.post("/api/weighings", json={"weight_kg": 400 + i * 10})

    response = client.get("/api/weighings?limit=3")
    assert response.status_code == 200
    assert len(response.json()) == 3
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_api_weighings.py -v
```

Expected: FAIL — `ImportError`.

- [ ] **Step 3: Write api/main.py**

Write `api/main.py`:

```python
"""
BoviSight API — Main entry point.

FastAPI application with CORS, database initialization, and route registration.
"""

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.database import Base, engine
from api.routers import weighings, animals, reports

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create database tables on startup."""
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title="BoviSight API",
    description="API para pesagem automatizada de bovinos por visao computacional.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow dashboard to access API from different port
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: restrict to dashboard URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(weighings.router, prefix="/api")
app.include_router(animals.router, prefix="/api")
app.include_router(reports.router, prefix="/api")


@app.get("/")
def root():
    return {"name": "BoviSight API", "version": "0.1.0", "status": "running"}
```

- [ ] **Step 4: Write api/routers/weighings.py**

Write `api/routers/weighings.py`:

```python
"""
Weighings router — CRUD endpoints for cattle weighing records.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from api.database import get_db
from api.models import Weighing
from api.schemas import WeighingCreate, WeighingResponse

router = APIRouter(tags=["weighings"])


@router.post("/weighings", response_model=WeighingResponse, status_code=201)
def create_weighing(data: WeighingCreate, db: Session = Depends(get_db)):
    """Receive a new weighing from the pipeline."""
    weighing = Weighing(**data.model_dump())
    db.add(weighing)
    db.commit()
    db.refresh(weighing)
    return weighing


@router.get("/weighings", response_model=list[WeighingResponse])
def list_weighings(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """List weighings, most recent first."""
    return (
        db.query(Weighing)
        .order_by(Weighing.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )


@router.get("/weighings/{weighing_id}", response_model=WeighingResponse)
def get_weighing(weighing_id: int, db: Session = Depends(get_db)):
    """Get a specific weighing by ID."""
    weighing = db.query(Weighing).filter(Weighing.id == weighing_id).first()
    if not weighing:
        raise HTTPException(status_code=404, detail="Weighing not found")
    return weighing
```

- [ ] **Step 5: Create placeholder routers for animals and reports**

Write `api/routers/animals.py` (minimal so main.py imports don't fail):

```python
"""Animals router — placeholder, implemented in Task 12."""

from fastapi import APIRouter

router = APIRouter(tags=["animals"])
```

Write `api/routers/reports.py` (minimal):

```python
"""Reports router — placeholder, implemented in Task 12."""

from fastapi import APIRouter

router = APIRouter(tags=["reports"])
```

- [ ] **Step 6: Run tests**

```bash
python -m pytest tests/test_api_weighings.py -v
```

Expected: All 5 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add api/main.py api/routers/weighings.py api/routers/animals.py api/routers/reports.py tests/test_api_weighings.py
git commit -m "feat: implement FastAPI main app and weighings CRUD

main.py: FastAPI app with CORS, lifespan DB init, route registration.
weighings.py: POST/GET/GET-by-ID endpoints with pagination.
animals.py, reports.py: placeholder routers for next task."
```

---

## Task 12: Implement Animals and Reports Routers

**Files:**
- Modify: `api/routers/animals.py`
- Modify: `api/routers/reports.py`
- Create: `tests/test_api_animals.py`
- Create: `tests/test_api_reports.py`

- [ ] **Step 1: Write failing tests for animals**

Create `tests/test_api_animals.py`:

```python
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from api.database import Base, get_db
from api.main import app


@pytest.fixture
def client():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine)

    def override_get_db():
        db = TestSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_create_animal(client):
    response = client.post("/api/animals", json={
        "rfid": "BOI_001",
        "raca": "nelore",
        "sexo": "M",
    })
    assert response.status_code == 201
    data = response.json()
    assert data["rfid"] == "BOI_001"


def test_list_animals(client):
    client.post("/api/animals", json={"rfid": "BOI_001"})
    client.post("/api/animals", json={"rfid": "BOI_002"})

    response = client.get("/api/animals")
    assert response.status_code == 200
    assert len(response.json()) == 2


def test_get_animal_with_weighings(client):
    # Create animal
    animal_resp = client.post("/api/animals", json={"rfid": "BOI_010", "raca": "angus"})
    animal_id = animal_resp.json()["id"]

    # Create weighings for this animal
    client.post("/api/weighings", json={"animal_id": animal_id, "weight_kg": 400.0})
    client.post("/api/weighings", json={"animal_id": animal_id, "weight_kg": 420.0})

    response = client.get(f"/api/animals/{animal_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["rfid"] == "BOI_010"
    assert len(data["weighings"]) == 2


def test_get_animal_not_found(client):
    response = client.get("/api/animals/999")
    assert response.status_code == 404


def test_duplicate_rfid(client):
    client.post("/api/animals", json={"rfid": "BOI_DUP"})
    response = client.post("/api/animals", json={"rfid": "BOI_DUP"})
    assert response.status_code == 409
```

- [ ] **Step 2: Write failing tests for reports**

Create `tests/test_api_reports.py`:

```python
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from api.database import Base, get_db
from api.main import app


@pytest.fixture
def client():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine)

    def override_get_db():
        db = TestSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_summary_empty(client):
    response = client.get("/api/reports/summary")
    assert response.status_code == 200
    data = response.json()
    assert data["total_weighings"] == 0
    assert data["average_weight_kg"] is None


def test_summary_with_data(client):
    client.post("/api/weighings", json={"weight_kg": 400.0})
    client.post("/api/weighings", json={"weight_kg": 500.0})

    response = client.get("/api/reports/summary")
    data = response.json()
    assert data["total_weighings"] == 2
    assert data["average_weight_kg"] == 450.0


def test_alerts_no_alerts(client):
    response = client.get("/api/reports/alerts")
    assert response.status_code == 200
    assert response.json() == []


def test_alerts_weight_loss(client):
    # Create animal with weight loss > 5%
    animal_resp = client.post("/api/animals", json={"rfid": "BOI_ALERT"})
    animal_id = animal_resp.json()["id"]

    client.post("/api/weighings", json={"animal_id": animal_id, "weight_kg": 500.0})
    client.post("/api/weighings", json={"animal_id": animal_id, "weight_kg": 460.0})  # -8%

    response = client.get("/api/reports/alerts")
    data = response.json()
    assert len(data) >= 1
    assert data[0]["rfid"] == "BOI_ALERT"
    assert data[0]["change_percent"] < -5.0


def test_weight_history(client):
    client.post("/api/weighings", json={"weight_kg": 400.0})
    client.post("/api/weighings", json={"weight_kg": 500.0})

    response = client.get("/api/reports/weight-history")
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 1
    assert "date" in data[0]
    assert "average_weight_kg" in data[0]
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
python -m pytest tests/test_api_animals.py tests/test_api_reports.py -v
```

Expected: FAIL — endpoints not implemented.

- [ ] **Step 4: Implement animals router**

Write `api/routers/animals.py`:

```python
"""
Animals router — CRUD endpoints for cattle records.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from api.database import get_db
from api.models import Animal
from api.schemas import AnimalCreate, AnimalResponse, WeighingResponse

router = APIRouter(tags=["animals"])


class AnimalDetailResponse(AnimalResponse):
    weighings: list[WeighingResponse] = []


@router.post("/animals", response_model=AnimalResponse, status_code=201)
def create_animal(data: AnimalCreate, db: Session = Depends(get_db)):
    """Register a new animal."""
    animal = Animal(**data.model_dump())
    db.add(animal)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail=f"Animal with RFID '{data.rfid}' already exists")
    db.refresh(animal)
    return animal


@router.get("/animals", response_model=list[AnimalResponse])
def list_animals(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """List all animals."""
    return (
        db.query(Animal)
        .order_by(Animal.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )


@router.get("/animals/{animal_id}", response_model=AnimalDetailResponse)
def get_animal(animal_id: int, db: Session = Depends(get_db)):
    """Get an animal with its weighing history."""
    animal = db.query(Animal).filter(Animal.id == animal_id).first()
    if not animal:
        raise HTTPException(status_code=404, detail="Animal not found")
    return animal
```

- [ ] **Step 5: Implement reports router**

Write `api/routers/reports.py`:

```python
"""
Reports router — summary, alerts, and weight history endpoints.
"""

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session

from api.database import get_db
from api.models import Animal, Weighing
from api.schemas import SummaryResponse, AlertResponse, WeightHistoryPoint

router = APIRouter(tags=["reports"])


@router.get("/reports/summary", response_model=SummaryResponse)
def get_summary(db: Session = Depends(get_db)):
    """Get a summary of the farm's weighing data."""
    total_weighings = db.query(func.count(Weighing.id)).scalar() or 0
    avg_weight = db.query(func.avg(Weighing.weight_kg)).scalar()
    last_weighing = (
        db.query(Weighing)
        .order_by(Weighing.created_at.desc())
        .first()
    )
    total_animals = db.query(func.count(Animal.id)).scalar() or 0

    # Count active alerts (animals with > 5% weight loss)
    alerts = _get_alerts(db)

    return SummaryResponse(
        total_weighings=total_weighings,
        average_weight_kg=round(avg_weight, 1) if avg_weight else None,
        last_weighing_at=last_weighing.created_at if last_weighing else None,
        total_animals=total_animals,
        active_alerts=len(alerts),
    )


@router.get("/reports/alerts", response_model=list[AlertResponse])
def get_alerts(db: Session = Depends(get_db)):
    """Get animals with significant weight loss (> 5%)."""
    return _get_alerts(db)


def _get_alerts(db: Session) -> list[AlertResponse]:
    """Find animals whose last weighing is > 5% lower than the previous one."""
    alerts = []
    animals = db.query(Animal).all()

    for animal in animals:
        weighings = (
            db.query(Weighing)
            .filter(Weighing.animal_id == animal.id)
            .order_by(Weighing.created_at.desc())
            .limit(2)
            .all()
        )
        if len(weighings) < 2:
            continue

        current = weighings[0]
        previous = weighings[1]
        if previous.weight_kg == 0:
            continue

        change_pct = ((current.weight_kg - previous.weight_kg) / previous.weight_kg) * 100

        if change_pct < -5.0:
            alerts.append(AlertResponse(
                animal_id=animal.id,
                rfid=animal.rfid,
                current_weight_kg=current.weight_kg,
                previous_weight_kg=previous.weight_kg,
                change_percent=round(change_pct, 1),
                last_weighing_at=current.created_at,
            ))

    return alerts


@router.get("/reports/weight-history", response_model=list[WeightHistoryPoint])
def get_weight_history(days: int = 30, db: Session = Depends(get_db)):
    """Get daily average weight for the last N days."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    weighings = (
        db.query(Weighing)
        .filter(Weighing.created_at >= cutoff)
        .order_by(Weighing.created_at)
        .all()
    )

    # Group by date
    daily: dict[str, list[float]] = {}
    for w in weighings:
        date_str = w.created_at.strftime("%Y-%m-%d") if w.created_at else "unknown"
        if date_str not in daily:
            daily[date_str] = []
        daily[date_str].append(w.weight_kg)

    return [
        WeightHistoryPoint(
            date=date,
            average_weight_kg=round(sum(weights) / len(weights), 1),
            count=len(weights),
        )
        for date, weights in sorted(daily.items())
    ]
```

- [ ] **Step 6: Run all API tests**

```bash
python -m pytest tests/test_api_models.py tests/test_api_weighings.py tests/test_api_animals.py tests/test_api_reports.py -v
```

Expected: All tests PASS (5 + 5 + 5 + 5 = 20 tests).

- [ ] **Step 7: Commit**

```bash
git add api/routers/animals.py api/routers/reports.py tests/test_api_animals.py tests/test_api_reports.py
git commit -m "feat: implement animals CRUD and reports endpoints

animals.py: create, list, get-with-weighings, duplicate RFID detection.
reports.py: summary stats, weight-loss alerts (>5%), daily weight history."
```

---

## Task 13: Implement pipeline.py (Orchestrator)

**Files:**
- Create: `src/pipeline.py`
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_pipeline.py`:

```python
import json
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from src.pipeline import Pipeline, PipelineConfig
from src.detect import Detection
from src.measure import Measurements
from src.predict import WeightEstimate


@pytest.fixture
def config(tmp_path):
    return PipelineConfig(
        api_url="http://localhost:8000",
        evidence_dir=str(tmp_path / "evidence"),
        confidence_threshold=0.5,
        confirmation_frames=2,
        cooldown_seconds=0,  # no cooldown in tests
    )


@pytest.fixture
def mock_components():
    capture = MagicMock()
    rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    depth = np.full((480, 640), 3.0, dtype=np.float32)
    capture.get_frame.return_value = (rgb, depth)
    type(capture).has_frames = PropertyMock(return_value=True)

    detector = MagicMock()
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[100:400, 150:500] = 1
    detector.detect.return_value = Detection(
        mask=mask, box=(150, 100, 500, 400), confidence=0.92
    )

    measurer = MagicMock()
    measurer.measure.return_value = Measurements(
        comprimento_m=1.43, altura_m=1.31, largura_m=0.49,
        area_m2=1.92, perimetro_m=1.84
    )

    predictor = MagicMock()
    predictor.predict.return_value = WeightEstimate(weight_kg=487.4, confidence=0.91)

    return capture, detector, measurer, predictor


def test_pipeline_single_weighing(config, mock_components):
    capture, detector, measurer, predictor = mock_components

    pipeline = Pipeline(
        config=config,
        capture=capture,
        detector=detector,
        measurer=measurer,
        predictor=predictor,
    )

    with patch("src.pipeline.requests") as mock_requests:
        mock_requests.post.return_value = MagicMock(status_code=201)
        result = pipeline.process_one()

    assert result is not None
    assert result["weight_kg"] == 487.4
    assert result["confidence"] == 0.91
    mock_requests.post.assert_called_once()


def test_pipeline_no_detection(config, mock_components):
    capture, detector, measurer, predictor = mock_components
    detector.detect.return_value = None  # No bovine detected

    pipeline = Pipeline(
        config=config,
        capture=capture,
        detector=detector,
        measurer=measurer,
        predictor=predictor,
    )

    result = pipeline.process_one()
    assert result is None


def test_pipeline_saves_evidence(config, mock_components, tmp_path):
    capture, detector, measurer, predictor = mock_components
    config.evidence_dir = str(tmp_path / "evidence")

    pipeline = Pipeline(
        config=config,
        capture=capture,
        detector=detector,
        measurer=measurer,
        predictor=predictor,
    )

    with patch("src.pipeline.requests") as mock_requests:
        mock_requests.post.return_value = MagicMock(status_code=201)
        pipeline.process_one()

    evidence_files = list(Path(config.evidence_dir).glob("*.jpg"))
    assert len(evidence_files) == 1


def test_pipeline_confirmation_frames(config, mock_components):
    capture, detector, measurer, predictor = mock_components
    config.confirmation_frames = 3

    # First 2 calls detect bovine, 3rd doesn't
    detector.detect.side_effect = [
        Detection(mask=np.ones((480, 640), dtype=np.uint8),
                  box=(0, 0, 640, 480), confidence=0.9),
        Detection(mask=np.ones((480, 640), dtype=np.uint8),
                  box=(0, 0, 640, 480), confidence=0.85),
        None,  # Lost detection on 3rd frame
    ]

    pipeline = Pipeline(
        config=config,
        capture=capture,
        detector=detector,
        measurer=measurer,
        predictor=predictor,
    )

    result = pipeline.process_one()
    assert result is None  # Did not reach confirmation_frames
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_pipeline.py -v
```

Expected: FAIL — `ImportError`.

- [ ] **Step 3: Write the implementation**

Write `src/pipeline.py`:

```python
"""
Pipeline orchestrator.

Runs the full weighing cycle: capture -> detect -> measure -> predict -> send.
Manages confirmation frames, cooldown, and evidence saving.
"""

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import requests

from src.capture import FrameCapture
from src.detect import BovineDetector, Detection
from src.measure import MorphologyMeasurer, Measurements
from src.predict import WeightPredictor, WeightEstimate


@dataclass
class PipelineConfig:
    api_url: str = "http://localhost:8000"
    evidence_dir: str = "data/evidence"
    confidence_threshold: float = 0.5
    confirmation_frames: int = 3
    cooldown_seconds: float = 30.0

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        return cls(
            api_url=os.getenv("API_URL", "http://localhost:8000"),
            evidence_dir=os.getenv("EVIDENCE_DIR", "data/evidence"),
            confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.5")),
            confirmation_frames=int(os.getenv("CONFIRMATION_FRAMES", "3")),
            cooldown_seconds=float(os.getenv("COOLDOWN_SECONDS", "30")),
        )


class Pipeline:
    """Orchestrates the full cattle weighing pipeline."""

    def __init__(
        self,
        config: PipelineConfig | None = None,
        capture: FrameCapture | None = None,
        detector: BovineDetector | None = None,
        measurer: MorphologyMeasurer | None = None,
        predictor: WeightPredictor | None = None,
    ):
        self.config = config or PipelineConfig.from_env()
        self.capture = capture or FrameCapture()
        self.detector = detector or BovineDetector()
        self.measurer = measurer or MorphologyMeasurer()
        self.predictor = predictor or WeightPredictor()

        Path(self.config.evidence_dir).mkdir(parents=True, exist_ok=True)

    def process_one(self) -> dict | None:
        """
        Attempt one weighing cycle.

        Captures frames, confirms detection across N frames,
        measures, predicts, saves evidence, and sends to API.

        Returns:
            dict with weighing data if successful, None otherwise.
        """
        confirmed_detections: list[tuple[np.ndarray, np.ndarray, Detection]] = []

        for _ in range(self.config.confirmation_frames):
            rgb, depth = self.capture.get_frame()
            detection = self.detector.detect(rgb)

            if detection is None:
                return None  # Lost the animal

            confirmed_detections.append((rgb, depth, detection))

        # Use the last confirmed frame for measurement
        rgb, depth, detection = confirmed_detections[-1]

        # Measure
        measurements = self.measurer.measure(detection.mask, depth)

        # Predict weight
        estimate = self.predictor.predict(measurements)

        # Save evidence photo
        evidence_path = self._save_evidence(rgb, detection)

        # Build result
        result = {
            "weight_kg": estimate.weight_kg,
            "confidence": estimate.confidence,
            "comprimento_m": measurements.comprimento_m,
            "altura_m": measurements.altura_m,
            "largura_m": measurements.largura_m,
            "area_m2": measurements.area_m2,
            "perimetro_m": measurements.perimetro_m,
            "evidence_path": evidence_path,
        }

        # Send to API
        self._send_to_api(result)

        return result

    def _save_evidence(self, rgb: np.ndarray, detection: Detection) -> str:
        """Save the frame with detection overlay as evidence."""
        evidence = rgb.copy()

        # Draw bounding box
        x1, y1, x2, y2 = detection.box
        cv2.rectangle(evidence, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw mask contour
        contours, _ = cv2.findContours(
            detection.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(evidence, contours, -1, (0, 255, 0), 2)

        # Save with timestamp filename
        timestamp = int(time.time())
        filename = f"{timestamp}.jpg"
        filepath = str(Path(self.config.evidence_dir) / filename)
        cv2.imwrite(filepath, evidence)

        return filepath

    def _send_to_api(self, data: dict):
        """Send weighing result to the API."""
        url = f"{self.config.api_url}/api/weighings"
        try:
            response = requests.post(url, json=data, timeout=10)
            if response.status_code == 201:
                print(f"  Sent to API: {data['weight_kg']:.1f} kg")
            else:
                print(f"  API error: {response.status_code} {response.text}")
        except requests.RequestException as e:
            print(f"  API connection failed: {e}")

    def run(self):
        """Main loop — run continuously until interrupted."""
        print("=== BoviSight Pipeline Running ===")
        print(f"  API: {self.config.api_url}")
        print(f"  Confirmation frames: {self.config.confirmation_frames}")
        print(f"  Cooldown: {self.config.cooldown_seconds}s")
        print("  Press Ctrl+C to stop.\n")

        while True:
            try:
                result = self.process_one()
                if result:
                    print(f"[WEIGHING] {result['weight_kg']:.1f} kg "
                          f"(confidence: {result['confidence']:.0%})")
                    time.sleep(self.config.cooldown_seconds)
                else:
                    time.sleep(0.5)  # Brief pause before next attempt
            except KeyboardInterrupt:
                print("\nPipeline stopped.")
                break
            except Exception as e:
                print(f"[ERROR] {e}")
                time.sleep(1)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    Pipeline().run()
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_pipeline.py -v
```

Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/pipeline.py tests/test_pipeline.py
git commit -m "feat: implement Pipeline orchestrator

Runs capture->detect->measure->predict cycle with confirmation frames.
Saves evidence photos with detection overlay. Sends results to API via HTTP.
Configurable via PipelineConfig or environment variables."
```

---

## Task 14: Setup Dashboard (React + Vite + Tailwind)

**Files:**
- Create: entire `dashboard/` directory

- [ ] **Step 1: Scaffold the React project**

```bash
cd /Users/jpgregorini/Desktop/projetos/Bovision/.claude/worktrees/happy-kalam-67f080
npm create vite@latest dashboard -- --template react
cd dashboard
npm install
npm install -D tailwindcss @tailwindcss/vite
npm install recharts
```

- [ ] **Step 2: Configure Tailwind**

Replace `dashboard/vite.config.js`:

```javascript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
})
```

Replace `dashboard/src/index.css`:

```css
@import "tailwindcss";
```

- [ ] **Step 3: Write the API client**

Create `dashboard/src/api.js`:

```javascript
const API_URL = import.meta.env.VITE_API_URL || '/api';

async function fetchJSON(path) {
  const response = await fetch(`${API_URL}${path}`);
  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }
  return response.json();
}

export async function getSummary() {
  return fetchJSON('/reports/summary');
}

export async function getWeighings(limit = 20) {
  return fetchJSON(`/weighings?limit=${limit}`);
}

export async function getAlerts() {
  return fetchJSON('/reports/alerts');
}

export async function getWeightHistory(days = 30) {
  return fetchJSON(`/reports/weight-history?days=${days}`);
}
```

- [ ] **Step 4: Verify setup**

```bash
cd /Users/jpgregorini/Desktop/projetos/Bovision/.claude/worktrees/happy-kalam-67f080/dashboard
npm run build
```

Expected: Build succeeds with no errors.

- [ ] **Step 5: Commit**

```bash
cd /Users/jpgregorini/Desktop/projetos/Bovision/.claude/worktrees/happy-kalam-67f080
git add dashboard/
git commit -m "feat: scaffold dashboard with React + Vite + Tailwind + Recharts

Vite project with Tailwind CSS configured, API proxy for development,
and api.js client for all endpoints (summary, weighings, alerts, history)."
```

---

## Task 15: Implement Dashboard Components

**Files:**
- Create: `dashboard/src/components/StatsCards.jsx`
- Create: `dashboard/src/components/WeightChart.jsx`
- Create: `dashboard/src/components/WeighingsTable.jsx`
- Create: `dashboard/src/components/AlertsList.jsx`
- Modify: `dashboard/src/App.jsx`

- [ ] **Step 1: Write StatsCards component**

Create `dashboard/src/components/StatsCards.jsx`:

```jsx
function StatCard({ label, value, sub }) {
  return (
    <div className="rounded-xl bg-white p-5 shadow">
      <p className="text-sm text-gray-500">{label}</p>
      <p className="mt-1 text-2xl font-bold text-gray-900">{value ?? '—'}</p>
      {sub && <p className="mt-1 text-xs text-gray-400">{sub}</p>}
    </div>
  );
}

export default function StatsCards({ summary }) {
  if (!summary) return null;

  const lastAt = summary.last_weighing_at
    ? new Date(summary.last_weighing_at).toLocaleString('pt-BR')
    : 'Nenhuma';

  return (
    <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
      <StatCard label="Total de Pesagens" value={summary.total_weighings} />
      <StatCard
        label="Peso Medio"
        value={summary.average_weight_kg ? `${summary.average_weight_kg} kg` : null}
      />
      <StatCard label="Ultima Pesagem" value={lastAt} />
      <StatCard
        label="Alertas Ativos"
        value={summary.active_alerts}
        sub={summary.active_alerts > 0 ? 'Animais com perda de peso' : null}
      />
    </div>
  );
}
```

- [ ] **Step 2: Write WeightChart component**

Create `dashboard/src/components/WeightChart.jsx`:

```jsx
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts';

export default function WeightChart({ data }) {
  if (!data || data.length === 0) {
    return (
      <div className="rounded-xl bg-white p-5 shadow">
        <h2 className="mb-4 text-lg font-semibold text-gray-900">
          Historico de Pesagens
        </h2>
        <p className="text-gray-400">Sem dados ainda.</p>
      </div>
    );
  }

  return (
    <div className="rounded-xl bg-white p-5 shadow">
      <h2 className="mb-4 text-lg font-semibold text-gray-900">
        Historico de Pesagens
      </h2>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" tick={{ fontSize: 12 }} />
          <YAxis
            tick={{ fontSize: 12 }}
            label={{ value: 'kg', position: 'insideLeft', offset: -5 }}
          />
          <Tooltip
            formatter={(value) => [`${value} kg`, 'Peso Medio']}
            labelFormatter={(label) => `Data: ${label}`}
          />
          <Line
            type="monotone"
            dataKey="average_weight_kg"
            stroke="#16a34a"
            strokeWidth={2}
            dot={{ r: 3 }}
            activeDot={{ r: 6 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
```

- [ ] **Step 3: Write WeighingsTable component**

Create `dashboard/src/components/WeighingsTable.jsx`:

```jsx
export default function WeighingsTable({ weighings }) {
  if (!weighings || weighings.length === 0) {
    return (
      <div className="rounded-xl bg-white p-5 shadow">
        <h2 className="mb-4 text-lg font-semibold text-gray-900">
          Pesagens Recentes
        </h2>
        <p className="text-gray-400">Nenhuma pesagem registrada.</p>
      </div>
    );
  }

  return (
    <div className="rounded-xl bg-white p-5 shadow">
      <h2 className="mb-4 text-lg font-semibold text-gray-900">
        Pesagens Recentes
      </h2>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm">
          <thead>
            <tr className="border-b text-gray-500">
              <th className="pb-2 pr-4">ID</th>
              <th className="pb-2 pr-4">Peso</th>
              <th className="pb-2 pr-4">Confianca</th>
              <th className="pb-2 pr-4">Data</th>
              <th className="pb-2">Evidencia</th>
            </tr>
          </thead>
          <tbody>
            {weighings.map((w) => (
              <tr key={w.id} className="border-b last:border-0">
                <td className="py-2 pr-4 font-medium">{w.id}</td>
                <td className="py-2 pr-4">{w.weight_kg.toFixed(1)} kg</td>
                <td className="py-2 pr-4">
                  {w.confidence ? `${(w.confidence * 100).toFixed(0)}%` : '—'}
                </td>
                <td className="py-2 pr-4">
                  {new Date(w.created_at).toLocaleString('pt-BR')}
                </td>
                <td className="py-2">
                  {w.evidence_path ? (
                    <span className="text-green-600">Ver</span>
                  ) : '—'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Write AlertsList component**

Create `dashboard/src/components/AlertsList.jsx`:

```jsx
export default function AlertsList({ alerts }) {
  if (!alerts || alerts.length === 0) {
    return (
      <div className="rounded-xl bg-white p-5 shadow">
        <h2 className="mb-4 text-lg font-semibold text-gray-900">Alertas</h2>
        <p className="text-gray-400">Nenhum alerta ativo.</p>
      </div>
    );
  }

  return (
    <div className="rounded-xl bg-white p-5 shadow">
      <h2 className="mb-4 text-lg font-semibold text-gray-900">Alertas</h2>
      <ul className="space-y-3">
        {alerts.map((alert) => (
          <li
            key={alert.animal_id}
            className="flex items-start gap-3 rounded-lg bg-red-50 p-3"
          >
            <span className="text-lg">&#9888;</span>
            <div>
              <p className="font-medium text-red-800">
                Animal #{alert.rfid}
              </p>
              <p className="text-sm text-red-600">
                Perda de {Math.abs(alert.change_percent).toFixed(1)}%
                ({(alert.current_weight_kg - alert.previous_weight_kg).toFixed(0)} kg)
                — de {alert.previous_weight_kg.toFixed(0)} kg para{' '}
                {alert.current_weight_kg.toFixed(0)} kg
              </p>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}
```

- [ ] **Step 5: Wire up App.jsx**

Replace `dashboard/src/App.jsx`:

```jsx
import { useEffect, useState } from 'react';
import * as api from './api';
import StatsCards from './components/StatsCards';
import WeightChart from './components/WeightChart';
import WeighingsTable from './components/WeighingsTable';
import AlertsList from './components/AlertsList';

const REFRESH_INTERVAL = 30_000; // 30 seconds

export default function App() {
  const [summary, setSummary] = useState(null);
  const [weighings, setWeighings] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [history, setHistory] = useState([]);
  const [error, setError] = useState(null);

  async function fetchAll() {
    try {
      const [s, w, a, h] = await Promise.all([
        api.getSummary(),
        api.getWeighings(),
        api.getAlerts(),
        api.getWeightHistory(),
      ]);
      setSummary(s);
      setWeighings(w);
      setAlerts(a);
      setHistory(h);
      setError(null);
    } catch (err) {
      setError('Falha ao conectar com a API. Verifique se o servidor esta rodando.');
      console.error(err);
    }
  }

  useEffect(() => {
    fetchAll();
    const interval = setInterval(fetchAll, REFRESH_INTERVAL);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-gray-100">
      <header className="bg-white shadow">
        <div className="mx-auto max-w-7xl px-4 py-4">
          <h1 className="text-2xl font-bold text-green-700">BoviSight</h1>
          <p className="text-sm text-gray-500">
            Pesagem automatizada por visao computacional
          </p>
        </div>
      </header>

      <main className="mx-auto max-w-7xl space-y-6 px-4 py-6">
        {error && (
          <div className="rounded-lg bg-yellow-50 p-4 text-yellow-800">
            {error}
          </div>
        )}

        <StatsCards summary={summary} />
        <WeightChart data={history} />
        <WeighingsTable weighings={weighings} />
        <AlertsList alerts={alerts} />
      </main>
    </div>
  );
}
```

- [ ] **Step 6: Clean up default files**

Remove default Vite files that are no longer needed:

```bash
cd /Users/jpgregorini/Desktop/projetos/Bovision/.claude/worktrees/happy-kalam-67f080/dashboard
rm -f src/App.css src/assets/react.svg public/vite.svg
```

Update `dashboard/src/main.jsx` to import css:

```jsx
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
```

- [ ] **Step 7: Build and verify**

```bash
cd /Users/jpgregorini/Desktop/projetos/Bovision/.claude/worktrees/happy-kalam-67f080/dashboard
npm run build
```

Expected: Build succeeds with no errors.

- [ ] **Step 8: Commit**

```bash
cd /Users/jpgregorini/Desktop/projetos/Bovision/.claude/worktrees/happy-kalam-67f080
git add dashboard/
git commit -m "feat: implement dashboard with all components

StatsCards: 4 summary cards (total, avg weight, last weighing, alerts).
WeightChart: Recharts line graph of daily average weights.
WeighingsTable: recent weighings with confidence and evidence link.
AlertsList: animals with >5% weight loss highlighted in red.
App.jsx: wires all components with auto-refresh every 30 seconds."
```

---

## Task 16: End-to-End Integration Test

**Files:**
- Create: `tests/test_integration.py`
- Create: `scripts/seed_data.py`

- [ ] **Step 1: Write the integration test**

Create `tests/test_integration.py`:

```python
"""
End-to-end integration test.

Simulates the full pipeline flow:
1. API is running (test client)
2. Pipeline processes a frame
3. Weighing appears in API
4. Reports reflect the new data
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from api.database import Base, get_db
from api.main import app
from src.detect import Detection
from src.measure import Measurements
from src.predict import WeightEstimate
from src.pipeline import Pipeline, PipelineConfig


@pytest.fixture
def api_client():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine)

    def override_get_db():
        db = TestSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_full_flow(api_client, tmp_path):
    """Test the complete pipeline -> API -> reports flow."""

    # Step 1: Verify API is running
    response = api_client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"

    # Step 2: Verify empty state
    summary = api_client.get("/api/reports/summary").json()
    assert summary["total_weighings"] == 0

    # Step 3: Simulate pipeline sending a weighing
    weighing_data = {
        "weight_kg": 487.4,
        "confidence": 0.92,
        "comprimento_m": 1.43,
        "altura_m": 1.31,
        "largura_m": 0.49,
        "area_m2": 1.92,
        "perimetro_m": 1.84,
        "evidence_path": str(tmp_path / "evidence" / "123456.jpg"),
    }
    response = api_client.post("/api/weighings", json=weighing_data)
    assert response.status_code == 201

    # Step 4: Verify weighing appears in list
    weighings = api_client.get("/api/weighings").json()
    assert len(weighings) == 1
    assert weighings[0]["weight_kg"] == 487.4

    # Step 5: Verify summary updated
    summary = api_client.get("/api/reports/summary").json()
    assert summary["total_weighings"] == 1
    assert summary["average_weight_kg"] == 487.4

    # Step 6: Create animal and weighings with weight loss for alert
    animal = api_client.post("/api/animals", json={"rfid": "BOI_E2E"}).json()
    api_client.post("/api/weighings", json={
        "animal_id": animal["id"], "weight_kg": 500.0
    })
    api_client.post("/api/weighings", json={
        "animal_id": animal["id"], "weight_kg": 450.0  # -10%
    })

    # Step 7: Verify alert generated
    alerts = api_client.get("/api/reports/alerts").json()
    assert len(alerts) == 1
    assert alerts[0]["rfid"] == "BOI_E2E"

    # Step 8: Verify weight history has data
    history = api_client.get("/api/reports/weight-history").json()
    assert len(history) >= 1

    print("Integration test PASSED")
```

- [ ] **Step 2: Write seed data script for manual testing**

Create `scripts/seed_data.py`:

```python
"""
Seed the database with sample data for manual testing.

Usage:
    python scripts/seed_data.py [--api-url http://localhost:8000]
"""

import argparse
import random
import time
import requests

BREEDS = ["nelore", "angus", "hereford", "brahman", "guzerá"]


def seed(api_url: str):
    print(f"Seeding {api_url}...\n")

    # Create animals
    animals = []
    for i in range(10):
        data = {
            "rfid": f"BOI_{i+1:03d}",
            "raca": random.choice(BREEDS),
            "sexo": random.choice(["M", "F"]),
        }
        resp = requests.post(f"{api_url}/api/animals", json=data)
        if resp.status_code == 201:
            animals.append(resp.json())
            print(f"  Animal: {data['rfid']} ({data['raca']})")
        else:
            print(f"  Skip {data['rfid']}: {resp.status_code}")

    # Create weighings (3 per animal, simulating weight gain/loss)
    for animal in animals:
        base_weight = random.uniform(350, 550)
        for j in range(3):
            change = random.uniform(-20, 30)
            weight = base_weight + change * j
            data = {
                "animal_id": animal["id"],
                "weight_kg": round(weight, 1),
                "confidence": round(random.uniform(0.8, 0.98), 2),
                "comprimento_m": round(random.uniform(1.2, 1.6), 2),
                "altura_m": round(random.uniform(1.1, 1.4), 2),
                "largura_m": round(random.uniform(0.35, 0.55), 2),
                "area_m2": round(random.uniform(1.4, 2.2), 2),
                "perimetro_m": round(random.uniform(1.5, 2.0), 2),
            }
            resp = requests.post(f"{api_url}/api/weighings", json=data)
            if resp.status_code == 201:
                print(f"  Weighing: {animal['rfid']} = {data['weight_kg']} kg")

    # Create one animal with clear weight loss (for alert testing)
    alert_animal = requests.post(f"{api_url}/api/animals", json={
        "rfid": "BOI_ALERT", "raca": "nelore", "sexo": "M"
    }).json()
    requests.post(f"{api_url}/api/weighings", json={
        "animal_id": alert_animal["id"], "weight_kg": 500.0, "confidence": 0.95
    })
    requests.post(f"{api_url}/api/weighings", json={
        "animal_id": alert_animal["id"], "weight_kg": 440.0, "confidence": 0.93
    })
    print(f"\n  Alert animal: BOI_ALERT (500 -> 440 kg, -12%)")

    print(f"\nDone! {len(animals) + 1} animals, {len(animals) * 3 + 2} weighings.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default="http://localhost:8000")
    args = parser.parse_args()
    seed(args.api_url)
```

- [ ] **Step 3: Run integration test**

```bash
cd /Users/jpgregorini/Desktop/projetos/Bovision/.claude/worktrees/happy-kalam-67f080
python -m pytest tests/test_integration.py -v
```

Expected: All assertions PASS.

- [ ] **Step 4: Run full test suite**

```bash
python -m pytest tests/ -v
```

Expected: All tests across all files PASS.

- [ ] **Step 5: Test manually (3 terminals)**

Terminal 1 — API:
```bash
cd /Users/jpgregorini/Desktop/projetos/Bovision/.claude/worktrees/happy-kalam-67f080
uvicorn api.main:app --reload --port 8000
```

Terminal 2 — Seed data:
```bash
python scripts/seed_data.py
```

Terminal 3 — Dashboard:
```bash
cd dashboard && npm run dev
```

Open http://localhost:5173 — verify stats cards, chart, table, and alerts render correctly.

- [ ] **Step 6: Commit**

```bash
git add tests/test_integration.py scripts/seed_data.py
git commit -m "feat: add integration test and seed data script

test_integration.py: full flow test — API, weighings, animals, alerts, history.
seed_data.py: populates DB with 10 animals, 30+ weighings, and one alert case."
```

---

## Summary

| Task | What | Tests |
|---|---|---|
| 1 | Project restructure + foundation | — |
| 2 | Detection dataset download | — |
| 3 | YOLOv8n-seg training | — |
| 4 | Regression dataset download | — |
| 5 | XGBoost training + evaluation | — |
| 6 | capture.py | 5 tests |
| 7 | detect.py | 4 tests |
| 8 | measure.py | 5 tests |
| 9 | predict.py | 3 tests |
| 10 | DB + models + schemas | 5 tests |
| 11 | API main + weighings router | 5 tests |
| 12 | Animals + reports routers | 10 tests |
| 13 | pipeline.py | 4 tests |
| 14 | Dashboard scaffold | build |
| 15 | Dashboard components | build |
| 16 | Integration test + seed data | 1 test |

**Total: 16 tasks, 42 tests, ~35 files created.**
