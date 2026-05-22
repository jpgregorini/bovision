#!/usr/bin/env python3
"""
Download cattle detection dataset from Kaggle, convert to YOLO format,
split into train/val/test and write data.yaml.
"""

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

# ── Project paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw" / "detection"
DATA_ANNOTATED = ROOT / "data" / "annotated"
DATA_SAMPLE = ROOT / "data" / "sample_images"
MODELS_DIR = ROOT / "models" / "detection"

KAGGLE_DATASETS = [
    "trainingdatapro/cattle-detection",
    "ayuraj/cow-detection-dataset",
]

RANDOM_SEED = 42
SPLIT = (0.80, 0.10, 0.10)  # train / val / test
N_SAMPLE = 10


# ── Kaggle helpers ─────────────────────────────────────────────────────────────

def check_kaggle_credentials() -> bool:
    """Return True if kaggle.json exists or env vars are set."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        return True
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return False


def download_from_kaggle(dataset: str, dest: Path) -> bool:
    """Try to download *dataset* using the Kaggle API. Returns True on success."""
    try:
        import kaggle  # noqa: F401 – just to check import
        from kaggle.api.kaggle_api_extended import KaggleApiExtended

        api = KaggleApiExtended()
        api.authenticate()
        dest.mkdir(parents=True, exist_ok=True)
        print(f"[download] Baixando dataset '{dataset}' …")
        api.dataset_download_files(dataset, path=str(dest), unzip=True, quiet=False)
        return True
    except Exception as exc:
        print(f"[aviso] Falha ao baixar '{dataset}': {exc}")
        return False


def print_manual_instructions():
    print("\n" + "=" * 70)
    print("DOWNLOAD MANUAL NECESSÁRIO")
    print("=" * 70)
    print("Não foi possível baixar automaticamente via API do Kaggle.")
    print("Faça o download manual de um dos datasets abaixo e extraia")
    print(f"o conteúdo em: {DATA_RAW}\n")
    for ds in KAGGLE_DATASETS:
        print(f"  https://www.kaggle.com/datasets/{ds}")
    print("\nAlternativamente, configure a API do Kaggle:")
    print("  1. Acesse https://www.kaggle.com/account e gere um token API")
    print("  2. Salve o arquivo kaggle.json em ~/.kaggle/kaggle.json")
    print("  3. Execute este script novamente")
    print("=" * 70 + "\n")


# ── Annotation conversion ──────────────────────────────────────────────────────

def coco_to_yolo(coco_json_path: Path, images_dir: Path, yolo_labels_dir: Path):
    """Convert COCO JSON annotations to per-image YOLO .txt files (bbox only)."""
    yolo_labels_dir.mkdir(parents=True, exist_ok=True)

    with open(coco_json_path) as f:
        coco = json.load(f)

    # Build lookup tables
    img_id_to_meta = {img["id"]: img for img in coco["images"]}
    # Map category id → 0 (only one class: bovino)
    annotations_by_image: dict[int, list] = {}
    for ann in coco.get("annotations", []):
        annotations_by_image.setdefault(ann["image_id"], []).append(ann)

    converted = 0
    for img_id, anns in annotations_by_image.items():
        meta = img_id_to_meta.get(img_id)
        if meta is None:
            continue
        w, h = meta["width"], meta["height"]
        stem = Path(meta["file_name"]).stem
        label_path = yolo_labels_dir / f"{stem}.txt"

        lines = []
        for ann in anns:
            bx, by, bw, bh = ann["bbox"]  # COCO: x_min, y_min, width, height
            cx = (bx + bw / 2) / w
            cy = (by + bh / 2) / h
            nw = bw / w
            nh = bh / h
            lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        label_path.write_text("\n".join(lines))
        converted += 1

    print(f"[convert] {converted} imagens convertidas de COCO para YOLO.")


def is_yolo_labels_dir(path: Path) -> bool:
    """Check if a directory contains YOLO-format .txt label files."""
    txts = list(path.glob("*.txt"))
    if not txts:
        return False
    # Peek at first file – YOLO lines are: <class> <cx> <cy> <w> <h>
    try:
        first = txts[0].read_text().strip().splitlines()
        if first:
            parts = first[0].split()
            return len(parts) == 5 and parts[0].isdigit()
    except Exception:
        pass
    return False


# ── Dataset organization ───────────────────────────────────────────────────────

def find_images_and_labels(base: Path):
    """
    Walk *base* and return (images, labels) lists of Paths.
    Tries to pair images with YOLO .txt files.
    """
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    all_images = [p for p in base.rglob("*") if p.suffix.lower() in image_exts]

    paired_images = []
    paired_labels = []
    unpaired_images = []

    for img in all_images:
        # Look for .txt sibling or in a parallel 'labels' directory
        candidates = [
            img.with_suffix(".txt"),
            img.parent.parent / "labels" / img.with_suffix(".txt").name,
        ]
        found_label = None
        for c in candidates:
            if c.exists():
                found_label = c
                break
        if found_label:
            paired_images.append(img)
            paired_labels.append(found_label)
        else:
            unpaired_images.append(img)

    if unpaired_images:
        print(f"[aviso] {len(unpaired_images)} imagens sem anotações encontradas – serão ignoradas.")

    return paired_images, paired_labels


def split_dataset(images, labels, seed=RANDOM_SEED, split=SPLIT):
    """Return (train, val, test) tuples of (images, labels) lists."""
    combined = list(zip(images, labels))
    random.seed(seed)
    random.shuffle(combined)

    n = len(combined)
    n_train = int(n * split[0])
    n_val = int(n * split[1])

    train = combined[:n_train]
    val = combined[n_train : n_train + n_val]
    test = combined[n_train + n_val :]
    return train, val, test


def copy_split(pairs, dest_images: Path, dest_labels: Path):
    dest_images.mkdir(parents=True, exist_ok=True)
    dest_labels.mkdir(parents=True, exist_ok=True)
    for img_path, lbl_path in pairs:
        shutil.copy2(img_path, dest_images / img_path.name)
        shutil.copy2(lbl_path, dest_labels / lbl_path.name)


# ── data.yaml ─────────────────────────────────────────────────────────────────

def write_data_yaml(dest: Path):
    train_path = (dest / "train" / "images").as_posix()
    val_path = (dest / "val" / "images").as_posix()
    test_path = (dest / "test" / "images").as_posix()

    yaml_content = (
        f"path: {dest.as_posix()}\n"
        f"train: {train_path}\n"
        f"val:   {val_path}\n"
        f"test:  {test_path}\n"
        "\n"
        "nc: 1\n"
        "names: ['bovino']\n"
    )
    yaml_path = dest / "data.yaml"
    yaml_path.write_text(yaml_content)
    print(f"[yaml] data.yaml salvo em {yaml_path}")


# ── Sample images ──────────────────────────────────────────────────────────────

def copy_sample_images(train_images_dir: Path, dest: Path, n: int = N_SAMPLE):
    dest.mkdir(parents=True, exist_ok=True)
    images = list(train_images_dir.glob("*"))
    random.seed(RANDOM_SEED)
    sample = random.sample(images, min(n, len(images)))
    for img in sample:
        shutil.copy2(img, dest / img.name)
    print(f"[sample] {len(sample)} imagens copiadas para {dest}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download e prepara dataset de detecção de bovinos")
    parser.add_argument("--raw-dir", type=Path, default=DATA_RAW,
                        help="Diretório onde o dataset bruto será/está salvo")
    parser.add_argument("--out-dir", type=Path, default=DATA_ANNOTATED,
                        help="Diretório de saída para dados organizados (data.yaml)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Pula o download e usa dados já presentes em --raw-dir")
    args = parser.parse_args()

    raw_dir: Path = args.raw_dir
    out_dir: Path = args.out_dir

    # ── 1. Download ────────────────────────────────────────────────────────────
    if not args.skip_download:
        if not check_kaggle_credentials():
            print("[aviso] Credenciais do Kaggle não encontradas.")
            print_manual_instructions()
            if not raw_dir.exists() or not any(raw_dir.rglob("*")):
                sys.exit(1)
            print("[info] Usando dados existentes em", raw_dir)
        else:
            downloaded = False
            for dataset in KAGGLE_DATASETS:
                if download_from_kaggle(dataset, raw_dir):
                    downloaded = True
                    break
            if not downloaded:
                print_manual_instructions()
                if not raw_dir.exists() or not any(raw_dir.rglob("*")):
                    sys.exit(1)
    else:
        print(f"[info] Usando dados existentes em {raw_dir}")

    # ── 2. Detect format and convert if needed ─────────────────────────────────
    coco_jsons = list(raw_dir.rglob("*.json"))
    yolo_label_dirs = [p for p in raw_dir.rglob("labels") if p.is_dir() and is_yolo_labels_dir(p)]

    if coco_jsons and not yolo_label_dirs:
        # Assume first JSON is the annotations
        coco_json = coco_jsons[0]
        print(f"[convert] Detectado formato COCO: {coco_json}")
        images_dir = coco_json.parent / "images"
        if not images_dir.exists():
            images_dir = coco_json.parent
        yolo_labels_out = raw_dir / "labels_yolo"
        coco_to_yolo(coco_json, images_dir, yolo_labels_out)
    elif yolo_label_dirs:
        print(f"[info] Formato YOLO detectado em {yolo_label_dirs[0]}")
    else:
        print("[aviso] Nenhuma anotação encontrada. Verifique o conteúdo de", raw_dir)

    # ── 3. Pair images and labels ──────────────────────────────────────────────
    images, labels = find_images_and_labels(raw_dir)
    print(f"[info] {len(images)} pares imagem/anotação encontrados.")
    if not images:
        print("[erro] Nenhum par imagem/anotação encontrado. Abortando.")
        sys.exit(1)

    # ── 4. Split ───────────────────────────────────────────────────────────────
    train, val, test = split_dataset(images, labels)
    print(f"[split] train={len(train)} | val={len(val)} | test={len(test)}")

    for pairs, split_name in [(train, "train"), (val, "val"), (test, "test")]:
        copy_split(pairs, out_dir / split_name / "images", out_dir / split_name / "labels")

    # ── 5. Write data.yaml ─────────────────────────────────────────────────────
    write_data_yaml(out_dir)

    # ── 6. Sample images for pipeline mock ────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    copy_sample_images(out_dir / "train" / "images", DATA_SAMPLE)

    print("\n[OK] Dataset de detecção preparado com sucesso!")
    print(f"     Configuração: {out_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()
