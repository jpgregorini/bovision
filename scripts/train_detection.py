#!/usr/bin/env python3
"""
Train YOLOv8n-seg for cattle instance segmentation.

Usage:
    python scripts/train_detection.py [--epochs 50] [--imgsz 640] \
                                      [--batch 16] [--device 0] [--resume]
"""

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "data" / "annotated" / "data.yaml"
MODELS_DIR = ROOT / "models" / "detection"
RUNS_DIR = ROOT / "runs" / "detect"


def parse_args():
    parser = argparse.ArgumentParser(description="Treina YOLOv8n-seg para detecção de bovinos")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Número de épocas de treinamento (padrão: 50)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Tamanho da imagem de entrada em pixels (padrão: 640)")
    parser.add_argument("--batch", type=int, default=16,
                        help="Tamanho do batch (padrão: 16; use -1 para AutoBatch)")
    parser.add_argument("--device", type=str, default=None,
                        help="Dispositivo: '0', '0,1', 'cpu', ou None para auto-detect")
    parser.add_argument("--resume", action="store_true",
                        help="Retoma treinamento a partir do último checkpoint")
    parser.add_argument("--data", type=Path, default=DATA_YAML,
                        help="Caminho para data.yaml")
    parser.add_argument("--project", type=str, default=str(RUNS_DIR),
                        help="Diretório raiz para salvar os runs do YOLO")
    parser.add_argument("--name", type=str, default="bovino_seg",
                        help="Nome do run")
    return parser.parse_args()


def check_dependencies():
    try:
        from ultralytics import YOLO  # noqa: F401
    except ImportError:
        print("[erro] 'ultralytics' não instalado. Execute:")
        print("       pip install ultralytics")
        sys.exit(1)


def check_data_yaml(path: Path):
    if not path.exists():
        print(f"[erro] data.yaml não encontrado em {path}")
        print("       Execute primeiro: python scripts/download_detection_data.py")
        sys.exit(1)


def train(args) -> Path:
    """Run YOLO training and return the run directory."""
    from ultralytics import YOLO

    print("[info] Carregando modelo base yolov8n-seg.pt ...")
    model = YOLO("yolov8n-seg.pt")

    train_kwargs = dict(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        resume=args.resume,
        exist_ok=True,
        verbose=True,
        # Augmentation / training hyper-params
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        mosaic=1.0,
        close_mosaic=10,
        amp=True,       # Automatic Mixed Precision
        seed=42,
    )

    if args.device is not None:
        train_kwargs["device"] = args.device

    print(f"[train] Iniciando treinamento: epochs={args.epochs} imgsz={args.imgsz} batch={args.batch}")
    model.train(**train_kwargs)

    # YOLO saves runs to <project>/<name>/
    run_dir = Path(args.project) / args.name
    return run_dir


def export_artifacts(run_dir: Path):
    """Copy best.pt to models/detection/ and export to ONNX."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    best_src = run_dir / "weights" / "best.pt"
    best_dst = MODELS_DIR / "best.pt"

    if not best_src.exists():
        print(f"[aviso] best.pt não encontrado em {best_src}")
        return

    shutil.copy2(best_src, best_dst)
    print(f"[export] best.pt copiado para {best_dst}")

    # Export to ONNX
    try:
        from ultralytics import YOLO
        model = YOLO(str(best_dst))
        onnx_path = model.export(format="onnx", imgsz=640, opset=12, simplify=True)
        onnx_dst = MODELS_DIR / "best.onnx"
        shutil.copy2(onnx_path, onnx_dst)
        print(f"[export] ONNX exportado para {onnx_dst}")
    except Exception as exc:
        print(f"[aviso] Falha ao exportar ONNX: {exc}")


def print_metrics_summary(run_dir: Path):
    """Print the final validation metrics from results.csv if available."""
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        return

    try:
        import csv
        with open(results_csv) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return
        last = rows[-1]
        print("\n" + "=" * 50)
        print("METRICAS FINAIS (ultima epoca)")
        print("=" * 50)
        for key, val in last.items():
            key = key.strip()
            if any(m in key for m in ["Box", "Mask", "mAP", "precision", "recall"]):
                try:
                    print(f"  {key:40s}: {float(val):.4f}")
                except ValueError:
                    pass
        print("=" * 50 + "\n")
    except Exception as exc:
        print(f"[aviso] Nao foi possivel ler metricas: {exc}")


def main():
    args = parse_args()

    check_dependencies()
    check_data_yaml(args.data)

    run_dir = train(args)
    export_artifacts(run_dir)
    print_metrics_summary(run_dir)

    print("[OK] Treinamento concluido!")
    print(f"     Modelo: {MODELS_DIR / 'best.pt'}")
    print(f"     ONNX  : {MODELS_DIR / 'best.onnx'}")


if __name__ == "__main__":
    main()
