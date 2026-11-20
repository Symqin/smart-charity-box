"""Train YOLO26n untuk nominal uang, lalu export ke ONNX.

    .venv/Scripts/python ai/yolo/train.py                       # train penuh
    .venv/Scripts/python ai/yolo/train.py --epochs 1 --frac 0.05 # uji cepat

Hasil: runs/<name>/weights/best.pt dan best.onnx
"""
import argparse
import os

from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=f"{ROOT}/ai/datasets/merged_v2/data.yaml")
    local = f"{ROOT}/ai/models/yolo26n.pt"   # tidak di-commit; kalau tidak ada, Ultralytics mengunduh otomatis
    ap.add_argument("--model", default=local if os.path.exists(local) else "yolo26n.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)     # RTX 2050 4 GB
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--name", default="uang_yolo26n")
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--workers", type=int, default=4)   # tiap worker ±1,5 GB RAM di Windows
    ap.add_argument("--resume", action="store_true")
    # fine-tune dari model yang sudah bagus: lr kecil + warmup pendek, kalau tidak val turun dulu
    ap.add_argument("--optimizer", default="auto")
    ap.add_argument("--lr0", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=3.0)
    args = ap.parse_args()

    if args.resume:
        model = YOLO(f"{ROOT}/runs/{args.name}/weights/last.pt")
        model.train(resume=True, workers=args.workers)   # workers boleh diubah saat resume
    else:
        model = YOLO(args.model)
        model.train(
            data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
            fraction=args.frac, project=f"{ROOT}/runs", name=args.name, exist_ok=True,
            patience=args.patience, workers=args.workers, seed=0, plots=True,
            # Augmentasi: JANGAN flip horizontal/vertikal (angka nominal jadi cermin).
            fliplr=0.0, flipud=0.0,
            degrees=10, translate=0.1, scale=0.5,
            hsv_h=0.015, hsv_s=0.6, hsv_v=0.5,   # variasi cahaya LED / pantulan nampan
            mosaic=1.0, close_mosaic=10,
            optimizer=args.optimizer, lr0=args.lr0, warmup_epochs=args.warmup,
        )

    best = YOLO(f"{ROOT}/runs/{args.name}/weights/best.pt")
    best.val(data=args.data, imgsz=args.imgsz, batch=args.batch, plots=True,
             project=f"{ROOT}/runs", name=f"{args.name}_val", exist_ok=True)
    best.export(format="onnx", imgsz=args.imgsz, simplify=True)


if __name__ == "__main__":
    main()
