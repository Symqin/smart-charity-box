"""Pakai model nominal uang: deteksi semua uang di foto + alur keputusan ambang 0,85.

Command line:
    .venv/Scripts/python ai/yolo/predict.py foto.jpg
    .venv/Scripts/python ai/yolo/predict.py folder_foto/ --save hasil/      # simpan foto berkotak
    .venv/Scripts/python ai/yolo/predict.py foto.jpg --model ai/models/uang_v2.pt --device 0

Dari kode Python lain (mis. server FastAPI):
    import sys; sys.path.insert(0, "ai/yolo")
    from predict import MoneyDetector
    det = MoneyDetector()                      # default: models/uang_v2.onnx di CPU
    hasil = det.predict("foto.jpg")            # path, bytes JPEG, array BGR, atau PIL.Image
    hasil["status"]   # "ok" | "ragu" | "kosong"
    hasil["total"]    # jumlah rupiah dari uang yang yakin (confidence >= ambang)
    hasil["items"]    # list dict per uang: kelas, nominal, jenis, confidence, yakin, box
"""
import argparse
import glob
import os

import cv2
import numpy as np
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(os.path.dirname(HERE), "models", "uang_v2.onnx")

NOMINAL = {"k1000": 1000, "k2000": 2000, "k5000": 5000, "k10000": 10000, "k20000": 20000,
           "k50000": 50000, "k100000": 100000,
           "c100": 100, "c200": 200, "c500": 500, "c1000": 1000, "lain": 0}
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class MoneyDetector:
    def __init__(self, model=DEFAULT_MODEL, thr=0.85, conf=0.25, imgsz=640, device="cpu"):
        self.model = YOLO(model, task="detect")
        self.thr, self.conf, self.imgsz, self.device = thr, conf, imgsz, device

    @staticmethod
    def _load(image):
        if isinstance(image, (bytes, bytearray)):            # body HTTP POST dari ESP32-CAM
            image = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("bytes bukan gambar yang valid")
        return image

    def predict(self, image):
        r = self.model.predict(self._load(image), imgsz=self.imgsz, conf=self.conf,
                               device=self.device, verbose=False)[0]
        items = []
        for box, c, cf in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            kelas = r.names[int(c)]
            items.append({
                "kelas": kelas,
                "nominal": NOMINAL[kelas],
                "jenis": "koin" if kelas.startswith("c") else ("lain" if kelas == "lain" else "kertas"),
                "confidence": round(cf, 4),
                "yakin": cf >= self.thr,
                "box": [round(v, 1) for v in box],            # x1, y1, x2, y2 (piksel)
            })
        items.sort(key=lambda d: (d["box"][1], d["box"][0]))
        if not items:
            status = "kosong"
        elif all(d["yakin"] for d in items):
            status = "ok"                                     # langsung catat
        else:
            status = "ragu"                                   # goyang nampan + foto ulang, lalu Gemini
        total = sum(d["nominal"] for d in items if d["yakin"])
        return {"status": status, "total": total, "items": items, "result": r}


def expand(paths):
    for p in paths:
        if os.path.isdir(p):
            yield from sorted(f for f in glob.glob(os.path.join(p, "*")) if f.lower().endswith(IMG_EXT))
        else:
            yield p


def main():
    ap = argparse.ArgumentParser(description="Deteksi nominal uang Rupiah di foto")
    ap.add_argument("source", nargs="+", help="file foto atau folder")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=".onnx (CPU) atau .pt")
    ap.add_argument("--thr", type=float, default=0.85, help="ambang yakin")
    ap.add_argument("--device", default="cpu", help="cpu atau 0 (GPU, hanya untuk .pt)")
    ap.add_argument("--save", help="folder untuk menyimpan foto berkotak")
    args = ap.parse_args()

    det = MoneyDetector(args.model, thr=args.thr, device=args.device)
    if args.save:
        os.makedirs(args.save, exist_ok=True)
    for path in expand(args.source):
        h = det.predict(path)
        print(f"\n{os.path.basename(path)}  ->  status={h['status']}  total=Rp{h['total']:,}".replace(",", "."))
        for d in h["items"]:
            print(f"   {d['kelas']:<8} conf {d['confidence']:.2f}  {'yakin' if d['yakin'] else 'RAGU'}")
        if args.save:
            cv2.imwrite(os.path.join(args.save, os.path.basename(path)), h["result"].plot())


if __name__ == "__main__":
    main()
