"""Uji model pada dataset eksternal yang TIDAK dipakai saat training.

Dataset (unduh ke ai/datasets/external/):
  cirengbasii/uang-2016-2022-1     (Roboflow fix-tmc9q, CC BY 4.0) -> split test saja
  cirengbasii/Rupiah-Detector-1    (Roboflow rupiah-detector, CC BY 4.0) -> split test saja
  (train/valid uang-2016-2022 & Rupiah-Detector dipakai di training v2)
  cirengbasii/deteksi-mata-uang-rupiah-1 (Roboflow skripsi-3kth2, CC BY 4.0) -> semua split
  drawrs                           (github.com/drawrs/dataset-indonesian-rupiah, VOC xml)

    .venv/Scripts/python ai/yolo/eval_external.py
"""
import argparse
import collections
import glob
import os
import re
import shutil
import xml.etree.ElementTree as ET

from ultralytics import YOLO

CLASSES = ["k1000", "k2000", "k5000", "k10000", "k20000", "k50000", "k100000",
           "c100", "c200", "c500", "c1000", "lain"]
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXT = f"{ROOT}/ai/datasets/external"
OUT = f"{EXT}/_test"

UANG2016 = ["k20000", "c200", "k2000", "k50000", "c500", "k5000", "k10000", "k100000",
            "c100", "k1000", "c1000"]
SEVEN = ["k20000", "k2000", "k50000", "k5000", "k10000", "k100000", "k1000"]
KERTAS_ONLY = False   # diisi dari --kertas
DRAWRS = {"1": "k1000", "2": "k2000", "5": "k5000", "10": "k10000", "20": "k20000",
          "50": "k50000", "100": "k100000"}


def to_box(v):
    v = list(map(float, v))
    if len(v) == 4:
        return v
    xs, ys = v[0::2], v[1::2]
    x1, x2, y1, y2 = max(0, min(xs)), min(1, max(xs)), max(0, min(ys)), min(1, max(ys))
    return [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]


def link(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build_yolo(name, base, splits, names):
    out = f"{OUT}/{name}"
    os.makedirs(f"{out}/images"); os.makedirs(f"{out}/labels")
    for sp in splits:
        for img in glob.glob(f"{base}/{sp}/images/*"):
            stem = f"{sp}_{os.path.splitext(os.path.basename(img))[0]}"
            lines = []
            for l in open(f"{base}/{sp}/labels/{os.path.splitext(os.path.basename(img))[0]}.txt"):
                p = l.split()
                if len(p) >= 5:
                    if KERTAS_ONLY and names[int(p[0])].startswith("c"):
                        continue                         # mode kertas: koin tidak dinilai
                    cx, cy, w, h = to_box(p[1:])
                    lines.append(f"{CLASSES.index(names[int(p[0])])} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            if lines:
                link(img, f"{out}/images/{stem}{os.path.splitext(img)[1]}")
                open(f"{out}/labels/{stem}.txt", "w").write("\n".join(lines) + "\n")
    return out


def build_drawrs():
    base, out = f"{EXT}/drawrs", f"{OUT}/drawrs"
    os.makedirs(f"{out}/images"); os.makedirs(f"{out}/labels")
    for x in glob.glob(f"{base}/annotations/*.xml"):
        r = ET.parse(x).getroot()
        fn = r.findtext("filename").strip()
        if not os.path.exists(f"{base}/{fn}"):
            continue
        W, H = float(r.findtext("size/width")), float(r.findtext("size/height"))
        lines = []
        for o in r.iter("object"):
            m = re.match(r"ID_(\d+)_", o.findtext("name").strip())
            bb = o.find("bndbox")
            x1, y1, x2, y2 = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
            lines.append(f"{CLASSES.index(DRAWRS[m.group(1)])} {(x1 + x2) / 2 / W:.6f} "
                         f"{(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
        stem = os.path.splitext(fn)[0].replace(" ", "_")
        link(f"{base}/{fn}", f"{out}/images/{stem}{os.path.splitext(fn)[1]}")
        open(f"{out}/labels/{stem}.txt", "w").write("\n".join(lines) + "\n")
    return out


def decision_stats(model, folder, thr):
    """Simulasi alur keputusan: per objek ground truth, apakah prediksi yang
    paling cocok (IoU>=0.5) benar & yakin, salah & yakin, atau ragu/hilang."""
    import torch
    from ultralytics.utils.metrics import box_iou
    st = collections.Counter()
    imgs = sorted(glob.glob(f"{folder}/images/*"))
    # Instance baru + CPU: memakai ulang model setelah .val() memicu OOM di GPU 4 GB.
    model = YOLO(model.ckpt_path)
    for r in model.predict(imgs, imgsz=640, conf=0.25, verbose=False, stream=True, device="cpu"):
        stem = os.path.splitext(os.path.basename(r.path))[0]
        gt = [l.split() for l in open(f"{folder}/labels/{stem}.txt") if l.strip()]
        if not gt:
            continue
        h, w = r.orig_shape
        g = torch.tensor([[(float(a) - float(c) / 2) * w, (float(b) - float(d) / 2) * h,
                           (float(a) + float(c) / 2) * w, (float(b) + float(d) / 2) * h]
                          for _, a, b, c, d in gt])
        gcls = [int(x[0]) for x in gt]
        pb = r.boxes.xyxy.cpu()
        iou = box_iou(g, pb) if len(pb) else torch.zeros((len(g), 0))
        for i, c in enumerate(gcls):
            st["n"] += 1
            if iou.shape[1] == 0 or iou[i].max() < 0.5:
                st["hilang"] += 1; continue
            j = int(iou[i].argmax())
            pc, cf = int(r.boxes.cls[j]), float(r.boxes.conf[j])
            if cf < thr:
                st["ragu"] += 1
            elif pc == c:
                st["benar"] += 1
            else:
                st["salah"] += 1
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=f"{ROOT}/ai/models/uang_v2.pt")
    ap.add_argument("--thr", type=float, default=0.85)
    ap.add_argument("--tag", default="v2", help="nama subfolder hasil di runs/external/")
    ap.add_argument("--kertas", action="store_true", help="nilai uang kertas saja (label koin dibuang)")
    args = ap.parse_args()
    global KERTAS_ONLY
    KERTAS_ONLY = args.kertas

    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    sets = {
        "uang2016_2022_test": build_yolo("uang2016_2022_test", f"{EXT}/cirengbasii/uang-2016-2022-1", ["test"], UANG2016),
        "rupiah_detector_test": build_yolo("rupiah_detector_test", f"{EXT}/cirengbasii/Rupiah-Detector-1", ["test"], SEVEN),
        "deteksi_mata_uang": build_yolo("deteksi_mata_uang", f"{EXT}/cirengbasii/deteksi-mata-uang-rupiah-1", ["train", "valid", "test"], SEVEN),
        "drawrs": build_drawrs(),
    }
    model = YOLO(args.model)
    names = [model.names[i] for i in range(len(model.names))]   # 12 kelas (v2) atau 7 (model kertas)
    summary = []
    for name, folder in sets.items():
        yml = f"{OUT}/{name}.yaml"
        open(yml, "w").write(f"path: {folder}\ntrain: images\nval: images\nnc: 12\nnames: {CLASSES}\n")
        n_img = len(os.listdir(f"{folder}/images"))
        print(f"\n######## {name} ({n_img} foto)")
        r = model.val(data=yml, imgsz=640, batch=16, workers=0, plots=True,
                      project=f"{ROOT}/runs/external/{args.tag}", name=name, exist_ok=True)
        st = decision_stats(model, folder, args.thr)
        summary.append((name, n_img, r.box.mp, r.box.mr, r.box.map50, r.box.map, st))

    print(f"\n{'dataset':<20}{'foto':>6}{'P':>7}{'R':>7}{'mAP50':>7}{'m50-95':>8}"
          f" | objek  benar  salah  ragu  hilang  (ambang {args.thr})")
    for name, n, p, rc, m50, m, st in summary:
        print(f"{name:<20}{n:>6}{p:>7.3f}{rc:>7.3f}{m50:>7.3f}{m:>8.3f} | {st['n']:>5}{st['benar']:>7}"
              f"{st['salah']:>7}{st['ragu']:>6}{st['hilang']:>8}")


if __name__ == "__main__":
    main()
