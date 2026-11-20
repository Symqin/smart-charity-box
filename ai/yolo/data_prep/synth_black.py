"""Buat data sintetis mirip nampan kotak amal: uang (lecek/terlipat) dan koin
ditempel ke latar hitam (doff atau mengkilap dengan pantulan LED).

Sumber potongan:
  uang kertas : foto rbd_* di ai/datasets/merged/{train,val} (latar putih ->
                mask dari autolabel_rbd.find_box)
  koin        : foto idr_* di ai/datasets/merged/{train,val} (mask lingkaran dari bbox)
Potongan dari split train hanya dipakai untuk sintetis train, dan val untuk val,
supaya tidak bocor.

    .venv/Scripts/python ai/yolo/data_prep/synth_black.py --n-train 3000 --n-val 300
"""
import argparse
import glob
import os
import random
import shutil
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from autolabel_rbd import find_box  # noqa: E402

CLASSES = ["k1000", "k2000", "k5000", "k10000", "k20000", "k50000", "k100000",
           "c100", "c200", "c500", "c1000", "lain"]
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
W, H = 640, 480                     # rasio 4:3 seperti VGA/SVGA ESP32-CAM


def read_label(path):
    return [l.split() for l in open(path) if l.strip()]


def note_cutouts(split):
    """Potongan uang kertas (BGRA, alpha = mask) dari foto rbd latar putih."""
    outs = []
    for img_path in sorted(glob.glob(f"{ROOT}/ai/datasets/merged/{split}/images/rbd_*.jpg")):
        lab = read_label(img_path.replace("images", "labels")[:-4] + ".txt")
        if len(lab) != 1:
            continue
        cls = int(lab[0][0])
        img = cv2.imread(img_path)
        s = 640 / max(img.shape[:2])
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        box, comps, mask = find_box(img)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        hh, ww = mask.shape
        if x1 <= 3 or y1 <= 3 or x2 >= ww - 3 or y2 >= hh - 3:
            continue                   # uang terpotong tepi foto -> tepi lurus tidak alami
        m = np.zeros_like(mask)
        m[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
        # GrabCut membuang sisa kertas/bayangan yang ikut masuk mask kasar.
        gc = np.full(m.shape, cv2.GC_BGD, np.uint8)
        gc[cv2.dilate(m, np.ones((15, 15), np.uint8)) > 0] = cv2.GC_PR_BGD
        gc[m > 0] = cv2.GC_PR_FGD
        gc[cv2.erode(m, np.ones((15, 15), np.uint8)) > 0] = cv2.GC_FGD
        try:
            cv2.grabCut(img, gc, None, np.zeros((1, 65), np.float64),
                        np.zeros((1, 65), np.float64), 3, cv2.GC_INIT_WITH_MASK)
        except cv2.error:
            continue
        m = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(m)
        if n < 2:
            continue
        m = np.where(lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA]), 255, 0).astype(np.uint8)
        # Isi lubang di dalam uang (bagian pucat) lewat flood fill dari luar.
        ff = cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        cv2.floodFill(ff, None, (0, 0), 255)
        m = m | cv2.bitwise_not(ff[1:-1, 1:-1])
        ys, xs = np.nonzero(m)
        x1, x2, y1, y2 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        m = cv2.erode(m, np.ones((3, 3), np.uint8))
        m = cv2.GaussianBlur(m, (5, 5), 0)
        crop = img[y1:y2, x1:x2]
        if crop.shape[0] < 20 or crop.shape[1] < 20:
            continue
        outs.append((cls, np.dstack([crop, m[y1:y2, x1:x2]])))
    return outs


def coin_cutouts(split):
    outs = []
    for img_path in sorted(glob.glob(f"{ROOT}/ai/datasets/merged/{split}/images/idr_*.jpg")):
        if "__dup" in img_path:
            continue
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        for c, cx, cy, bw, bh in read_label(img_path.replace("images", "labels")[:-4] + ".txt"):
            cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
            x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
            x2, y2 = int(cx + bw / 2), int(cy + bh / 2)
            crop = img[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                continue
            m = np.zeros(crop.shape[:2], np.uint8)
            cv2.ellipse(m, (crop.shape[1] // 2, crop.shape[0] // 2),
                        (crop.shape[1] // 2 - 1, crop.shape[0] // 2 - 1), 0, 0, 360, 255, -1)
            m = cv2.GaussianBlur(m, (5, 5), 0)
            outs.append((int(c), np.dstack([crop, m])))
    return outs


def background(rng):
    base = rng.uniform(4, 40)
    tint = rng.uniform(-4, 4, 3)
    bg = np.full((H, W, 3), base, np.float32) + tint
    # gradien cahaya
    gx, gy = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H))
    a, b = rng.uniform(-1, 1, 2)
    bg += (a * gx + b * gy)[..., None] * rng.uniform(0, 15)
    # tekstur (kain/plastik) + noise sensor
    tex = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), rng.uniform(0.5, 3))
    bg += tex[..., None] * rng.uniform(0, 6)
    # nampan mengkilap: pantulan LED (bercak terang)
    if rng.random() < 0.5:
        for _ in range(rng.integers(1, 3)):
            cx, cy = rng.uniform(0, W), rng.uniform(0, H)
            sx, sy = rng.uniform(15, 120), rng.uniform(10, 80)
            blob = np.exp(-(((gx + 1) * W / 2 - cx) ** 2 / (2 * sx ** 2) +
                            ((gy + 1) * H / 2 - cy) ** 2 / (2 * sy ** 2)))
            bg += blob[..., None] * rng.uniform(60, 220)
    return bg


def transform(cut, target_len, angle, rng):
    """Skala + rotasi (bukan cermin) + variasi cahaya per objek."""
    h, w = cut.shape[:2]
    s = target_len / max(h, w)
    cut = cv2.resize(cut, (max(2, int(w * s)), max(2, int(h * s))), interpolation=cv2.INTER_AREA)
    h, w = cut.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    cut = cv2.warpAffine(cut, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
    rgb = cut[..., :3].astype(np.float32) * rng.uniform(0.6, 1.15) + rng.uniform(-20, 15)
    return np.clip(rgb, 0, 255), cut[..., 3].astype(np.float32) / 255


def compose(notes, coins, rng):
    canvas = background(rng)
    layers = []                                   # (cls, full-frame alpha)
    k = rng.choice([1, 1, 1, 2, 2, 2, 3, 3, 4])
    items = [("note", notes[rng.integers(len(notes))]) for _ in range(k)]
    if coins and rng.random() < 0.35:
        items += [("coin", coins[rng.integers(len(coins))]) for _ in range(rng.integers(1, 3))]
    rng.shuffle(items)
    for kind, (cls, cut) in items:
        if kind == "note":
            target = rng.uniform(0.35, 1.15) * W   # 1.15: uang memenuhi/terpotong frame
        else:
            target = rng.uniform(0.10, 0.22) * W
        rgb, a = transform(cut, target, rng.uniform(0, 360), rng)
        h, w = a.shape
        for _ in range(25):
            x0 = int(rng.uniform(-w * 0.3, W - w * 0.7))
            y0 = int(rng.uniform(-h * 0.3, H - h * 0.7))
            full = np.zeros((H, W), np.float32)
            xs, ys = max(0, x0), max(0, y0)
            xe, ye = min(W, x0 + w), min(H, y0 + h)
            if xe <= xs or ye <= ys:
                continue
            full[ys:ye, xs:xe] = a[ys - y0:ye - y0, xs - x0:xe - x0]
            if (full > 0.5).sum() < 0.4 * (a > 0.5).sum():
                continue                           # terlalu banyak di luar frame
            # jangan tutupi >50% objek yang sudah ada
            ok = all(((lay > 0.5) & ~(full > 0.5)).sum() >= 0.5 * (lay0 > 0.5).sum()
                     for _, lay, lay0 in layers)
            if not ok:
                continue
            img = np.zeros((H, W, 3), np.float32)
            img[ys:ye, xs:xe] = rgb[ys - y0:ye - y0, xs - x0:xe - x0]
            canvas = canvas * (1 - full[..., None]) + img * full[..., None]
            layers = [(c, lay * (1 - full), lay0) for c, lay, lay0 in layers]
            layers.append((cls, full, full.copy()))
            break
    # bayangan halus + vignette LED + blur/noise/JPEG seperti OV2640
    gx, gy = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H))
    canvas *= (1 - rng.uniform(0, 0.35) * (gx ** 2 + gy ** 2) / 2)[..., None]
    canvas = cv2.GaussianBlur(canvas, (0, 0), rng.uniform(0.3, 1.3))
    canvas += rng.normal(0, rng.uniform(1, 6), canvas.shape)
    canvas = np.clip(canvas, 0, 255).astype(np.uint8)
    q = int(rng.uniform(35, 90))
    canvas = cv2.imdecode(cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, q])[1], 1)
    labels = []
    for c, lay, _ in layers:
        ys, xs = np.nonzero(lay > 0.5)
        if len(xs) < 30:
            continue
        x1, x2, y1, y2 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        labels.append(f"{c} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
                      f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
    return canvas, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{ROOT}/ai/datasets/synth")
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-val", type=int, default=300)
    ap.add_argument("--preview", type=int, default=0, help="hanya buat N contoh ke --out")
    args = ap.parse_args()
    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    rng = np.random.default_rng(0)
    for split, n in (("train", args.preview or args.n_train), ("val", 0 if args.preview else args.n_val)):
        if n == 0:
            continue
        notes, coins = note_cutouts(split), coin_cutouts(split)
        print(f"{split}: {len(notes)} potongan uang, {len(coins)} potongan koin")
        os.makedirs(f"{args.out}/{split}/images"); os.makedirs(f"{args.out}/{split}/labels")
        for i in range(n):
            img, labels = compose(notes, coins, rng)
            cv2.imwrite(f"{args.out}/{split}/images/syn_{split}_{i:05d}.jpg", img)
            open(f"{args.out}/{split}/labels/syn_{split}_{i:05d}.txt", "w").write("\n".join(labels) + "\n")
    print("selesai")


if __name__ == "__main__":
    main()
