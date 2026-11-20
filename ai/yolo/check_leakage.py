"""Cek kebocoran test -> train: cari foto test yang hampir identik dengan foto train/val (dHash).

    .venv/Scripts/python ai/yolo/check_leakage.py --test "ai/datasets/kaggle_uang_emisi_2022/Uang Baru" \
        --train ai/datasets/merged_kertas/train/images ai/datasets/merged_kertas/val/images

Prefix nama file train (rbd_, rk_, u1622_, ...) dipakai untuk menunjukkan sumber yang bocor.
Ambang default 20 dari 256 bit; 0 = identik.
"""
import argparse
import collections
import glob
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def dhash(path):
    im = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None
    im = cv2.resize(im, (17, 16), interpolation=cv2.INTER_AREA)
    return np.packbits((im[:, 1:] > im[:, :-1]).ravel())


def collect(folders):
    files = [f for d in folders for f in glob.glob(os.path.join(d, "**", "*"), recursive=True)
             if f.lower().endswith(IMG_EXT)]
    with ThreadPoolExecutor(12) as ex:
        hs = list(ex.map(dhash, files))
    return [(f, h) for f, h in zip(files, hs) if h is not None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", nargs="+", required=True)
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--thr", type=int, default=20)
    args = ap.parse_args()

    train = collect(args.train)
    T = np.array([h for _, h in train])
    src = [os.path.basename(f).split("_")[0] for f, _ in train]
    test = collect(args.test)
    leak = collections.Counter()
    for f, h in test:
        d = np.unpackbits(T ^ h, axis=1).sum(1)
        i = int(d.argmin())
        if d[i] <= args.thr:
            leak[src[i]] += 1
    n = sum(leak.values())
    print(f"{len(test)} foto test, {len(train)} foto train/val")
    print(f"hampir identik (<= {args.thr}/256 bit): {n} ({n / max(1, len(test)):.1%})  per sumber: {dict(leak)}")


if __name__ == "__main__":
    main()
