"""Dataset KHUSUS UANG KERTAS (7 kelas) diturunkan dari merged_v2.

- Label koin (c100..c1000) dan 'lain' dibuang; indeks k1000..k100000 tetap 0..6.
- Foto yang berisi uang kertas dipakai semua (koin di foto itu jadi latar).
- Foto yang HANYA berisi koin -> sebagian dipakai sebagai foto negatif (label kosong),
  supaya model belajar bahwa koin bukan uang kertas. Salinan __dup tidak dipakai.
- Ditambah foto nampan hitam kosong sintetis (doff/mengkilap + pantulan LED) sebagai negatif.

    .venv/Scripts/python ai/yolo/data_prep/make_kertas.py
Hasil: ai/datasets/merged_kertas/{train,val}/{images,labels} + data.yaml
"""
import argparse
import collections
import glob
import os
import random
import shutil
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth_black import background  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
KERTAS = ["k1000", "k2000", "k5000", "k10000", "k20000", "k50000", "k100000"]


def link(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def empty_tray(rng):
    img = background(rng)
    img += rng.normal(0, rng.uniform(1, 5), img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.5))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(35, 91))])
    return buf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=f"{ROOT}/ai/datasets/merged_v2")
    ap.add_argument("--out", default=f"{ROOT}/ai/datasets/merged_kertas")
    ap.add_argument("--neg-frac", type=float, default=0.08, help="foto koin-saja, relatif thd jumlah foto kertas")
    ap.add_argument("--empty", type=int, nargs=2, default=[300, 50], help="nampan kosong train val")
    args = ap.parse_args()

    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    rnd, rng = random.Random(0), np.random.default_rng(0)
    for split, n_empty in zip(("train", "val"), args.empty):
        os.makedirs(f"{args.out}/{split}/images"); os.makedirs(f"{args.out}/{split}/labels")
        per_cls, per_src, coin_only, n_pos = collections.Counter(), collections.Counter(), [], 0
        for lab in sorted(glob.glob(f"{args.src}/{split}/labels/*.txt")):
            stem = os.path.splitext(os.path.basename(lab))[0]
            img = glob.glob(f"{args.src}/{split}/images/{stem}.*")[0]
            keep = [l for l in open(lab) if l.strip() and int(l.split()[0]) < len(KERTAS)]
            if keep:
                link(img, f"{args.out}/{split}/images/{os.path.basename(img)}")
                open(f"{args.out}/{split}/labels/{stem}.txt", "w").write("".join(
                    l if l.endswith("\n") else l + "\n" for l in keep))
                n_pos += 1
                per_src[stem.split("_")[0]] += 1
                per_cls.update(KERTAS[int(l.split()[0])] for l in keep)
            elif "__dup" not in stem:
                coin_only.append(img)
        negs = rnd.sample(coin_only, min(len(coin_only), int(n_pos * args.neg_frac)))
        for img in negs:
            link(img, f"{args.out}/{split}/images/neg_{os.path.basename(img)}")
            open(f"{args.out}/{split}/labels/neg_{os.path.splitext(os.path.basename(img))[0]}.txt", "w").close()
        for i in range(n_empty):
            empty_tray(rng).tofile(f"{args.out}/{split}/images/kosong_{split}_{i:04d}.jpg")
            open(f"{args.out}/{split}/labels/kosong_{split}_{i:04d}.txt", "w").close()
        print(f"\n{split}: {n_pos} foto uang kertas + {len(negs)} negatif koin + {n_empty} nampan kosong"
              f" = {n_pos + len(negs) + n_empty}")
        print("  per sumber:", dict(per_src))
        print("  per kelas :", {k: per_cls[k] for k in KERTAS})

    open(f"{args.out}/data.yaml", "w").write(
        f"path: {args.out}\ntrain: train/images\nval: val/images\nnc: {len(KERTAS)}\nnames: {KERTAS}\n")
    print(f"\ndata.yaml -> {args.out}/data.yaml")


if __name__ == "__main__":
    main()
