"""Uji per foto untuk foto berisi SATU uang/koin yang dikelompokkan per folder (tanpa bbox).

Struktur folder: <root>/<nama folder kelas>/[subfolder...]/*.jpg
Nama folder dipetakan ke kelas lewat angka di dalamnya, mis. "Koin Rp 500" -> c500
(pakai --kertas kalau isinya uang kertas: "Rp 5000" -> k5000).

    .venv/Scripts/python ai/yolo/eval_koin.py ai/datasets/external/rikisanjaya/raw/dataset_splitted

Kategori per foto (ambang --thr, default 0,85), dari deteksi dengan confidence tertinggi:
  benar  = kelas benar & yakin      salah = kelas salah & yakin (paling berbahaya)
  ragu   = confidence < ambang       hilang = tidak ada deteksi sama sekali
"""
import argparse
import collections
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict import DEFAULT_MODEL, IMG_EXT, MoneyDetector  # noqa: E402


def folder_class(name, kertas):
    m = re.search(r"(\d+)\s*(k|rb|ribu)?", name.replace(".", "").lower())
    if not m:
        return None
    n = int(m.group(1)) * (1000 if m.group(2) else 1)
    if kertas and n < 1000:
        n *= 1000                                      # "10" -> 10000
    return ("k" if kertas else "c") + str(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--thr", type=float, default=0.85)
    ap.add_argument("--kertas", action="store_true")
    ap.add_argument("--save-wrong", help="folder untuk menyimpan foto yang salah/ragu/hilang")
    args = ap.parse_args()

    det = MoneyDetector(args.model, thr=args.thr)
    st = collections.defaultdict(collections.Counter)
    cm = collections.defaultdict(collections.Counter)          # gt -> prediksi teratas (tanpa ambang)
    for d in sorted(os.listdir(args.root)):
        gt = folder_class(d, args.kertas)
        if not gt:
            continue
        files = [f for f in glob.glob(os.path.join(args.root, d, "**", "*"), recursive=True)
                 if f.lower().endswith(IMG_EXT)]
        for f in sorted(files):
            h = det.predict(f)
            top = max(h["items"], key=lambda x: x["confidence"], default=None)
            if top is None:
                r = "hilang"; cm[gt]["(tidak ada)"] += 1
            else:
                cm[gt][top["kelas"]] += 1
                r = "ragu" if not top["yakin"] else ("benar" if top["kelas"] == gt else "salah")
            st[gt][r] += 1
            st[gt]["n"] += 1
            if args.save_wrong and r != "benar":
                import cv2
                out = os.path.join(args.save_wrong, f"{gt}_{r}")
                os.makedirs(out, exist_ok=True)
                cv2.imwrite(os.path.join(out, os.path.basename(f)), h["result"].plot())
        print(f"{gt}: {dict(st[gt])}", flush=True)

    print(f"\nmodel: {args.model}\n{'kelas':<8}{'foto':>6}{'benar':>7}{'salah':>7}{'ragu':>6}{'hilang':>8}"
          f"{'top-1':>8}   prediksi teratas")
    tot = collections.Counter()
    for gt, s in st.items():
        tot.update(s)
        top1 = cm[gt][gt] / s["n"]
        print(f"{gt:<8}{s['n']:>6}{s['benar']:>7}{s['salah']:>7}{s['ragu']:>6}{s['hilang']:>8}{top1:>8.1%}   "
              + ", ".join(f"{k} {v}" for k, v in cm[gt].most_common()))
    top1 = sum(cm[g][g] for g in cm) / tot["n"]
    print(f"{'total':<8}{tot['n']:>6}{tot['benar']:>7}{tot['salah']:>7}{tot['ragu']:>6}{tot['hilang']:>8}{top1:>8.1%}")


if __name__ == "__main__":
    main()
