"""Auto-label bounding box untuk rupiah-banknotes-dataset (latar kertas putih).

Setiap folder idr_<nominal> berisi foto satu uang kertas, jadi kelas diambil dari
nama folder. Bounding box dicari dengan OpenCV: uang lebih gelap dan/atau lebih
berwarna daripada kertas putih di belakangnya. idr_mix dilewati (labeli manual).

Hasil (format YOLO, siap diunggah ke Roboflow untuk dicek):
    <out>/images/*.jpg      foto yang sudah diputar sesuai EXIF dan diperkecil
    <out>/labels/*.txt      label YOLO, indeks kelas mengikuti 12 kelas proyek
    <out>/preview/ok|cek/   foto kecil dengan kotak tergambar untuk dicek cepat
    <out>/review.csv        daftar foto + alasan perlu dicek
    <out>/data.yaml

Contoh:
    .venv/Scripts/python ai/yolo/data_prep/autolabel_rbd.py \
        --src ai/datasets/rupiah-banknotes-dataset \
        --out ai/datasets/rbd_autolabel
"""
import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from PIL import Image, ImageOps

CLASSES = ["k1000", "k2000", "k5000", "k10000", "k20000", "k50000", "k100000",
           "c100", "c200", "c500", "c1000", "lain"]
FOLDER_TO_CLASS = {
    "idr_1000": "k1000", "idr_2000": "k2000", "idr_5000": "k5000",
    "idr_10000": "k10000", "idr_20000": "k20000", "idr_50000": "k50000",
    "idr_100000": "k100000",
}
WORK = 640          # sisi terpanjang gambar kerja untuk segmentasi
T_DARK = 22         # selisih kecerahan (L) terhadap kertas
T_VERY_DARK = 70    # sangat gelap: pasti objek walau tidak bertekstur
T_TEX = 5           # std lokal L minimum agar area gelap dianggap uang (bukan bayangan)
T_CHROMA = 10       # selisih warna (a,b) terhadap kertas
MIN_AREA = 0.004    # komponen lebih kecil dari ini (fraksi gambar) diabaikan


def load(path, max_side):
    im = Image.open(path)
    im.draft("RGB", (max_side, max_side))       # decode JPEG lebih cepat
    im = ImageOps.exif_transpose(im).convert("RGB")
    im.thumbnail((max_side, max_side), Image.LANCZOS)
    return im


def find_box(bgr):
    """Kembalikan (x1, y1, x2, y2) dalam piksel gambar kerja, info komponen, mask."""
    h, w = bgr.shape[:2]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]

    # Perkiraan kecerahan kertas: closing besar menghapus objek gelap (uang),
    # sisa gradien cahaya/vignette kertas.
    small = cv2.resize(L, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    bgL = cv2.morphologyEx(small, cv2.MORPH_CLOSE, k)
    bgL = cv2.GaussianBlur(bgL, (0, 0), 6)
    bgL = cv2.resize(bgL, (w, h), interpolation=cv2.INTER_LINEAR)

    diffL = bgL - L
    # Tekstur lokal: uang penuh detail cetakan, bayangan/pojok gelap halus.
    mean = cv2.blur(L, (9, 9))
    std = np.sqrt(np.maximum(cv2.blur(L * L, (9, 9)) - mean * mean, 0))
    textured = cv2.dilate((std > T_TEX).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    dark = ((diffL > T_DARK) & textured) | (diffL > T_VERY_DARK)
    a0, b0 = np.median(a), np.median(b)          # warna kertas (dominan)
    chroma = np.sqrt((a - a0) ** 2 + (b - b0) ** 2) > T_CHROMA
    mask = (dark | chroma).astype(np.uint8) * 255

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))

    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    comps = []
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        if area < MIN_AREA * w * h:
            continue
        # Buang strip tipis yang menempel di tepi foto (tepi meja/kertas).
        at_edge = x <= 2 or y <= 2 or x + cw >= w - 2 or y + ch >= h - 2
        if at_edge and min(cw, ch) < 0.06 * max(w, h) and max(cw, ch) > 5 * min(cw, ch):
            continue
        comps.append(stats[i])
    if not comps:
        return None, [], mask
    comps.sort(key=lambda s: -s[cv2.CC_STAT_AREA])
    big = comps[0][cv2.CC_STAT_AREA]
    # Uang terlipat kadang terpecah jadi beberapa bagian: gabungkan bagian besar.
    keep = [s for s in comps if s[cv2.CC_STAT_AREA] >= 0.15 * big]
    x1 = min(s[0] for s in keep); y1 = min(s[1] for s in keep)
    x2 = max(s[0] + s[2] for s in keep); y2 = max(s[1] + s[3] for s in keep)
    return (x1, y1, x2, y2), keep, mask


def process(job):
    path, folder, out, max_side = job
    cls = FOLDER_TO_CLASS[folder]
    stem = f"{cls}_{os.path.splitext(os.path.basename(path))[0]}"
    im = load(path, max_side)
    W, H = im.size
    rgb = np.array(im)
    work = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    s = WORK / max(W, H)
    work = cv2.resize(work, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA)
    h, w = work.shape[:2]

    box, comps, _ = find_box(work)
    reasons = []
    if box is None:
        reasons.append("tidak_ada_objek")
    else:
        x1, y1, x2, y2 = box
        frac = (x2 - x1) * (y2 - y1) / (w * h)
        if frac < 0.02:
            reasons.append("kotak_kecil")
        if frac > 0.6:
            reasons.append("kotak_besar")
        if len(comps) > 1:
            reasons.append(f"{len(comps)}_bagian")
        m = 3
        if x1 <= m or y1 <= m or x2 >= w - m or y2 >= h - m:
            reasons.append("tepi")    # info saja: uang terpotong di tepi foto itu wajar

    im.save(f"{out}/images/{stem}.jpg", quality=92)
    with open(f"{out}/labels/{stem}.txt", "w") as f:
        if box is not None:
            x1, y1, x2, y2 = box
            f.write(f"{CLASSES.index(cls)} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                    f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}\n")

    need_check = any(r != "tepi" for r in reasons)
    prev = work.copy()
    if box is not None:
        cv2.rectangle(prev, box[:2], box[2:], (0, 0, 255) if need_check else (0, 200, 0), 3)
    cv2.putText(prev, f"{cls} {' '.join(reasons)}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 2)
    cv2.imwrite(f"{out}/preview/{'cek' if need_check else 'ok'}/{stem}.jpg", prev,
                [cv2.IMWRITE_JPEG_QUALITY, 75])
    return stem, folder, os.path.basename(path), need_check, " ".join(reasons)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-side", type=int, default=1280)
    ap.add_argument("--limit", type=int, default=0, help="maks foto per folder (uji coba)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    for d in ["images", "labels", "preview/ok", "preview/cek"]:
        os.makedirs(f"{args.out}/{d}", exist_ok=True)

    jobs = []
    for folder in sorted(FOLDER_TO_CLASS):
        files = sorted(f for f in os.listdir(f"{args.src}/{folder}")
                       if f.lower().endswith((".jpg", ".jpeg", ".png")))
        if args.limit:
            files = files[:: max(1, len(files) // args.limit)][: args.limit]
        jobs += [(f"{args.src}/{folder}/{f}", folder, args.out, args.max_side) for f in files]

    with ProcessPoolExecutor(args.workers) as ex:
        rows = list(ex.map(process, jobs, chunksize=4))

    with open(f"{args.out}/review.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["file", "folder", "asli", "perlu_cek", "alasan"])
        wr.writerows(rows)
    with open(f"{args.out}/data.yaml", "w") as f:
        f.write("train: images\nval: images\n")
        f.write(f"nc: {len(CLASSES)}\nnames: [{', '.join(CLASSES)}]\n")

    n_cek = sum(r[3] for r in rows)
    print(f"Selesai: {len(rows)} foto, {n_cek} perlu dicek ({args.out}/preview/cek)")


if __name__ == "__main__":
    main()
