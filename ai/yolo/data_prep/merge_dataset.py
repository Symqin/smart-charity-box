"""Gabungkan semua dataset publik ke 12 kelas proyek, format YOLO detect.

Sumber (di folder ai/datasets/):
  rbd_autolabel/          hasil autolabel_rbd.py (foto yang ditandai 'cek' dilewati)
  roboflow_kertas_koin/   frame video; diambil tiap N frame, split per blok waktu per video
  roboflow_koin/          split asli dipertahankan (valid+test -> val)
  idr-coins/              COCO -> YOLO, diperkecil ke 1280 px

Split dibuat per *blok berurutan* (foto/frame yang diambil berdekatan masuk ke
split yang sama) supaya val tidak berisi hampir-duplikat dari train.
Tidak ada split test: test set nanti hanya dari foto kotak amal sendiri.

    .venv/Scripts/python ai/yolo/data_prep/merge_dataset.py --root ai/datasets --out ai/datasets/merged
"""
import argparse
import collections
import csv
import json
import os
import re
import shutil

from PIL import Image, ImageOps

CLASSES = ["k1000", "k2000", "k5000", "k10000", "k20000", "k50000", "k100000",
           "c100", "c200", "c500", "c1000", "lain"]
CID = {c: i for i, c in enumerate(CLASSES)}

RK_MAP = {"1000": "c1000", "100rb": "k100000", "10rb": "k10000", "1rb": "k1000",
          "20rb": "k20000", "2rb": "k2000", "500": "c500", "50rb": "k50000", "5rb": "k5000"}
KOIN_MAP = {"100": "c100", "1000": "c1000", "200": "c200", "500": "c500"}
COCO_MAP = {"one_hundred_rupiah": "c100", "two_hundred_rupiah": "c200",
            "five_hundred_rupiah": "c500", "one_thousand_rupiah": "c1000"}


def yaml_names(path):
    return re.findall(r"^- '?([^'\n]+?)'?$", open(path).read(), re.M)


def poly_to_box(parts):
    """Satu baris label YOLO (bbox atau polygon) -> (cx, cy, w, h)."""
    v = list(map(float, parts))
    if len(v) == 4:
        return tuple(v)
    xs, ys = v[0::2], v[1::2]
    x1, x2 = max(0, min(xs)), min(1, max(xs))
    y1, y2 = max(0, min(ys)), min(1, max(ys))
    return ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)


def block_split(keys, block, val_every=5, offset=2):
    """keys berurutan -> dict key->split. Tiap blok ke-`val_every` masuk val."""
    return {k: ("val" if (i // block) % val_every == offset else "train")
            for i, k in enumerate(keys)}


class Writer:
    def __init__(self, out, max_side=0):
        self.out = out
        self.max_side = max_side    # perkecil gambar besar -> loading data lebih cepat
        self.count = collections.Counter()   # (split, class) -> instance
        self.nimg = collections.Counter()
        for sp in ("train", "val"):
            for d in ("images", "labels"):
                os.makedirs(f"{out}/{sp}/{d}", exist_ok=True)

    def add(self, split, name, src_img, boxes, resize=None, copies=1):
        """boxes: list of (class_name, cx, cy, w, h)."""
        for c in range(copies):
            stem = name if c == 0 else f"{name}__dup{c}"
            dst = f"{self.out}/{split}/images/{stem}.jpg"
            if not os.path.exists(dst):
                resize = min(resize or 10 ** 9, self.max_side or 10 ** 9)
                if c > 0:
                    src_img = f"{self.out}/{split}/images/{name}.jpg"   # salinan dari hasil pertama
                    resize = 10 ** 9
                if max(Image.open(src_img).size) > resize:
                    im = ImageOps.exif_transpose(Image.open(src_img)).convert("RGB")
                    im.thumbnail((resize, resize), Image.LANCZOS)
                    im.save(dst, quality=92)
                else:
                    try:
                        os.link(src_img, dst)        # hardlink: hemat disk
                    except OSError:
                        shutil.copy2(src_img, dst)
            with open(f"{self.out}/{split}/labels/{stem}.txt", "w") as f:
                for cls, cx, cy, w, h in boxes:
                    f.write(f"{CID[cls]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                    self.count[(split, cls)] += 1
            self.nimg[split] += 1


def rare_copies(boxes, rare, n):
    return n if any(b[0] in rare for b in boxes) else 1


def do_rbd(root, wr):
    d = f"{root}/rbd_autolabel"
    skip = {r["file"] for r in csv.DictReader(open(f"{d}/review.csv")) if r["perlu_cek"] == "True"}
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(f"{d}/images"))
    by_cls = collections.defaultdict(list)
    for s in stems:
        if s not in skip:
            by_cls[s.split("_")[0]].append(s)
    for cls, ss in by_cls.items():        # nama file = timestamp -> urutan waktu
        split = block_split(ss, block=8)
        for s in ss:
            lines = open(f"{d}/labels/{s}.txt").read().split("\n")
            boxes = [(CLASSES[int(p[0])], *map(float, p[1:5])) for p in
                     (l.split() for l in lines) if len(p) == 5]
            if boxes:
                wr.add(split[s], f"rbd_{s}", f"{d}/images/{s}.jpg", boxes)
    print(f"rbd: {sum(map(len, by_cls.values()))} foto ({len(skip)} 'cek' dilewati)")


def do_rk(root, wr, every):
    d = f"{root}/roboflow_kertas_koin"
    names = yaml_names(f"{d}/data.yaml")
    videos = collections.defaultdict(list)
    for sp in ("train", "valid", "test"):
        for f in os.listdir(f"{d}/{sp}/images"):
            m = re.match(r"(.+?)_mp4-(\d+)_jpg", f)
            videos[m.group(1)].append((int(m.group(2)), sp, os.path.splitext(f)[0]))
    n = 0
    for vid, frames in videos.items():
        frames.sort()
        frames = frames[::every]
        split = block_split([fr[0] for fr in frames], block=max(4, 40 // every))
        for idx, sp, stem in frames:
            boxes = []
            for line in open(f"{d}/{sp}/labels/{stem}.txt").read().splitlines():
                p = line.split()
                if len(p) >= 5:
                    boxes.append((RK_MAP[names[int(p[0])]], *poly_to_box(p[1:])))
            if boxes:
                wr.add(split[idx], f"rk_{vid}_{idx:04d}", f"{d}/{sp}/images/{stem}.jpg", boxes)
                n += 1
    print(f"roboflow_kertas_koin: {n} frame dari {len(videos)} video (tiap {every} frame)")


def do_koin(root, wr, rare, dup):
    d = f"{root}/roboflow_koin"
    names = yaml_names(f"{d}/data.yaml")
    n = 0
    for sp, out_sp in (("train", "train"), ("valid", "val"), ("test", "val")):
        for f in sorted(os.listdir(f"{d}/{sp}/images")):
            stem = os.path.splitext(f)[0]
            boxes = [(KOIN_MAP[names[int(p[0])]], *poly_to_box(p[1:])) for p in
                     (l.split() for l in open(f"{d}/{sp}/labels/{stem}.txt").read().splitlines())
                     if len(p) >= 5]
            if boxes:
                cp = rare_copies(boxes, rare, dup) if out_sp == "train" else 1
                wr.add(out_sp, f"koin_{n:04d}", f"{d}/{sp}/images/{f}", boxes, copies=cp)
                n += 1
    print(f"roboflow_koin: {n} foto")


def do_idr(root, wr, rare, dup):
    d = f"{root}/idr-coins"
    coco = json.load(open(f"{d}/_annotations.coco.json"))
    cat = {c["id"]: c["name"] for c in coco["categories"]}
    anns = collections.defaultdict(list)
    for a in coco["annotations"]:
        anns[a["image_id"]].append(a)
    imgs = sorted(coco["images"], key=lambda i: i.get("extra", {}).get("name", i["file_name"]))
    split = block_split([i["id"] for i in imgs], block=6)
    for im in imgs:
        W, H = im["width"], im["height"]
        boxes = []
        for a in anns[im["id"]]:
            name = cat[a["category_id"]]
            if name not in COCO_MAP:
                continue
            x, y, w, h = a["bbox"]
            boxes.append((COCO_MAP[name], (x + w / 2) / W, (y + h / 2) / H, w / W, h / H))
        if boxes:
            sp = split[im["id"]]
            cp = rare_copies(boxes, rare, dup) if sp == "train" else 1
            wr.add(sp, f"idr_{os.path.splitext(im['file_name'])[0]}", f"{d}/{im['file_name']}",
                   boxes, resize=1280, copies=cp)
    print(f"idr-coins: {len(imgs)} foto")


def drop_part_boxes(boxes):
    """Buang kotak yang hanya menandai BAGIAN uang (angka/potret): kotak yang
    >=85% luasnya berada di dalam kotak lain berkelas sama yang lebih besar."""
    keep = []
    for i, (c, cx, cy, w, h) in enumerate(boxes):
        a1 = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
        inside = False
        for j, (c2, cx2, cy2, w2, h2) in enumerate(boxes):
            if i == j or c2 != c or w2 * h2 <= w * h:
                continue
            b = (cx2 - w2 / 2, cy2 - h2 / 2, cx2 + w2 / 2, cy2 + h2 / 2)
            iw = max(0, min(a1[2], b[2]) - max(a1[0], b[0]))
            ih = max(0, min(a1[3], b[3]) - max(a1[1], b[1]))
            if iw * ih >= 0.85 * w * h:
                inside = True
                break
        if not inside and w * h > 0.001:
            keep.append((c, cx, cy, w, h))
    return keep


def do_roboflow(root, wr, name, rel, names, rare, dup):
    """Ekspor Roboflow eksternal: train->train, valid->val. Split test TIDAK dipakai
    (disimpan untuk eval_external.py)."""
    d = f"{root}/{rel}"
    n = dropped = 0
    for sp, out_sp in (("train", "train"), ("valid", "val")):
        for f in sorted(os.listdir(f"{d}/{sp}/images")):
            stem = os.path.splitext(f)[0]
            raw = [(names[int(p[0])], *poly_to_box(p[1:])) for p in
                   (l.split() for l in open(f"{d}/{sp}/labels/{stem}.txt").read().splitlines())
                   if len(p) >= 5]
            boxes = drop_part_boxes(raw)
            dropped += len(raw) - len(boxes)
            if boxes:
                cp = rare_copies(boxes, rare, dup) if out_sp == "train" else 1
                wr.add(out_sp, f"{name}_{n:05d}", f"{d}/{sp}/images/{f}", boxes, copies=cp)
                n += 1
    print(f"{name}: {n} foto, {dropped} kotak 'bagian uang' dibuang")


def do_synth(root, wr):
    d = f"{root}/synth"
    for sp in ("train", "val"):
        for f in sorted(os.listdir(f"{d}/{sp}/images")):
            stem = os.path.splitext(f)[0]
            boxes = [(CLASSES[int(p[0])], *map(float, p[1:5])) for p in
                     (l.split() for l in open(f"{d}/{sp}/labels/{stem}.txt").read().splitlines())
                     if len(p) == 5]
            if boxes:
                wr.add(sp, stem, f"{d}/{sp}/images/{f}", boxes)
    print("synth: selesai")


UANG2016 = ["k20000", "c200", "k2000", "k50000", "c500", "k5000", "k10000", "k100000",
            "c100", "k1000", "c1000"]
RDET = ["k20000", "k2000", "k50000", "k5000", "k10000", "k100000", "k1000"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="ai/datasets")
    ap.add_argument("--out", default="ai/datasets/merged")
    ap.add_argument("--rk-every", type=int, default=2, help="ambil tiap N frame video")
    ap.add_argument("--rare-dup", type=int, default=3, help="duplikasi train c100/c200")
    ap.add_argument("--v2", action="store_true",
                    help="tambah uang-2016-2022, Rupiah-Detector, dan data sintetis latar hitam")
    ap.add_argument("--max-side", type=int, default=0, help="perkecil gambar > N px")
    args = ap.parse_args()
    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    wr = Writer(args.out, args.max_side)
    rare = {"c100", "c200"}
    do_rbd(args.root, wr)
    do_rk(args.root, wr, args.rk_every)
    do_koin(args.root, wr, rare, args.rare_dup)
    do_idr(args.root, wr, rare, args.rare_dup)
    if args.v2:
        do_roboflow(args.root, wr, "u1622", "external/cirengbasii/uang-2016-2022-1", UANG2016,
                    rare, 2)
        do_roboflow(args.root, wr, "rdet", "external/cirengbasii/Rupiah-Detector-1", RDET, rare, 1)
        do_synth(args.root, wr)

    with open(f"{args.out}/data.yaml", "w") as f:
        f.write(f"path: {os.path.abspath(args.out).replace(os.sep, '/')}\n")
        f.write("train: train/images\nval: val/images\n")
        f.write(f"nc: {len(CLASSES)}\nnames: {CLASSES}\n")

    print(f"\nGambar: train={wr.nimg['train']}  val={wr.nimg['val']}")
    print(f"{'kelas':<9}{'train':>7}{'val':>6}")
    for c in CLASSES:
        print(f"{c:<9}{wr.count[('train', c)]:>7}{wr.count[('val', c)]:>6}")


if __name__ == "__main__":
    main()
