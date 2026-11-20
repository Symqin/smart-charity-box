# Smart Kotak Amal (Smart Charity Box)

Kotak amal berbasis IoT + AI: ESP32-CAM memotret uang di nampan, model YOLO di server laptop
mengenali nominal, setiap donasi dicatat (dengan foto bukti) dan tampil di dashboard secara real-time.
Final project Samsung Innovation Campus Batch 8 Stage 3 — Tim **BesokKamis**.


## Struktur repo

```
smart-kotak-amal/
├── README.md, requirements.txt
├── firmware/
│   ├── esp32_main/   # sensor, servo, OLED, MQTT, antrean offline, pins.h      (belum)
│   └── esp32_cam/    # foto nampan -> POST ke server                           (belum)
├── ai/
│   ├── yolo/         # model nominal uang (siap)
│   │   ├── predict.py          pakai model: CLI + class MoneyDetector
│   │   ├── train.py            training + val + export ONNX
│   │   ├── eval_external.py    uji di test set ber-bbox yang tidak ikut training
│   │   ├── eval_koin.py        uji per foto (1 uang/koin per foto, folder per kelas)
│   │   ├── check_leakage.py    cek foto test yang hampir identik dengan train
│   │   └── data_prep/          autolabel_rbd -> merge_dataset -> synth_black -> make_kertas
│   ├── anomaly/      # Isolation Forest MPU6050                                (belum)
│   ├── datasets/     # dataset (tidak di-commit)
│   └── models/       # model hasil training (ONNX + PyTorch)
├── server/           # FastAPI /deteksi + MQTT + SQLite                         (belum)
├── dashboard/        # Streamlit                                                (belum)
├── docs/             # arsitektur, topik MQTT, wiring, catatan keputusan        (belum)
└── scripts/          # run_server.sh, run_dashboard.sh                          (placeholder)
```

## Model

YOLO26n (Ultralytics), input 640×640, dilatih di RTX 2050.

| File | Kelas | Keterangan |
|---|---|---|
| `ai/models/uang_v2.onnx` / `.pt` | 12: `k1000 k2000 k5000 k10000 k20000 k50000 k100000 c100 c200 c500 c1000 lain` | model utama, default `predict.py` |
| `ai/models/uang_kertas_v2.onnx` / `.pt` | 7: uang kertas saja | fine-tune dari v2 + negatif koin & nampan kosong; cocok bila koin hanya dicatat IR |

### Hasil uji (test set yang tidak ikut training, ambang confidence 0,85)

| Test | uang_v2 | uang_kertas_v2 |
|---|---|---|
| val (sumber sama dengan train), mAP50-95 | 0,950 | 0,962 |
| Rupiah-PCR, 700 foto uang kertas, top-1 | 86,7% | 86,7% |
| deteksi-mata-uang-rupiah, mAP50 (kertas) | 0,345 | 0,327 |
| drawrs, mAP50 | 0,386 | 0,375 |
| Koin, 630 foto emisi 2016, top-1 | 39,0% | – |

Catatan jujur:
- Angka val tinggi karena val satu sumber dengan train; data asing jauh lebih rendah.
- **Koin belum layak** dibaca YOLO (Rp100 hampir selalu salah) karena data koin minim variasi fisik.
- Angka resmi untuk laporan akan diambil dari **foto kotak sendiri** (ESP32-CAM, nampan hitam).

## Instalasi

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt        # Windows (Linux/Mac: .venv/bin/pip)
# GPU (opsional): pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## Cara pakai model

**Terminal**

```bash
.venv/Scripts/python ai/yolo/predict.py foto.jpg
.venv/Scripts/python ai/yolo/predict.py folder_foto/ --save hasil/                     # simpan foto berkotak
.venv/Scripts/python ai/yolo/predict.py foto.jpg --model ai/models/uang_kertas_v2.onnx  # model kertas
```

```
foto.jpg  ->  status=ragu  total=Rp50.000
   k20000   conf 0.74  RAGU
   k50000   conf 0.86  yakin
```

**Python (mis. server FastAPI)**

```python
import sys; sys.path.insert(0, "ai/yolo")
from predict import MoneyDetector

det = MoneyDetector("ai/models/uang_v2.onnx")   # load sekali saat server start
hasil = det.predict(jpeg_bytes)                 # path, bytes JPEG dari ESP32-CAM, array BGR, atau PIL.Image

hasil["status"]   # "ok"     -> semua conf >= 0,85, langsung catat
                  # "ragu"   -> ada yang < 0,85: goyang nampan + foto ulang, lalu Gemini, lalu unknown
                  # "kosong" -> tidak ada uang terdeteksi
hasil["total"]    # total rupiah dari uang yang yakin
hasil["items"]    # [{"kelas","nominal","jenis","confidence","yakin","box":[x1,y1,x2,y2]}, ...]
```

ONNX di CPU (i5-13500H) ±25 ms per foto.

## Training ulang

Dataset tidak di-commit; unduh sumber di bawah ke `ai/datasets/`, lalu jalankan `ai/yolo/data_prep/` berurutan.

```bash
.venv/Scripts/python ai/yolo/train.py --name uang_kertas_v3 --data ai/datasets/merged_kertas/data.yaml \
    --model ai/models/uang_kertas_v2.pt --epochs 25 --patience 25 --optimizer SGD --lr0 0.002 --warmup 0.5 --workers 2
.venv/Scripts/python ai/yolo/eval_external.py --kertas --model runs/uang_kertas_v3/weights/best.pt --tag kertas_v3
```

Fine-tune dari model yang sudah bagus harus memakai lr kecil (0,002); dengan lr otomatis (0,01) hasil uji justru turun.
`--workers` > 2 bisa kehabisan RAM di Windows (±1,5 GB per worker).

## Sumber dataset

Training:
- [mkaspulanwar/rupiah-banknotes-dataset](https://github.com/mkaspulanwar/rupiah-banknotes-dataset) — MIT (bbox dibuat otomatis oleh `autolabel_rbd.py`)
- [Uang Rupiah Kertas dan Koin (Adelia Clarissa)](https://universe.roboflow.com/coin-new/uang-rupiah-kertas-dan-koin-by-adelia-clarissa-zzt2j) — CC BY 4.0
- [coin (class-tnvuh)](https://universe.roboflow.com/class-tnvuh/coin-yxjpa) — CC BY 4.0
- idr-coins — MIT
- [uang-2016-2022](https://universe.roboflow.com/fix-tmc9q/uang-2016-2022) — CC BY 4.0 (train/valid)
- [Rupiah Detector](https://universe.roboflow.com/rupiah-detector/rupiah-detector-qzmb7) — CC BY 4.0 (train/valid)

Hanya untuk uji:
- [deteksi-mata-uang-rupiah](https://universe.roboflow.com/skripsi-3kth2/deteksi-mata-uang-rupiah-nerog) — CC BY 4.0
- [drawrs/dataset-indonesian-rupiah](https://github.com/drawrs/dataset-indonesian-rupiah)
- [Rupiah-PCR-Using-Image-Processing-and-CNN](https://github.com/ardinadnn/Rupiah-PCR-Using-Image-Processing-and-CNN) (700 foto uang kertas)
- [RikiSanjayaa/projek-edge-detection](https://github.com/RikiSanjayaa/projek-edge-detection) (630 foto koin)
