# CLAUDE.md — Smart Kotak Amal (Team BesokKamis)

Konteks proyek untuk Claude Code. Fokus file ini: **bagian AI** yang akan kita bangun, plus konteks IoT secukupnya supaya kode AI nyambung dengan perangkat.

## 1. Ringkasan proyek

- **Nama:** Smart Kotak Amal berbasis IoT dan AI.
- **Tim:** BesokKamis (4 orang), final project Samsung Innovation Campus (SIC) Batch 8, Stage 3 (IoT + AI), tema **Isu Sosial**.
- **Judul riset (mata kuliah Metodologi Riset):** "Rancang Bangun Smart Kotak Amal Berbasis Internet of Things untuk Monitoring dan Penghitungan Otomatis Donasi Uang Rupiah Menggunakan ESP32".
- **Masalah:** penghitungan kotak amal masjid masih manual, lambat, rawan salah dan kurang transparan. Penelitian sebelumnya (sensor warna TCS3200/TCS34725) sering salah membaca Rp1.000 vs Rp5.000, sangat sensitif jarak, dan gagal pada uang terlipat lebih dari 2 kali.
- **Solusi:** kamera ESP32-CAM memotret uang di nampan dalam kotak, model YOLO di server laptop mengenali nominal, setiap donasi dicatat, dan dashboard menampilkan total secara real-time.
- **Wajib SIC:** model AI dilatih sendiri (bukan hanya API pihak ketiga). Yang dikumpulkan: kode, prototipe hardware, video demo.
- **Deadline Study Case:** 14 Desember 2026.

## 2. Arsitektur sistem

```
[Kotak amal]                                  [Laptop / AI server]           [Dashboard]
 IR FC-51 (uang masuk) ─┐
 MPU6050 (gerak)        ├─ ESP32 DevKit ──MQTT── HiveMQ Cloud ──┬── FastAPI + YOLO ──┬── Streamlit
 VL53L0X (kepenuhan)    │      │                                 │   Isolation Forest │
 OLED, servo, buzzer  ──┘      │ trigger                         │   SQLite + foto    │
                         ESP32-CAM + LED putih ──HTTP POST foto──┘
```

- **ESP32 tidak menjalankan YOLO.** Model terlalu berat untuk ESP32, jadi inference di laptop (FastAPI).
- ESP32-CAM: OV2640, resolusi VGA/SVGA, menghadap ke bawah ke nampan hitam (permukaan doff atau mengkilap, belum final; model harus tahan keduanya, termasuk pantulan LED), jarak ±12 cm, diterangi LED putih + kertas kalkir (difuser).
- Setelah dibaca, servo MG90S memiringkan nampan ±60° sehingga uang jatuh ke brankas.

## 3. Hardware (referensi pin/komponen)

| Komponen | Fungsi |
|---|---|
| ESP32 DevKit | Pengendali utama, MQTT, antrean offline |
| ESP32-CAM (OV2640) | Foto nampan, kirim ke AI server |
| IR FC-51 × 3 | 2 di celah uang kertas, 1 di jalur koin |
| MPU6050 | Data gerak untuk deteksi anomali (pencurian/guncangan) |
| VL53L0X | Ukur kepenuhan brankas (I2C 0x29) |
| Servo MG90S | Goyang nampan (foto ulang) dan miringkan nampan |
| OLED SSD1306 | Ucapan terima kasih + nominal |
| Buzzer, LED status | Alarm dan indikator |
| Adaptor 5V 3A | Catu daya |

Catatan I2C: TCS34725 dan VL53L0X sama-sama 0x29. Kita **tidak** memakai sensor warna, jadi aman.

## 4. Model AI #1 — YOLO untuk nominal uang

### Spesifikasi
- **Model utama:** YOLO26n (Ultralytics) — ±2,4 juta parameter, ±5,5 GFLOPs, ±38,9 ms per gambar di CPU (ONNX). Cadangan bila akurasi kurang: YOLO26s.
- **Tugas:** object detection (bukan klasifikasi), karena satu foto bisa berisi beberapa uang sekaligus.
- **Input:** 640 × 640.

### Kelas (12)
```
k1000, k2000, k5000, k10000, k20000, k50000, k100000   # uang kertas
c100, c200, c500, c1000                                # koin (lihat bagian 7, status koin belum final)
lain                                                   # benda bukan uang / tidak dikenali
```

### Alur keputusan inference
1. Foto masuk → YOLO deteksi semua objek.
2. Semua confidence ≥ **0,85** → catat setiap uang + simpan foto bukti.
3. Ada yang < 0,85 → servo goyang nampan, foto ulang, deteksi lagi.
4. Masih ragu → crop objek itu, minta **second opinion ke Gemini Decision 1** (Google AI Studio). Periksa dokumentasi terbaru di https://aistudio.google.com/docs/models/gemini-decision-1 sebelum menulis kode pemanggilannya.
5. Masih tidak yakin → catat sebagai `unknown` (nominal 0, status perlu verifikasi takmir).

### Dataset
- **Tahap 1 (pretrain/fine-tune awal):** dataset publik uang Rupiah (Roboflow, Hugging Face, Kaggle, GitHub). Kekurangan: hampir semua uang kertas rapi, tidak ada uang terlipat/lecek, koin sangat sedikit.
- **Tahap 2 (wajib):** foto sendiri dari dalam kotak (nampan, LED, jarak kamera yang sama). Sertakan uang **terlipat, lecek, tertumpuk, sebagian tertutup**, depan dan belakang, banyak uang dalam satu foto.
- **Test set hanya dari foto kotak sendiri**, tidak dicampur dataset publik.
- Labeling di **Roboflow**, export format **YOLO**. Bounding box mengikuti bentuk uang yang terlihat (termasuk saat terlipat).
- Augmentasi: brightness, blur, rotasi; **`fliplr=0`** (jangan flip horizontal, angka nominal jadi terbalik).

### Training
- Google Colab (GPU), Ultralytics.
- Metrik: mAP@0.5, mAP@0.5:0.95, precision, recall per kelas, confusion matrix (perhatikan k1000 vs k5000, dan antar koin).
- Export ke ONNX untuk inference CPU di laptop.

## 5. Model AI #2 — Isolation Forest untuk anomali gerak

- Input: data MPU6050, window **50 sampel**, diekstrak menjadi **24 fitur** (statistik per sumbu akselerometer + giroskop, mis. mean, std, min, max).
- Dilatih hanya dengan data **normal** (kotak diam, orang memasukkan uang).
- Output anomali → buzzer + alert MQTT ke dashboard.
- Library: scikit-learn.

## 6. Data, MQTT, dan penyimpanan

### Topik MQTT (HiveMQ Cloud)
`donation`, `status`, `alert`, `cmd` (prefix per kotak, mis. `kotakamal/<id_kotak>/donation`).

### SQLite — tabel `donasi` (satu baris per uang)
| Kolom | Isi |
|---|---|
| id | primary key |
| waktu | timestamp |
| id_kotak | id perangkat |
| sumber | kertas / koin |
| nominal | angka rupiah (0 jika unknown) |
| confidence | skor YOLO / Gemini |
| status | ok / verifikasi / unknown |
| foto | path foto bukti |
| sinkron | sudah terkirim ke dashboard atau belum |

Ada juga tabel rekonsiliasi (hitungan manual takmir vs sistem) untuk evaluasi.

### Offline
ESP32 menyimpan antrean lokal saat Wi-Fi putus, lalu mengirim ulang saat online.

## 7. Keputusan yang masih terbuka

- **Koin:** belum final. Pilihan:
  1. Koin dibaca kamera + YOLO (kelas c100–c1000) + cek diameter (Rp100 23 mm, Rp200 25 mm, Rp500 27 mm, Rp1000 24,1 mm seri 2016). Risiko: dataset koin harus dibuat sendiri, koin lama (1999–2003) sisi Garuda nyaris sama.
  2. Sortir mekanis berdasarkan diameter + sensor IR (seperti Anggraini dkk., 2024), tanpa YOLO.
  3. IR hanya mencatat "ada koin masuk", nominal tidak diidentifikasi.
- **Judul:** mungkin ditambah "dan Computer Vision".

## 8. Penelitian pembanding (untuk gap)

- **Anggraini, Zulkifli, Hakiem (2024)**, MATRIK 24(1):11–24, DOI 10.30812/matrik.v24i1.4209 — Raspberry Pi 3, TCS34725 + 2 IR, Firebase + Telegram. Akurasi 100% hanya pada jarak 1 cm (42,86% di 0,5 cm; 28,57% di 1,5 cm), uang terlipat maks. 2 kali, koin hanya Rp500/Rp1.000 lewat lubang diameter, gagal 16% pada uji pengguna.
- **Gushardi & Faiza (2022)**, Jurnal Pendidikan Tambusai 6(1), DOI 10.31004/jptam.v6i1.3132 — TCS3200 (kertas) + LDR (koin), Telegram.
- **Amrullah (2026)**, tugas akhir Politeknik Negeri Jember — ESP32-CAM → Flask → YOLO11s, uang kertas emisi 2022, 924 gambar, mAP@0.5 0,983, tanpa koin dan tanpa uang terlipat. Paling mirip dengan kita.
- **Charleen & Kusuma (2024)**, IJCDS 16(1):315–327, DOI 10.12785/ijcds/160125 — perbandingan YOLOv5n/YOLOv7-tiny/YOLOv8n/EfficientDet-d0, YOLOv8n mAP@0.5 0,995.
- **Hadisukmana dkk. (2026)**, IJEECS 42(3):856–864, DOI 10.11591/ijeecs.v42.i3.pp856-864 — deteksi koin Indonesia dengan transfer learning, dataset dibuat sendiri.

**Pembeda kita:** pengenalan berbasis kamera (bukan warna), uang terlipat/lecek, fallback (foto ulang + Gemini + unknown), pencatatan per donasi dengan foto bukti, deteksi anomali gerak, pemantauan kepenuhan, antrean offline, ESP32.

## 9. Jadwal kelas SIC terkait

22 Nov Data Prep · 28 Nov Building Model · 29 Nov Evaluating · 5 Des ML + Sensor · 6 Des Visualization · 12 Des Edge AI · 14 Des deadline Study Case.

## 10. Rencana struktur repo (usulan)

```
smart-kotak-amal/
├── firmware/
│   ├── esp32_main/        # sensor, MQTT, servo, OLED, antrean offline
│   └── esp32_cam/         # foto + HTTP POST ke server
├── ai/
│   ├── yolo/              # notebook Colab, data.yaml, hasil training, export ONNX
│   └── anomaly/           # Isolation Forest MPU6050
├── server/                # FastAPI: /predict, fallback Gemini, SQLite, MQTT publish
├── dashboard/             # Streamlit
└── docs/
```

## 11. Cara kerja dengan Claude

- Tulis penjelasan dalam Bahasa Indonesia yang natural; istilah teknis tetap dalam bahasa Inggris.
- Jangan menebak API Gemini Decision; cek dokumentasi resminya dulu.
- Utamakan solusi yang bisa selesai sebelum 14 Desember 2026 dengan hardware di atas.
