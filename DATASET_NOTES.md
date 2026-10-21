# CakraWatch — Dataset Notes & Baseline Results
Dicatat: 2026-10-21. Semua angka di bawah adalah hasil inspeksi dan run aktual, bukan estimasi.

## 1. Dataset yang berhasil diunduh

### A. `raihankalla_turnbackhoax_raw.csv` (1.830.603 byte)
- Sumber: link "Raw Data" dari https://raihankalla.id/hoax-analysis-2024 → Google Sheets "Scraping turnbackhoax.id" (ID `1uDFRbWFzMJjYxs1D13o79g726uROOVcJj41B4axGmWU`), diekspor sebagai CSV.
- Isi: **3.760 baris** artikel debunk TurnBackHoax/Mafindo (2024 + awal 2025).
- Kolom: `TITLE, URL, PREVIEW, IMAGE URL, DATE, AUTHOR, PAGE, DAY` (semua terisi penuh).
- Distribusi prefix vonis pada TITLE: SALAH 3.007 · PENIPUAN 718 · BELUM TERBUKTI 31 · lain-lain 4.
- **Kelas: HOAKS saja.** Tidak ada kelas valid/fakta di file ini.
- Contoh baris:
  - `[SALAH] Sidang Jokowi Dituntut Ratusan Triliun` — PREVIEW: "Tidak ditemukan informasi atau pemberitaan kredibel yang membenarkan klaim."
  - `[SALAH] Hakim Kasus Harvey Moeis Terima Uang Suap Rp50 Miliar` — PREVIEW: "Tidak ditemukan artikel atau informasi yang memberitakan klaim."
- Caveat penting: TITLE memuat vonis pemeriksa fakta dan PREVIEW adalah suara pemeriksa fakta (penjelasan debunk), bukan teks hoaks aslinya. Prefix vonis adalah *label leakage* — wajib dibuang sebelum dipakai training (sudah dilakukan di script baseline).

### B. `github_jibranfawaid_scrapping.csv` (3.481.039 byte)
- Sumber: repo GitHub `JibranFawaid/turnbackhoax-dataset` (data diambil dari turnbackhoax.id, 25 April 2021), file `Scrapping.csv`.
- Format asli: delimiter **titik-koma** (`;`), encoding lama (Windows-1252/latin-1), kolom `Label;Headline;Body`.
- Isi: **1.116 baris**. Distribusi label: `0` = 683 · `1` = 433.
- Semantik label **diverifikasi dari isi** (bukan asumsi):
  - `0` = **hoaks** — headline/body berupa klaim dan narasi hoaks, mis. "Viral Video Sekawanan Monyet di Arab Saudi Pertanda Ada Bencana Alam Besar", "Penculikan Di Bekasi Oleh Wanita Memakai Mukena Putih".
  - `1` = **valid/fakta** — artikel klarifikasi/debunk resmi, mis. "Pertamina dan Pemerintah Klarifikasi Isu BBM Naik Malam Ini", "ISU PENCULIKAN ... TIDAK TERBUKTI" (KATEGORI: KLARIFIKASI).
  - Cross-check: jumlah label `1` (433) persis sama dengan jumlah kategori BENAR (433) di sheet 'Kategori' file XLSX pendamping.

### C. `github_jibranfawaid_metadata.xlsx` (11.166.441 byte)
- File `Meta Data.xlsx` dari repo yang sama. 3 sheet:
  - `Hasil Scrapping` — 6.975 baris, kolom Header/Content/Link (mentah, sebagian berbahasa Inggris dari versi .id berbahasa Inggris).
  - `Kategori` — 6.975 baris, kolom Kategori/Header/Content/Link. Distribusi kategori teratas: SALAH 4.689 · HOAX 670 · BENAR 433 · KLARIFIKASI 359 · DISINFORMASI 269 · BERITA 183 · False 90 · EDUKASI 82 · FITNAH 36 · TOP 34. Ini pool mentah yang lebih besar, belum dibersihkan.
  - `Fix` — 1.116 baris, kolom Label/Headline/Body — **identik** dengan `Scrapping.csv` (versi bersih final yang dipakai baseline).
- Repo clone lengkap tersimpan di `hidden_files/build/turnbackhoax-dataset-repo/` untuk provenance.

## 2. Baseline TF-IDF — hasil run aktual
Script: `hidden_files/build/baseline_tfidf.py` · Log: `hidden_files/build/baseline_run_log.txt`
Konfigurasi: TF-IDF word n-gram 1–2, max_features 20.000, sublinear TF, min_df 2 + Logistic Regression (max_iter 1000, class_weight balanced) · split stratifikasi 80/20, random_state 42 · teks = Headline + Body.

### Run A — GitHub JibranFawaid saja (1.116 data; train 892 / test 224)
Kelas: hoaks 683 · valid 433.
- **Accuracy: 0,9375**
- hoaks: precision 0,9843 · recall 0,9124 · **F1 0,9470** (support 137)
- valid: precision 0,8763 · recall 0,9770 · **F1 0,9239** (support 87)
- Macro F1: 0,9354

### Run B — Augmented: GitHub + raihankalla sebagai hoaks (4.876 data; train 3.900 / test 976)
Kelas: hoaks 4.443 · valid 433 (perbandingan ±10:1, prefix vonis sudah dibuang dari judul).
- Accuracy: 0,9498 (terlihat naik, tetapi menipu karena mayoritas kelas hoaks)
- hoaks: precision 1,0000 · recall 0,9449 · F1 0,9717 (support 889)
- valid: precision 0,6397 · recall 1,0000 · **F1 0,7803** (support 87) — **turun jauh dari 0,9239**
- Macro F1: 0,8760 (turun dari 0,9354)

### Interpretasi jujur
1. Baseline TF-IDF di 1.116 data berlabel sudah kuat (accuracy 93,75%, macro F1 0,9354) dan cocok sebagai *safety net* bila fine-tune IndoBERT bermasalah.
2. Menambah 3.760 judul debunk sebagai kelas hoaks **merusak** kelas valid (F1 valid 0,92 → 0,78). Penyebabnya ganda: ketidakseimbangan kelas ekstrem, dan judul/preview raihankalla bersuara pemeriksa fakta — mirip suara kelas valid. Augmentasi naif seperti ini jangan dipakai untuk model final.
3. Untuk tahap IndoBERT: pakai dataset GitHub sebagai inti; bila menambah data hoaks 2024, wajib diimbangi kelas valid sepadan (berita valid Kompas/Tempo/Detik/CNN Indonesia) dan/atau class weighting/undersampling.
4. Pool mentah 6.975 baris di sheet 'Kategori' XLSX adalah cadangan bila ingin memperbesar dataset berlabel (kategori BENAR/KLARIFIKASI → valid; SALAH/HOAX/DISINFORMASI/FITNAH → hoaks) dengan pembersihan manual.
