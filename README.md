# CakraWatch — Deteksi Hoaks & Buzzer Berbahasa Indonesia

**Submission Kartika Cyber Hackathon 2026 · Kategori 04 Siber Sosial**
PUSSIBERAD (Pusat Siber TNI AD) x CodePolitan
Tim CakraWatch (solo): **Nugroho Riziq Darmawan Nata Saputra**

CakraWatch adalah sistem pemantau disinformasi berbahasa Indonesia. Pengguna menempelkan teks berita, klaim, atau pesan berantai, lalu sistem memberi **vonis HOAKS / MERAGUKU / VALID**, **skor keyakinan**, dan **penjelasan frasa pemicu** yang paling memengaruhi keputusan model. Tujuannya membantu analis dan masyarakat menyaring hoaks secara cepat dan transparan, sebagai bagian dari pertahanan domain kognitif (Siber Sosial).

---

## 1. Latar Belakang dan Relevansi

Disinformasi adalah ancaman keamanan yang nyata: hoaks memicu kepanikan, menggerus kepercayaan publik, dan dapat digerakkan secara terorganisir oleh jaringan buzzer. Bagi TNI AD, ruang informasi adalah salah satu domain pertahanan. CakraWatch dirancang ringan (berjalan di laptop standar dan browser HP), berbahasa Indonesia, dan dapat diadopsi sebagai alat bantu awal pemantauan isu oleh satuan atau lembaga yang membutuhkan triase hoaks yang cepat dan dapat dijelaskan.

## 2. Fitur Utama

- Klasifikasi teks Bahasa Indonesia: HOAKS / MERAGUKU / VALID (kelas MERAGUKU berasal dari ambang keyakinan, lihat Bagian 4).
- Skor keyakinan terkalibrasi dari probabilitas model.
- Penjelasan frasa pemicu: kata/frasa dengan kontribusi terbesar terhadap vonis disorot dan didaftarkan, agar keputusan model dapat diaudit manusia.
- Dashboard web interaktif: panel analisis, contoh cepat sekali klik, panel perbandingan dua model, dan riwayat sesi.
- Dua model yang saling mengasuransikan: baseline TF-IDF yang sangat ringan dan model utama IndoBERT dengan akurasi tertinggi.

## 3. Arsitektur: Dua Model

### 3.1 Model Demo Langsung — TF-IDF + Logistic Regression
- Representasi: TF-IDF kata (n-gram 1-2, sublinear, maks 20.000 fitur, min_df 2).
- Klasifier: Logistic Regression (class_weight balanced).
- Seluruh parameter model ini (leksikon 20.000 term beserta idf dan koefisiennya) diekspor ke `data/cakrawatch_lexicon.json` dan ditanam di dashboard, sehingga inferensi berjalan **sepenuhnya di browser pengunjung** tanpa server. Porting fungsi skornya terverifikasi identik dengan scikit-learn (simpangan probabilitas 0,0000 pada 224 teks uji).
- File: `src/baseline_tfidf.py`, `data/cakrawatch_lexicon.json`, `data/cakrawatch_samples.json`.

### 3.2 Model Utama — IndoBERT (fine-tune)
- Basis: `indolem/indobert-base-uncased` (IndoBERT, IndoLEM).
- Fine-tune di Google Colab (GPU T4): learning rate 2e-5, batch 16, 5 epoch, panjang maksimum 256 token, loss berbobot kelas [HOAKS 0,817 · VALID 1,289], seed 42, model terbaik dipilih dari macro F1 per epoch (terbaik: epoch 3). Waktu training 2 menit 28 detik.
- Penjelasan frasa pemicu memakai metode occlusion (kata dicabut satu per satu, penurunan probabilitas diukur).
- File: `notebook/CakraWatch_IndoBERT_Colab.ipynb`.

## 4. Aturan Vonis Dashboard

Probabilitas HOAKS `p` dari model dipetakan menjadi:
- `p >= 0,75` → **HOAKS**
- `p <= 0,35` → **VALID**
- di antaranya → **MERAGUKU** (perlu verifikasi manusia)

Ambang ini dipilih mengikuti distribusi probabilitas model TF-IDF yang terkompresi (rentang efektif sekitar 0,15-0,89), sehingga ketiga kelas benar-benar terpakai.

## 5. Dataset

- Sumber inti: dataset TurnBackHoax pada repo GitHub `JibranFawaid/turnbackhoax-dataset` (data Mafindo/TurnBackHoax). 1.116 data berlabel dua kelas: 683 HOAKS (klaim/narasi) dan 433 VALID (artikel klarifikasi resmi). Dibagi stratifikasi 80/20 (892 latih, 224 uji), seed 42.
- Cadangan ekspansi: 3.760 judul hoaks TurnBackHoax 2024 dari raihankalla.id (hanya kelas hoaks; judul memuat prefix vonis pemeriksa fakta yang harus dibuang karena label leakage).
- Temuan data penting: augmentasi naif dengan 3.760 data hoaks menaikkan accuracy mentah tetapi **menjatuhkan F1 kelas VALID ke 0,78** (imbalance sekitar 10:1). Kesimpulan: penambahan data wajib diimbangi kelas valid sepadan atau class weighting. Rincian di `data/DATASET_NOTES.md`.

## 6. Hasil Evaluasi (data uji 224, terukur, bukan klaim)

| Model | Accuracy | Macro F1 | F1 HOAKS | F1 VALID |
|---|---|---|---|---|
| TF-IDF + Logistic Regression | 0,9375 | 0,9354 | 0,9470 | 0,9239 |
| IndoBERT fine-tune | **0,9955** | **0,9953** | 1,00 (P 1,00 / R 0,99) | 0,99 (P 0,99 / R 1,00) |

Confusion matrix IndoBERT: `[[136, 1], [0, 87]]` — satu kesalahan dari 224.

Catatan proses (transparansi): fine-tune pertama gagal total (macro F1 0,2831, model menebak satu kelas) karena fungsi loss kustom tidak kompatibel dengan argumen baru library Trainer. Setelah tanda tangan fungsi diperbaiki dan learning rate diturunkan ke 2e-5, run kedua mencapai hasil pada tabel di atas. Riwayat kegagalan dan perbaikan ini didokumentasikan apa adanya sebagai bagian dari proses rekayasa.

## 7. Instalasi dan Cara Pakai

### 7.1 Dashboard (tanpa instalasi)
Buka halaman dashboard CakraWatch (tautan pada halaman submission CodePolitan), tempel teks pada panel analisis, tekan **Analisis Sekarang**. Vonis, meter keyakinan, frasa pemicu, dan sorotan teks muncul seketika. Spesifikasi teknis dashboard ada di `docs/DASHBOARD_SPEC.md`.

### 7.2 Melatih ulang IndoBERT
1. Buka `notebook/CakraWatch_IndoBERT_Colab.ipynb` di Google Colab.
2. Pilih runtime GPU (T4): Runtime → Change runtime type → T4 GPU.
3. Jalankan semua sel dari atas (dataset terunduh otomatis dari repo sumber).
4. Sel evaluasi mencetak classification report dan perbandingan dengan baseline; sel terakhir menyimpan model terlatih.

### 7.3 Baseline TF-IDF
```
pip install pandas scikit-learn
python src/baseline_tfidf.py
```
Script melatih baseline pada dataset GitHub dan mencetak metrik Run A (dataset inti) dan Run B (augmentasi), termasuk demonstrasi dampak imbalance.

## 8. Keterbatasan (dicatat jujur)

- Kelas VALID pada dataset berbentuk **artikel klarifikasi pemeriksa fakta**, bukan berita media arus utama murni. Model dapat sebagian belajar dari gaya bahasa klarifikasi. Bukti terukur: pada uji manual, pernyataan resmi singkat yang bertopik sama dengan sebuah hoaks pernah divonis HOAKS oleh IndoBERT (domain shift), walaupun metrik data ujinya 0,9953.
- Data latih masih kecil (1.116) dan berasal dari satu sumber ekosistem (TurnBackHoax).
- Modul deteksi buzzer (analisis pola akun dan penyebaran) saat ini masih **rencana pengembangan**; versi submission berfokus pada klasifikasi hoaks dan penjelasan frasa pemicu.

## 9. Roadmap

- Menambah berita valid dari media arus utama (Kompas, Tempo, Detik, CNN Indonesia) dan teks pendek gaya media sosial agar model tahan domain shift.
- Menyeimbangkan ekspansi data hoaks 2024 dengan kelas valid sepadan / class weighting.
- Modul buzzer: fitur perilaku akun (usia akun, ritme posting, kemiripan teks antar akun) sebagai sinyal pendukung.
- API ringan agar dashboard dan aplikasi lain dapat memakai model yang sama.

## 10. Kredit dan Lisensi Komponen

- Dataset: Mafindo / TurnBackHoax, dikompilasi pada repo `JibranFawaid/turnbackhoax-dataset` dan analisis `raihankalla.id/hoax-analysis-2024`.
- Model dasar: IndoBERT oleh IndoLEM (`indolem/indobert-base-uncased`), dipakai sesuai lisensi modelnya.
- Karya ini dibuat selama program Kartika Cyber Hackathon 2026; hak kekayaan intelektual tetap pada peserta sesuai S&K penyelenggara.
