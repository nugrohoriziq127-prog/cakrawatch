# DASHBOARD_SPEC — CakraWatch Dashboard (artifact web_static, goal Kartika)

## Tujuan
Dashboard demo publik CakraWatch untuk form Submit Karya Kartika Cyber Hackathon 2026 (Link Karya/Project). Bahasa UI: Bahasa Indonesia. Tema visual: gelap ala pusat pemantauan (SOC), aksen cyan/teal + warna status (merah HOAKS, kuning MERAGUKAN, hijau VALID), peta/radar hanya dekorasi ringan. Harus nyaman dibuka di HP.

## Mesin analisis (WAJIB persis seperti ini, bukan simulasi bebas)
Data model ada di `data/cakrawatch_lexicon.json` (relatif terhadap folder goal ini):
`{intercept, meta{...}, terms: [[term, idf, coef], ...]}`.
Tanam seluruh isi JSON itu ke halaman sebagai konstanta `LEX`.

Fungsi skor (port setia dari pipeline sklearn TF-IDF sublinear + Logistic Regression):
1. `text.toLowerCase()`, token = semua kecocokan regex `[\p{L}\p{N}_]{2,}` dengan flag `gu` (urutan kemunculan dipertahankan).
2. Bentuk unigram (token) + bigram (pasangan token berurutan digabung spasi).
3. Hitung frekuensi hanya untuk term yang ada di LEX. Nilai term = `(1 + ln(freq)) * idf`. Normalisasi L2 atas seluruh nilai term dokumen.
4. `logit = intercept + Σ (nilai_ternormalisasi * coef)`. `p_hoaks = 1 / (1 + exp(-logit))`.
5. Kontribusi frasa pemicu per term = `nilai_ternormalisasi * coef` (positif = mendorong HOAKS, negatif = mendorong VALID).

Vonis dari p_hoaks:
- `p >= 0.75` → HOAKS (keyakinan = p)
- `p <= 0.35` → VALID (keyakinan = 1 - p)
- selain itu → MERAGUKAN (keyakinan = probabilitas kelas terpilih ditampilkan apa adanya)

## Komponen halaman
1. Header: nama CakraWatch, subjudul "Deteksi Hoaks & Buzzer Berbahasa Indonesia", lencana "Kartika Cyber Hackathon 2026 · Kategori 04 Siber Sosial".
2. Panel analisis: textarea besar (tempel teks berita/klaim/pesan berantai), tombol "Analisis Sekarang", tombol contoh cepat dari `data/cakrawatch_samples.json` (chip yang mengisi textarea dengan teks contoh).
3. Kartu hasil: label vonis besar berwarna, meter keyakinan (persen), probabilitas HOAKS vs VALID, daftar "Frasa pemicu terkuat" (5 term dengan kontribusi positif terbesar, tampilkan term + nilai kontribusi), dan teks input yang dirender ulang dengan kata/frasa pemicu disorot (highlight untuk unigram yang masuk top kontribusi; bigram cukup di daftar).
4. Panel "Dua Model CakraWatch" (angka HARUS persis ini):
   - Model Demo Langsung (di halaman ini): TF-IDF + Logistic Regression, dilatih 892 data TurnBackHoax, Accuracy 0,9375, Macro F1 0,9354. Catatan transparan: model ringan ini yang berjalan langsung di browser pengunjung.
   - Model Utama IndoBERT (fine-tune `indolem/indobert-base-uncased` di Google Colab, 21 Okt 2026): Accuracy 0,9955, Macro F1 0,9953, F1 HOAKS 1,00, F1 VALID 0,99, data uji 224 (1 meleset). Catatan: model penuh dijalankan lewat notebook/repo pendamping.
   - Keterbatasan yang dicatat jujur: kelas VALID data latih berbentuk artikel klarifikasi pemeriksa fakta; teks resmi singkat bertopik sama dengan hoaks dapat tertukar (domain shift). Roadmap: tambah berita valid media arus utama + teks pendek media sosial.
5. Panel "Cara kerja": 3 langkah ringkas (teks masuk → TF-IDF memberi bobot kata → regresi logistik menghitung probabilitas; IndoBERT memahami konteks kalimat penuh).
6. Footer: "Dibangun oleh Nugroho Riziq Darmawan Nata Saputra (tim CakraWatch, solo) untuk Kartika Cyber Hackathon 2026 — PUSSIBERAD TNI AD x CodePolitan. Dataset: TurnBackHoax/Mafindo (repo JibranFawaid). Model dasar: IndoBERT (IndoLEM)."

## Aturan kejujuran
- Jangan mengklaim halaman ini menjalankan IndoBERT. Yang berjalan di halaman = model TF-IDF (disebut eksplisit).
- Semua angka metrik persis seperti di spec ini; jangan menambah metrik karangan.
- Riwayat analisis sesi (daftar hasil analisis pengunjung selama halaman terbuka) boleh ditampilkan lokal, tanpa penyimpanan.
