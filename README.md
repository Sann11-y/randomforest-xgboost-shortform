<<<<<<< HEAD
# Klasifikasi Tingkat Penurunan Kognitif Pengguna *Short-Form Video* (Random Forest vs XGBoost)

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Random%20Forest-F7931E?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-189FDD)
![Methodology](https://img.shields.io/badge/Methodology-CRISP--DM-2EA44F)
![Status](https://img.shields.io/badge/Status-Selesai-success)

Sistem *early screening* berbasis *Machine Learning* untuk mengklasifikasikan tingkat penurunan kognitif (*cognitive decay / brain rot*) pengguna platform video pendek (TikTok, Instagram Reels, YouTube Shorts, dan Snapchat Spotlight) ke dalam empat tingkatan risiko hanya berdasarkan **11 indikator perilaku layar harian dan demografi** tanpa tes klinis laboratorium.

---

## Ringkasan Eksekutif & Temuan Utama

* **Model Terbaik:** **Random Forest Tuned + Cost-Sensitive (`rf_tuned`)** (`n_estimators=150`, `max_depth=12`, `min_samples_split=5`, `min_samples_leaf=2`).
* **Performa pada Data Uji ($n = 1.747$):**
  * **Accuracy:** `87,64%`
  * **Macro F1-Score:** `0,8636`
  * **Recall Kelas Kritis (`Tier 3: Acute Brain Rot`):** `75,00%` (15 dari 20 kasus akut terdeteksi tepat)
  * **Precision Kelas Kritis (`Tier 3`):** `94,00%` (hanya 1 *false positive*, jauh lebih stabil dibandingkan XGBoost Tuned sebesar `65,00%` dengan 8 *false positive*)
  * **Keamanan Klinis:** `0` kesalahan prediksi lintas ekstrem antara `Tier 0 (Normal)` dengan `Tier 2 (Severe)` maupun `Tier 3 (Acute)`.
* **Empat Faktor Perilaku Paling Dominan (*Feature Importance*):**
  1. Durasi menonton video pendek harian (`daily_short_form_hours`): **22,82%**
  2. Rata-rata durasi klip (`avg_clip_duration_sec`): **13,48%**
  3. Durasi *scrolling* sebelum tidur (`bedtime_scroll_min`): **13,40%**
  4. Kecepatan *swipe* per menit (`swipes_per_minute`): **12,44%**
  *(Total **96,33%** keputusan model berasal dari 8 fitur kebiasaan layar dan tidur, sedangkan demografi dan merek platform hanya menyumbang **3,68%**).*

---

## Gambaran Dataset & Alur Metodologi (CRISP-DM)

Dataset sekunder diperoleh dari Kaggle: [**Brain Rot: Short-Form Video & Attention Span**](https://www.kaggle.com/datasets/samartalwar/brain-rot-short-form-video-and-attention-span) (`short_form_brain_rot_cognitive_decay.csv`).

| Tahap CRISP-DM | Keputusan Teknis & Ringkasan Proses |
| :--- | :--- |
| **1. Business Understanding** | Membangun skrining dini non-klinis 4 kelas: `Tier 0` (*Normal / Intact Focus*), `Tier 1` (*Mild Attentional Strain*), `Tier 2` (*Severe Focus Fragmentation*), dan `Tier 3` (*Acute Brain Rot / Executive Burnout*). |
| **2. Data Understanding** | Eksplorasi 9.000 baris dan 19 atribut; ditemukan multikolinearitas sempurna ($r = 1,0000$) dan *target leakage* pada 6 skor kognitif laboratorium serta ketimpangan kelas tajam ($36,56 : 1$). |
| **3. Data Preparation** | Penyaringan 267 baris kategori minoritas `Non-Binary` -> **8.733 baris bersih**; penghapusan `participant_id` + 6 fitur *leakage* -> **11 fitur prediktor**; retensi seluruh *outlier* (menghapus *outlier* IQR terbukti membuang **99,02%** atau 101 dari 102 sampel `Tier 3`); *Stratified Split* 80:20 (**6.986 latih** / **1.747 uji**); *post-split* `StandardScaler` & `OneHotEncoder(drop='first')` -> **13 fitur akhir**; *Cost-Sensitive Weighting* dengan penalti kelas `Tier 3` sebesar `21,2988`. |
| **4. Modelling** | Eksperimen komparatif 3 skenario (*Baseline*, *Cost-Sensitive*, dan *Tuned + Cost-Sensitive* via `RandomizedSearchCV`, `cv=3`, `scoring='f1_macro'`) pada algoritma **Random Forest** (*Bagging*) vs **XGBoost** (*Boosting*). |
| **5. Evaluation & Deployment** | Evaluasi metrik makro, laporan per kelas, pengecekan *Train vs Test* (*fitting*), *Confusion Matrix*, *Feature Importance*, serta ekspor bundel model ke `artifacts/model_brainrot.joblib` untuk aplikasi CLI interaktif `deploy.py`. |

---

## Hasil Evaluasi Komparatif (6 Skenario Model)

Pengujian dilakukan pada **1.747 baris Data Uji** (*Testing Set*) beserta perbandingan terhadap **6.986 baris Data Latih** (*Training Set*) untuk memverifikasi status *fitting*:

| Algoritma & Skenario | Train Acc | Test Acc | Macro Prec | Macro Rec | Recall Kelas 3 | Train F1 | Test Macro-F1 | Gap F1 | Status Fitting |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| RF Model 1 (*Baseline*) | 1.0000 | 0.8752 | 0.8850 | 0.7738 | 0.5000 | 1.0000 | 0.8141 | 0.1859 | Overfitting |
| RF Model 2 (*Cost-Sensitive*) | 0.9890 | 0.8798 | 0.9041 | 0.8009 | 0.5500 | 0.9927 | 0.8360 | 0.1567 | Overfitting |
| **RF Model 3 (*Tuned + Weighted*)** | **0.9399** | **0.8764** | **0.8849** | **0.8481** | **0.7500** | **0.9388** | **0.8636** | **0.0752** | **Overfitting Ringan (Terbaik)** |
| XGB Model 1 (*Baseline*) | 0.9951 | 0.8741 | 0.8855 | 0.8041 | 0.6000 | 0.9971 | 0.8362 | 0.1609 | Overfitting |
| XGB Model 2 (*Cost-Sensitive*) | 0.9860 | 0.8706 | 0.8685 | 0.8264 | 0.7000 | 0.9916 | 0.8450 | 0.1466 | Overfitting |
| XGB Model 3 (*Tuned + Weighted*) | 0.8806 | 0.8552 | 0.7910 | 0.8370 | 0.7500 | 0.8565 | 0.8112 | 0.0453 | Good Fitting (*Over-Sensitive*) |

> **Catatan Analisis:** Pembatasan kedalaman pohon (`max_depth=12`) pada **RF Model 3** memangkas *Gap F1* lebih dari separuhnya dibandingkan *Baseline* (`0,1859` -> `0,0752`). Meskipun **XGB Model 3** mencatat *Gap F1* paling tipis (`0,0453`), kapasitas prediksinya turun drastis (*Precision Kelas 3* anjlok ke `0,65`). Skor **uji** RF Model 3 (`0,8636`) tetap lebih tinggi bahkan dibandingkan skor **latih** XGB Model 3 (`0,8565`).

---

## Struktur Direktori Repositori

```text
randomforest-xgboost-shortform/
=======
Klasifikasi Tingkat Penurunan Kognitif Pengguna Short-Form Video (Random Forest vs XGBoost)Sistem early screening berbasis Machine Learning untuk mengklasifikasikan tingkat penurunan kognitif (cognitive decay / brain rot) pengguna platform video pendek (TikTok, Instagram Reels, YouTube Shorts, dan Snapchat Spotlight) ke dalam empat tingkatan risiko hanya berdasarkan 11 indikator perilaku layar harian dan demografi tanpa tes klinis laboratorium.Ringkasan Eksekutif & Temuan UtamaModel Terbaik: Random Forest Tuned + Cost-Sensitive (rf_tuned) (n_estimators=150, max_depth=12, min_samples_split=5, min_samples_leaf=2).Performa pada Data Uji ($n = 1.747$):Accuracy: 87,64%Macro F1-Score: 0,8636Recall Kelas Kritis (Tier 3: Acute Brain Rot): 75,00% (15 dari 20 kasus akut terdeteksi tepat)Precision Kelas Kritis (Tier 3): 94,00% (hanya 1 false positive, jauh lebih stabil dibandingkan XGBoost Tuned sebesar 65,00% dengan 8 false positive)Keamanan Klinis: 0 kesalahan prediksi lintas ekstrem antara Tier 0 (Normal) dengan Tier 2 (Severe) maupun Tier 3 (Acute).Empat Faktor Perilaku Paling Dominan (Feature Importance):Durasi menonton video pendek harian (daily_short_form_hours): 22,82%Rata-rata durasi klip (avg_clip_duration_sec): 13,48%Durasi scrolling sebelum tidur (bedtime_scroll_min): 13,40%Kecepatan swipe per menit (swipes_per_minute): 12,44%
(Total 96,33% keputusan model berasal dari 8 fitur kebiasaan layar dan tidur, sedangkan demografi dan merek platform hanya menyumbang 3,68%).Gambaran Dataset & Alur Metodologi (CRISP-DM)Dataset sekunder diperoleh dari Kaggle: Brain Rot: Short-Form Video & Attention Span (short_form_brain_rot_cognitive_decay.csv).Tahap CRISP-DMKeputusan Teknis & Ringkasan Proses1. Business UnderstandingMembangun skrining dini non-klinis 4 kelas: Tier 0 (Normal / Intact Focus), Tier 1 (Mild Attentional Strain), Tier 2 (Severe Focus Fragmentation), dan Tier 3 (Acute Brain Rot / Executive Burnout).2. Data UnderstandingEksplorasi 9.000 baris dan 19 atribut; ditemukan multikolinearitas sempurna ($r = 1,0000$) dan target leakage pada 6 skor kognitif laboratorium serta ketimpangan kelas tajam ($36,56 : 1$).3. Data PreparationPenyaringan 267 baris kategori minoritas Non-Binary -> 8.733 baris bersih; penghapusan participant_id + 6 fitur leakage -> 11 fitur prediktor; retensi seluruh outlier (menghapus outlier IQR terbukti membuang 99,02% atau 101 dari 102 sampel Tier 3); Stratified Split 80:20 (6.986 latih / 1.747 uji); post-split StandardScaler & OneHotEncoder(drop='first') -> 13 fitur akhir; Cost-Sensitive Weighting dengan penalti kelas Tier 3 sebesar 21,2988.4. ModellingEksperimen komparatif 3 skenario (Baseline, Cost-Sensitive, dan Tuned + Cost-Sensitive via RandomizedSearchCV, cv=3, scoring='f1_macro') pada algoritma Random Forest (Bagging) vs XGBoost (Boosting).5. Evaluation & DeploymentEvaluasi metrik makro, laporan per kelas, pengecekan Train vs Test (fitting), Confusion Matrix, Feature Importance, serta ekspor bundel model ke artifacts/model_brainrot.joblib untuk aplikasi CLI interaktif deploy.py.Hasil Evaluasi Komparatif (6 Skenario Model)Pengujian dilakukan pada 1.747 baris Data Uji (Testing Set) beserta perbandingan terhadap 6.986 baris Data Latih (Training Set) untuk memverifikasi status fitting:Algoritma & SkenarioTrain AccTest AccMacro PrecMacro RecRecall Kelas 3Train F1Test Macro-F1Gap F1Status FittingRF Model 1 (Baseline)1.00000.87520.88500.77380.50001.00000.81410.1859OverfittingRF Model 2 (Cost-Sensitive)0.98900.87980.90410.80090.55000.99270.83600.1567OverfittingRF Model 3 (Tuned + Weighted)0.93990.87640.88490.84810.75000.93880.86360.0752Overfitting Ringan (Terbaik)XGB Model 1 (Baseline)0.99510.87410.88550.80410.60000.99710.83620.1609OverfittingXGB Model 2 (Cost-Sensitive)0.98600.87060.86850.82640.70000.99160.84500.1466OverfittingXGB Model 3 (Tuned + Weighted)0.88060.85520.79100.83700.75000.85650.81120.0453Good Fitting (Over-Sensitive)Catatan Analisis: Pembatasan kedalaman pohon (max_depth=12) pada RF Model 3 memangkas Gap F1 lebih dari separuhnya dibandingkan Baseline (0,1859 -> 0,0752). Meskipun XGB Model 3 mencatat Gap F1 paling tipis (0,0453), kapasitas prediksinya turun drastis (Precision Kelas 3 anjlok ke 0,65). Skor uji RF Model 3 (0,8636) tetap lebih tinggi bahkan dibandingkan skor latih XGB Model 3 (0,8565).Struktur Direktori Repositorirandomforest-xgboost-shortform/
>>>>>>> aec59942a0ef5dac5059b1b730ff676159b5599e
├── artifacts/
│   └── model_brainrot.joblib          # Bundel preprocessor & model terlatih (siap demo)
├── dataset/                           # (Unduh manual dari Kaggle, tidak di-commit ke Git)
│   └── short_form_brain_rot_cognitive_decay.csv
├── deploy.py                          # Aplikasi CLI Early Screening (Menu Uji Skenario & Input Manual)
├── main.ipynb                         # Pipeline lengkap CRISP-DM (Bab 2 s/d Bab 5)
├── requirements.txt                   # Daftar dependensi pustaka Python
├── .gitignore                         # Konfigurasi file yang diabaikan Git
└── README.md                          # Dokumentasi utama repositori
<<<<<<< HEAD
```

---

## Panduan Instalasi & Cara Menjalankan (*Quick Start*)

### 1. Kloning Repositori & Instalasi Dependensi

```bash
git clone https://github.com/sann11-y/randomforest-xgboost-shortform.git
=======
Panduan Instalasi & Cara Menjalankan (Quick Start)1. Kloning Repositori & Instalasi Dependensigit clone https://github.com/sann11-y/randomforest-xgboost-shortform.git
>>>>>>> aec59942a0ef5dac5059b1b730ff676159b5599e
cd randomforest-xgboost-shortform

# Buat dan aktifkan virtual environment (opsional namun disarankan)
python -m venv .venv
# Windows (PowerShell/CMD):
.venv\Scripts\activate
# macOS / Linux:
# source .venv/bin/activate

pip install -r requirements.txt
<<<<<<< HEAD
```

### 2. Mode A — Demo Cepat *Early Screening* (Tanpa Unduh Dataset)

Berkas `artifacts/model_brainrot.joblib` sudah disertakan di repositori sehingga aplikasi skrining dapat langsung dijalankan:

```bash
python deploy.py
```

Aplikasi menyediakan dua mode:

* **Menu `1`:** Pengujian serentak 5 skenario pengguna baru (`UJI-01` s/d `UJI-05`) berbasis zona kuartil ($Q_1$, Median, $Q_3$) dan batas *outlier* IQR.
* **Menu `2`:** Input manual 11 indikator perilaku layar secara interaktif beserta distribusi probabilitas dan catatan indikator risiko.

**Contoh Cuplikan Output Menu `1`:**

```text
Simulasi Skrining 5 Profil Pengguna Baru (Model: Random Forest Tuned)
=======
2. Mode A — Demo Cepat Early Screening (Tanpa Unduh Dataset)Berkas artifacts/model_brainrot.joblib sudah disertakan di repositori sehingga aplikasi skrining dapat langsung dijalankan:python deploy.py
Aplikasi menyediakan dua mode:Menu 1: Pengujian serentak 5 skenario pengguna baru (UJI-01 s/d UJI-05) berbasis zona kuartil ($Q_1$, Median, $Q_3$) dan batas outlier IQR.Menu 2: Input manual 11 indikator perilaku layar secara interaktif beserta distribusi probabilitas dan catatan indikator risiko.Contoh Cuplikan Output Menu 1:Simulasi Skrining 5 Profil Pengguna Baru (Model: Random Forest Tuned)
>>>>>>> aec59942a0ef5dac5059b1b730ff676159b5599e
------------------------------------------------------------------------------------------------------------------------------
Data Profil Input (11 Fitur):
    ID  Usia Gender           Platform  Video(j)  Klip(d)  Swipe/m  Sesi  Mlm(m)  Tidur(j)  Multi  Layar(j)
UJI-01    21 Female     YouTube Shorts       1.2     18.5      3.2     4      15       7.8    1.5       4.5
UJI-02    23   Male    Instagram Reels       3.5     12.0      5.8    10      65       6.8    3.5       7.2
UJI-03    20 Female             TikTok       6.0      7.5     10.5    17     115       5.5    6.2      10.0
UJI-04    29   Male             TikTok       9.2      4.2     15.5    25     168       4.2    8.8      13.5
UJI-05    32   Male Snapchat Spotlight       8.5      5.0     14.0    23     155       4.8    9.2      12.8

Hasil Klasifikasi:
    ID                           Profil                                 Hasil Prediksi Keyakinan                         Saran
UJI-01      Pengguna Sehat / Terkontrol    Tier 0: Penurunan Nol / Fokus Utuh (Normal)    99.61%  Aman, pertahankan pola sehat
UJI-02    Pengguna Kasual / Gejala Awal         Tier 1: Penurunan Ringan (Mild Strain)    91.20%     Mulai batasi durasi layar
UJI-03     Pecandu Berat / Doomscroller Tier 2: Penurunan Parah (Severe Fragmentation)    97.60%    Perlu pengurangan intensif
UJI-04 Pecandu Ekstrem / Begadang Parah       Tier 3: Penurunan Akut (Acute Brain Rot)    92.63% Segera lakukan detoks digital
UJI-05     Pekerja Multitasking Ekstrem       Tier 3: Penurunan Akut (Acute Brain Rot)    90.18% Segera lakukan detoks digital
------------------------------------------------------------------------------------------------------------------------------
<<<<<<< HEAD
```

### 3. Mode B — Reproduksi Pelatihan Model dari Awal (`main.ipynb`)

1. Unduh berkas `short_form_brain_rot_cognitive_decay.csv` dari [halaman Kaggle berikut](https://www.kaggle.com/datasets/samartalwar/brain-rot-short-form-video-and-attention-span).
2. Buat folder `dataset/` di direktori utama repositori dan letakkan file CSV tersebut di `dataset/short_form_brain_rot_cognitive_decay.csv`.
3. Buka `main.ipynb` di VS Code atau Jupyter Notebook lalu pilih **Restart & Run All** untuk meregenerasi seluruh analisis dan berkas `artifacts/model_brainrot.joblib`.

---

## Rencana Pengembangan Lanjutan (*Roadmap*)

* [ ] **Validasi Data Lapangan Nyata:** Melakukan pengujian model terhadap data telemetri *Digital Wellbeing* / *Screen Time* dan kuesioner primer mahasiswa di Indonesia guna menguji daya generalisasi di luar dataset sekunder Kaggle.
* [ ] **Penyetelan Regularisasi Lanjutan:** Mengeksplorasi kombinasi *cost-complexity pruning* (`ccp_alpha`) dan pembatasan `max_samples` untuk menekan *overfitting* ringan pada `rf_tuned` (`Gap F1` dari `0,0752` menuju `< 0,05`) tanpa menurunkan *Recall Kelas 3*.
* [ ] **Migrasi Antarmuka Web:** Mengembangkan purwarupa aplikasi berbasis web (Streamlit / FastAPI) agar pengguna awam dapat melakukan skrining mandiri tanpa perlu menggunakan terminal.

---

## Tim Penyusun & Konteks Akademik

Proyek Akhir Mata Kuliah **Fundamental Data Analyst (FDA)** — **Kelompok 2**  
Program Studi Sistem Informasi, Fakultas Teknik dan Informatika, Universitas Bina Sarana Informatika (2026).

* **Anggota Kelompok 2:**
  * Tata Dwi Tama Putra
  * Said Hasan Nasrallah
  * Hon Zie Kaori
  * Shabryan Khaylani
  * Sheren Putri Toisuta
* **Dosen Pengampu:** Widiarina, M.Kom

<!-- markdownlint-disable-file MD013 -->
=======
3. Mode B — Reproduksi Pelatihan Model dari Awal (main.ipynb)Unduh berkas short_form_brain_rot_cognitive_decay.csv dari halaman Kaggle berikut.Buat folder dataset/ di direktori utama repositori dan letakkan file CSV tersebut di dataset/short_form_brain_rot_cognitive_decay.csv.Buka main.ipynb di VS Code atau Jupyter Notebook lalu pilih Restart & Run All untuk meregenerasi seluruh analisis dan berkas artifacts/model_brainrot.joblib.Rencana Pengembangan Lanjutan (Roadmap)[ ] Validasi Data Lapangan Nyata: Melakukan pengujian model terhadap data telemetri Digital Wellbeing / Screen Time dan kuesioner primer mahasiswa di Indonesia guna menguji daya generalisasi di luar dataset sekunder Kaggle.[ ] Penyetelan Regularisasi Lanjutan: Mengeksplorasi kombinasi cost-complexity pruning (ccp_alpha) dan pembatasan max_samples untuk menekan overfitting ringan pada rf_tuned (Gap F1 dari 0,0752 menuju < 0,05) tanpa menurunkan Recall Kelas 3.[ ] Migrasi Antarmuka Web: Mengembangkan purwarupa aplikasi berbasis web (Streamlit / FastAPI) agar pengguna awam dapat melakukan skrining mandiri tanpa perlu menggunakan terminal.Tim Penyusun & Konteks AkademikProyek Akhir Mata Kuliah Fundamental Data Analyst (FDA) — Kelompok 2Program Studi Sistem Informasi, Fakultas Teknik dan Informatika, Universitas Bina Sarana Informatika (2026).Anggota Kelompok 2:Tata Dwi Tama PutraSaid Hasan NasrallahHon Zie KaoriShabryan KhaylaniSheren Putri ToisutaDosen Pengampu: Widiarina, M.Kom
>>>>>>> aec59942a0ef5dac5059b1b730ff676159b5599e
