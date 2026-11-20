# ==============================================================================
# FILE: deploy.py
# SISTEM EARLY SCREENING PENURUNAN KOGNITIF (TERMINAL VS CODE)
# Model Utama: Random Forest Tuned + Cost-Sensitive
# Cara jalankan: python deploy.py
# ==============================================================================

import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

LOKASI_MODEL = Path("artifacts/model_brainrot.joblib")
DAFTAR_GENDER = ("Male", "Female")
DAFTAR_PLATFORM = ("TikTok", "Instagram Reels", "YouTube Shorts", "Snapchat Spotlight")

REKOMENDASI_TIER = {
    0: "Aman / Pertahankan",
    1: "Pantau & Batasi",
    2: "Perlu Intervensi",
    3: "Penanganan Segera",
}

# 5 Skenario Uji Cepat berbasis zona kuartil & batas outlier IQR (Bab 2)
SKENARIO_UJI = [
    {
        "id": "UJI-01", "profil": "Pengguna Sehat / Terkontrol",
        "age": 21, "gender": "Female", "primary_platform": "YouTube Shorts",
        "daily_short_form_hours": 1.2, "avg_clip_duration_sec": 18.5,
        "swipes_per_minute": 3.2, "scroll_sessions_day": 4,
        "bedtime_scroll_min": 15, "total_sleep_hours": 7.8,
        "multitasking_score": 1.5, "total_screen_time_hours": 4.5,
    },
    {
        "id": "UJI-02", "profil": "Pengguna Kasual / Gejala Awal",
        "age": 23, "gender": "Male", "primary_platform": "Instagram Reels",
        "daily_short_form_hours": 3.5, "avg_clip_duration_sec": 12.0,
        "swipes_per_minute": 5.8, "scroll_sessions_day": 10,
        "bedtime_scroll_min": 65, "total_sleep_hours": 6.8,
        "multitasking_score": 3.5, "total_screen_time_hours": 7.2,
    },
    {
        "id": "UJI-03", "profil": "Pecandu Berat / Doomscroller",
        "age": 20, "gender": "Female", "primary_platform": "TikTok",
        "daily_short_form_hours": 6.0, "avg_clip_duration_sec": 7.5,
        "swipes_per_minute": 10.5, "scroll_sessions_day": 17,
        "bedtime_scroll_min": 115, "total_sleep_hours": 5.5,
        "multitasking_score": 6.2, "total_screen_time_hours": 10.0,
    },
    {
        "id": "UJI-04", "profil": "Pecandu Ekstrem / Begadang Parah",
        "age": 29, "gender": "Male", "primary_platform": "TikTok",
        "daily_short_form_hours": 9.2, "avg_clip_duration_sec": 4.2,
        "swipes_per_minute": 15.5, "scroll_sessions_day": 25,
        "bedtime_scroll_min": 168, "total_sleep_hours": 4.2,
        "multitasking_score": 8.8, "total_screen_time_hours": 13.5,
    },
    {
        "id": "UJI-05", "profil": "Pekerja Multitasking Ekstrem",
        "age": 32, "gender": "Male", "primary_platform": "Snapchat Spotlight",
        "daily_short_form_hours": 8.5, "avg_clip_duration_sec": 5.0,
        "swipes_per_minute": 14.0, "scroll_sessions_day": 23,
        "bedtime_scroll_min": 155, "total_sleep_hours": 4.8,
        "multitasking_score": 9.2, "total_screen_time_hours": 12.8,
    },
]

# Aturan deteksi indikator berisiko (> Q3 / < Q1 pada fitur protektif)
ATURAN_BAHAYA = [
    ("daily_short_form_hours", lambda v: v > 4.2, "Durasi Video Pendek Tinggi ({v} jam/hari)"),
    ("avg_clip_duration_sec",  lambda v: v < 8.2, "Durasi Klip Sangat Pendek ({v} detik/klip)"),
    ("bedtime_scroll_min",     lambda v: v > 79,  "Bedtime Scrolling Berlebih ({v} menit)"),
    ("swipes_per_minute",      lambda v: v > 7.3, "Kecepatan Swipe Kompulsif ({v} swipe/menit)"),
    ("total_sleep_hours",      lambda v: v < 6.5, "Defisit Jam Tidur ({v} jam)"),
    ("multitasking_score",     lambda v: v > 4.5, "Intensitas Multitasking Tinggi (Skor {v}/10)"),
]

# Skema deklaratif untuk 11 input manual: (kolom, label, tipe/opsi)
SKEMA_INPUT = [
    ("age",                     "1. Usia (16-55 tahun)",                        int),
    ("gender",                  "2. Gender (Male / Female)",                    DAFTAR_GENDER),
    ("primary_platform",        f"3. Platform ({' | '.join(DAFTAR_PLATFORM)})", DAFTAR_PLATFORM),
    ("daily_short_form_hours",  "4. Durasi Nonton Video Pendek (jam/hari)",     float),
    ("avg_clip_duration_sec",   "5. Rata-rata Durasi Klip (detik)",             float),
    ("swipes_per_minute",       "6. Kecepatan Swipe per Menit",                 float),
    ("scroll_sessions_day",     "7. Jumlah Sesi Buka Aplikasi per Hari",        int),
    ("bedtime_scroll_min",      "8. Durasi Scrolling Sebelum Tidur (menit)",    int),
    ("total_sleep_hours",       "9. Total Jam Tidur Malam (jam)",               float),
    ("multitasking_score",      "10. Skor Multitasking (1.0 - 10.0)",           float),
    ("total_screen_time_hours", "11. Total Screen Time HP (jam/hari)",          float),
]


def muat_bundel_model() -> dict:
    """Memuat objek preprocessor dan model terbaik (Random Forest Tuned)."""
    if not LOKASI_MODEL.exists():
        sys.exit(
            f"[ERROR] Berkas '{LOKASI_MODEL}' tidak ditemukan.\n"
            "Jalankan sel penyimpanan model di 'main.ipynb' terlebih dahulu."
        )
    return joblib.load(LOKASI_MODEL)


def prediksi_data(bundel: dict, df_mentah: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Mentransformasi data mentah lalu memprediksi kelas & probabilitas memakai Random Forest Tuned."""
    praproses = bundel["preprocessor"]
    model_terbaik = bundel["rf_tuned"]
    matriks_uji = pd.DataFrame(
        praproses.transform(df_mentah[bundel["fitur"]]),
        columns=praproses.get_feature_names_out(),
    )
    return model_terbaik.predict(matriks_uji), model_terbaik.predict_proba(matriks_uji) * 100


def periksa_indikator_bahaya(data_pengguna: dict) -> list[str]:
    """Mengembalikan daftar indikator perilaku yang melewati ambang batas sehat."""
    return [
        pesan.format(v=data_pengguna[kolom])
        for kolom, kondisi, pesan in ATURAN_BAHAYA
        if kondisi(data_pengguna[kolom])
    ]


def minta_input(label: str, aturan):
    """Membaca dan memvalidasi satu input pengguna secara aman tanpa crash."""
    while True:
        isian = input(f"{label:<46}: ").strip()
        if isinstance(aturan, tuple):
            cocok = next((opsi for opsi in aturan if opsi.lower() == isian.lower()), None)
            if cocok:
                return cocok
            print(f"   [!] Pilihan tidak dikenali. Gunakan salah satu: {', '.join(aturan)}")
        else:
            try:
                return aturan(isian)
            except ValueError:
                print("   [!] Masukkan angka yang valid.")


def jalankan_uji_skenario(bundel: dict) -> None:
    """Menampilkan 11 fitur profil lengkap beserta hasil klasifikasi model terbaik (RF Tuned)."""
    label_kelas = bundel["label_tingkat_kognitif"]
    pred_rf, prob_rf = prediksi_data(bundel, pd.DataFrame(SKENARIO_UJI))

    tabel_input = pd.DataFrame([
        {
            "ID": d["id"],
            "Usia": d["age"],
            "Gender": d["gender"],
            "Platform": d["primary_platform"],
            "Video(j)": d["daily_short_form_hours"],
            "Klip(d)": d["avg_clip_duration_sec"],
            "Swipe/m": d["swipes_per_minute"],
            "Sesi": d["scroll_sessions_day"],
            "Mlm(m)": d["bedtime_scroll_min"],
            "Tidur(j)": d["total_sleep_hours"],
            "Multi": d["multitasking_score"],
            "Layar(j)": d["total_screen_time_hours"],
        }
        for d in SKENARIO_UJI
    ])

    tabel_hasil = pd.DataFrame([
        {
            "ID": d["id"],
            "Profil Pengguna Baru": d["profil"],
            "Hasil Klasifikasi (Random Forest Tuned)": label_kelas[int(k)],
            "Confidence": f"{prob_rf[i].max():.2f}%",
            "Rekomendasi": REKOMENDASI_TIER[int(k)],
        }
        for i, (d, k) in enumerate(zip(SKENARIO_UJI, pred_rf))
    ])

    teks_input = tabel_input.to_string(index=False)
    teks_hasil = tabel_hasil.to_string(index=False)
    lebar = max(
        max(len(baris) for baris in teks_input.splitlines()),
        max(len(baris) for baris in teks_hasil.splitlines()),
    )

    garis = "=" * lebar
    print(f"\n{garis}")
    print("SIMULASI DEPLOYMENT EARLY SCREENING - MODEL TERBAIK: RANDOM FOREST TUNED".center(lebar))
    print(garis)
    print("[A] PROFIL LENGKAP 5 DATA UJI BARU (11 FITUR INPUT):")
    print(teks_input)
    print("-" * lebar)
    print("[B] HASIL PREDIKSI & REKOMENDASI (RANDOM FOREST TUNED):")
    print(teks_hasil)
    print(garis)


def jalankan_input_manual(bundel: dict) -> None:
    """Menerima 11 input manual dari terminal dan menampilkan hasil skrining lengkap."""
    print("\n--- MASUKKAN 11 INDIKATOR PERILAKU PENGGUNA BARU ---")
    profil_baru = {kolom: minta_input(label, aturan) for kolom, label, aturan in SKEMA_INPUT}

    pred_rf, prob_rf = prediksi_data(bundel, pd.DataFrame([profil_baru]))
    kelas_rf, persen_rf = int(pred_rf[0]), prob_rf[0]
    label_kelas = bundel["label_tingkat_kognitif"]
    peringatan = periksa_indikator_bahaya(profil_baru)

    garis = "=" * 78
    print(f"\n{garis}")
    print("        HASIL EARLY SCREENING (MODEL TERBAIK: RANDOM FOREST TUNED)")
    print(garis)
    print(f"Hasil Klasifikasi  : {label_kelas[kelas_rf]}")
    print(f"Tingkat Keyakinan  : {persen_rf[kelas_rf]:.2f}%")
    print(f"Rekomendasi        : {REKOMENDASI_TIER[kelas_rf]}")
    print("-" * 78)
    print("Distribusi Probabilitas Risiko (150 Pohon Keputusan):")
    for label, nilai in zip(label_kelas, persen_rf):
        batang = "#" * int(nilai // 4)
        print(f"  {label:<48}: {nilai:6.2f}% | {batang}")
    print("-" * 78)
    if peringatan:
        print("Peringatan Indikator Berisiko (Red Flags):")
        print("\n".join(f"  [!] {item}" for item in peringatan))
    else:
        print("Audit Perilaku: [AMAN] Tidak ada indikator perilaku berisiko tinggi.")
    print(garis)


def utama() -> None:
    """Menu interaktif utama di terminal VS Code."""
    bundel = muat_bundel_model()
    daftar_menu = {"1": jalankan_uji_skenario, "2": jalankan_input_manual}

    try:
        while True:
            print("\n=== SISTEM EARLY SCREENING PENURUNAN KOGNITIF (KELOMPOK 2 - FDA) ===")
            print("[1] Jalankan Uji Cepat 5 Skenario Profil Baru (UJI-01 s/d UJI-05)")
            print("[2] Input Manual Profil Pengguna Baru (Custom Demo)")
            print("[0] Keluar")

            pilihan = input("Pilih menu (0/1/2): ").strip()
            if pilihan == "0":
                print("Selesai. Terima kasih!")
                break
            aksi = daftar_menu.get(pilihan)
            aksi(bundel) if aksi else print("[!] Pilihan tidak valid, silakan ketik 0, 1, atau 2.")
    except (KeyboardInterrupt, EOFError):
        print("\nProgram dihentikan oleh pengguna.")


if __name__ == "__main__":
    utama()