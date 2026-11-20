import sys
from pathlib import Path
import joblib
import pandas as pd

LOKASI_MODEL = Path("artifacts/model_brainrot.joblib")
OPSI_GENDER = ("Male", "Female")
OPSI_PLATFORM = ("TikTok", "Instagram Reels", "YouTube Shorts", "Snapchat Spotlight")

KOLOM_DATA = [
    "id", "profil", "age", "gender", "primary_platform",
    "daily_short_form_hours", "avg_clip_duration_sec", "swipes_per_minute",
    "scroll_sessions_day", "bedtime_scroll_min", "total_sleep_hours",
    "multitasking_score", "total_screen_time_hours",
]

# 5 profil uji berdasarkan kuartil (Q1, Median, Q3) & batas outlier IQR Bab 2
DATA_SKENARIO = [
    ("UJI-01", "Pengguna Sehat / Terkontrol",      21, "Female", "YouTube Shorts",     1.2, 18.5,  3.2,  4,  15, 7.8, 1.5,  4.5),
    ("UJI-02", "Pengguna Kasual / Gejala Awal",    23, "Male",   "Instagram Reels",    3.5, 12.0,  5.8, 10,  65, 6.8, 3.5,  7.2),
    ("UJI-03", "Pecandu Berat / Doomscroller",     20, "Female", "TikTok",             6.0,  7.5, 10.5, 17, 115, 5.5, 6.2, 10.0),
    ("UJI-04", "Pecandu Ekstrem / Begadang Parah", 29, "Male",   "TikTok",             9.2,  4.2, 15.5, 25, 168, 4.2, 8.8, 13.5),
    ("UJI-05", "Pekerja Multitasking Ekstrem",     32, "Male",   "Snapchat Spotlight", 8.5,  5.0, 14.0, 23, 155, 4.8, 9.2, 12.8),
]

LABEL_PROFIL = {
    "id": "ID", "age": "Usia", "gender": "Gender", "primary_platform": "Platform",
    "daily_short_form_hours": "Video(j)", "avg_clip_duration_sec": "Klip(d)",
    "swipes_per_minute": "Swipe/m", "scroll_sessions_day": "Sesi",
    "bedtime_scroll_min": "Mlm(m)", "total_sleep_hours": "Tidur(j)",
    "multitasking_score": "Multi", "total_screen_time_hours": "Layar(j)",
}

SARAN_TIER = {
    0: "Aman, pertahankan pola sehat",
    1: "Mulai batasi durasi layar",
    2: "Perlu pengurangan intensif",
    3: "Segera lakukan detoks digital",
}

ATURAN_RISIKO = [
    ("daily_short_form_hours", lambda v: v > 4.2, "Durasi video pendek tinggi ({v} jam/hari)"),
    ("avg_clip_duration_sec",  lambda v: v < 8.2, "Durasi tonton per klip terlalu singkat ({v} detik)"),
    ("bedtime_scroll_min",     lambda v: v > 79,  "Kebiasaan scroll sebelum tidur lama ({v} menit)"),
    ("swipes_per_minute",      lambda v: v > 7.3, "Kecepatan swipe terlalu tinggi ({v} kali/menit)"),
    ("total_sleep_hours",      lambda v: v < 6.5, "Jam tidur kurang dari ideal ({v} jam)"),
    ("multitasking_score",     lambda v: v > 4.5, "Skor multitasking tinggi ({v}/10)"),
]

DAFTAR_INPUT = [
    ("age",                     "1. Usia (16-55 tahun)",                         int),
    ("gender",                  "2. Gender (Male / Female)",                     OPSI_GENDER),
    ("primary_platform",        f"3. Platform ({' / '.join(OPSI_PLATFORM)})",    OPSI_PLATFORM),
    ("daily_short_form_hours",  "4. Durasi video pendek (jam/hari)",             float),
    ("avg_clip_duration_sec",   "5. Rata-rata durasi klip (detik)",              float),
    ("swipes_per_minute",       "6. Kecepatan swipe per menit",                  float),
    ("scroll_sessions_day",     "7. Jumlah sesi buka aplikasi per hari",         int),
    ("bedtime_scroll_min",      "8. Durasi scroll sebelum tidur (menit)",        int),
    ("total_sleep_hours",       "9. Total jam tidur malam (jam)",                float),
    ("multitasking_score",      "10. Skor multitasking (1.0 - 10.0)",            float),
    ("total_screen_time_hours", "11. Total screen time HP (jam/hari)",           float),
]


def muat_model():
    if not LOKASI_MODEL.exists():
        sys.exit(f"File '{LOKASI_MODEL}' belum ada. Jalankan sel export model di main.ipynb dulu.")
    return joblib.load(LOKASI_MODEL)


def prediksi(bundel, df):
    praproses = bundel["preprocessor"]
    X = pd.DataFrame(
        praproses.transform(df[bundel["fitur"]]),
        columns=praproses.get_feature_names_out(),
    )
    model = bundel["rf_tuned"]
    return model.predict(X), model.predict_proba(X) * 100


def baca_input(teks, aturan):
    while True:
        isian = input(f"{teks:<44}: ").strip()
        if isinstance(aturan, tuple):
            cocok = next((o for o in aturan if o.lower() == isian.lower()), None)
            if cocok:
                return cocok
            print(f"  Pilihan tidak sesuai. Gunakan: {', '.join(aturan)}")
            continue
        try:
            nilai = aturan(isian)
            if nilai < 0:
                print("  Nilai tidak boleh negatif.")
                continue
            return nilai
        except ValueError:
            print("  Masukkan angka yang valid.")


def uji_skenario(bundel):
    df = pd.DataFrame(DATA_SKENARIO, columns=KOLOM_DATA)
    kelas, persen = prediksi(bundel, df)
    label_tier = bundel["label_tingkat_kognitif"]

    teks_profil = df[list(LABEL_PROFIL)].rename(columns=LABEL_PROFIL).to_string(index=False)
    teks_hasil = pd.DataFrame({
        "ID": df["id"],
        "Profil": df["profil"],
        "Hasil Prediksi": [label_tier[int(k)] for k in kelas],
        "Keyakinan": [f"{p.max():.2f}%" for p in persen],
        "Saran": [SARAN_TIER[int(k)] for k in kelas],
    }).to_string(index=False)

    lebar = max(
        max(len(b) for b in teks_profil.splitlines()),
        max(len(b) for b in teks_hasil.splitlines()),
    )

    print("\nSimulasi Skrining 5 Profil Pengguna Baru (Model: Random Forest Tuned)")
    print("-" * lebar)
    print("Data Profil Input (11 Fitur):")
    print(teks_profil)
    print("\nHasil Klasifikasi:")
    print(teks_hasil)
    print("-" * lebar)


def uji_manual(bundel):
    print("\nMasukkan data kebiasaan layar pengguna:")
    profil = {kolom: baca_input(teks, aturan) for kolom, teks, aturan in DAFTAR_INPUT}

    kelas, persen = prediksi(bundel, pd.DataFrame([profil]))
    k, p = int(kelas[0]), persen[0]
    label_tier = bundel["label_tingkat_kognitif"]
    risiko = [pesan.format(v=profil[kol]) for kol, cek, pesan in ATURAN_RISIKO if cek(profil[kol])]

    print("\nHasil Skrining Pengguna (Random Forest Tuned)")
    print("-" * 68)
    print(f"Kategori  : {label_tier[k]}")
    print(f"Keyakinan : {p[k]:.2f}%")
    print(f"Saran     : {SARAN_TIER[k]}")

    print("\nProbabilitas per Tingkat:")
    for label, nilai in zip(label_tier, p):
        print(f"  - {label:<47} {nilai:6.2f}%")

    print("\nCatatan Risiko:")
    if risiko:
        for item in risiko:
            print(f"  - {item}")
    else:
        print("  - Tidak ada indikator yang melewati batas normal.")
    print("-" * 68)


def main():
    bundel = muat_model()
    menu = {"1": uji_skenario, "2": uji_manual}

    try:
        while True:
            print("\nSkrining Dini Penurunan Kognitif — Kelompok 2 FDA")
            print("1. Uji 5 profil skenario baru (UJI-01 s/d UJI-05)")
            print("2. Input data pengguna manual")
            print("0. Keluar")

            pilih = input("Pilih menu: ").strip()
            if pilih == "0":
                print("Program selesai.")
                break
            aksi = menu.get(pilih)
            aksi(bundel) if aksi else print("Menu tidak tersedia, pilih 0, 1, atau 2.")
    except (KeyboardInterrupt, EOFError):
        print("\nProgram dihentikan.")


if __name__ == "__main__":
    main()