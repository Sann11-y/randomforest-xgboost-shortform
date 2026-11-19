# ==============================================================================
# FILE: deploy_terminal.py
# SIMULASI DEPLOYMENT EARLY SCREENING PENURUNAN KOGNITIF (TERMINAL VS CODE)
# Jalankan di Terminal VS Code: python deploy_terminal.py
# ==============================================================================

import os
import joblib
import numpy as np
import pandas as pd

BUNDLE_PATH = "artifacts/model_brainrot.joblib"

PRESETS = [
    {
        "id": "UJI-01 (Pengguna Sehat / Terkontrol)",
        "age": 21, "gender": "Female", "primary_platform": "YouTube Shorts",
        "daily_short_form_hours": 1.2, "avg_clip_duration_sec": 18.5,
        "swipes_per_minute": 3.2, "scroll_sessions_day": 4,
        "bedtime_scroll_min": 15, "total_sleep_hours": 7.8,
        "multitasking_score": 1.8, "total_screen_time_hours": 4.5
    },
    {
        "id": "UJI-02 (Pengguna Kasual / Gejala Awal)",
        "age": 23, "gender": "Male", "primary_platform": "Instagram Reels",
        "daily_short_form_hours": 3.5, "avg_clip_duration_sec": 12.0,
        "swipes_per_minute": 5.8, "scroll_sessions_day": 10,
        "bedtime_scroll_min": 65, "total_sleep_hours": 6.8,
        "multitasking_score": 3.5, "total_screen_time_hours": 7.2
    },
    {
        "id": "UJI-03 (Pecandu Berat / Doomscroller)",
        "age": 20, "gender": "Female", "primary_platform": "TikTok",
        "daily_short_form_hours": 6.0, "avg_clip_duration_sec": 7.5,
        "swipes_per_minute": 10.5, "scroll_sessions_day": 17,
        "bedtime_scroll_min": 115, "total_sleep_hours": 5.5,
        "multitasking_score": 6.2, "total_screen_time_hours": 10.0
    },
    {
        "id": "UJI-04 (Pecandu Ekstrem / Begadang Parah)",
        "age": 29, "gender": "Male", "primary_platform": "TikTok",
        "daily_short_form_hours": 9.2, "avg_clip_duration_sec": 4.2,
        "swipes_per_minute": 15.5, "scroll_sessions_day": 25,
        "bedtime_scroll_min": 168, "total_sleep_hours": 4.2,
        "multitasking_score": 8.8, "total_screen_time_hours": 13.5
    },
    {
        "id": "UJI-05 (Pekerja Multitasking Ekstrem)",
        "age": 32, "gender": "Male", "primary_platform": "Snapchat Spotlight",
        "daily_short_form_hours": 8.5, "avg_clip_duration_sec": 5.0,
        "swipes_per_minute": 14.0, "scroll_sessions_day": 23,
        "bedtime_scroll_min": 155, "total_sleep_hours": 4.8,
        "multitasking_score": 8.2, "total_screen_time_hours": 12.8
    }
]

def load_bundle():
    if not os.path.exists(BUNDLE_PATH):
        print(f"\n[ERROR] File '{BUNDLE_PATH}' belum ditemukan!")
        print("Jalankan sel terakhir Bab 5 di 'main.ipynb' terlebih dahulu untuk menyimpan model.")
        exit(1)
    return joblib.load(BUNDLE_PATH)

def audit_red_flags(row):
    flags = []
    if row["daily_short_form_hours"] > 4.2:
        flags.append(f"Durasi Video Pendek Tinggi ({row['daily_short_form_hours']} jam/hari)")
    if row["bedtime_scroll_min"] > 79:
        flags.append(f"Bedtime Scrolling Berlebih ({row['bedtime_scroll_min']} menit)")
    if row["swipes_per_minute"] > 7.3:
        flags.append(f"Kecepatan Swipe Kompulsif ({row['swipes_per_minute']} swipe/menit)")
    if row["total_sleep_hours"] < 6.5:
        flags.append(f"Defisit Jam Tidur ({row['total_sleep_hours']} jam)")
    return flags

def run_batch_presets(bundle):
    prep = bundle["preprocessor"]
    rf = bundle["rf_tuned"]
    xgb = bundle["xgb_tuned"]
    fitur = bundle["fitur"]
    labels = bundle["label_tingkat_kognitif"]

    df_raw = pd.DataFrame(PRESETS)
    X_proc = pd.DataFrame(prep.transform(df_raw[fitur]), columns=prep.get_feature_names_out())

    pred_rf = rf.predict(X_proc)
    prob_rf = rf.predict_proba(X_proc)
    pred_xgb = xgb.predict(X_proc)

    print("\n" + "=" * 95)
    print("   HASIL SIMULASI DEPLOYMENT: 5 PROFIL DATA UJI BARU (MODEL UTAMA: RANDOM FOREST TUNED)")
    print("=" * 95)
    rows = []
    for i, item in enumerate(PRESETS):
        k_rf = int(pred_rf[i])
        conf_rf = float(np.max(prob_rf[i]) * 100)
        k_xgb = int(pred_xgb[i])
        rows.append({
            "Kasus Uji": item["id"],
            "Jam Video": item["daily_short_form_hours"],
            "Swipe/Mnt": item["swipes_per_minute"],
            "Scroll Mlm": item["bedtime_scroll_min"],
            "Tidur": item["total_sleep_hours"],
            "Prediksi RF Tuned": labels[k_rf],
            "Conf. RF": f"{conf_rf:.2f}%",
            "Prediksi XGB": f"Tier {k_xgb}"
        })
    print(pd.DataFrame(rows).to_string(index=False))
    print("=" * 95)

def run_manual_input(bundle):
    prep = bundle["preprocessor"]
    rf = bundle["rf_tuned"]
    xgb = bundle["xgb_tuned"]
    fitur = bundle["fitur"]
    labels = bundle["label_tingkat_kognitif"]

    print("\n--- MASUKKAN 11 INDIKATOR PERILAKU PENGGUNA BARU ---")
    try:
        age = int(input("1. Usia (16-55 tahun)                     : "))
        gender = input("2. Gender (Male / Female)                 : ").strip().capitalize()
        if gender not in ["Male", "Female"]:
            gender = "Male"
        print("   Pilihan Platform: TikTok | Instagram Reels | YouTube Shorts | Snapchat Spotlight")
        platform = input("3. Platform Utama                         : ").strip()
        if platform not in ["TikTok", "Instagram Reels", "YouTube Shorts", "Snapchat Spotlight"]:
            platform = "TikTok"
        daily_hrs = float(input("4. Durasi Nonton Video Pendek (jam/hari)  : "))
        clip_sec = float(input("5. Rata-rata Durasi Klip (detik)          : "))
        swipes = float(input("6. Kecepatan Swipe per Menit              : "))
        sessions = int(input("7. Jumlah Sesi Buka Aplikasi per Hari     : "))
        bedtime = int(input("8. Durasi Scrolling Sebelum Tidur (menit) : "))
        sleep_hrs = float(input("9. Total Jam Tidur Malam (jam)            : "))
        multi = float(input("10. Skor Multitasking (1.0 - 10.0)        : "))
        screen_hrs = float(input("11. Total Screen Time HP (jam/hari)       : "))
    except ValueError:
        print("[ERROR] Input harus berupa angka yang valid!")
        return

    user_row = {
        "age": age, "gender": gender, "primary_platform": platform,
        "daily_short_form_hours": daily_hrs, "avg_clip_duration_sec": clip_sec,
        "swipes_per_minute": swipes, "scroll_sessions_day": sessions,
        "bedtime_scroll_min": bedtime, "total_sleep_hours": sleep_hrs,
        "multitasking_score": multi, "total_screen_time_hours": screen_hrs
    }
    df_u = pd.DataFrame([user_row])[fitur]
    X_u = pd.DataFrame(prep.transform(df_u), columns=prep.get_feature_names_out())

    k_rf = int(rf.predict(X_u)[0])
    p_rf = rf.predict_proba(X_u)[0] * 100
    k_xgb = int(xgb.predict(X_u)[0])
    p_xgb = xgb.predict_proba(X_u)[0] * 100

    print("\n" + "=" * 75)
    print("               HASIL EARLY SCREENING PENURUNAN KOGNITIF")
    print("=" * 75)
    print(f"Vonis Random Forest (Utama) : {labels[k_rf]}")
    print(f"Tingkat Keyakinan (RF)      : {p_rf[k_rf]:.2f}%")
    print(f"Pembanding XGBoost Tuned    : {labels[k_xgb]} ({p_xgb[k_xgb]:.2f}%)")
    print("-" * 75)
    print("Distribusi Probabilitas Risiko (RF Tuned):")
    for idx, lbl in enumerate(labels):
        bar = "#" * int(p_rf[idx] // 4)
        print(f"  {lbl:48s}: {p_rf[idx]:6.2f}% | {bar}")
    print("-" * 75)
    flags = audit_red_flags(user_row)
    if flags:
        print("Peringatan Indikator Berisiko (Red Flags):")
        for f in flags:
            print(f"  [!] {f}")
    else:
        print("Audit Perilaku: [AMAN] Tidak ada indikator perilaku berisiko tinggi.")
    print("=" * 75)

def main():
    bundle = load_bundle()
    while True:
        print("\n=== SISTEM EARLY SCREENING PENURUNAN KOGNITIF (KELOMPOK 2 - FDA) ===")
        print("[1] Jalankan Uji Cepat 5 Skenario Profil Baru (UJI-01 s/d UJI-05)")
        print("[2] Input Manual Profil Pengguna Baru (Custom Demo)")
        print("[0] Keluar")
        pilihan = input("Pilih menu (0/1/2): ").strip()
        if pilihan == "1":
            run_batch_presets(bundle)
        elif pilihan == "2":
            run_manual_input(bundle)
        elif pilihan == "0":
            print("Selesai. Terima kasih!")
            break
        else:
            print("Pilihan tidak valid.")

if __name__ == "__main__":
    main()