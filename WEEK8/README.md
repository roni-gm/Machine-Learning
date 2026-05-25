# Week 8 - K-Means Clustering: Wine Quality

Tugas praktikum clustering menggunakan algoritma K-Means pada dataset Wine Quality.

**Nama:** Roni Gunawan Muhammad  
**NIM:** 4222311024  
**Kelas:** Robotika A Malam

---

## Dataset

`WineQT.csv` — dataset kualitas wine dengan fitur utama yang digunakan:
- `alcohol` — kadar alkohol
- `volatile acidity` — tingkat keasaman volatil
- `quality` — label kualitas wine (dari anotator)

## Alur Kerja

1. **EDA** — scatter plot, box plot, dan histogram untuk memahami distribusi data
2. **Feature Engineering** — drop duplikat dan standardisasi fitur menggunakan `StandardScaler`
3. **K-Means Clustering** dengan dua metode pemilihan nilai K:
   - **Elbow Method** → K=3 (Low / Medium / High Quality)
   - **Via Score Plot (Yellowbrick)** → K=4
4. **Evaluasi** — membandingkan hasil cluster dengan label `quality` secara visual

## Library

```
numpy, pandas, matplotlib, seaborn, scikit-learn, yellowbrick
```
