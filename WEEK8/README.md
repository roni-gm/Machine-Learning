# Week 8 - K-Means Clustering: Wine Quality

Tugas praktikum clustering menggunakan algoritma K-Means pada dataset Wine Quality. Tujuannya adalah mengelompokkan data wine secara unsupervised berdasarkan fitur `alcohol` dan `volatile acidity`, lalu membandingkan hasilnya dengan label kualitas asli dari anotator.

**Nama:** Roni Gunawan Muhammad  
**NIM:** 4222311024  
**Kelas:** Robotika A Malam

---

## Dataset

`WineQT.csv` — dataset kualitas wine dengan fitur utama yang digunakan:
- `alcohol` — kadar alkohol
- `volatile acidity` — tingkat keasaman volatil
- `quality` — label kualitas wine (dari anotator), digunakan sebagai pembanding

---

## Alur Kerja

### 1. Exploratory Data Analysis (EDA)
Melihat distribusi dan sebaran data menggunakan scatter plot, box plot, dan histogram pada fitur `alcohol` dan `volatile acidity`.

### 2. Feature Engineering
Karena ini merupakan task unsupervised, tidak diperlukan train-test split. Tahapan yang dilakukan:
- Drop data duplikat
- Standardisasi fitur menggunakan `StandardScaler` agar skala kedua fitur seimbang saat perhitungan jarak K-Means

### 3. K-Means Clustering
Dua metode digunakan untuk menentukan nilai K optimal:

| Metode | K Optimal | Keterangan |
|---|---|---|
| Elbow Method | 3 | Low / Medium / High Quality |
| Via Score Plot (Yellowbrick) | 4 | Pembagian lebih detail |

### 4. Evaluasi
Hasil cluster divisualisasikan menggunakan scatter plot dan dibandingkan langsung dengan sebaran label `quality` dari anotator.

---

## Kesimpulan

- Elbow Method menghasilkan K=3 yang cukup merepresentasikan kualitas wine (rendah, sedang, tinggi).
- Via Score Plot menghasilkan K=4 yang memberikan pembagian lebih rinci, namun tidak sempurna karena label `quality` asli memiliki 6 kategori.
- Setelah feature scaling, kedua fitur memiliki kontribusi yang seimbang dalam proses clustering.

---

## Library

```
numpy, pandas, matplotlib, seaborn, scikit-learn, yellowbrick
```