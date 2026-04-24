# 📘 README — Supervised Learning: Classification

> **Dibuat untuk:** Mahasiswa Teknik Rekayasa Robotika — Politeknik Negeri Batam
> **Topik:** Supervised Learning | Klasifikasi dengan KNN, Naive Bayes, dan Decision Tree
> **Dataset:** `cardata.csv` (data jual-beli mobil bekas)

---

## 🧩 Apa Sebenarnya Program ini?

Program ini adalah sebuah **notebook Python (Google Colab)** yang mengajarkan cara membuat **model machine learning** untuk mengklasifikasikan jenis transmisi mobil — apakah mobil itu **Manual** atau **Automatic** — berdasarkan data-data seperti tahun pembuatan, harga, jarak tempuh, jenis bahan bakar, dan lainnya.

Ini termasuk dalam kategori **Supervised Learning** (Pembelajaran Terawasi), artinya kita melatih komputer menggunakan data yang sudah diketahui jawabannya (berlabel), supaya nanti komputer bisa menebak jawaban dari data baru yang belum pernah dilihat sebelumnya.

---

## 📂 File yang Terlibat

| File | Keterangan |
|---|---|
| `Week_5_6_7_Supervised_Learning_Hands_On_Classification.ipynb` | Notebook utama berisi seluruh kode program |
| `cardata.csv` | Dataset mobil bekas yang digunakan untuk melatih model |
| `hasil_gridsearch_knn.xlsx` | Hasil percobaan otomatis mencari kombinasi parameter KNN terbaik |

---

## 🗂️ Struktur Program (Alur Besar)

```
1. Persiapan (Import Library)
        ↓
2. Memuat Dataset
        ↓
3. Exploratory Data Analysis (EDA) — Mengenal Data
        ↓
4. Feature Engineering — Membersihkan & Menyiapkan Data
        ↓
5. Training (Melatih Model)
     ├── KNN (K-Nearest Neighbors)
     ├── Naive Bayes
     └── Decision Tree
        ↓
6. Evaluasi (Mengukur Performa Model)
        ↓
7. Prediksi Data Baru
        ↓
8. Optimasi dengan GridSearchCV (Khusus KNN)
```

---

## 🔧 BAGIAN 1 — Persiapan (Import Library)

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
```

**Penjelasan:**
Library adalah "kotak peralatan" yang kita pinjam supaya tidak perlu membuat semuanya dari nol.

| Library | Fungsi |
|---|---|
| `pandas` | Membaca dan mengolah data tabel (seperti Excel) |
| `numpy` | Operasi matematika pada angka dan array |
| `matplotlib` & `seaborn` | Membuat grafik dan visualisasi data |
| `sklearn` | Kumpulan algoritma machine learning siap pakai |

---

## 📊 BAGIAN 2 — Memuat Dataset

```python
train = pd.read_csv('cardata.csv')
train.head()
```

**Penjelasan:**
Program membaca file `cardata.csv` yang berisi data mobil bekas. Perintah `.head()` menampilkan 5 baris pertama agar kita bisa mengintip isi datanya.

**Kolom-kolom dalam dataset:**

| Kolom | Keterangan |
|---|---|
| `Year` | Tahun pembuatan mobil |
| `Selling_Price` | Harga jual (dalam Lakh Rupee India) |
| `Present_Price` | Harga baru mobil tersebut saat ini |
| `Kms_Driven` | Total jarak yang sudah ditempuh (km) |
| `Fuel_Type` | Jenis bahan bakar (Petrol / Diesel / CNG) |
| `Seller_Type` | Penjual dealer atau individu |
| `Owner` | Pernah berganti tangan berapa kali |
| `Transmission` | **TARGET** — Manual atau Automatic |

> ⚠️ Kolom `Transmission` adalah yang ingin kita prediksi (disebut *target* atau *label*).

---

## 🔍 BAGIAN 3 — Exploratory Data Analysis (EDA)

EDA adalah proses **mengenal data lebih dalam** sebelum melatih model. Ibarat kita membaca buku dulu sebelum ujian.

Program membuat beberapa grafik:

### 3.1 Distribusi Transmisi
```python
sns.countplot(x='Transmission', data=train, palette='RdBu_r')
```
Menghitung berapa banyak mobil Manual vs Automatic dalam dataset.

### 3.2 Transmisi vs Jenis Bahan Bakar
```python
sns.countplot(x='Transmission', hue='Fuel_Type', data=train, palette='RdBu_r')
```
Melihat apakah jenis bahan bakar memengaruhi jenis transmisi.

### 3.3 Transmisi vs Kepemilikan
```python
sns.countplot(x='Transmission', hue='Owner', data=train, palette='rainbow')
```
Apakah mobil yang sering berpindah tangan lebih banyak Manual atau Automatic?

### 3.4 Distribusi Jarak Tempuh
```python
train['Kms_Driven'].hist(bins=30, alpha=0.7)
```
Histogram yang menunjukkan sebaran jarak tempuh seluruh mobil.

### 3.5 Boxplot Jarak Tempuh vs Transmisi & Bahan Bakar
Visualisasi untuk melihat rata-rata dan sebaran jarak tempuh berdasarkan kelompok.

---

## 🛠️ BAGIAN 4 — Feature Engineering (Menyiapkan Data)

Sebelum data bisa dimasukkan ke model, data perlu "dibersihkan" dan "diformat".

### 4.1 Mengisi Nilai yang Hilang (Missing Value)

```python
km_mean = train.groupby('Fuel_Type')['Kms_Driven'].mean().to_dict()

def impute_km(cols):
    Kms = cols['Kms_Driven']
    Fuel = cols['Fuel_Type']
    if pd.isnull(Kms):
        return km_mean[Fuel]
    else:
        return Kms

train['Kms_Driven'] = train[['Kms_Driven', 'Fuel_Type']].apply(impute_km, axis=1)
```

**Penjelasan:**
Beberapa baris data mungkin tidak memiliki nilai `Kms_Driven`. Daripada dihapus, nilainya **diisi otomatis** menggunakan rata-rata jarak tempuh berdasarkan jenis bahan bakar yang sama. Ini lebih cerdas daripada mengisi semua dengan rata-rata global.

### 4.2 Menghapus Baris yang Masih Kosong

```python
if train.isnull().sum().sum() > 0:
    train = train.dropna()
else:
    print("Tidak ada missing value, tidak perlu drop data")
```

Setelah pengisian tadi, jika masih ada baris kosong yang tidak bisa diisi, baris tersebut dihapus.

### 4.3 Menghapus Kolom Tidak Relevan

```python
train.drop('Car_Name', axis=1, inplace=True)
```

Kolom `Car_Name` (nama mobil) tidak digunakan karena terlalu banyak variasinya dan tidak membantu model belajar pola.

### 4.4 Mengubah Teks Menjadi Angka (Label Encoding)

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

train['Fuel_Type'] = le.fit_transform(train['Fuel_Type'])
train['Seller_Type'] = le.fit_transform(train['Seller_Type'])
train['Transmission'] = le.fit_transform(train['Transmission'])
```

**Penjelasan:**
Komputer hanya bisa bekerja dengan angka. Kolom bertipe teks (kategori) diubah jadi angka:
- `Fuel_Type`: Diesel → 0, Petrol → 1, CNG → 2 (contoh)
- `Seller_Type`: Dealer → 0, Individual → 1 (contoh)
- `Transmission`: Manual → 0, Automatic → 1 (contoh)

---

## ✂️ BAGIAN 5 — Membagi Data: Training & Testing

```python
from sklearn.model_selection import train_test_split

X = train.drop('Transmission', axis=1)   # Fitur (input)
y = train['Transmission']                 # Target (output)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)
```

**Penjelasan:**
Data dibagi menjadi dua bagian:
- **70% → Data Training** — Digunakan untuk melatih model (belajar)
- **30% → Data Testing** — Digunakan untuk menguji seberapa akurat model (ujian)

Ini penting agar kita tahu apakah model benar-benar belajar atau sekadar "menghafal".

---

## 🤖 BAGIAN 6 — Tiga Algoritma Machine Learning

### 🔵 6.1 KNN (K-Nearest Neighbors)

```python
from sklearn.neighbors import KNeighborsClassifier

clf1 = KNeighborsClassifier()
clf1.fit(X_train, y_train)
y_pred = clf1.predict(X_test)
```

**Cara kerja (analogi):**
Bayangkan kamu ingin tahu apakah seorang mahasiswa baru akan lulus atau tidak. KNN melihat "tetangga terdekat" — yaitu mahasiswa-mahasiswa sebelumnya yang paling mirip karakternya. Jika mayoritas tetangga terdekat lulus, maka mahasiswa baru itu diprediksi akan lulus juga.

Dalam kasus ini: model melihat mobil-mobil di data training yang paling mirip fiturnya, lalu menyimpulkan transmisinya.

---

### 🟢 6.2 Naive Bayes

```python
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
```

**Cara kerja (analogi):**
Naive Bayes bekerja berdasarkan **probabilitas**. Seperti dokter yang mendiagnosis penyakit: "Pasien ini demam + batuk + pilek → kemungkinan 80% flu biasa". Model menghitung peluang setiap kelas berdasarkan fitur yang diberikan, lalu memilih kelas dengan peluang tertinggi.

Disebut "Naive" (polos) karena ia mengasumsikan setiap fitur **bebas satu sama lain**, padahal di dunia nyata hal itu jarang terjadi.

---

### 🔴 6.3 Decision Tree (Pohon Keputusan)

```python
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
```

**Cara kerja (analogi):**
Seperti sebuah flowchart atau kuis tebak-tebakan:
```
Apakah harga jual > 5 Lakh?
├── YA → Apakah tahun > 2015?
│         ├── YA → Kemungkinan AUTOMATIC
│         └── TIDAK → Kemungkinan MANUAL
└── TIDAK → Kemungkinan MANUAL
```
Model membuat serangkaian pertanyaan/aturan berdasarkan data training untuk mengklasifikasikan data baru.

---

## 📏 BAGIAN 7 — Evaluasi Model

Setelah model dilatih, kita ukur performanya menggunakan beberapa metrik:

```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

accuracy = accuracy_score(y_test, y_pred)
print(f'Akurasi Model KNN: {accuracy:.2f}')
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

### Penjelasan Metrik:

**1. Akurasi (Accuracy)**
Persentase prediksi yang benar dari keseluruhan prediksi.
```
Akurasi = Prediksi Benar / Total Data × 100%
```

**2. Confusion Matrix**
Tabel 2×2 yang menunjukkan detail prediksi benar dan salah:

|  | Prediksi Manual | Prediksi Automatic |
|---|---|---|
| **Aktual Manual** | ✅ True Negative (TN) | ❌ False Positive (FP) |
| **Aktual Automatic** | ❌ False Negative (FN) | ✅ True Positive (TP) |

**3. Classification Report**
Laporan lengkap per kelas yang mencakup:
- **Precision** — Dari semua yang diprediksi kelas X, berapa persen yang benar?
- **Recall** — Dari semua data kelas X, berapa persen yang berhasil terdeteksi?
- **F1-Score** — Rata-rata harmonis dari Precision dan Recall (semakin tinggi semakin baik)

---

### 🏆 Hasil Perbandingan Model

Berdasarkan notebook, **Naive Bayes dipilih sebagai model terbaik** karena memiliki nilai evaluasi tertinggi di antara ketiga model yang diuji.

---

## 🔮 BAGIAN 8 — Prediksi Data Mobil Baru

```python
new_data = np.array([[2015, 4.5, 6.8, 35000, 1, 0, 0]])
# Format: [Year, Selling_Price, Present_Price, Kms_Driven, Fuel_Type, Seller_Type, Owner]

new_prediction = nb.predict(new_data)
print("Prediksi Transmission (0 = Manual, 1 = Automatic):", new_prediction[0])

proba = nb.predict_proba(new_data)
print("Probabilitas [Manual, Automatic]:", proba[0])
```

**Penjelasan:**
Setelah model dilatih, kita bisa memprediksi jenis transmisi sebuah mobil baru yang datanya kita masukkan manual. Model Naive Bayes juga bisa memberikan nilai **probabilitas** (misalnya: 75% Manual, 25% Automatic), bukan hanya jawaban ya/tidak.

---

## ⚙️ BAGIAN 9 — Optimasi KNN dengan GridSearchCV

Ini adalah bagian paling canggih dalam notebook ini.

### Apa itu GridSearchCV?

Bayangkan kamu ingin mencari resep kopi terbaik. Kamu mencoba semua kombinasi:
- Gula: sedikit / sedang / banyak
- Susu: ada / tidak ada
- Es: ada / tidak ada

GridSearchCV melakukan hal yang sama untuk parameter machine learning — mencoba **semua kombinasi parameter** secara otomatis, lalu memilih yang terbaik.

### Parameter yang Diuji untuk KNN:

```python
param_grid = {
    'n_neighbors': [3, 5, 7],       # Berapa "tetangga" yang dilihat
    'weights': ['uniform', 'distance'],  # Cara menghitung bobot tetangga
    'metric': ['euclidean', 'manhattan'] # Cara mengukur jarak antar data
}
```

| Parameter | Pilihan | Penjelasan |
|---|---|---|
| `n_neighbors` | 3, 5, 7 | Jumlah tetangga terdekat yang dipertimbangkan |
| `weights='uniform'` | — | Semua tetangga dihitung sama |
| `weights='distance'` | — | Tetangga yang lebih dekat diberi bobot lebih besar |
| `metric='euclidean'` | — | Jarak garis lurus (seperti Pythagoras) |
| `metric='manhattan'` | — | Jarak grid (seperti jalan kota kotak-kotak) |

Total: **3 × 2 × 2 = 12 kombinasi** yang dicoba.

### Cross-Validation (CV=5):

```python
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring=scoring, refit='f1')
```

Data training dibagi menjadi 5 bagian. Setiap kombinasi parameter diuji 5 kali, dengan bergantian bagian mana yang jadi data uji. Ini mencegah hasil yang kebetulan bagus hanya karena pemilihan data yang beruntung.

---

## 📋 BAGIAN 10 — Hasil GridSearchCV (File Excel)

File `hasil_gridsearch_knn.xlsx` berisi hasil dari 12 kombinasi parameter KNN yang dicoba. Berikut rangkumannya:

| Rank | n_neighbors | weights | metric | Accuracy | F1-Score |
|---|---|---|---|---|---|
| 🥇 1 | 3 | uniform | euclidean | 89.05% | 0.5677 |
| 🥇 1 | 3 | uniform | manhattan | 89.05% | 0.5677 |
| 🥉 3 | 5 | distance | euclidean | 85.24% | 0.5377 |
| 4 | 5 | distance | manhattan | 84.76% | 0.5335 |
| ... | ... | ... | ... | ... | ... |
| 12 | 7 | uniform | euclidean | 89.05% | 0.4994 |

**Kesimpulan dari GridSearch:**
- Parameter terbaik: **n_neighbors=3, weights=uniform, metric=euclidean atau manhattan**
- Akurasi tertinggi: **89.05%**
- F1-Score terbaik: **0.5677**

> **Catatan Menarik:** Beberapa kombinasi memiliki akurasi yang sama (89.05%), tetapi F1-Score berbeda. Ini menunjukkan bahwa akurasi saja tidak cukup — terutama jika data tidak seimbang (lebih banyak Manual daripada Automatic atau sebaliknya).

---

## 📌 Kesimpulan Keseluruhan

| Aspek | Detail |
|---|---|
| **Masalah** | Klasifikasi jenis transmisi mobil (Manual/Automatic) |
| **Dataset** | Data penjualan mobil bekas (cardata.csv) |
| **Algoritma yang Diuji** | KNN, Naive Bayes, Decision Tree |
| **Model Terbaik (awal)** | Naive Bayes (evaluasi terbaik) |
| **Optimasi** | GridSearchCV pada KNN → parameter terbaik: k=3, uniform, euclidean |
| **Akurasi KNN Optimal** | ~89% |

---

## 💡 Konsep Penting yang Dipelajari

1. **Supervised Learning** — Belajar dari data berlabel untuk memprediksi data baru
2. **EDA** — Memahami data sebelum modeling
3. **Feature Engineering** — Membersihkan dan menyiapkan data
4. **Train-Test Split** — Memisahkan data latih dan uji
5. **KNN** — Klasifikasi berdasarkan kedekatan/kemiripan data
6. **Naive Bayes** — Klasifikasi berbasis probabilitas
7. **Decision Tree** — Klasifikasi berbasis aturan if-else bertingkat
8. **Confusion Matrix & Classification Report** — Evaluasi model secara menyeluruh
9. **GridSearchCV** — Pencarian otomatis parameter model terbaik
10. **Cross-Validation** — Teknik validasi yang lebih robust dan terpercaya

---

## 🔗 Library yang Dibutuhkan

```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
```

---

*README ini dibuat berdasarkan analisis notebook `Week_5_6_7_Supervised_Learning_Hands_On_Classification_All_tanpa_tuning.ipynb` dan file hasil `hasil_gridsearch_knn.xlsx`.*
