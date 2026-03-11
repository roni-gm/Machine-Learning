# Machine Learning Project – Week 3
## Data Exploration & Feature Engineering

Project ini berisi proses **eksplorasi data dan feature engineering** yang dilakukan pada beberapa dataset sebelum digunakan pada tahap **Machine Learning Modeling**.

Fokus utama project ini adalah melakukan **data preprocessing** agar dataset lebih bersih, terstruktur, dan siap digunakan untuk proses analisis maupun pembuatan model.

---

# Project Overview

Pada project Week 3 ini dilakukan beberapa tahapan penting dalam pengolahan data, yaitu:

- Eksplorasi dataset
- Visualisasi distribusi data
- Deteksi dan analisis outlier
- Penanganan missing value
- Encoding data kategorikal
- Persiapan dataset sebelum modeling

Seluruh proses dilakukan menggunakan **Python** dan **Jupyter Notebook**.

---

# Dataset yang Digunakan

Project ini menggunakan beberapa dataset untuk melakukan berbagai teknik preprocessing.

## 1. California Housing Dataset

Digunakan untuk melakukan **analisis distribusi data dan identifikasi outlier**.

**File:**
california_dataset.csv

**Kolom yang dianalisis:**

- MedInc
- HouseAge
- AveRooms
- AveBedrms
- AveOccup

**Langkah yang dilakukan:**

- Visualisasi distribusi menggunakan **histogram**
- Deteksi **outlier menggunakan boxplot**
- Analisis karakteristik data

---

## 2. Company Dataset
Dataset ini digunakan untuk melakukan **analisis dan penanganan missing value**.
**File:**
company.csv

**Langkah yang dilakukan:**

- Menghitung jumlah missing value
- Menghitung persentase missing value
- Menentukan apakah kolom perlu dihapus atau tidak
- Mengisi nilai kosong menggunakan metode yang sesuai

---

## 3. Telco Customer Churn Dataset

Dataset ini digunakan untuk melakukan **encoding data kategorikal** agar dapat digunakan dalam proses machine learning.
**File:**
TelcoCustomerChurn.csv
TelcoCustomerChurn.xlsx
**Langkah yang dilakukan:**

- Mengubah nilai kategori menjadi nilai numerik
- Membersihkan nilai kategori yang tidak konsisten
- Menggunakan beberapa teknik encoding

**Contoh transformasi data:**


Yes → 1
No → 0


**Beberapa kolom yang diproses:**

- StreamingMovies
- StreamingTV
- TechSupport
- DeviceProtection
- OnlineBackup
- OnlineSecurity
- MultipleLines

---

# Proses Data Preparation

Beberapa teknik preprocessing yang digunakan dalam project ini antara lain:

## 1. Data Splitting

Dataset dibagi menjadi dua bagian:

- **Training Data**
- **Testing Data**

Dengan rasio umum:


80% Training Data
20% Testing Data


Hal ini dilakukan agar model nantinya dapat diuji menggunakan **data yang belum pernah dilihat sebelumnya**.

---

## 2. Data Cleaning

Beberapa proses pembersihan data yang dilakukan:

- Mengganti nilai kategori yang tidak konsisten  

Contoh:


"No internet service" → "No"


- Menghapus atau memperbaiki nilai yang tidak valid

---

## 3. Feature Encoding

Beberapa teknik encoding yang digunakan:

### One Hot Encoding
Digunakan untuk kolom kategorikal tanpa urutan.

### Label Encoding
Digunakan untuk nilai kategori sederhana seperti:
Yes / No

yang kemudian diubah menjadi:
Yes = 1
No = 0


---

# Tools yang Digunakan

Project ini dibuat menggunakan beberapa tools berikut:

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

# Struktur Repository

Machine-Learning/
│
└── WEEK3
├── Assignment_Week 3.ipynb
├── Hands On_Week_3_Feature_Engineering.ipynb
├── california_dataset.csv
├── company.csv
├── TelcoCustomerChurn.csv
├── TelcoCustomerChurn.xlsx
├── titanic.xlsx
└── EDA with Python.pdf


---

# Project Result

Melalui project ini dataset berhasil diproses melalui beberapa tahapan preprocessing sehingga:

- Data menjadi lebih bersih
- Nilai kategorikal berhasil dikonversi menjadi numerik
- Outlier dapat teridentifikasi
- Missing value dapat ditangani

Dataset yang telah diproses ini siap digunakan untuk tahap **Machine Learning Modeling** pada tahap selanjutnya.

---

# Author

Project ini dibuat sebagai bagian dari latihan praktikum **Machine Learning menggunakan Python**.

