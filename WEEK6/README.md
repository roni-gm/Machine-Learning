# Machine Learning - Classification (Minggu 5, 6, 7)

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-orange)
![Algorithms](https://img.shields.io/badge/Algorithms-Classification-yellow)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

*Implementasi praktis supervised learning untuk algoritma klasifikasi seperti Naive Bayes dan K-Nearest Neighbors, menggunakan dataset nyata seperti Titanic.*

</div>

---

## 📚 Daftar Isi

* [Tentang Proyek](#tentang-proyek)
* [Materi yang Dipelajari](#materi-yang-dipelajari)
* [Struktur Repository](#struktur-repository)
* [Cara Menjalankan](#cara-menjalankan)
* [Penjelasan Algoritma](#penjelasan-algoritma)
* [Hasil & Analisis](#hasil--analisis)
* [Referensi](#referensi)

---

## 📌 Tentang Proyek

Repository ini berisi implementasi dan materi pembelajaran untuk **Supervised Learning - Classification**, dengan fokus pada algoritma dasar dalam machine learning.

Proyek ini bertujuan untuk memahami bagaimana model dapat belajar dari data berlabel dan melakukan prediksi terhadap data baru.

Salah satu studi kasus utama yang digunakan adalah **dataset Titanic**, untuk memprediksi apakah seorang penumpang selamat atau tidak berdasarkan fitur tertentu.

**Dikembangkan untuk:** Mata Kuliah Machine Learning
**Fokus:** Algoritma Klasifikasi (Naive Bayes, KNN)
**Pendekatan:** Teori + Praktik Langsung

---

## 🧠 Materi yang Dipelajari

* Pengenalan Classification
* Konsep Supervised Learning
* Algoritma Naive Bayes
* K-Nearest Neighbors (KNN)
* Data Preprocessing & Cleaning
* Penanganan Missing Value (Imputasi)
* Evaluasi Model
* Hyperparameter Tuning (Grid Search)

---

## 📁 Struktur Repository

```bash
WEEK6/
│
├── README.md
│
├── Week 5 - 7 - Introduction to classification.pptx
│   ← Slide materi teori klasifikasi
│
├── Week_5_6_7_Supervised_Learning_Hands_On_Classification_All_tanpa_tuning.ipynb
│   ← Notebook utama (implementasi & eksperimen)
│
├── titanic.csv
│   ← Dataset untuk klasifikasi
│
├── cara kerja perhitungan naive bayes.xlsx
│   ← Perhitungan manual Naive Bayes
│
└── hasil_gridsearch_knn.xlsx
    ← Hasil tuning hyperparameter KNN
```

---

## 🚀 Cara Menjalankan

### 1. Persiapan

Pastikan Python sudah terinstall (disarankan versi 3.8 ke atas), lalu install library:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### 2. Menjalankan Project

Buka Jupyter Notebook:

```bash
jupyter notebook
```

Kemudian jalankan file:

```
Week_5_6_7_Supervised_Learning_Hands_On_Classification_All_tanpa_tuning.ipynb
```

---

## ⚙️ Penjelasan Algoritma

### 1. Naive Bayes

* Berdasarkan Teorema Bayes
* Mengasumsikan setiap fitur saling independen
* Cepat dan efisien untuk klasifikasi

Digunakan untuk:

* Prediksi berbasis probabilitas
* Memahami konsep klasifikasi statistik

---

### 2. K-Nearest Neighbors (KNN)

* Algoritma berbasis instance
* Mengklasifikasikan berdasarkan tetangga terdekat
* Sensitif terhadap nilai K

Digunakan untuk:

* Pengenalan pola (pattern recognition)
* Klasifikasi sederhana namun efektif

---

## 📊 Hasil & Analisis

### Dataset yang Digunakan

* Titanic Dataset

### Tahapan Utama:

* Data cleaning (menangani missing value)
* Seleksi fitur
* Training model
* Evaluasi model

### Insight:

* Missing value pada fitur **Age** ditangani dengan imputasi berdasarkan kelas penumpang
* KNN dioptimasi menggunakan **Grid Search**
* Dilakukan perbandingan performa antar model

---

## 📚 Referensi

* Pattern Recognition and Machine Learning
* Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow
* Scikit-learn Documentation — https://scikit-learn.org/
* Kaggle — Titanic Dataset

---

## 👨‍💻 Author

**Roni Gunawan Muhammad**

---

<div align="center"> 
<i>Machine Learning · Classification Study · Politeknik Negeri Batam</i> 
</div>
