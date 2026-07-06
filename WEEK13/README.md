# Week 10 - Handling Overfitting & Underfitting with Transfer Learning
Tugas praktikum mengenai identifikasi dan penanganan **overfitting** serta **underfitting** pada model klasifikasi gambar menggunakan metode **Transfer Learning** dengan arsitektur **MobileNetV2**. Tujuannya adalah membandingkan performa model sebelum dan sesudah dilakukan teknik penanganan overfitting seperti data augmentation, regularisasi, dropout, early stopping, dan fine-tuning.

**Nama:** Roni Gunawan Muhammad  
**NIM:** 4222311024  
**Kelas:** Robotika A Malam

---

## Dataset
[`trash-type-image-dataset`](https://www.kaggle.com/datasets/farzadnekouei/trash-type-image-dataset) (diunduh otomatis menggunakan `kagglehub`) — dataset klasifikasi gambar sampah yang terdiri dari 6 kategori:
- `cardboard`
- `glass`
- `metal`
- `paper`
- `plastic`
- `trash`

---

## Alur Kerja

### 1. Persiapan Dataset
- Dataset diunduh otomatis menggunakan `kagglehub.dataset_download()`
- Struktur folder dataset diperiksa untuk memastikan lokasi dataset benar.
- Dataset dibagi menjadi data **training** dan **validation** menggunakan `ImageDataGenerator`.
- Ukuran gambar diubah menjadi **224 × 224 piksel** agar sesuai dengan input MobileNetV2.

### 2. Eksplorasi Dataset
- Menampilkan struktur folder dataset.
- Memastikan setiap kelas berhasil terbaca.
- Menghitung jumlah data pada masing-masing kelas.

### 3. Membangun Model Baseline
Model menggunakan **Transfer Learning MobileNetV2** dengan bobot pretrained ImageNet.

Arsitektur model:

| Layer | Detail |
|---|---|
| MobileNetV2 | Pre-trained ImageNet (feature extractor) |
| GlobalAveragePooling2D | Mengubah feature map menjadi vector |
| Dense | 128 unit, ReLU |
| Dense Output | Softmax sebanyak jumlah kelas |

Model dikompilasi menggunakan:
- Optimizer : `Adam`
- Loss : `categorical_crossentropy`
- Metric : `accuracy`

### 4. Training Model Baseline
- Model dilatih menggunakan data training.
- Data validasi digunakan untuk memantau performa model.
- Kurva **Accuracy** dan **Loss** diamati untuk mendeteksi gejala overfitting maupun underfitting.

### 5. Evaluasi Model
Evaluasi dilakukan menggunakan:
- Accuracy Training
- Accuracy Validation
- Loss Training
- Loss Validation
- Grafik Accuracy
- Grafik Loss

---

## Handling Overfitting

Beberapa teknik diterapkan untuk meningkatkan kemampuan generalisasi model.

### 1. Data Augmentation
Menggunakan `ImageDataGenerator` dengan:
- Rotation
- Width Shift
- Height Shift
- Zoom
- Horizontal Flip
- Shear

Tujuannya agar model memperoleh variasi data yang lebih banyak sehingga mengurangi overfitting.

### 2. Dropout & L2 Regularization
Model diperbaiki dengan menambahkan:
- Layer Dropout
- L2 Regularization pada Dense Layer

Teknik ini membantu mengurangi kompleksitas model dan mencegah model menghafal data training.

### 3. Early Stopping
Menggunakan callback **EarlyStopping** untuk:
- Menghentikan proses training saat validation loss tidak lagi membaik.
- Mengembalikan bobot terbaik selama proses training.

---

## Fine-Tuning Model

Setelah model stabil, beberapa layer pada MobileNetV2 dibuka kembali (**unfreeze**) kemudian dilakukan proses **Fine-Tuning** dengan learning rate yang lebih kecil.

Tujuannya adalah:
- Menyesuaikan feature extractor terhadap dataset.
- Meningkatkan akurasi klasifikasi.
- Memperoleh performa model yang lebih optimal.

---

## Evaluasi Akhir

Model akhir dievaluasi menggunakan:
- Accuracy
- Loss
- Grafik perbandingan training dan validation
- Perbandingan performa sebelum dan sesudah handling overfitting

Hasil evaluasi menunjukkan bahwa penggunaan **Data Augmentation**, **Dropout**, **L2 Regularization**, **Early Stopping**, dan **Fine-Tuning** mampu meningkatkan kemampuan generalisasi model dibandingkan model baseline.

---

## Library

```
tensorflow
keras
kagglehub
numpy
matplotlib
scikit-learn
```