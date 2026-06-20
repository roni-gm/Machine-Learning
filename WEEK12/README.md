# Week 9 - CNN Image Classification: Trash Type
Tugas praktikum klasifikasi gambar menggunakan Convolutional Neural Network (CNN) pada dataset Trash Type Image Dataset. Tujuannya adalah mengklasifikasikan gambar sampah ke dalam beberapa kategori jenis sampah menggunakan model CNN yang dibangun dengan TensorFlow/Keras.

**Nama:** Roni Gunawan Muhammad  
**NIM:** 4222311024  
**Kelas:** Robotika A Malam

---

## Dataset
[`trash-type-image-dataset`](https://www.kaggle.com/datasets/farzadnekouei/trash-type-image-dataset) (diunduh otomatis via `kagglehub`) — dataset gambar sampah dengan 6 kategori kelas:
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
- Struktur folder dataset diperiksa dengan `os.walk()` untuk memastikan path kelas yang benar
- Gambar dimuat menggunakan `keras.utils.image_dataset_from_directory()` dengan parameter:
  - `IMG_SIZE = (128, 128)`
  - `BATCH_SIZE = 32`
  - `validation_split = 0.2` (80% train, 20% validation)
  - `SEED = 42`

### 2. Eksplorasi & Preprocessing
- Visualisasi contoh gambar dari `train_dataset` untuk masing-masing label
- Normalisasi piksel gambar (`Rescaling(1./255)`)
- Optimasi pipeline data dengan `.cache()`, `.shuffle()`, dan `.prefetch()` agar training lebih efisien

### 3. Membangun Model CNN
Arsitektur model `Sequential`:
| Layer | Detail |
|---|---|
| Rescaling | Normalisasi input |
| Conv2D + MaxPooling2D | 32 filter |
| Conv2D + MaxPooling2D | 64 filter |
| Conv2D + MaxPooling2D | 64 filter |
| Flatten | — |
| Dense | 64 unit, ReLU |
| Dropout | 0.3 |
| Dense (output) | softmax, sejumlah kelas |

Model dikompilasi dengan optimizer `adam`, loss `sparse_categorical_crossentropy`, dan metrik `accuracy`.

### 4. Training
Model dilatih selama **15 epoch** menggunakan `train_dataset` dan divalidasi dengan `val_dataset`.

### 5. Evaluasi
- Plot kurva akurasi dan loss (training vs validation) per epoch
- Evaluasi akhir pada `val_dataset` (loss & accuracy)
- `classification_report` dan `confusion_matrix` untuk melihat performa per kelas
- Visualisasi prediksi vs label asli pada sampel gambar validasi

### 6. Simpan Model
Model akhir disimpan ke file `trash_classifier_model.h5` / `.keras`.

---

## Catatan Penting
⚠️ Pada eksekusi notebook ini, log training menunjukkan `accuracy: 0.0000e+00` dan `loss: 0.0000e+00` di semua epoch, serta saat memuat dataset tertulis **"Found 2527 files belonging to 1 classes"**. Ini menandakan `DATASET_DIR` mengarah ke folder induk (`.../versions/1`) bukan ke folder yang langsung berisi subfolder kelas (`.../versions/1/TrashType_Image_Dataset`), sehingga seluruh gambar terbaca sebagai satu kelas saja. Perbaikannya:

```python
DATASET_DIR = os.path.join(path, "TrashType_Image_Dataset")
```

Gunakan path ini sebelum memanggil `image_dataset_from_directory()` agar model membaca ke-6 subfolder kelas dengan benar dan training menghasilkan accuracy/loss yang valid.

---

## Library
```
tensorflow, keras, kagglehub, scikit-learn, seaborn, matplotlib, numpy
```
