# LASA Pill Classifier

Klasifikasi gambar obat **LASA** (*Look-Alike Sound-Alike*) ke dalam dua kategori:
**NSAID (Pereda Nyeri)** vs **Antibiotik**, menggunakan *transfer learning* berbasis CNN.

Sistem hanya untuk keperluan penelitian/akademik (UAS Machine Learning)

---

## Penjelasan Singkat Proyek

Obat LASA adalah obat yang memiliki rupa/kemasan mirip sehingga rawan tertukar.
Proyek ini membangun model *computer vision* untuk membedakan dua golongan obat secara otomatis
dari foto sediaan maupun kemasannya. Selain itu, konsumsi obat LASA terutaman antibiotik dan pereda nyeri harus memiliki jeda waktu yang tepat agar tidak menyebabkan efek samping berbahaya pada tubuh.

**Dataset** terdiri dari 4 folder (2 kelas):

| Folder | Label | Kelas |
|---|---|---|
| `pereda_nyeri/` | 0 | NSAID (Pereda Nyeri) |
| `pereda_nyeri_bungkus/` | 0 | NSAID (Pereda Nyeri) |
| `antibiotik/` | 1 | Antibiotik |
| `antibiotik_Bungkus/` | 1 | Antibiotik |

Data dibagi **60% train / 20% validation / 20% test** (stratified). Karena jumlah data
terbatas dan tidak seimbang, digunakan *class weighting*, *weighted sampling*, augmentasi
(Albumentations), serta *Test-Time Augmentation* (TTA).

**Metodologi (notebook `UAS_ML_1.ipynb`):**
1. **Data pipeline** (load gambar, augmentasi, split, penanganan imbalance).
2. **Baseline** EfficientNet-B0 (*frozen backbone* + classifier head).
3. **Cell A Komparasi arsitektur**: EfficientNet-B0, MobileNetV3-Small, ResNet50,
   EfficientNet-B1, plus baseline klasik (CNN-Feature + SMOTE + Random Forest).
4. **Cell B Hyperparameter tuning** dengan Optuna (TPE sampler + Median pruner).
5. **Cell C Evaluasi lengkap**: accuracy, precision, recall, F1, confusion matrix, training curves.
6. **Cell D Interpretasi model** dengan Grad-CAM.
7. **Cell E Simpan model terbaik** (checkpoint + state_dict).

Model hasil pelatihan kemudian di-*deploy* sebagai aplikasi web sederhana (Streamlit) di folder `demo-run/`.

---

## Struktur Folder

```
UAS_ML1/
├── UAS_ML_1.ipynb            # Notebook pipeline lengkap (training & evaluasi)
├── antibiotik/               # Dataset: foto antibiotik (sediaan)
├── antibiotik_Bungkus/       # Dataset: foto antibiotik (kemasan)
├── pereda_nyeri/             # Dataset: foto NSAID (sediaan)
├── pereda_nyeri_bungkus/     # Dataset: foto NSAID (kemasan)
└── demo-run/                 # Aplikasi demo
    ├── app.py                        # Streamlit app
    ├── MobileNetV3_S_state_dict.pth  # Bobot model siap inferensi
    ├── req.txt                       # Dependensi demo
    └── pereda-nyeri.jpg              # Contoh gambar uji
```

---

## Cara Menjalankan Program

### A. Menjalankan Demo (Streamlit)

```bash
cd demo-run

# (opsional) buat virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# install dependensi
pip install -r req.txt

# jalankan aplikasi
streamlit run app.py
```

Aplikasi akan terbuka di browser (`http://localhost:8501`).
Upload foto obat (JPG/PNG), lalu aplikasi menampilkan **prediksi kelas** beserta
**confidence** dan probabilitas tiap kelas.

### B. Menjalankan Notebook Pelatihan

Notebook `UAS_ML_1.ipynb` dirancang untuk **Google Colab** (mount Google Drive):

1. Upload `UAS_ML_1.ipynb` ke Google Colab.
2. Letakkan dataset di `MyDrive/DatasetLASA/` dengan 4 subfolder kelas di atas.
3. Aktifkan runtime **GPU** (Runtime → Change runtime type → GPU).
4. Jalankan sel berurutan. Dependensi diinstal otomatis:
   ```bash
   pip install torch torchvision albumentations imbalanced-learn optuna --quiet
   ```
5. Model terbaik akan tersimpan di `MyDrive/DatasetLASA/saved_model/`.

---

## Hasil Utama Model

Perbandingan akurasi seluruh model yang diuji (pada data uji):

| Model | Akurasi |
|---|---|
| Random Forest (klasik, terbaik) | 70.37% |
| KNN (klasik) | 59.26% |
| SVM (klasik) | 70.37% |
| **EfficientNet-B0 (baseline)** | **80.00%** ⭐ |
| MobileNetV3-Small | 45.00% |
| ResNet50 | 50.00% |
| EfficientNet-B1 | 60.00% |
| EfficientNet-B0 (Tuned/Optuna) | 60.00% |

**Kesimpulan utama:**
- **EfficientNet-B0** memberikan akurasi model tertinggi (**80%**), mengungguli arsitektur lain
  maupun model machine learning tradisional (RF/KNN/SVM).
- Model *deep learning* dengan *transfer learning* secara umum lebih unggul dibanding
  pendekatan klasik untuk tugas klasifikasi citra obat ini.
- **Grad-CAM** menunjukkan model berfokus pada area obat/kemasan yang relevan saat
  mengambil keputusan, sehingga prediksi cukup dapat diinterpretasi.
- Untuk *deployment*, dipilih **MobileNetV3-Small** karena ringan dan cepat untuk inferensi
  (lihat `demo-run/`), cocok untuk aplikasi web real-time meskipun akurasinya lebih rendah
  dibanding EfficientNet-B0.

> Catatan: akurasi terbatas oleh ukuran dataset yang kecil. Penambahan data dan
> *fine-tuning backbone* berpotensi meningkatkan performa lebih lanjut.

---

## 🛠️ Teknologi

- Python, PyTorch, TorchVision
- Albumentations (augmentasi), imbalanced-learn (SMOTE), scikit-learn
- Optuna (hyperparameter tuning)
- Streamlit (demo), OpenCV, Pillow, NumPy, Matplotlib
