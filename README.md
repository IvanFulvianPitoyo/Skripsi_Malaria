# Sistem Deteksi Malaria Berbasis YOLOv8
**Oleh: Ivan Fulvian Pitoyo (122430064)**
**Program Studi Teknik Biomedis - Institut Teknologi Sumatera**

Repositori ini berisi keseluruhan *source code*, dataset sampel, hasil pelatihan, dan model Deep Learning (YOLOv8m) untuk sistem deteksi parasit malaria (*Plasmodium falciparum* dan *Plasmodium vivax*) pada citra mikroskopis sediaan darah tepi.

File ini adalah panduan singkat untuk memahami struktur direktori, keputusan metodologis arsitektur, dan cara menjalankan model ini untuk keperluan pengujian (testing/inference) di perangkat lain.

---

## 1. Struktur Direktori dan File

Berikut adalah penjelasan mengenai isi dari repositori proyek ini yang telah dirapikan sesuai standar:

**Folder Utama**
* `dataset_sample/` : Berisi beberapa citra sediaan darah mikroskopis contoh untuk keperluan uji coba (*testing*).
* `db/` : Berisi database studi Optuna (`malaria_yolo_study.db`) untuk perekaman hasil pencarian *hyperparameter* secara persisten.
* `models/` : Folder khusus yang menyimpan file bobot (*weights*) model (`.pt`), termasuk model *pre-trained* (YOLOv8m) dan model final terbaik hasil penelitian (`yolov8m_malaria.pt`).
* `runs/` : Folder default bawaan library Ultralytics YOLOv8 untuk menyimpan riwayat hasil training, grafik evaluasi, matriks konfusi, dan inferensi deteksi.
* `src/` : Folder utama kode sumber (*source code*) yang berisi skrip Python, terbagi menjadi:
    * `1_preprocessing/` : Skrip untuk pembagian dataset 80:20 (Hold-Out) dan augmentasi citra (Noise, Brightness, dan Flip).
    * `2_utilities/` : Skrip alat bantu tambahan untuk mempermudah alur kerja evaluasi dan pembersihan data.
    * `3_training/` : Skrip utama untuk pelatihan model dan optimasi *hyperparameter* menggunakan Optuna.

**File Konfigurasi & Skrip Utama di Root Directory**
* `requirements.txt` : Daftar seluruh pustaka Python (*libraries*) beserta versinya yang dibutuhkan untuk menjalankan sistem ini.
* `.gitignore` : Konfigurasi untuk mengabaikan file sistem, bobot raksasa, dan virtual environment agar tidak terunggah ke *repository*.
* `uji_model.py` : Skrip untuk menjalankan pengujian deteksi secara interaktif (Dialog File).
* `GUI.py` : Skrip untuk menjalankan Antarmuka Pengguna Grafis (GUI) lengkap dengan fitur kamera, *zoom*, rotasi, dan *export*.

---

## 2. Catatan Konfigurasi Training & Evaluasi

Pembimbing yang terhormat, berikut adalah catatan penting terkait metodologi dan konfigurasi akhir dari pelatihan model pada sistem ini:

1. **Penggunaan YOLOv8 Medium:** Model dilatih menggunakan arsitektur `yolov8m.pt` (Medium) alih-alih tipe Nano, guna memaksimalkan performa ekstraksi fitur dan pelokalan parasit yang berukuran sangat mikro.
2. **Pemilihan Hyperparameter:** Pencarian *hyperparameter* terbaik dilakukan menggunakan framework Optuna. Resep parameter yang digunakan untuk model final diambil dari **Trial 36**, karena trial ini terbukti secara empiris menghasilkan evaluasi mAP50 tertinggi.
3. **Penanganan Overfitting & Pemilihan Bobot (Weights):** Pada proses pelatihan final (100 Epochs), observasi kurva evaluasi menunjukkan bahwa model mencapai puncak akurasi dan kepintarannya pada **Epoch 66** (mAP50: 0.890). Setelah melewati epoch tersebut, model mulai mengalami *overfitting* ringan yang ditandai dengan melebarnya jarak nilai *Validation Loss*. Sebagai mitigasi ilmiah yang tepat, penelitian ini mengekstraksi dan menggunakan file bobot **`best.pt`** (rekaman kondisi model di Epoch 66) yang kemudian diubah namanya menjadi `yolov8m_malaria.pt` untuk evaluasi akhir, dan secara sadar membuang bobot `last.pt` (kondisi *overfit* di Epoch 100).

---

## 3. Panduan Pengujian Model (Inferensi) di Komputer Lain

Agar model ini dapat diuji coba secara langsung di komputer Bapak Pembimbing atau Penguji, silakan ikuti langkah-langkah berikut:

**Prasyarat (Requirements)**
1. Komputer telah terinstal Python 3.8 atau lebih baru.
2. Terdapat aplikasi Terminal (Command Prompt) atau IDE seperti Visual Studio Code.
3. Untuk sekadar pengujian deteksi (inferensi citra), CPU standar sudah memadai.

**Langkah 1: Persiapan Folder**
1. Unduh atau salin (copy) seluruh folder "Skripsi_Malaria" ini ke dalam komputer.
2. Buka Terminal/Command Prompt, lalu arahkan (*cd*) ke dalam direktori folder ini.

**Langkah 2: Instalasi Library Sesuai Standar Lingkungan**
Untuk menyamakan *environment*, jalankan perintah berikut di Terminal agar semua pustaka terinstal otomatis:
> pip install -r requirements.txt

**Langkah 3: Menguji Model Melalui GUI (Sangat Disarankan)**
Sistem ini telah dilengkapi dengan Graphical User Interface (GUI) yang interaktif. Cukup jalankan perintah:
> python gui_malaria.py

Jendela aplikasi akan terbuka. Pengguna dapat menekan "Start Camera" untuk deteksi *real-time* berbasis mikroskop, menyesuaikan parameter rotasi dan pembesaran (*zoom*), atau mengekspor hasil tangkapan.

**Langkah 4: Menguji Model Cepat (Skrip Interaktif)**
Sebagai alternatif, Bapak bisa menggunakan skrip *File Dialog* sederhana dengan perintah:
> python uji_model.py

Sistem akan memunculkan *pop-up*, Bapak cukup memilih salah satu citra uji dari folder `dataset_sample/`. Program akan memproses citra tersebut dengan model `yolov8m_malaria.pt`, menampilkan hasilnya di layar, dan menyimpan citranya ke dalam folder `runs/`.

---

## Kontak
Jika terdapat kendala teknis saat menjalankan sistem ini, silakan menghubungi:
**Ivan Fulvian Pitoyo** (122430064)
Program Studi Teknik Biomedis - ITERA
+6281380093355

Terima kasih yang sebesar-besarnya atas bimbingan dan arahan dari Bapak Dosen Pembimbing selama proses pengerjaan Tugas Akhir ini.