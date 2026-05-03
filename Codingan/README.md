# Sistem Deteksi Malaria Berbasis YOLOv8
**Oleh: Ivan Fulvian Pitoyo (122430064)**
**Program Studi Teknik Biomedis - Institut Teknologi Sumatera**

Repositori ini berisi keseluruhan source code, dataset, hasil pelatihan, dan model Deep Learning (YOLOv8) untuk sistem deteksi parasit malaria (Plasmodium falciparum dan Plasmodium vivax) pada citra mikroskopis sediaan darah tepi.

File ini adalah panduan singkat untuk memahami struktur direktori, keputusan arsitektur, dan cara menjalankan model ini untuk keperluan pengujian (testing/inference) di perangkat lain.

---

## 1. Struktur Direktori dan File

Berikut adalah penjelasan mengenai isi dari folder utama proyek ini yang telah disusun ulang:

**Folder Utama**
* `db/` : Berisi database studi Optuna (`malaria_yolo_study`) untuk perekaman hasil pencarian hyperparameter.
* `Hasil_Training/` : Tempat disimpannya luaran cadangan dari proses training (grafik evaluasi, confusion matrix, dll).
* `models/` : Folder khusus yang menyimpan semua file *weights* model (`.pt`), termasuk model pre-trained (YOLOv8n, YOLOv8m) dan model final terbaik hasil penelitian (`yolo26m.pt`).
* `runs/` : Folder default bawaan library Ultralytics YOLOv8 untuk menyimpan riwayat hasil training dan inferensi deteksi.
* `src/` : Folder utama (*source*) yang berisi seluruh skrip Python, terbagi menjadi sub-folder:
    * `1_preprocessing/` : Skrip untuk membagi data (*split*), augmentasi, dan penyeimbangan kelas.
    * `2_utilities/` : Skrip alat bantu tambahan untuk mempermudah alur kerja.
    * `3_training/` : Skrip utama untuk pelatihan model dan optimasi *hyperparameter*.
* `venv/` : Lingkungan virtual Python (Virtual Environment) yang berisi semua instalasi library yang dibutuhkan. (Penting: Folder ini spesifik untuk konfigurasi sistem lokal dan mungkin perlu dibuat ulang jika dijalankan di komputer yang berbeda OS/Arsitektur).

**Skrip di Root Directory**
* `cek_gpu.py` / `Tes GPU bisa nggak.py` : Skrip singkat untuk memverifikasi apakah sistem berhasil membaca kartu grafis (GPU) NVIDIA.

---

## 2. Catatan Konfigurasi Training

Pembimbing yang terhormat, berikut adalah dua catatan penting terkait konfigurasi training akhir pada skrip utama:

1. **Penggunaan YOLOv8 Medium:** Model akhir dilatih menggunakan arsitektur `yolov8m.pt`, bukan `yolov8n.pt`, guna memaksimalkan performa pelokalan fitur parasit yang berukuran sangat mikro.
2. **Mekanisme Early Stopping:** Pada eksperimen final (Trial 20), parameter epochs diatur ke 100 dan patience juga diatur ke 100. Pengaturan ini dilakukan secara sengaja untuk memaksa model menyelesaikan 100 putaran iterasi secara utuh tanpa interupsi pemotongan prematur. Hal ini krusial agar peneliti dapat mengekstraksi dan menganalisis kurva Validation Loss secara penuh dari awal hingga akhir untuk membuktikan tidak terjadinya anomali overfitting yang ekstrem.

---

## 3. Panduan Pengujian Model (Inferensi) di Komputer Lain

Agar model ini dapat diuji coba di komputer Bapak Pembimbing, silakan ikuti langkah-langkah berikut:

**Prasyarat (Requirements)**
1. Komputer sudah terinstal Python 3.12.3 atau lebih baru.
2. Terdapat aplikasi Terminal (Command Prompt) atau IDE seperti Visual Studio Code (VS Code).
3. Untuk sekadar pengujian deteksi (inferensi gambar), CPU standar tanpa kartu grafis NVIDIA sudah memadai.

**Langkah 1: Persiapan Folder**
1. Salin (copy) seluruh folder "Skripsi_Malaria" ini ke dalam komputer tujuan.
2. Buka folder tersebut menggunakan VS Code atau arahkan direktori Terminal ke dalam folder ini.

**Langkah 2: Instalasi Library**
Buka Terminal/Command Prompt, lalu ketik dan jalankan dua perintah berikut:
> pip install ultralytics
> pip install opencv-python

**Langkah 3: Menguji Model Melalui Terminal**
Untuk menguji kemampuan model (`yolo26m.pt`) dalam mendeteksi gambar malaria baru, jalankan perintah ini di Terminal (pastikan file gambar uji tersedia di dalam folder datasets dan sesuaikan path model ke folder `models/`):

> yolo task=detect mode=predict model="models/yolo26m.pt" source="datasets/gambar_uji.jpg" show=True

(Catatan: Parameter show=True akan memunculkan jendela baru yang menampilkan gambar sediaan darah lengkap dengan kotak merah deteksi parasit malaria).

**Langkah 4 (Opsional): Menguji Model Melalui Skrip Interaktif**
Sebagai alternatif yang lebih mudah, sistem ini telah dilengkapi dengan antarmuka pemilihan file (File Dialog). Buka Terminal/Command Prompt, lalu jalankan perintah berikut:

> python uji_model.py

Setelah dijalankan, sebuah jendela pop-up akan muncul. Bapak cukup memilih gambar sediaan darah mana pun yang ada di komputer Bapak. Sistem akan otomatis memuat model `yolo26m.pt`, mendeteksi parasit, menampilkan hasilnya di layar, dan menyimpan gambar hasil deteksinya.


---

## Kontak
Jika terdapat kendala teknis saat menjalankan sistem ini, silakan menghubungi:
Ivan Fulvian Pitoyo (122430064)
Program Studi Teknik Biomedis - ITERA
+6281380093355

Terima kasih yang sebesar-besarnya atas bimbingan dan arahan Bapak selama proses pengerjaan Tugas Akhir ini.