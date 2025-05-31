# Panduan Resume Training PhaseNet Indonesia

Script `resume_training_indonesia.sh` telah dimodifikasi untuk memberikan fleksibilitas dalam memilih model yang akan digunakan sebagai base untuk melanjutkan training.

## 🚀 Cara Penggunaan

### 1. Menggunakan Model Terbaru (Otomatis)
```bash
bash resume_training_indonesia.sh
```
Script akan otomatis menggunakan model terbaru di direktori `model_indonesia/scratch/`.

### 2. Menggunakan Model Tertentu (Manual)
```bash
bash resume_training_indonesia.sh <nama_folder_model>
```

**Contoh:**
```bash
bash resume_training_indonesia.sh 241215-123456
```

## 📋 Melihat Daftar Model yang Tersedia

Gunakan script utility untuk melihat semua model yang tersedia:

```bash
bash list_models.sh
```

Script ini akan menampilkan:
- Daftar semua model di direktori `model_indonesia/scratch/`
- Daftar semua model di direktori `model_indonesia/resume/`
- Informasi detail setiap model (ukuran, tanggal, file yang tersedia)
- Cara penggunaan yang benar

## 🔍 Contoh Output

### Melihat Daftar Model:
```bash
$ bash list_models.sh

==================================================
📋 DAFTAR MODEL YANG TERSEDIA
==================================================

🔹 Scratch Training Models:
   Directory: model_indonesia/scratch

   Available models (sorted by modification time):
   ================================================
   📂 241215-143022
      🕒 Created: 2024-12-15 14:30
      💾 Size: 245M
      📄 Files: checkpoint config.json
      🎯 Checkpoints: 12 files

   📂 241215-120844
      🕒 Created: 2024-12-15 12:08
      💾 Size: 189M
      📄 Files: checkpoint config.json
      🎯 Checkpoints: 8 files
```

### Menggunakan Model Tertentu:
```bash
$ bash resume_training_indonesia.sh 241215-120844

==================================================
🔄 RESUME TRAINING PHASENET INDONESIA
==================================================
🎯 Using SPECIFIED model: 241215-120844

🎯 Configuration:
   Dataset: dataset_phasenet_aug
   Base model directory: model_indonesia/scratch/241215-120844
   Output model: model_indonesia/resume
   ...
```

## ⚠️ Error Handling

### Model Tidak Ditemukan:
Jika model yang ditentukan tidak ada, script akan memberikan error dan menampilkan daftar model yang tersedia:

```bash
$ bash resume_training_indonesia.sh model_yang_tidak_ada

❌ Specified model directory not found: model_indonesia/scratch/model_yang_tidak_ada

Available models in model_indonesia/scratch/:
total 16
drwxr-xr-x 4 user user 4096 Dec 15 14:30 241215-143022
drwxr-xr-x 4 user user 4096 Dec 15 12:08 241215-120844

Usage:
   bash resume_training_indonesia.sh                    # Use latest model automatically
   bash resume_training_indonesia.sh model_folder_name  # Use specific model folder
```

## 🎯 Fitur Baru

1. **Fleksibilitas Pemilihan Model**: Bisa menggunakan model terbaru atau model tertentu
2. **Validasi Model**: Script akan memverifikasi bahwa model yang ditentukan ada sebelum memulai training
3. **Error Handling**: Pesan error yang informatif dengan panduan penggunaan
4. **Utility Script**: Script `list_models.sh` untuk melihat daftar model dengan detail
5. **Backward Compatibility**: Tetap mendukung penggunaan tanpa parameter (behavior asli)

## 📝 Catatan

- Nama folder model biasanya dalam format `YYMMDD-HHMMSS` (contoh: `241215-143022`)
- Model yang digunakan sebagai base harus berada di direktori `model_indonesia/scratch/`
- Hasil resume training akan disimpan di direktori `model_indonesia/resume/`
- Script akan otomatis melakukan testing setelah training selesai 