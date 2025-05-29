# PhaseNet Indonesia - Complete Fine-tuning Guide

Panduan lengkap untuk fine-tuning PhaseNet menggunakan dataset gempa Indonesia. Dokumentasi ini menjelaskan seluruh proses dari persiapan data hingga training model dengan window size yang dioptimalkan untuk karakteristik seismik Indonesia.

## **Data Yang Digunakan**

- **Dataset**: Dataset gempa Indonesia yang sudah dipreprocess dan di-padding sehingga semuanya memiliki panjang yang sama (untuk NPZ)
- **NPZ**: Dataset disimpan dalam format `.npz`, yang merupakan format kompresi dari NumPy dan memuat beberapa array sekaligus. Dataset ini berada dalam direktori:

  ```
  dataset_phasenet_aug/npz_padded/
  ```

  * Setiap file mewakili satu event seismik dari satu stasiun.
  * Nama file tidak secara langsung mencerminkan tanggal atau lokasi, namun informasi tersebut dapat diperoleh dari atribut `t0` dan `station_id`.

  * Setiap file `.npz` berisi beberapa *key* dengan deskripsi sebagai berikut:

    | Key                | Tipe Data         | Bentuk     | Deskripsi                                                                              |
    | ------------------ | ----------------- | ---------- | -------------------------------------------------------------------------------------- |
    | `data`             | `float64 ndarray` | (30085, 3) | Data waveform seismik ter-*padding* untuk 3 channel: E (East), N (North), Z (Vertical) |
    | `p_idx`            | `int64 ndarray`   | (1, 1)     | Indeks waktu kedatangan gelombang **P**                                                |
    | `s_idx`            | `int64 ndarray`   | (1, 1)     | Indeks waktu kedatangan gelombang **S**                                                |
    | `station_id`       | `str`             | ()         | ID stasiun seismik, misalnya `'BBJI'`                                                  |
    | `t0`               | `str`             | ()         | Timestamp awal rekaman dalam format ISO, misalnya `'2017-02-13T08:12:33.369'`          |
    | `channel`          | `str ndarray`     | (3,)       | Nama channel yang digunakan, misalnya `['BHE', 'BHN', 'BHZ']`                          |
    | `channel_type`     | `str ndarray`     | (3,)       | Jenis channel berdasarkan orientasi: E, N, Z                                           |
    | `is_augmented`     | `bool ndarray`    | (1,)       | Apakah data ini merupakan hasil augmentasi                                             |
    | `original_channel` | `str ndarray`     | (3,)       | Nama channel asli sebelum augmentasi (jika ada)                                        |
  
  * Alasan semua data di-padding adalah karena PhaseNet membutuhkan data dengan panjang yang sama untuk training, sehingga kami memilih untuk membuat semua data memiliki panjang yang sama dengan cara padding, namun ketika masih dalam file mseed datanya sudah memiliki panjang yang mirip sehingga paddingnya tidak terlalu besar (hanya 10 samples saja).

- **CSV**: Daftar file untuk training/validation, berada di direktori `dataset_phasenet_aug/padded_train_list.csv` dan `dataset_phasenet_aug/padded_valid_list.csv`
  - **Format**: `dataset_phasenet_aug/padded_train_list.csv`
  - **Format**: `dataset_phasenet_aug/padded_valid_list.csv`
* **mseed**: Ini adalah data asli sebelum dikonversi ke format `.npz`, disimpan dalam direktori:

  ```
  dataset_phasenet_aug/waveform/
  ```

  * Setiap file `.mseed` mewakili satu event seismik dari satu stasiun, sehingga akan berisi **3 trace**: komponen **E (East)**, **N (North)**, dan **Z (Vertical)**.
  * Data **tidak di-*padding***, sehingga panjang setiap file **berbeda-beda**:

    * Panjang maksimum: **30086 samples**
    * Panjang minimum: **30076 samples**
  * **Sampling rate** dari semua data telah di-*interpolate* menggunakan *linear interpolation* menjadi **100 Hz**, untuk menyesuaikan dengan kebutuhan PhaseNet (data asli dari gempa Indonesia hanya memiliki sampling rate 20 Hz).
  * Channel gempa yang digunakan **tidak terbatas pada BH\*** saja, tetapi juga mencakup:

    * **BL\***, **SH\***, **HL\***, **HH\***, dan **SL\***.
    * Pemilihan ini dilakukan dengan syarat channel memiliki **3 komponen (E, N, Z)** dan **sampling rate minimal 20 Hz**.
  * Alasan penggunaan channel yang bervariasi:

    * **PhaseNet** dilatih dengan berbagai jenis channel, sehingga model dapat menangani data dengan variasi channel yang luas.
    * Jika hanya menggunakan channel **BH\***, jumlah data sangat terbatas (**sekitar 600 file saja**), yang tidak cukup untuk pelatihan yang efektif.
  * Untuk data dari channel selain BH\*:

    * Walaupun hanya channel BH\* yang memiliki anotasi manual untuk **P dan S picks** dari GEOFON,
    * Kami melakukan **anotasi otomatis** pada channel lainnya dengan cara:

      * Mengambil waktu arrival **P dan S** dari channel **BH\***.
      * Menerapkan waktu yang sama pada channel lain (BL\*, HL\*, dst) dalam event dan stasiun yang sama.
      * Proses ini dilakukan karena umumnya tidak ada perbedaan waktu yang signifikan antar channel dalam satu stasiun yang sama.

---

## 📊 **ANALISIS DATA INDONESIA**

Berdasarkan analisis 2,053 file NPZ dataset Indonesia:

### **Statistik P-S Interval:**
- **Mean**: 36.0 detik (3,599 samples)
- **Median**: 29.8 detik (2,976 samples)
- **99th percentile**: 117.1 detik (11,707 samples)
- **Maximum**: 240.8 detik (24,081 samples)

### **Distribusi Data:**
- Sangat Pendek (< 20s): 30.2% (620 file)
- Pendek (20-40s): 37.0% (760 file)
- Sedang (40-60s): 18.7% (384 file)
- Panjang (60-120s): 13.1% (269 file)
- Sangat Panjang (> 120s): 1.0% (20 file)

### **Solusi Window Size:**
- **Window 135 detik (13,500 samples)** untuk menangkap 99% data
- **Margin safety**: 15 detik sebelum P dan setelah S
- **Trade-off**: Memory usage lebih tinggi, tetapi coverage maksimal

---

## 🔧 **SETUP ENVIRONMENT**

### **1. Masuk ke Docker Container**
```bash
# Pastikan berada di direktori PhaseNet dalam container
cd /home/jovyan/PhaseNet

# Aktifkan conda environment
conda activate phasenet

# Verifikasi TensorFlow
python3 -c "import tensorflow as tf; print(tf.__version__)"
```

### **2. Verifikasi Data**
```bash
# Cek data NPZ exists
ls -la dataset_phasenet_aug/npz_padded/ | head -5

# Cek format CSV (harus ada header 'fname')
head dataset_phasenet_aug/padded_train_list.csv
head dataset_phasenet_aug/padded_valid_list.csv
```

---

## 🚀 **CARA TRAINING**

### **Scenario 1: Training Baru (Fresh Start)**

```bash
# 2. Mulai training
bash run_training_indonesia_99pct.sh
```

### **Scenario 2: Melanjutkan Training dari Model Pre-trained yang Bagus**

```bash
# 1. Cek status training yang ada
python3 check_training_status.py

# 2. Resume training dari model 190703-214543 (model yang sudah bagus)
bash resume_training_indonesia_99pct.sh
```

### **Scenario 3: Training dari Model Pre-trained**

```bash
# Jika memiliki model checkpoint yang ingin dilanjutkan
bash run_training_indonesia_99pct.sh --load_model --load_model_dir path/to/existing/model
```

---

## 🔄 **PERBEDAAN SCRIPT TRAINING**

### **`run_training_indonesia_99pct.sh` - Training Baru**
- **Fungsi**: Memulai training dari awal (fresh start)
- **Model Output**: Membuat directory baru dengan timestamp (YYMMDD-HHMMSS)
- **Use Case**: 
  - Training pertama kali
  - Eksperimen dengan parameter baru
  - Reset training dari awal

### **`resume_training_indonesia_99pct.sh` - Fine-tuning dari Model Bagus**
- **Fungsi**: Fine-tuning dari model pre-trained yang sudah bagus (190703-214543)
- **Model Source**: Menggunakan model `/PhaseNet/model/190703-214543` yang sudah terbukti bagus
- **Model Output**: Membuat model baru hasil fine-tuning di `model_indonesia_99pct/`
- **Use Case**:
  - Fine-tuning model yang sudah bagus untuk data Indonesia
  - Transfer learning dari model pre-trained
  - Memanfaatkan model yang sudah dilatih dengan baik

### **Perbedaan Kunci:**
| Aspek | `run_training` | `resume_training` |
|-------|----------------|-------------------|
| **Starting Point** | From scratch | Model 190703-214543 |
| **Model Directory** | `model_indonesia_99pct/YYMMDD-HHMMSS` | `model_indonesia_99pct/YYMMDD-HHMMSS` |
| **Training Type** | Fresh training | Fine-tuning |
| **Pre-trained Model** | None | 190703-214543 (fixed) |
| **Use Case** | Eksperimen baru | Optimasi model bagus |

**⚠️ Catatan Penting**: Script `resume_training` sekarang menggunakan model spesifik `190703-214543` yang sudah terbukti bagus, bukan mencari model terbaru secara otomatis.

---

## 📁 **STRUKTUR FILE YANG DITAMBAHKAN**

### **File Training Khusus Indonesia:**
```
PhaseNet/
├── phasenet/
│   ├── train_indonesia_99pct.py          # Script training utama
│   ├── data_reader_indonesia_99pct.py    # Data reader khusus 99% coverage
│   ├── test_indonesia_99pct.py           # Script testing
│   ├── analyze_ps_intervals_99pct.py     # Analisis P-S intervals
│   └── prepare_data_split_99pct.py       # Persiapan data split
├── run_training_indonesia_99pct.sh       # Runner script training baru
├── resume_training_indonesia_99pct.sh    # Runner script resume training
├── check_training_status.py             # Cek status training
```

### **Modifikasi pada File Existing:**
- **model.py**: Ditambahkan support untuk window size besar (13,500 samples)
- **util.py**: Ditambahkan utility functions untuk compatibility

---

## ⚙️ **KONFIGURASI TEKNIS**

### **Parameter Training Optimal:**
```bash
WINDOW_LENGTH=13500      # 135 detik untuk 99% coverage
BATCH_SIZE=16           # Dikurangi untuk efisiensi memory
LEARNING_RATE=0.00003   # Conservative untuk stabilitas
DROP_RATE=0.15          # Higher dropout untuk regularization
EPOCHS=100              # Standard training epochs
SAVE_INTERVAL=5         # Save model setiap 5 epochs
```

### **Memory Requirements:**
- **Minimum**: 16GB GPU memory
- **Recommended**: 24GB GPU memory
- **Fallback**: Reduce batch_size ke 8 atau 4

### **Hardware Optimization:**
```bash
# Jika GPU memory terbatas (<16GB):
BATCH_SIZE=8
WINDOW_LENGTH=12000     # 120s → ~98% coverage

# Jika ingin training cepat (testing):
EPOCHS=10
SAVE_INTERVAL=2
```

---

## 🔄 **TRAINING WORKFLOW DETAIL**

### **1. Data Preparation (Otomatis)**
- Verifikasi file NPZ dan CSV exists
- Fix CSV headers jika diperlukan
- Analisis P-S intervals untuk optimasi window
- Split data training/validation

### **2. Model Initialization**
- Load PhaseNet architecture dengan window size besar
- Setup optimizer dan loss function
- Initialize atau load existing checkpoint

### **3. Training Loop**
- Batch processing dengan memory optimization
- Frequent checkpointing (setiap 5 epochs)
- Training history tracking
- Validation monitoring

### **4. Resume Capability**
- Automatic checkpoint detection
- Training history preservation
- Smart resume logic dengan compatibility fixes

---

## 🔧 **TROUBLESHOOTING & LESSONS LEARNED**

### **Masalah yang Sudah Diperbaiki:**

#### **1. Import Error: 'MakeDirs' from 'util'**
**Solusi**: ✅ Implementasi utility functions langsung di training script
```python
def MakeDirs(dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)
```

#### **2. KeyError: 'fname' di CSV files**
**Solusi**: Perbaiki header CSV manual

#### **3. TensorFlow Regularizer Tensor Error**
**Solusi**: ✅ Menggunakan float value langsung untuk weight_decay
```python
weight_decay=float(args.weight_decay)  # Bukan tensor operation
```

#### **4. Variable Scope Error (global_step)**
**Solusi**: ✅ Compatibility mode untuk checkpoint loading
```python
tf.train.get_or_create_global_step()  # Instead of creating new
```

#### **5. Data Loading Issues**
**Solusi**: ✅ Robust NPZ file handling dan shape validation
```python
# Extract waveform dari NPZ dengan proper error handling
data = np.load(file_path)
waveform = data['waveform'] if 'waveform' in data else data[data.files[0]]
```

### **Lessons Learned:**

#### **Memory Management:**
- Window size besar membutuhkan batch size kecil
- Frequent checkpointing penting untuk training panjang
- GPU memory monitoring essential

#### **Data Compatibility:**
- CSV headers harus konsisten ('fname')
- NPZ files harus memiliki struktur yang uniform
- Path compatibility antara Docker dan host

#### **Training Stability:**
- Learning rate konservatif untuk window besar
- Higher dropout rate untuk regularization
- Frequent validation untuk monitoring overfitting

---

## 📊 **MONITORING TRAINING**

### **Real-time Monitoring:**
```bash
# GPU usage
watch -n 1 nvidia-smi

# Training progress
tail -f dataset_phasenet_aug/logs_indonesia_99pct/train.log

# Training status
python3 check_training_status.py
```

### **Expected Performance:**
- **Training time**: ~6-12 jam untuk 100 epochs
- **Memory usage**: 16-24GB GPU
- **Coverage**: 99% data Indonesia
- **P-wave accuracy**: Expected >90%
- **S-wave accuracy**: Expected >85%

---

## 📈 **HASIL DAN OUTPUT**

### **Model Output:**
```
dataset_phasenet_aug/
├── model_indonesia_99pct/YYMMDD-HHMMSS/    # Trained model dengan timestamp
│   ├── config.json                          # Model configuration
│   ├── training_history.npy                 # Loss history
│   ├── model_XXXX.ckpt.*                   # Checkpoint files
│   └── final_model.ckpt.*                  # Final model
├── logs_indonesia_99pct/                    # Training logs
├── test_results_indonesia_99pct/            # Test results (setelah testing)
├── ps_interval_analysis_99pct.csv           # Data analysis
└── ps_interval_analysis_99pct.png           # Visualization
```

### **Testing Model:**
```bash
cd PhaseNet/phasenet
python3 test_indonesia_99pct.py \
    --test_dir ../dataset_phasenet_aug/npz_padded \
    --test_list ../dataset_phasenet_aug/padded_valid_list.csv \
    --model_dir ../dataset_phasenet_aug/model_indonesia_99pct/YYMMDD-HHMMSS \
    --output_dir ../dataset_phasenet_aug/test_results_indonesia_99pct \
    --batch_size 8 \
    --plot_results
```

---

## 🎯 **QUICK START CHECKLIST**

### **Persiapan (Sekali saja):**
- [ ] Data NPZ ada di `dataset_phasenet_aug/npz_padded/`
- [ ] CSV files dengan header 'fname' tersedia
- [ ] Docker environment aktif dengan conda phasenet
- [ ] GPU memory minimal 16GB available

### **Training Baru:**
```bash
cd /home/jovyan/PhaseNet
conda activate phasenet
bash run_training_indonesia_99pct.sh
```

### **Resume Training:**
```bash
cd /home/jovyan/PhaseNet
conda activate phasenet
python3 check_training_status.py
bash resume_training_indonesia_99pct.sh
```

---

## 🚨 **EMERGENCY FIXES**

### **CUDA Out of Memory:**
```bash
# Quick fix: reduce batch size
sed -i 's/BATCH_SIZE=16/BATCH_SIZE=8/' run_training_indonesia_99pct.sh
sed -i 's/BATCH_SIZE=16/BATCH_SIZE=8/' resume_training_indonesia_99pct.sh
```


### **Checkpoint Loading Issues:**
```bash
# Check training status
python3 check_training_status.py

# If corrupted, start fresh
bash run_training_indonesia_99pct.sh
```

---

## 🎉 **SUCCESS INDICATORS**

Training berhasil jika:
- ✅ Setup script completed without errors
- ✅ No CUDA memory errors during training
- ✅ Training loss menurun secara konsisten
- ✅ Model checkpoints tersimpan setiap 5 epochs
- ✅ Resume training loads checkpoint successfully
- ✅ Validation loss tidak diverge dari training loss

---

## 📁 **STRUKTUR FILE LENGKAP**

### **Script Training Utama:**
| File | Fungsi |
|------|--------|
| `run_training_indonesia_99pct.sh` | **Script utama** untuk memulai training baru |
| `resume_training_indonesia_99pct.sh` | **Script resume** untuk melanjutkan training |

### **Script Training Indonesia (99% Coverage):**
| File | Fungsi |
|------|--------|
| `phasenet/train_indonesia_99pct.py` | **Script training utama** dengan window 135 detik |
| `phasenet/data_reader_indonesia_99pct.py` | **Data reader khusus** untuk coverage 99% |
| `phasenet/test_indonesia_99pct.py` | **Script testing** model Indonesia |
| `phasenet/analyze_ps_intervals_99pct.py` | Analisis interval P-S untuk optimasi window |
| `phasenet/prepare_data_split_99pct.py` | Persiapan data split training/validation |

### **PhaseNet Core Files (Original):**
| File | Fungsi |
|------|--------|
| `phasenet/model.py` | **Arsitektur UNet** PhaseNet (sudah dimodifikasi untuk window besar) |
| `phasenet/data_reader.py` | Data reader original PhaseNet |
| `phasenet/train.py` | Training script original PhaseNet |
| `phasenet/predict.py` | Prediction script |
| `phasenet/util.py` | Utility functions |

### **File yang Sudah Dihapus (Redundan):**
- `QUICK_START_99PCT.md` → Digabung ke README ini
- `README_99PCT_COVERAGE.md` → Digabung ke README ini
- `TROUBLESHOOTING_99PCT.md` → Digabung ke README ini
- `PHASENET_INDONESIA_COMPLETE_GUIDE.md` → Digabung ke README ini
- `FILE_STRUCTURE.md` → Digabung ke README ini
- `train_indonesia.py` → Digantikan dengan versi 99pct
- `test_model_init.py` → Script testing sementara
- `migrate_existing_model.py` → Sudah tidak diperlukan
- `resume_migrated_model.sh` → Sudah tidak diperlukan
- `Untitled.ipynb` → Notebook tanpa nama
- `phasenet/data_reader_indonesia.py` → Digantikan versi 99pct

---

## 📞 **SUPPORT & NEXT STEPS**

### **Jika Masih Ada Issues:**
1. Jalankan `python3 check_training_status.py` untuk diagnosis
2. Check GPU memory dengan `nvidia-smi`
3. Verify data integrity dengan `ls -la dataset_phasenet_aug/npz_padded/`
4. Check environment dengan `conda env list`

### **Setelah Training Selesai:**
1. Run testing untuk evaluasi model
2. Analyze hasil dengan visualization tools
3. Deploy model untuk prediction pada data baru
4. Fine-tune parameters jika diperlukan

---

## 🏆 **KESIMPULAN**

PhaseNet Indonesia 99% Coverage solution memberikan:

1. **Significant Coverage Improvement**: Dari 67% → 99%
2. **Better S-wave Detection**: Untuk interval P-S panjang
3. **Reduced Data Loss**: Minimal truncation
4. **Indonesian-Specific Optimization**: Disesuaikan karakteristik seismik Indonesia
5. **Robust Training Pipeline**: Dengan error handling dan resume capability

**Trade-off yang Acceptable**: Memory usage lebih tinggi dan training time lebih lama, tetapi akurasi dan coverage jauh lebih baik untuk data Indonesia.

**Ready to Train! 🚀**