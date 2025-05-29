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


## **ANALISIS DATA INDONESIA**

Berdasarkan analisis 2,053 file NPZ dataset Indonesia:

### **Statistik P-S Interval:**

- P-S interval di sini maksudnya adalah interval waktu antara gelombang P dan S.
- **Mean**: 36.0 detik (3,599 samples)
- **Median**: 29.8 detik (2,976 samples)
- **99th percentile**: 117.1 detik (11,707 samples) ⭐
- **Maximum**: 240.8 detik (24,081 samples)

### **Distribusi Data:**

- Sangat Pendek (< 20s): 30.2% (620 file)
- Pendek (20-40s): 37.0% (760 file)
- Sedang (40-60s): 18.7% (384 file)
- Panjang (60-120s): 13.1% (269 file)
- Sangat Panjang (> 120s): 1.0% (20 file)

### **Permasalahan yang muncul saat training:**

- PhaseNet original dilatih hanya dengan window **3000 samples (30 detik)**, namun data gempa Indonesia memiliki P-S interval yang jauh lebih panjang.
- **Coverage Original PhaseNet**: Hanya ~70% data Indonesia yang dapat dideteksi dengan baik
- **Data Loss**: 30% event dengan P-S interval panjang akan ter-truncate atau hilang

### **Solusi Window Size 99% Coverage:**

- **Window 135 detik (13,500 samples)** untuk menangkap **99% data Indonesia**
- **Berdasarkan**: 99th percentile (117.1 detik) + margin safety 18 detik
- **Trade-off**: Memory usage lebih tinggi, tetapi coverage maksimal
- **Hasil**: Dari ~70% → **99% coverage** untuk data seismik Indonesia



## 🚀 **CARA TRAINING**

### **Scenario 1: Training Baru (From Scratch)**

```bash
# Training dari awal dengan window 135 detik
bash run_training_indonesia_99pct.sh
```

**Karakteristik:**
- Memulai training dari random weights
- Use case: Eksperimen parameter baru

### **Scenario 2: Fine-tuning dari Model Pre-trained (Recommended)**

```bash
# Fine-tuning dari model PhaseNet yang sudah bagus (190703-214543)
bash resume_training_indonesia_99pct.sh
```

**Karakteristik:**
- Transfer learning dari model pre-trained yang telah dilatih sebelumnya oleh pembuat PhaseNet.

## 🔄 **PERBEDAAN SCRIPT TRAINING**

### **`run_training_indonesia_99pct.sh` - Training Baru**
- **Fungsi**: Memulai training dari awal (fresh start)
- **Starting Point**: Random weights initialization
- **Model Output**: `model_indonesia_99pct/YYMMDD-HHMMSS/`
- **Use Case**: 
  - Training pertama kali
  - Eksperimen dengan parameter baru

### **`resume_training_indonesia_99pct.sh` - Fine-tuning**
- **Fungsi**: Fine-tuning dari model pre-trained yang sudah bagus
- **Starting Point**: Model `190703-214543` (pre-trained PhaseNet)
- **Model Output**: `model_indonesia_99pct/YYMMDD-HHMMSS/`
- Transfer learning yang efisien


### **Modifikasi pada File Existing:**
- **model.py**: Ditambahkan support untuk window size besar (13,500 samples)
- **util.py**: Ditambahkan utility functions untuk compatibility

---

## ⚙️ **KONFIGURASI TEKNIS**

### **Parameter Training:**
- Parameter training dapat diubah di file `run_training_indonesia_99pct.sh` dan `resume_training_indonesia_99pct.sh`


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
