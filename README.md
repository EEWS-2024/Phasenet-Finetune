# PhaseNet Indonesia

Repository ini adalah adaptasi dari [PhaseNet](https://github.com/AI4EPS/PhaseNet) yang telah dioptimalkan untuk dataset gempa Indonesia menggunakan **sliding window strategy** dan **transfer learning** dari model pretrained NCEDC dataset.

## **Overview**

PhaseNet Indonesia menggunakan **dua strategi windowing** yang telah dioptimalkan untuk karakteristik seismik Indonesia:

1. **Sliding Window 3000 samples (30s)**
   - Compatible dengan model pretrained PhaseNet
   - Transfer learning dari NCEDC dataset

2. **Fixed Window 170s** - Legacy approach
   - Custom architecture untuk data Indonesia

---

## **Flexible Architecture: Multiple Window Sizes**

### **Pertanyaan Teknis: Bagaimana Model Bisa Menerima 17,000 Samples?**

Model PhaseNet menggunakan **Dynamic Input Placeholders** dan **Convolutional Architecture** yang inherently **size-agnostic**, sehingga bisa menerima input dengan ukuran berapapun tanpa mengubah kode model.

### **1. Dynamic Input Shape - Kunci Utama**

**Code di `model.py` line 123-124:**
```python
def add_placeholders(self, input_batch=None, mode="train"):
    if input_batch is None:
        # KUNCI: None berarti dimensi bisa berubah-ubah!
        self.X = tf.placeholder(dtype=tf.float32, 
                               shape=[None, None, None, self.X_shape[-1]], 
                               name='X')
        self.Y = tf.placeholder(dtype=tf.float32, 
                               shape=[None, None, None, self.n_class], 
                               name='y')
```

**Shape Explanation:**
```python
[None, None, None, 3]
 │     │     │    └── Channels (FIXED = 3 untuk Z,N,E)
 │     │     └────── Dummy dimension (selalu 1)
 │     └──────────── Time samples (DYNAMIC = 3000 atau 17000)
 └────────────────── Batch size (DYNAMIC)
```

**Mengapa `None` bisa bekerja?**
- TensorFlow `None` artinya dimensi **runtime-determined**
- Model tidak perlu tau ukuran pasti saat compile time
- Input shape ditentukan saat **feed data** ke model

### **2. Convolutional Networks: Size-Agnostic by Design**

**Prinsip Fundamental CNN:**
```python
# Convolutional operation bekerja pada local patches
# Tidak peduli total ukuran input!

# Contoh Conv2D dengan kernel 7x1:
input_3000:  [batch, 3000, 1, 3] → conv2d → [batch, 3000, 1, filters]
input_17000: [batch, 17000, 1, 3] → conv2d → [batch, 17000, 1, filters]

# SAMA kernel weights, BEDA output size!
```

### **3. UNet Architecture: Proportional Scaling**

**Encoder Path (Downsampling):**
```python
# Pool size = [4, 1] artinya reduce time dimension by factor 4
# Depth = 5 levels, jadi total reduction = 4^4 = 256x

# Untuk 3,000 samples:
Level 0: 3000 samples (input)
Level 1: 750 samples  (3000 ÷ 4)
Level 2: 187 samples  (750 ÷ 4)  
Level 3: 46 samples   (187 ÷ 4)
Level 4: 11 samples   (46 ÷ 4)
Level 5: 2 samples    (11 ÷ 4, rounded)

# Untuk 17,000 samples:
Level 0: 17000 samples (input)
Level 1: 4250 samples  (17000 ÷ 4)
Level 2: 1062 samples  (4250 ÷ 4)
Level 3: 265 samples   (1062 ÷ 4)
Level 4: 66 samples    (265 ÷ 4)
Level 5: 16 samples    (66 ÷ 4)
```

### **4. Parameter Sharing: Key Insight**

**Model parameters TIDAK berubah antara 3,000 vs 17,000 samples!**

```python
# Convolutional weights yang sama digunakan untuk kedua ukuran:

# Input convolution:
kernel_7x1x3x8 = [7, 1, 3, 8]     # SAMA untuk 3k dan 17k

# DownConv layers:  
down_conv1_weights = [7, 1, 8, 8]   # SAMA
down_conv2_weights = [7, 1, 8, 16]  # SAMA  
down_conv3_weights = [7, 1, 16, 32] # SAMA

# Output layer:
output_weights = [1, 1, 8, 3]       # SAMA
```

**Total parameters:**
- **3,000 samples model**: ~269,675 parameters
- **17,000 samples model**: ~269,675 parameters (**IDENTIK!**)

**Yang berbeda hanya:**
- **Computational cost** (more pixels to process)
- **Memory usage** (larger activation maps)  
- **Training time** (more forward/backward passes)

### **5. Practical Implementation Differences**

**3,000 samples (`train_indonesia_3000.py`):**
```python
# Create config for 3k samples
config = ModelConfig(
    X_shape=[3000, 1, 3],   # Tells model expected input size
    Y_shape=[3000, 1, 3]
)

# Model automatically handles 3000-sample inputs
model = UNet(config=config, input_batch=(X_placeholder, Y_placeholder))
```

**17,000 samples (`train_indonesia.py`):**
```python
# Create config for 17k samples
config = ModelConfig(
    X_shape=[17000, 1, 3],  # Tells model expected input size
    Y_shape=[17000, 1, 3]
)

# SAME model code, different input size!
model = UNet(config=config, input_batch=(X_placeholder, Y_placeholder))
```


### **Kesimpulan Flexible Architecture**

**PhaseNet bisa menerima 17,000 samples karena:**

1. **Dynamic placeholders** dengan `None` dimensions
2. **Convolutional architecture** yang inherently size-agnostic  
3. **Parameter sharing** - weights yang sama untuk semua ukuran
4. **Proportional scaling** dalam UNet encoder-decoder
5. **1x1 output convolution** yang preserve input dimensions

**Key Takeaway:** Model TIDAK perlu diubah sama sekali. Yang berubah hanya **input configuration** dan **computational resources** yang dibutuhkan. Arsitektur UNet inherently flexible untuk handle berbagai ukuran input.

---

## **Dataset Indonesia**

### **Format Data:**
- **NPZ Files**: `dataset_phasenet_aug/npz_padded/`
- **Training List**: `dataset_phasenet_aug/padded_train_list.csv`
- **Validation List**: `dataset_phasenet_aug/padded_valid_list.csv`

### **Struktur NPZ File:**
| Key | Type | Shape | Description |
|-----|------|-------|-------------|
| `data` | `float64` | (30085, 3) | Waveform data (E, N, Z channels) |
| `p_idx` | `int64` | (1, 1) | P-wave arrival index |
| `s_idx` | `int64` | (1, 1) | S-wave arrival index |
| `station_id` | `str` | () | Station ID (e.g., 'BBJI') |
| `t0` | `str` | () | Start timestamp |
| `channel` | `str array` | (3,) | Channel names |

### **Statistik P-S Interval:**
- **Mean**: 36.0 detik
- **Median**: 29.8 detik  
- **P99**: 117.1 detik
- **Maximum**: 240.8 detik
- **Total Files**: 2,053 NPZ files

---

## **Decoder-Only Fine-tuning Strategy**

### **Pendekatan Fine-tuning Decoder-Only**

**Fine-tuning decoder-only** adalah pendekatan pelatihan model di mana hanya bagian decoder yang dilatih, sementara bagian encoder dibekukan (freeze). Pendekatan ini memberikan sejumlah keuntungan:

* **Mengurangi risiko *catastrophic forgetting*** terhadap kemampuan *feature extraction*.
* **Memfokuskan adaptasi model** pada karakteristik output data seismik Indonesia.
* **Efisiensi parameter**: Hanya 45.2% parameters yang trainable (153,307 dari 269,675 total).

### **Arsitektur Decoder-Only**

```
PhaseNet Architecture:
┌─────────────────┐
│   Input Layer   │  FROZEN (Encoder)
├─────────────────┤
│   DownConv_0    │  FROZEN (116,368 parameters)
│   DownConv_1    │  FROZEN
│   DownConv_2    │  FROZEN   } ENCODER
│   DownConv_3    │  FROZEN   
│   DownConv_4    │  FROZEN   
├─────────────────┤
│    UpConv_3     │  TRAINABLE
│    UpConv_2     │  TRAINABLE
│    UpConv_1     │  TRAINABLE    } DECODER (153,307 parameters)
│    UpConv_0     │  TRAINABLE
├─────────────────┤
│  Output Layer   │  TRAINABLE
└─────────────────┘
```

### **1. Decoder-Only Fine-tuning:**

```bash
# Basic usage
bash run_finetuning_decoder_only.sh \
    ../augmentasi-data-phasenet-full/dataset_phasenet_aug/npz_padded \
    ../augmentasi-data-phasenet-full/dataset_phasenet_aug/padded_train_list.csv

# With validation data
bash run_finetuning_decoder_only.sh \
    ../augmentasi-data-phasenet-full/dataset_phasenet_aug/npz_padded \
    ../augmentasi-data-phasenet-full/dataset_phasenet_aug/padded_train_list.csv \
    ../augmentasi-data-phasenet-full/dataset_phasenet_aug/npz_padded \
    ../augmentasi-data-phasenet-full/dataset_phasenet_aug/padded_valid_list.csv

# Output: model_indonesia/decoder_only/decoder3000_YYMMDD-HHMMSS/
```

### **2. Training From Scratch (Full Model):**
```bash
bash run_training_scratch_indonesia.sh

# Output: model_indonesia/scratch/YYMMDD-HHMMSS/
```

### **3. Decoder-Only Training Parameters:**
```bash
EPOCHS=50                 
BATCH_SIZE=512            # Larger batch size karena fewer parameters
LEARNING_RATE=0.0001      # Higher LR untuk decoder-only
DROP_RATE=0.05             
DECAY_STEP=10              
DECAY_RATE=0.98           
```

### **4. Monitor Decoder-Only Training:**
```bash
# Check training progress
tail -f model_indonesia/decoder_only/decoder3000_*/training_history.csv

# View loss curves
eog model_indonesia/decoder_only/decoder3000_*/loss_curves.png

# Check frozen layers info
cat model_indonesia/decoder_only/decoder3000_*/frozen_layers.txt
```

### **5. Decoder-Only Output Structure:**
```
model_indonesia/decoder_only/decoder3000_YYMMDD-HHMMSS/
├── config.json                 # Konfigurasi model
├── frozen_layers.txt           # Informasi layer yang dibekukan dan yang dilatih
├── training_history.csv        # Riwayat loss selama pelatihan
├── loss_curves.png             # Grafik kurva loss
├── decoder_model_final.ckpt    # Model hasil akhir
└── test_results/               # Hasil pengujian
    ├── decoder_only_comprehensive_results.csv
    └── decoder_only_comprehensive_results.png
```

---

## **Sliding Window Strategy - Technical Details**

### **Strategy Overview:**
```
Original Data Indonesia: [--------- 30,000+ samples (300+ detik) ---------]

Sliding Windows (3000 samples each):
Window 1: [--- 3000 samples ---]
Window 2:      [--- 3000 samples ---]  (50% overlap)
Window 3:           [--- 3000 samples ---]
...
Window N:                          [--- 3000 samples ---]
```

### **Benefits:**
- **20x More Training Data**: 2,053 files → 37,050+ training windows
- **Compatible Architecture**: Langsung compatible dengan model pretrained
- **Decoder-Only Efficiency**: Parameter trainable berkurang 54.8%

### **Configuration:**
```python
# DataConfig_Indonesia_3000
window_length = 3000    # 30 seconds (same as pretrained)
window_step = 1500      # 15 seconds (50% overlap)
sampling_rate = 100     # 100 Hz
X_shape = [3000, 1, 3]  # Compatible dengan model pretrained NCEDC
Y_shape = [3000, 1, 3]
```

---

## **Files & Architecture**

### **Output Directory Structure:**
```
model_indonesia/
├── decoder_only/                    # Decoder-only fine-tuning results
│   └── decoder3000_YYMMDD-HHMMSS/   # Timestamped model directories
├── scratch/                         # Training from scratch results  
│   └── YYMMDD-HHMMSS/              # Timestamped model directories
└── resume/                          # Resumed training results (legacy)

logs_indonesia/
├── decoder_only/                    # Decoder-only fine-tuning logs
└── scratch/                         # Training from scratch logs
```


### **Configuration Files:**
```
model_indonesia/
├── decoder_only/
│   └── decoder3000_YYMMDD-HHMMSS/
│       ├── config.json
│       ├── frozen_layers.txt
│       ├── training_history.csv
│       ├── loss_curves.png
│       └── decoder_model_final.ckpt
├── scratch/
│   └── YYMMDD-HHMMSS/
└── resume/
   └── YYMMDD-HHMMSS/
```
