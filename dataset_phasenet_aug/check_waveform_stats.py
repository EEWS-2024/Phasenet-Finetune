import os
import pandas as pd
from obspy import read

# Folder tempat file .mseed berada
folder_path = "./waveform"

# List untuk menyimpan data
data = []

# Iterasi semua file dalam folder
for filename in os.listdir(folder_path):
    if filename.endswith(".mseed"):
        file_path = os.path.join(folder_path, filename)
        try:
            st = read(file_path)
            # Ambil metadata dari trace pertama
            tr0 = st[0]
            duration = tr0.stats.endtime - tr0.stats.starttime
            bandcode = tr0.stats.channel[:2]  # Dua huruf pertama channel

            data.append({
                "filename": filename,
                "station": tr0.stats.station,
                "bandcode": bandcode,
                "duration": float(duration),
                "sampling_rate": tr0.stats.sampling_rate,
                "samples": tr0.stats.npts
            })
        except Exception as e:
            print(f"Failed to read {filename}: {e}")
            continue

# Buat DataFrame
df = pd.DataFrame(data)

# Statistik deskriptif
stats = df[["duration", "sampling_rate", "samples"]].describe()

# Total file per stasiun
files_per_station = df["station"].value_counts().rename_axis("station").reset_index(name="total_files")

# Total file per bandcode (2 huruf pertama channel)
files_per_bandcode = df["bandcode"].value_counts().rename_axis("bandcode").reset_index(name="total_files")

# Total keseluruhan file unik
total_files = df["filename"].nunique()

# Tampilkan hasil
print("=== Descriptive Statistics ===")
print(stats)

print("\n=== Total Files per Station ===")
print(files_per_station)

print("\n=== Total Files per Bandcode (2 huruf pertama channel) ===")
print(files_per_bandcode)

print(f"\n=== Total Unique MiniSEED Files: {total_files} ===")


import os
from obspy import read

folder_path = "./waveform"
# Hanya tampilkan 3 file saja
for filename in list(filter(lambda x: x.endswith('.mseed'), os.listdir(folder_path)))[:3]:
    file_path = os.path.join(folder_path, filename)
    print(f"\n=== Reading: {filename} ===")
    try:
        st = read(file_path)
        print(st)
    except Exception as e:
        print(f"Failed to read {filename}: {e}")
