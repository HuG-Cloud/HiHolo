import numpy as np
import h5py
import hiholo

# Input/output files
input_file = "/home/hug/Downloads/HoloTomo_Data/holo_purephase.h5"
output_file1 = "/home/hug/Downloads/HoloTomo_Data/visiblelight/board_holo.h5"
output_file2 = "/home/hug/Downloads/HoloTomo_Data/visiblelight/board_back_holo.h5"
dataset_name = "holodata"

# Read holograms
with h5py.File(input_file1, 'r') as f:
    holo_data = np.array(f[dataset_name], dtype=np.float32)
    holo_data = hiholo.removeOutliers(holo_data)

with h5py.File(input_file2, 'r') as f:
    back_data = np.array(f[dataset_name], dtype=np.float32)
    back_data = hiholo.removeOutliers(back_data)

holo_data = holo_data / back_data
holo_data = holo_data[0:2704, 0:2704]
# back_data = back_data[0:2048, 400:2448]

# Save processed holograms
with h5py.File(output_file1, 'w') as f:
    f.create_dataset(dataset_name, data=holo_data, dtype=np.float32)