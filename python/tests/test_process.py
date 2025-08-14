import numpy as np
import h5py
import matplotlib.pyplot as plt
import fastholo

# Input/output files
input_file1 = "/home/hug/Downloads/HoloTomo_Data/visiblelight/wing.h5"
input_file2 = "/home/hug/Downloads/HoloTomo_Data/visiblelight/wing_back.h5"
output_file1 = "/home/hug/Downloads/HoloTomo_Data/visiblelight/wing_holo.h5"
output_file2 = "/home/hug/Downloads/HoloTomo_Data/visiblelight/wing_back_holo.h5"
dataset_name = "holodata"

# Read holograms
with h5py.File(input_file1, 'r') as f:
    holo_data = np.array(f[dataset_name], dtype=np.float32)
    holo_data = fastholo.removeOutliers(holo_data)

with h5py.File(input_file2, 'r') as f:
    back_data = np.array(f[dataset_name], dtype=np.float32)
    back_data = fastholo.removeOutliers(back_data)

holo_data = holo_data / back_data
holo_data = holo_data[0:2048, 400:2448]
# back_data = back_data[0:2048, 400:2448]

# Save processed holograms
with h5py.File(output_file1, 'w') as f:
    f.create_dataset(dataset_name, data=holo_data, dtype=np.float32)

# with h5py.File(output_file2, 'w') as f:
#     f.create_dataset(dataset_name, data=back_data, dtype=np.float32)