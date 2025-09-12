import numpy as np
import h5py
import matplotlib.pyplot as plt
import mytools

# Input/output files
input_file = "/home/hug/Downloads/HoloTomo_Data/holo_purephase.h5"
output_file = "/home/hug/Downloads/HoloTomo_Data/holo_purephase_processed.h5"
dataset_name = "holodata"

# Read holograms
holo_data = mytools.read_h5_to_float(input_file, dataset_name)
print(holo_data.shape)

mytools.remove_outliers(holo_data)
mytools.remove_stripes(holo_data)

display_data = mytools.scale_display_data(holo_data[1])
plt.figure(figsize=(8, 8))
plt.imshow(display_data, cmap='viridis')
plt.colorbar()
plt.title("holo_data first frame")
plt.show()

# Save processed holograms
mytools.save_h5_from_float(output_file, dataset_name, holo_data)