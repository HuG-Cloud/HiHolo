import numpy as np
import h5py
import matplotlib.pyplot as plt
import mytools

# Input/output files
input_file = "/home/hug/Downloads/HoloTomo_Data/holo_200angles_simu_format.h5"
datasets = "holodata_distance_0,holodata_distance_1,holodata_distance_2,holodata_distance_3"
distance = 3
angle = 199

data_angle = mytools.get_angle_data(input_file, datasets, angle)
print(data_angle.shape)

mytools.remove_outliers(data_angle)
mytools.remove_stripes(data_angle)

display_data = mytools.scale_display_data(data_angle[distance])
plt.figure(figsize=(8, 8))
plt.imshow(display_data, cmap='viridis')
plt.colorbar()
plt.title("holo_data first frame")
plt.show()
