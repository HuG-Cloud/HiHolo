import hiholo
import matplotlib.pyplot as plt
import mytools

# Read holograms
input_file = "/home/hug/Downloads/HoloTomo_Data/holo_200angles_simu_format.h5"
# input_dataset = "holodata"
# holo_data = mytools.read_phasedata_frame(input_file, input_dataset, 0)
# print(holo_data.shape)

datasets = "holodata_distance_0,holodata_distance_1,holodata_distance_2,holodata_distance_3"
angles, img_1 = mytools.read_holodata_info(input_file, datasets)
print(angles)

img_2 = mytools.read_holodata_frame(input_file, datasets, 1, 127)

display_data = mytools.scale_display_data(img_2)
print(display_data.shape)

# 绘制 holo_data的图像
plt.figure(figsize=(8, 8))
plt.imshow(display_data, cmap='viridis')
plt.title("holo_data first frame")
plt.colorbar()
plt.show()
