import numpy as np
import mytools

# Read holograms
input_file = "/home/hug/Downloads/HoloTomo_Data/holo_200angles_simu_format.h5"
phase_file = "/home/hug/Downloads/HoloTomo_Data/ctf_200angles_result.h5"

holo_datasets = "holodata_distance_0,holodata_distance_1,holodata_distance_2,holodata_distance_3"
back_dataset = "backgrounds"
dark_dataset = "darks"
phase_dataset = "phasedata"

angles, holo_first = mytools.read_holodata_info(input_file, holo_datasets)
print(angles)

distance = 2
angle = 127
holo_frame = mytools.read_holodata_frame(input_file, holo_datasets, distance, angle)


distances = 4
distacne = 3
back_first = mytools.read_3d_data_info(input_file, back_dataset, distances)
back_frame = mytools.read_3d_data_frame(input_file, back_dataset, distance)

dark_data = mytools.read_dark_data(input_file, dark_dataset)
print(dark_data.shape)

angles = 200
angle = 127
phase_first = mytools.read_3d_data_info(phase_file, phase_dataset, angles)
phase_frame = mytools.read_3d_data_frame(phase_file, phase_dataset, angle)

processed_file = "/home/hug/Downloads/HoloTomo_Data/processed_data.h5"
processed_dataset = "holodata"
processed_first = mytools.read_4d_data_first(processed_file, processed_dataset, angles, distances)
angle = 127
distance = 1
processed_frame = mytools.read_4d_data_frame(processed_file, processed_dataset, angle, distance)

# k represents the number of times the array is rotated counterclockwise
image = np.rot90(processed_frame, k=1)
image = np.fliplr(image) # Flip left-right
image = np.flipud(image) # Flip up-down
display_data = mytools.scale_display_data(image)
# mytools.display_image(display_data, "processed_data frame")
