import numpy as np
import h5py
import sys
import os

# Add the parent directory to Python path to import mytools
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mytools
import hiholo


# Input/output files
# vinput_file1 = "/home/hug/Downloads/HoloTomo_Data/visiblelight/board.h5"
# input_file2 = "/home/hug/Downloads/HoloTomo_Data/visiblelight/board_back.h5"
# output_file1 = "/home/hug/Downloads/HoloTomo_Data/visiblelight/board_holo.h5"
# output_file2 = "/home/hug/Downloads/HoloTomo_Data/visiblelight/board_back_holo.h5"
size = 500

input_file1 = "/home/hug/Downloads/HoloTomo_Data/holo_data.h5"
input_file2 = "/home/hug/Downloads/HoloTomo_Data/probe_data.h5"
output_file1 = "/home/hug/Downloads/HoloTomo_Data/processed_holo.h5"
output_file2 = "/home/hug/Downloads/HoloTomo_Data/processed_probe.h5"

dataset_name = "holodata"

holo_data = mytools.read_h5_to_float(input_file1, dataset_name)
probe_data = mytools.read_h5_to_float(input_file2, dataset_name)
# mytools.remove_outliers(holo_data)
# mytools.remove_outliers(probe_data)

# processed = holo_data / probe_data
# mytools.remove_outliers(processed)
holo_temp = mytools.scale_display_data(holo_data[2], 500)
probe_temp = mytools.scale_display_data(probe_data[2], 500)
mytools.save_h5_from_float(output_file1, dataset_name, holo_temp)
mytools.save_h5_from_float(output_file2, dataset_name, probe_temp)
