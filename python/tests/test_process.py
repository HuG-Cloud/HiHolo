import numpy as np
import sys
import os

# Add the parent directory to Python path to import mytools
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mytools
import hiholo

# Input/output files
input_file1 = "/home/hug/Downloads/HoloTomo_Data/visiblelight/foot.h5"
input_file2 = "/home/hug/Downloads/HoloTomo_Data/visiblelight/foot_back.h5"
output_file1 = "/home/hug/Downloads/HoloTomo_Data/visiblelight/foot_holo.h5"
output_file2 = "/home/hug/Downloads/HoloTomo_Data/visiblelight/foot_back_holo.h5"
dataset_name = "holodata"

# Read holograms
holo_data = mytools.read_h5_to_float(input_file1, dataset_name)
holo_data = hiholo.removeOutliers(holo_data)
# holo_data = hiholo.removeStripes(holo_data)

back_data = mytools.read_h5_to_float(input_file2, dataset_name)
back_data = hiholo.removeOutliers(back_data)
# back_data = hiholo.removeStripes(back_data)

# holo_data = holo_data / back_data
holo_data = holo_data[672:3376, 0:2704]
back_data = back_data[672:3376, 0:2704]

# Save processed holograms
mytools.save_h5_from_float(output_file1, dataset_name, holo_data)
mytools.save_h5_from_float(output_file2, dataset_name, back_data)