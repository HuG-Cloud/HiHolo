import matplotlib.pyplot as plt
import mytools
import hiholo

# def display_image(phase, title="Phase"):
#     """Display image"""
#     plt.figure(figsize=(8, 8))
#     plt.imshow(phase, cmap='viridis')
#     plt.colorbar()
#     plt.title(title)
#     plt.pause(3)
#     plt.close()

# Input/output files
input_file = "/home/hug/Downloads/HoloTomo_Data/holo_200angles_simu_format.h5"
datasets = "holodata_distance_0,holodata_distance_1,holodata_distance_2,holodata_distance_3"
back_dataset = "backgrounds"
dark_dataset = "darks"

# The first interface (required)
angle = 127
holo_data = mytools.get_angle_data(input_file, datasets, angle)
back_data = mytools.read_h5_to_float(input_file, back_dataset)
dark_data = mytools.read_h5_to_float(input_file, dark_dataset)

# The second interface (optional)
kernel_size = 5
threshold = 2.0
range_value = 0
window_size = 5
method = "mul"

holo_data = mytools.remove_outliers(holo_data, kernel_size, threshold)
holo_data = mytools.remove_stripes(holo_data, range_value, range_value, window_size, method)
back_data = mytools.remove_outliers(back_data, kernel_size, threshold)
dark_data = mytools.remove_outliers(dark_data, kernel_size, threshold)

# The third interface (optional)
isAPWP = False
holo_data, probe_data = mytools.dark_flat_correction(holo_data, dark_data, back_data, isAPWP)

# The fourth interface (optional)
holo_data, translations = mytools.register_images(holo_data)

fresnel_numbers = [[1.6667e-3], [8.3333e-4], [4.83333e-4], [2.66667e-4]]
# Algorithm selection (0:AP, 1:RAAR, 2:HIO, 3:DRAP, 4:APWP, 5:EPI, 100:CTF)
algorithm = hiholo.Algorithm.CTF

# Padding
pad_size = [50, 50]
# Padding type (0:Constant, 1:Replicate, 2:Fadeout)
pad_type = hiholo.PaddingType.Replicate
pad_value = 0.0

low_freq_lim = 1e-3      # 低频正则化参数
high_freq_lim = 1e-1     # 高频正则化参数  
beta_delta_ratio = 0.1   # β/δ比值（吸收与相位偏移的比值）

output_file = "ctf_result.h5"
output_dataset = "phasedata"

result = hiholo.reconstruct_ctf(
    holograms=holo_data,
    fresnelNumbers=fresnel_numbers,
    lowFreqLim=low_freq_lim,
    highFreqLim=high_freq_lim,
    betaDeltaRatio=beta_delta_ratio,
    padSize=pad_size,
    padType=pad_type,
    padValue=pad_value
)

mytools.save_h5_from_float(output_file, output_dataset, result)