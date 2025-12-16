import matplotlib.pyplot as plt
import mytools
import hiholo

# Input/output files
input_file = "/home/hug/Downloads/HoloTomo_Data/holo_200angles_simu_format.h5"
datasets = "holodata_distance_0,holodata_distance_1,holodata_distance_2,holodata_distance_3"
processed_output_file = "/home/hug/Downloads/HoloTomo_Data/processed_data.h5"
processed_output_dataset = "holodata"
back_dataset = "backgrounds"
dark_dataset = "darks"

distances = 4
angles = 200

batch_size = 50
kernel_size = 5
threshold = 2.0
range_value = 0
window_size = 5
method = "mul"

back_data = mytools.read_h5_to_float(input_file, back_dataset)
dark_data = mytools.read_h5_to_float(input_file, dark_dataset)
back_data = mytools.remove_outliers(back_data, kernel_size, threshold)
dark_data = mytools.remove_outliers(dark_data, kernel_size, threshold)

im_size = [back_data.shape[1], back_data.shape[2]]

mytools.create_h5_file_dataset(processed_output_file, processed_output_dataset, 
                               (angles, distances, im_size[0], im_size[1]))

for i in range(0, angles, batch_size):
    data_batch = mytools.get_batch_raw_data(input_file, datasets, i, batch_size)
    print(f"Processing batch {i} to {i + batch_size}")
    data_batch = mytools.remove_outliers(data_batch, kernel_size, threshold)
    data_batch = mytools.remove_stripes(data_batch, range_value, range_value, window_size, method)
    
    for j in range(data_batch.shape[0]):
        data_batch[j], _ = mytools.dark_flat_correction(data_batch[j], dark_data, back_data)
        data_batch[j], _ = mytools.register_images(data_batch[j])
    
    print(f"Saving batch {i} to {i + batch_size}")
    mytools.save_4d_batch_data(processed_output_file, processed_output_dataset, data_batch, i)

batch_size_recon = 100
fresnel_numbers = [[1.6667e-3], [8.3333e-4], [4.83333e-4], [2.66667e-4]]
# Algorithm selection (0:AP, 1:RAAR, 2:HIO, 3:DRAP, 100:CTF)
algorithm = hiholo.Algorithm.CTF

# Padding
pad_size = [50, 50]
# Padding type (0:Constant, 1:Replicate, 2:Fadeout)
pad_type = hiholo.PaddingType.Replicate
pad_value = 0.0

low_freq_lim = 1e-3      # 低频正则化参数
high_freq_lim = 1e-1     # 高频正则化参数  
beta_delta_ratio = 0.1   # β/δ比值（吸收与相位偏移的比值）

ctf_output_file = "/home/hug/Downloads/HoloTomo_Data/ctf_result.h5"
ctf_output_dataset = "phasedata"

ctf_reconstructor = hiholo.CTFReconstructor(
    batchSize=batch_size_recon,
    images=distances,
    imSize=im_size,
    fresnelNumbers=fresnel_numbers,
    lowFreqLim=low_freq_lim,
    highFreqLim=high_freq_lim,
    ratio=beta_delta_ratio,
    padSize=pad_size,
    padType=pad_type,
    padValue=pad_value
)

mytools.create_h5_file_dataset(ctf_output_file, ctf_output_dataset,
                               (angles, im_size[0], im_size[1]))

for i in range(0, angles, batch_size_recon):
    data_batch = mytools.get_batch_recon_data(processed_output_file,
                                              processed_output_dataset,
                                              i, batch_size_recon)

    recon_batch = ctf_reconstructor.reconsBatch(data_batch)
    mytools.save_3d_batch_data(ctf_output_file, ctf_output_dataset, recon_batch, i)