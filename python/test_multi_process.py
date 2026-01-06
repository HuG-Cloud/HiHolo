import mytools
import hiholo
import matplotlib.pyplot as plt
import time
import multiprocessing

def display_image(phase, title="Phase"):
    """Display image"""
    plt.figure(figsize=(8, 8))
    plt.imshow(phase, cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.pause(3)
    plt.close()

input_file = "/home/hug/Downloads/HoloTomo_Data/holo_200angles_simu_format.h5"
datasets = "holodata_distance_0,holodata_distance_1,holodata_distance_2,holodata_distance_3"
processed_output_file = "/home/hug/Downloads/HoloTomo_Data/processed_data.h5"
processed_output_dataset = "holodata"
back_dataset = "backgrounds"
dark_dataset = "darks"

distances = 4
angles = 200
im_size = [500, 500]

batch_size = 50
kernel_size = 5
threshold = 2.0
range_value = 0
window_size = 5
method = "mul"


def preprocess():
    back_data = mytools.read_h5_to_float(input_file, back_dataset)
    dark_data = mytools.read_h5_to_float(input_file, dark_dataset)
    back_data = mytools.remove_outliers(back_data, kernel_size, threshold)
    dark_data = mytools.remove_outliers(dark_data, kernel_size, threshold)

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
    
    #angle = 127
    #distance = 1
    #processed_frame = mytools.read_4d_data_frame(processed_file, processed_dataset, angle, distance)

def ctf_reconstruction_worker(processed_file,
                              processed_dataset,
                              distances_value,
                              angles_value,
                              size_value,
                              batch_size_recon,
                              fresnel_numbers,
                              pad_size,
                              pad_type,
                              pad_value,
                              low_freq_lim,
                              high_freq_lim,
                              beta_delta_ratio,
                              output_file,
                              output_dataset):
    
    ctf_reconstructor = hiholo.CTFReconstructor(
        batchSize=batch_size_recon,
        images=distances_value,
        imSize=size_value,
        fresnelNumbers=fresnel_numbers,
        lowFreqLim=low_freq_lim,
        highFreqLim=high_freq_lim,
        ratio=beta_delta_ratio,
        padSize=pad_size,
        padType=pad_type,
        padValue=pad_value
    )

    mytools.create_h5_file_dataset(output_file, output_dataset,
                                   (angles_value, size_value[0], size_value[1]))

    for i in range(0, angles_value, batch_size_recon):
        data_batch = mytools.get_batch_recon_data(processed_file,
                                                  processed_dataset,
                                                  i, batch_size_recon)

        recon_batch = ctf_reconstructor.reconsBatch(data_batch)
        #time.sleep(3)
        #print(f"Saving batch {i} to {i + batch_size_recon}")
        mytools.save_3d_batch_data(output_file, output_dataset, recon_batch, i)


def main():
    #preprocess()
    batch_size_recon = 100
    fresnel_numbers = [[1.6667e-3], [8.3333e-4], [4.83333e-4], [2.66667e-4]]
    pad_size = [50, 50]
    pad_type = hiholo.PaddingType.Replicate
    pad_value = 0.0
    low_freq_lim = 1e-3
    high_freq_lim = 1e-1
    beta_delta_ratio = 0.1
    ctf_output_file = "/home/hug/Downloads/HoloTomo_Data/ctf_result.h5"
    ctf_output_dataset = "phasedata"
    p = multiprocessing.Process(
        target=ctf_reconstruction_worker,
        args=(
            processed_output_file,
            processed_output_dataset,
            distances,
            angles,
            im_size,
            batch_size_recon,
            fresnel_numbers,
            pad_size,
            pad_type,
            pad_value,
            low_freq_lim,
            high_freq_lim,
            beta_delta_ratio,
            ctf_output_file,
            ctf_output_dataset,
        ),
    )
    
    p.start()
    #p.terminate()
    p.join()

    angle = 12
    phase_frame = mytools.read_3d_data_frame(ctf_output_file, ctf_output_dataset, angle)
    display_image(phase_frame)

if __name__ == "__main__":
    main()
