"""
Test script for hiholo with improved Python API.

新的 reconstruct_iter API 返回值结构：
result = [phase_2d, amplitude_2d, probe_phase_2d, step_errors_1d, pm_errors_1d]
- result[0]: phase (2D numpy array)
- result[1]: amplitude (2D numpy array) 
- result[2]: probe_phase (2D numpy array, 仅APWP算法)
- result[3]: step_errors (1D numpy array, 当calcError=True时)
- result[4]: pm_errors (1D numpy array, 当calcError=True时)

新的 reconstruct_epi API 返回值结构：
result = [phase_2d, amplitude_2d, step_errors_1d, pm_errors_1d]
- result[0]: phase (2D numpy array, 包含padding的测量尺寸)
- result[1]: amplitude (2D numpy array, 包含padding的测量尺寸)
- result[2]: step_errors (1D numpy array, 当calcError=True时)
- result[3]: pm_errors (1D numpy array, 当calcError=True时)
"""

import numpy as np
import h5py
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import hiholo
import mytools

def display_image(phase, title="Phase", cmap='gray'):
    """Display image"""
    plt.figure(figsize=(8, 8))
    plt.imshow(phase, cmap=cmap)
    plt.colorbar()
    plt.title(title)
    plt.pause(1)
    plt.close()

def test_reconstruction():
    """Test holographic reconstruction with hiholo"""
    
    #############################################################
    # Parameters (modify this section)
    #############################################################
    holo_file = "/home/hug/Downloads/data/logo/holo_obj.h5"
    probe_file = "/home/hug/Downloads/data/logo/holo_probe.h5"
    fn_file = "/home/hug/Downloads/data/logo/FN.h5"
    
    holo_dataset = "holo_data_p0_flatcorr_org"
    probe_dataset = "probe_data_p0_flatcorr_org"
    fn_dataset = "FN"
    #input_file = "/home/hug/Downloads/HoloTomo_Data/dog_cat_dataset/only_phase/holo_probewithobj3.h5"
    #input_file = "/home/hug/Downloads/HoloTomo_Data/holo_data.h5"
    #input_file = "/home/hug/Downloads/data/second/holo_obj.h5"
    
    #input_dataset = "holodata"
    #input_dataset = "holo_obj"
    #input_dataset = "hologramCTF_objwithprobe"
    output_dataset = "phasedata"
    
    # List of fresnel numbers
    #fresnel_numbers=[[5.56e-4]]
    #fresnel_numbers = [[0.0126], [0.00725], [0.00426], [0.00886]]
    
    # Reconstruction parameters
    iterations = 300            # Number of iterations
    plot_interval = 300         # Interval for displaying results
    
    # Initial guess (optional)
    #initial_phase_file = "/home/hug/Downloads/HoloTomo_Data/purephase_ctf_result.h5"
    #initial_phase_dataset = "phasedata"

    initial_phase_file = None
    initial_phase_dataset = None
    
    # Algorithm selection (0:AP, 1:RAAR, 2:HIO, 3:DRAP, 4:APWP, 5:EPI)
    algorithm = hiholo.Algorithm.APWP
    
    # Algorithm parameters
    if algorithm == hiholo.Algorithm.RAAR:
        algo_params = [0.75, 0.99, 20]
    else:
        algo_params = [0.7]
    
    # Constraints
    amp_limits = [1, 1]  # [min, max] amplitude
    phase_limits = [-float('inf'), float('inf')]  # [min, max] phase
    support = []  # Support constraint region size
    outside_value = 0.0  # Value outside support region
    
    # Padding
    pad_size = [300, 300]  # Padding size
    pad_type = hiholo.PaddingType.Replicate
    pad_value = 1.0
    
    # Probe parameters (for APWP algorithm)
    #probe_file = "/home/hug/Downloads/HoloTomo_Data/dog_cat_dataset/only_phase/holo_probe2.h5"
    #probe_file = "/home/hug/Downloads/HoloTomo_Data/probe_data.h5"
    #probe_file = "/home/hug/Downloads/data/second/holo_probe.h5"
    #probe_dataset = "hologramCTF_probe"
    #probe_dataset = "holodata"
    #probe_dataset = "holo_probe"
    probe_phase_file = None
    probe_phase_dataset = None
    
    # Projection type, Kernel method, Error calculation
    projection_type = hiholo.ProjectionType.Averaged
    kernel_type = hiholo.PropKernelType.Fourier
    
    # Error calculation
    calc_error = False
    
    #############################################################
    # End of parameters section
    #############################################################
    holo_temp = mytools.read_h5_to_float(holo_file, holo_dataset)
    probe_temp = mytools.read_h5_to_float(probe_file, probe_dataset)

    fresnel_number = mytools.read_h5_to_float(fn_file, fn_dataset)[0][0]    
    fresnel_numbers = [[fresnel_number]]
    print(f"Using {len(fresnel_numbers)} fresnel numbers: {fresnel_numbers}")

    holo_data = holo_temp
    print(f"Loaded hologram of size {holo_data.shape}")

    # holo_data = holo_data / holo_data.max()
    # display_image(holo_data, "Hologram")
    plt.imsave("holodata.png", holo_data, cmap='viridis')
    # save_image_with_colorbar(holo_data[3], "holodata.png", cmap='gray', display_range=None)
    # display_image(probe_data, "Probe")
    plt.imsave("probeholo.png", probe_temp, cmap='viridis')

    # Read initial phase if provided
    initial_phase_array = np.array([])

    # Read probe grams if provided
    probe_array = probe_temp
    probe_phase_array = np.array([])

    #probe_array = probe_data

    # Output algorithm info
    algorithm_names = {
        hiholo.Algorithm.AP: "AP",
        hiholo.Algorithm.RAAR: "RAAR",
        hiholo.Algorithm.HIO: "HIO",
        hiholo.Algorithm.DRAP: "DRAP",
        hiholo.Algorithm.APWP: "APWP",
        hiholo.Algorithm.EPI: "EPI"
    }
    print(f"Using algorithm: {algorithm_names.get(algorithm, 'Unknown')}")
    
    # Initialize results storage
    result = None
    residuals = [[], []] if calc_error else None
    
    initial_amplitude_array = np.array([])
    # Perform reconstruction in intervals
    for i in range(iterations // plot_interval):
        if algorithm == hiholo.Algorithm.EPI:
            result = hiholo.reconstruct_epi(
                holograms=holo_data,                    
                fresnelNumbers=fresnel_numbers,
                iterations=plot_interval,
                initialPhase=initial_phase_array,       
                initialAmplitude=initial_amplitude_array,          
                minPhase=phase_limits[0],
                maxPhase=phase_limits[1],
                minAmplitude=amp_limits[0],
                maxAmplitude=amp_limits[1],
                support=support,
                outsideValue=outside_value,
                padSize=pad_size,                       
                projectionType=projection_type,
                kernelType=kernel_type,
                calcError=calc_error
            )
            
            # result现在是2D numpy数组的列表：[phase, amplitude, step_errors?, pm_errors?]
            initial_phase_array = result[0]        
            initial_amplitude_array = result[1]

            if calc_error:
                residuals[0].extend(result[2].tolist())
                residuals[1].extend(result[3].tolist())
            
            display_image(result[0], f"Phase reconstructed by {(i+1)*plot_interval} iterations")
        else:            
            # New iterative reconstruction API
            result = hiholo.reconstruct_iter( 
                holograms=holo_data,                    
                fresnelNumbers=fresnel_numbers,
                iterations=plot_interval,
                initialPhase=initial_phase_array,
                initialAmplitude=initial_amplitude_array,
                algorithm=algorithm,
                algoParameters=algo_params,
                minPhase=phase_limits[0],
                maxPhase=phase_limits[1], 
                minAmplitude=amp_limits[0],
                maxAmplitude=amp_limits[1],
                support=support,
                outsideValue=outside_value,
                padSize=pad_size,
                padType=pad_type,
                padValue=pad_value,
                projectionType=projection_type,
                kernelType=kernel_type,
                holoProbes=probe_array,                 
                initProbePhase=probe_phase_array,       
                calcError=calc_error
            )
            
            # result现在是2D numpy数组的列表：[phase, amplitude, ...]
            initial_phase_array = result[0]
            initial_amplitude_array = result[1]

            if algorithm == hiholo.Algorithm.APWP:
                probe_phase_array = result[2]
            
            if calc_error:
                residuals[0].extend(result[3].tolist())
                residuals[1].extend(result[4].tolist())
            
            display_image(result[0], f"Phase reconstructed by {(i+1)*plot_interval} iterations")
    
    # Display error if calculated
    if calc_error:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(residuals[0])
        plt.title("Step Error")
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(residuals[1])
        plt.title("PM Error")
        plt.grid(True)
        plt.tight_layout()
        plt.pause(3)
        plt.close()
    
    # Save images
    #save_image_with_colorbar(result[0], "phase_with_cb.png", cmap='viridis')
    plt.imsave(algorithm_names.get(algorithm, 'Unknown') + "_phase.png", result[0], cmap='viridis')
    #plt.imsave("amplitude.png", result[1], cmap='viridis')
    if algorithm == hiholo.Algorithm.APWP:
        plt.imsave(algorithm_names.get(algorithm, 'Unknown') + "_probe_phase.png", result[2], cmap='gray')
        
    # Save reconstructed holograms
    with h5py.File("obj_" + algorithm_names.get(algorithm, 'Unknown') + ".h5", 'w') as f:
        f.create_dataset(output_dataset, data=result[0], dtype=np.float32)

    with h5py.File("probe_" + algorithm_names.get(algorithm, 'Unknown') + ".h5", 'w') as f:
        f.create_dataset(output_dataset, data=result[2], dtype=np.float32)
        
    #output_phase_tiff = "recons_phase.tiff"
    #mytools.save_tiff_from_float(output_phase_tiff, result[0])
    
    #data = mytools.read_tiff_to_float(output_phase_tiff)
    #display_image(data, "Phase reconstructed from TIFF")

if __name__ == "__main__":
    test_reconstruction()