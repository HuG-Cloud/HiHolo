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

def save_image_with_colorbar(img, filename, cmap='gray', display_range=None):
    """
    Save an image with a colorbar, normalizing the image display but
    allowing a custom range for the colorbar labels.
    If display_range is None, no colorbar is added.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Normalize the entire image data to the [0, 1] range first
    min_val = np.min(img)
    max_val = np.max(img)
    
    if max_val == min_val:
        # Handle constant image case
        normalized_img = np.zeros_like(img)
    else:
        normalized_img = (img - min_val) / (max_val - min_val)

    # Display the normalized image with a fixed color range [0, 1]
    im = ax.imshow(normalized_img, cmap=cmap, vmin=0, vmax=1)
    
    # Hide axes (no width/height scales)
    ax.axis('off')
    
    if display_range is not None:
        # Colorbar aligned to image axis height
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.25)
        cbar = fig.colorbar(im, cax=cax)
        
        # Map normalized ticks to the actual display range
        ticks = np.linspace(0, 1, 6)
        tick_labels = np.linspace(display_range[0], display_range[1], 6)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{val:.1f}" for val in tick_labels])
        cbar.ax.tick_params(labelsize=16)

    # # Add a 2mm scale bar
    # # Pixel size in micrometers
    # pixel_size_um = 3.45
    # # Image dimensions
    # img_height, img_width = normalized_img.shape
    # # Scale bar length in micrometers (2mm = 2000um)
    # scale_bar_length_um = 2000
    # # Scale bar length in pixels
    # scale_bar_length_px = scale_bar_length_um / pixel_size_um

    # # Position the scale bar at the bottom-left
    # # Margin from the edge
    # margin_px = 50
    # # Scale bar dimensions (increased height)
    # scale_bar_height_px = 40
    # # Rectangle position (x, y)
    # rect_x = margin_px
    # rect_y = img_height - margin_px - scale_bar_height_px
    
    # # Create and add the scale bar rectangle
    # rect = patches.Rectangle((rect_x, rect_y), scale_bar_length_px, scale_bar_height_px, linewidth=1, edgecolor='white', facecolor='white')
    # ax.add_patch(rect)

    # # Add the scale bar label with larger font
    # ax.text(rect_x + scale_bar_length_px / 2, rect_y - 10, '2 mm', color='white', ha='center', va='bottom', fontsize=25)
    
    fig.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def test_reconstruction():
    """Test holographic reconstruction with hiholo"""
    
    #############################################################
    # Parameters (modify this section)
    #############################################################
    
    #input_file = "/home/hug/Downloads/HoloTomo_Data/holo_regist_new.h5"
    #input_file = "/home/hug/Downloads/HoloTomo_Data/holo_purephase.h5"
    #input_file = "/home/hug/Downloads/HoloTomo_Data/holopadw1.h5"
    input_file = "/home/hug/Downloads/HoloTomo_Data/dog_cat_dataset/only_phase/holo_probewithobj3.h5"
    #input_file = "/home/hug/Downloads/HoloTomo_Data/diff_1.tif"
    #input_file = "/home/hug/Downloads/HoloTomo_Data/holo_data.h5"
    #input_file = "/home/hug/Downloads/HoloTomo_Data/processed_data.h5"    
    #input_dataset = "holodata"
    #input_file = "/home/hug/Downloads/HoloTomo_Data/visiblelight/wing_holo.h5"
    input_dataset = "hologramCTF_objwithprobe"
    #output_file = "/home/hug/Downloads/HoloTomo_Data/visiblelight/board_result.h5"
    output_file = "/home/hug/Downloads/HoloTomo_Data/result.h5"
    output_dataset = "phasedata"
    
    # List of fresnel numbers
    #fresnel_numbers = [[1.6667e-3], [8.3333e-4], [4.83333e-4], [2.66667e-4]]
    #fresnel_numbers = [[2.906977e-4], [1.453488e-4], [8.4302325e-5], [4.651163e-5]]
    #fresnel_numbers = [[0.003], [0.0015], [0.00087], [0.00039], [0.000216]]
    #fresnel_numbers = [[0.00087]]
    fresnel_numbers = [[0.0126], [0.00725], [0.00426], [0.00886]]
    #fresnel_numbers = [[2.987e-4]]
    print(f"Using {len(fresnel_numbers)} fresnel numbers: {fresnel_numbers}")
    
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
    amp_limits = [0, float('inf')]  # [min, max] amplitude
    phase_limits = [-float('inf'), float('inf')]  # [min, max] phase
    support = [2048, 2048]  # Support constraint region size
    outside_value = 1.0  # Value outside support region
    
    # Padding
    pad_size = [250, 250]  # Padding size
    pad_type = hiholo.PaddingType.Replicate
    pad_value = 0.0
    
    # Probe parameters (for APWP algorithm)
    #probe_file = None
    probe_file = "/home/hug/Downloads/HoloTomo_Data/dog_cat_dataset/only_phase/holo_probe2.h5"
    #probe_file = "/home/hug/Downloads/HoloTomo_Data/probe_data.h5"
    probe_dataset = "hologramCTF_probe"
    #probe_dataset = "holodata"
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
    
    holo_data = mytools.read_h5_to_float(input_file, input_dataset)
    #probe_data = mytools.read_h5_to_float(probe_file, probe_dataset)
    print(f"Loaded hologram of size {holo_data.shape}")

    # holo_data = holo_data / holo_data.max()
    # display_image(holo_data, "Hologram")
    # plt.imsave("holodata.png", holo_data[0], cmap='gray')
    save_image_with_colorbar(holo_data[3], "holodata.png", cmap='gray', display_range=None)
    # display_image(probe_data, "Probe")
    # plt.imsave("probeholo.png", probe_data, cmap='gray')

    # Read initial phase if provided
    initial_phase_array = np.array([])
    if initial_phase_file is not None:
        initial_phase_array = mytools.read_h5_to_float(initial_phase_file,
                                                       initial_phase_dataset)

    # Read probe grams if provided
    probe_array = np.array([])
    probe_phase_array = np.array([])
    if algorithm == hiholo.Algorithm.APWP:
        if probe_file is not None:
            probe_array = mytools.read_h5_to_float(probe_file, probe_dataset)
            plt.imsave("probeholo_0.png", probe_array, cmap='gray')
            #probe_array = probe_array / probe_array.max()
        
        if probe_phase_file is not None:
            with h5py.File(probe_phase_file, 'r') as f:
                probe_phase_array = np.array(f[probe_phase_dataset], dtype=np.float32)
    
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
            
            #display_image(result[0], f"Amplitude reconstructed by {(i+1)*plot_interval} iterations")
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
    amplitude = result[1][250:2298, 250:2298]
    save_image_with_colorbar(amplitude, "amplitude.png", cmap='viridis', display_range=[0, 1])
    plt.imsave("phase.png", result[0], cmap='viridis')
    #plt.imsave("amplitude.png", result[1], cmap='viridis')
    if algorithm == hiholo.Algorithm.APWP:
        plt.imsave("probe_phase.png", result[2], cmap='viridis')
        
    # Save reconstructed holograms
    with h5py.File(output_file, 'w') as f:
        f.create_dataset(output_dataset, data=result[0], dtype=np.float32)

if __name__ == "__main__":
    test_reconstruction()