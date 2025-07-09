"""
Test script for fastholo with improved Python API.

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
import matplotlib.pyplot as plt
import fastholo

def display_image(phase, title="Phase"):
    """Display image"""
    plt.figure(figsize=(8, 8))
    plt.imshow(phase, cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.pause(3)
    plt.close()

def test_reconstruction():
    """Test holographic reconstruction with fastholo"""
    
    #############################################################
    # Parameters (modify this section)
    #############################################################
    
    # Input/output files
    #input_file = "/home/hug/Downloads/HoloTomo_Data/visiblelight/wing_holo.h5"
    input_file = "/home/hug/Downloads/HoloTomo_Data/holo_data.h5"
    input_dataset = "holodata"
    #output_file = "/home/hug/Downloads/HoloTomo_Data/visiblelight/wing_result.h5"
    output_file = "/home/hug/Downloads/HoloTomo_Data/result.h5"
    output_dataset = "phasedata"
    
    # List of fresnel numbers
    #fresnel_numbers = [[0.0016667], [0.00083333], [0.000483333], [0.000266667]]
    # fresnel_numbers = [[2.906977e-4], [1.453488e-4], [8.4302325e-5], [4.651163e-5]]
    fresnel_numbers = [[0.003], [0.0015], [0.00087], [0.00039], [0.000216]]
    # fresnel_numbers = [[2.987065e-4]] # wing of dragonfly
    # 确保fresnel_numbers的格式正确
    print(f"Using {len(fresnel_numbers)} fresnel numbers: {fresnel_numbers}")
    
    # Reconstruction parameters
    iterations = 300            # Number of iterations
    plot_interval = 300          # Interval for displaying results
    
    # Initial guess (optional)
    #initial_phase_file = "/home/hug/Downloads/HoloTomo_Data/purephase_ctf_result.h5"
    #initial_phase_dataset = "phasedata"

    initial_phase_file = None
    initial_phase_dataset = None
    
    # Algorithm selection (0:AP, 1:RAAR, 2:HIO, 3:DRAP, 4:APWP, 5:EPI)
    algorithm = fastholo.Algorithm.APWP
    
    # Algorithm parameters
    if algorithm == fastholo.Algorithm.RAAR:
        algo_params = [0.75, 0.99, 20]
    else:
        algo_params = [0.7]
    
    # Constraints
    amp_limits = [0.0, float('inf')]  # [min, max] amplitude
    phase_limits = [-float('inf'), float('inf')]  # [min, max] phase
    support = []  # Support constraint region size
    outside_value = 1.0  # Value outside support region
    
    # Padding
    pad_size = [100, 100]  # Padding size
    pad_type = fastholo.PaddingType.Replicate
    pad_value = 0.0
    
    # Probe parameters (for APWP algorithm)
    probe_file = "/home/hug/Downloads/HoloTomo_Data/probe_data.h5"
    probe_dataset = "holodata"
    probe_phase_file = None
    probe_phase_dataset = None
    
    # Projection type, Kernel method, Error calculation
    projection_type = fastholo.ProjectionType.Averaged
    kernel_type = fastholo.PropKernelType.Fourier
    
    # Error calculation
    calc_error = True
    
    #############################################################
    # End of parameters section
    #############################################################
    
    # Read holograms
    with h5py.File(input_file, 'r') as f:
        holo_data = np.array(f[input_dataset], dtype=np.float32)
    
    print(f"Loaded hologram of size {holo_data.shape}")
    
    # Read initial phase if provided
    initial_phase_array = np.array([])
    if initial_phase_file is not None:
        with h5py.File(initial_phase_file, 'r') as f:
            initial_phase_array = np.array(f[initial_phase_dataset], dtype=np.float32)
    
    # Read probe grams if provided
    probe_array = np.array([])
    probe_phase_array = np.array([])
    if algorithm == fastholo.Algorithm.APWP:
        if probe_file is not None:
            with h5py.File(probe_file, 'r') as f:
                probe_array = np.array(f[probe_dataset], dtype=np.float32)
        
        if probe_phase_file is not None:
            with h5py.File(probe_phase_file, 'r') as f:
                probe_phase_array = np.array(f[probe_phase_dataset], dtype=np.float32)
    
    # Output algorithm info
    algorithm_names = {
        fastholo.Algorithm.AP: "AP",
        fastholo.Algorithm.RAAR: "RAAR",
        fastholo.Algorithm.HIO: "HIO",
        fastholo.Algorithm.DRAP: "DRAP",
        fastholo.Algorithm.APWP: "APWP",
        fastholo.Algorithm.EPI: "EPI"
    }
    print(f"Using algorithm: {algorithm_names.get(algorithm, 'Unknown')}")
    
    # Initialize results storage
    result = None
    residuals = [[], []] if calc_error else None
    
    initial_amplitude_array = np.array([])
    # Perform reconstruction in intervals
    for i in range(iterations // plot_interval):
        if algorithm == fastholo.Algorithm.EPI:
            result = fastholo.reconstruct_epi(
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
            # 注意：EPI的phase和amplitude尺寸包含padding，比原始图像大
            initial_phase_array = result[0]        
            initial_amplitude_array = result[1]

            if calc_error:
                residuals[0].extend(result[2].tolist()) 
                residuals[1].extend(result[3].tolist())
            
            display_image(result[1], f"Amplitude reconstructed by {(i+1)*plot_interval} iterations")
        else:            
            # New iterative reconstruction API
            result = fastholo.reconstruct_iter( 
                holograms=holo_data,                    
                fresnelNumbers=fresnel_numbers,
                iterations=plot_interval,
                initialPhase=initial_phase_array,      
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
            
            if algorithm == fastholo.Algorithm.APWP:
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
    plt.imsave("phase.png", result[0], cmap='viridis')
    plt.imsave("amplitude.png", result[1], cmap='viridis')
        
    # Save reconstructed holograms
    with h5py.File(output_file, 'w') as f:
        f.create_dataset(output_dataset, data=result[1], dtype=np.float32)

if __name__ == "__main__":
    test_reconstruction()
