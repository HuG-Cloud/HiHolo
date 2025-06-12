import numpy as np
import h5py
import matplotlib.pyplot as plt
import fastholo

def display_phase(phase, title="Phase", im_shape=None):
    """Display phase image"""
    # 如果phase是一维数据且提供了shape信息，将其重塑为二维
    if isinstance(phase, list) and im_shape is not None:
        phase = np.array(phase).reshape(im_shape)
    elif isinstance(phase, np.ndarray) and phase.ndim == 1 and im_shape is not None:
        phase = phase.reshape(im_shape)
        
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
    input_file = "/home/hug/Downloads/HoloTomo_Data/visiblelight/wing_holo.h5"
    input_dataset = "holodata"
    output_file = "/home/hug/Downloads/HoloTomo_Data/visiblelight/wing_result.h5"
    output_dataset = "phasedata"
    
    # List of fresnel numbers
    # fresnel_numbers = [[0.0016667], [0.00083333], [0.000483333], [0.000266667]]
    # fresnel_numbers = [[2.906977e-4], [1.453488e-4], [8.4302325e-5], [4.651163e-5]]
    fresnel_numbers = [[2.987065e-4]] # wing of dragonfly
    # 确保fresnel_numbers的格式正确
    print(f"Using {len(fresnel_numbers)} fresnel numbers: {fresnel_numbers}")
    
    # Reconstruction parameters
    iterations = 300            # Number of iterations
    plot_interval = 300          # Interval for displaying results
    
    # Initial guess (optional)
    # initial_phase_file = "/home/hug/Downloads/HoloTomo_Data/ctf_result.h5"
    # initial_phase_dataset = "phasedata"

    initial_phase_file = None
    initial_phase_dataset = None
    
    # Algorithm selection (0:AP, 1:RAAR, 2:HIO, 3:DRAP, 4:APWP, 5:BIPEPI)
    algorithm = fastholo.Algorithm.AP
    
    # Algorithm parameters
    if algorithm == fastholo.Algorithm.RAAR:
        algo_params = [0.75, 0.99, 20]
    else:
        algo_params = [0.7]
    
    # Constraints
    amp_limits = [0.0, float('inf')]  # [min, max] amplitude
    phase_limits = [-float('inf'), float('inf')]  # [min, max] phase
    support = []  # Support constraint region size
    outside_value = 0.0  # Value outside support region
    
    # Padding
    pad_size = [250, 250]  # Padding size
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
    calc_error = False
    
    #############################################################
    # End of parameters section
    #############################################################
    
    # Read holograms
    with h5py.File(input_file, 'r') as f:
        # 将多维数组转换为一维数组
        holo_data = np.array(f[input_dataset], dtype=np.float32)
        holograms = holo_data.flatten().tolist()
    
    num_holograms = 1
    im_size = [holo_data.shape[0], holo_data.shape[1]]

    print(f"Loaded {num_holograms} holograms of size {im_size}")
    print(f"Holograms type: {type(holograms)}, length: {len(holograms)}")
    
    # 检查holograms是否正确
    if len(holograms) != num_holograms * im_size[0] * im_size[1]:
        print(f"警告: holograms长度 ({len(holograms)}) 与预期的 ({num_holograms * im_size[0] * im_size[1]}) 不符!")
    
    # Read initial phase if provided
    initial_phase = []
    if initial_phase_file is not None:
        with h5py.File(initial_phase_file, 'r') as f:
            initial_phase_array = np.array(f[initial_phase_dataset], dtype=np.float32)
            # 确保是一维列表
            initial_phase = initial_phase_array.flatten().tolist()
    
    # Read probe grams if provided
    probe_grams = []
    init_probe_phase = []
    if algorithm == fastholo.Algorithm.APWP:
        if probe_file is not None:
            with h5py.File(probe_file, 'r') as f:
                probe_array = np.array(f[probe_dataset], dtype=np.float32)
                probe_grams = probe_array.flatten().tolist()
        
        if probe_phase_file is not None:
            with h5py.File(probe_phase_file, 'r') as f:
                probe_phase_array = np.array(f[probe_phase_dataset], dtype=np.float32)
                init_probe_phase = probe_phase_array.flatten().tolist()
    
    # Output algorithm info
    algorithm_names = {
        fastholo.Algorithm.AP: "AP",
        fastholo.Algorithm.RAAR: "RAAR",
        fastholo.Algorithm.HIO: "HIO",
        fastholo.Algorithm.DRAP: "DRAP",
        fastholo.Algorithm.APWP: "APWP",
        fastholo.Algorithm.BIPEPI: "BIPEPI"
    }
    print(f"Using algorithm: {algorithm_names.get(algorithm, 'Unknown')}")
    
    # Initialize results storage
    result = None
    residuals = [[], []] if calc_error else None
    
    # For BIPEPI algorithm
    initial_amplitude = []
    new_size = None
    if algorithm == fastholo.Algorithm.BIPEPI:
        if not pad_size:
            raise ValueError("Padding size is required for BIPEPI algorithm")
        new_size = [im_size[0] + 2 * pad_size[0], im_size[1] + 2 * pad_size[1]]
    
    # Perform reconstruction in intervals
    for i in range(iterations // plot_interval):
        if algorithm == fastholo.Algorithm.BIPEPI:
            result = fastholo.reconstruct_bipepi(
                holograms, num_holograms, im_size, fresnel_numbers, plot_interval, new_size,
                initial_phase, initial_amplitude, phase_limits[0], phase_limits[1], amp_limits[0],
                amp_limits[1], support, outside_value, projection_type, kernel_type, calc_error
            )
            
            initial_phase = result[0]
            initial_amplitude = result[1]
            
            if calc_error:
                residuals[0].extend(result[3])
                residuals[1].extend(result[4])
            
            display_phase(result[1], f"Amplitude reconstructed by {(i+1)*plot_interval} iterations", new_size)
        else:            
            result = fastholo.reconstruct_iter(
                holograms, num_holograms, im_size, fresnel_numbers, plot_interval, initial_phase,
                algorithm, algo_params, phase_limits[0], phase_limits[1], amp_limits[0], amp_limits[1],
                support, outside_value, pad_size, pad_type, pad_value, projection_type, kernel_type,
                probe_grams, init_probe_phase, calc_error
            )
            
            initial_phase = result[0]
            
            if algorithm == fastholo.Algorithm.APWP:
                init_probe_phase = result[2]
            
            if calc_error:
                residuals[0].extend(result[3])
                residuals[1].extend(result[4])
            
            display_phase(result[1], f"Amplitude reconstructed by {(i+1)*plot_interval} iterations", im_size)
    
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
    
    # Save results
    if algorithm == fastholo.Algorithm.BIPEPI:
        im_size = new_size
    
    # 确保结果是二维的
    phase_2d = np.array(result[0]).reshape(im_size)
    amplitude_2d = np.array(result[1]).reshape(im_size)
    
    # Save images
    plt.imsave("phase.png", phase_2d, cmap='viridis')
    plt.imsave("amplitude.png", amplitude_2d, cmap='viridis')
    
    # Save reconstructed holograms
    with h5py.File(output_file, 'w') as f:
        f.create_dataset(output_dataset, data=phase_2d, dtype=np.float32)

if __name__ == "__main__":
    test_reconstruction()
