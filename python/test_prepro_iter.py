import numpy as np
import h5py
import matplotlib.pyplot as plt
import mytools
import hiholo

def display_image(phase, title="Phase"):
    """Display image"""
    plt.figure(figsize=(8, 8))
    plt.imshow(phase, cmap='gray')
    plt.colorbar()
    plt.title(title)
    plt.pause(3)
    plt.close()

# Input/output files
input_file = "/home/hug/Downloads/HoloTomo_Data/holo_200angles_simu_format.h5"
phase_file = "/home/hug/Downloads/HoloTomo_Data/holo_200angles_phase.h5"
#phase_file = None

datasets = "holodata_distance_0,holodata_distance_1,holodata_distance_2,holodata_distance_3"
back_dataset = "backgrounds"
dark_dataset = "darks"
phase_dataset = "phasedata"

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
isAPWP = True
probe_data = np.array([])
holo_data, probe_data = mytools.dark_flat_correction(holo_data, dark_data, back_data, isAPWP)

# The fourth interface (optional)
# holo_data, translations = mytools.register_images(holo_data)

fresnel_numbers = [[1.6667e-3], [8.3333e-4], [4.83333e-4], [2.66667e-4]]

# Algorithm selection (0:AP, 1:RAAR, 2:HIO, 3:DRAP, 4:APWP, 5:EPI, 100:CTF)
algorithm = hiholo.Algorithm.APWP

# Padding
pad_size = [50, 50]
# Padding type (0:Constant, 1:Replicate, 2:Fadeout)
pad_type = hiholo.PaddingType.Replicate
pad_value = 0.0

output_file = "iter_result.h5"
output_dataset = "phasedata"

iterations = 200
plot_interval = 50

# Algorithm parameters
if algorithm == hiholo.Algorithm.RAAR:
    algo_params = [0.75, 0.99, 20]
else:
    algo_params = [0.7]

# Kernel type (0:Fourier, 1:Chirp, 2:ChirpLimited)
kernel_type = hiholo.PropKernelType.Fourier
# Projection type (0:Averaged, 1:Sequential, 2:Cyclic)
projection_type = hiholo.ProjectionType.Averaged

# Default Constraint Values
phase_limits = [-float('inf'), float('inf')]  # [min, max] phase
amp_limits = [0, float('inf')]  # [min, max] amplitude
support = []  # Support constraint region size
outside_value = 0.0  # Value outside support region

init_phase = np.array([])
if phase_file is not None and phase_dataset is not None:
    init_phase = mytools.read_3d_data_frame(phase_file, phase_dataset, angle)
    #display_image(init_phase)

calc_error = True

result = None
residuals = [[], []] if calc_error else None
init_amplitude = np.array([])
probe_phase = np.array([])

 # Perform reconstruction in intervals
for i in range(iterations // plot_interval):
    if algorithm == hiholo.Algorithm.EPI:
        result = hiholo.reconstruct_epi(
            holograms=holo_data,                    
            fresnelNumbers=fresnel_numbers,
            iterations=plot_interval,
            initialPhase=init_phase,       
            initialAmplitude=init_amplitude,          
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
        
        init_phase = result[0]
        init_amplitude = result[1]

        if calc_error:
            residuals[0].extend(result[2].tolist())
            residuals[1].extend(result[3].tolist())
        
    else:            
        result = hiholo.reconstruct_iter( 
            holograms=holo_data,                    
            fresnelNumbers=fresnel_numbers,
            iterations=plot_interval,
            initialPhase=init_phase,
            initialAmplitude=init_amplitude,
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
            holoProbes=probe_data,                 
            initProbePhase=probe_phase,
            calcError=calc_error
        )
        
        init_phase = result[0]
        init_amplitude = result[1]
        
        if algorithm == hiholo.Algorithm.APWP:
            probe_phase = result[2]
        
        if calc_error:
            residuals[0].extend(result[3].tolist())
            residuals[1].extend(result[4].tolist())
        
        #display_image(result[0], f"Phase reconstructed by {(i+1)*plot_interval} iterations")

mytools.save_h5_from_float(output_file, output_dataset, result[0])
if algorithm == hiholo.Algorithm.APWP:
    mytools.append_h5_from_float(output_file, "probe_" + output_dataset, result[2])
