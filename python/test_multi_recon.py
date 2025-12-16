import matplotlib.pyplot as plt
import mytools
import hiholo
import numpy as np

# Input/output files
processed_output_file = "/home/hug/Downloads/HoloTomo_Data/processed_data.h5"
processed_output_dataset = "holodata"
ite_output_file = "/home/hug/Downloads/HoloTomo_Data/ite_result.h5"
ite_output_dataset = "phasedata"

# phase_file = "/home/hug/Downloads/HoloTomo_Data/ctf_result.h5"
# phase_dataset = "phasedata"
phase_file = None
phase_dataset = None

distances = 4
angles = 200
im_size = [500, 500]

batch_size_recon = 100
fresnel_numbers = [[1.6667e-3], [8.3333e-4], [4.83333e-4], [2.66667e-4]]
# Algorithm selection (0:AP, 1:RAAR, 2:HIO, 3:DRAP)
algorithm = hiholo.Algorithm.RAAR

# Padding
pad_size = [50, 50]
# Padding type (0:Constant, 1:Replicate, 2:Fadeout)
pad_type = hiholo.PaddingType.Replicate
pad_value = 0.0

iterations = 200

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
phase_limits = [-float('inf'), float('inf')]
amp_limits = [0, float('inf')]
support = []
outside_value = 0.0

reconstructor = hiholo.Reconstructor(
    batchSize=batch_size_recon,
    images=distances,
    imSize=im_size,
    fresnelNumbers=fresnel_numbers,
    iter=iterations,
    algo=algorithm,
    algoParams=algo_params,
    minPhase=phase_limits[0],
    maxPhase=phase_limits[1],
    minAmplitude=amp_limits[0],
    maxAmplitude=amp_limits[1],
    support=support,
    outsideValue=outside_value,
    padSize=pad_size,
    padType=pad_type,
    padValue=pad_value,
    projType=projection_type,
    kernelType=kernel_type
)

mytools.create_h5_file_dataset(ite_output_file, ite_output_dataset,
                               (angles, im_size[0], im_size[1]))

for i in range(0, angles, batch_size_recon):
    data_batch = mytools.get_batch_recon_data(processed_output_file,
                                              processed_output_dataset,
                                              i, batch_size_recon)
    
    init_batch = np.array([])
    if phase_file is not None and phase_dataset is not None:
        init_batch = mytools.get_batch_recon_data(phase_file, phase_dataset,
                                                  i, batch_size_recon)
    
    print(f"Processing batch {i} to {i + batch_size_recon}")
    recon_batch = reconstructor.reconsBatch(data_batch, init_batch)
    print(f"Saving batch {i} to {i + batch_size_recon}")
    mytools.save_3d_batch_data(ite_output_file, ite_output_dataset, recon_batch, i)
