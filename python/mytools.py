import h5py
import numpy as np
from PIL import Image
import SimpleITK as sitk
import hiholo

def read_float_from_tiff(file_path):
    img = Image.open(file_path)
    # 支持 'F', 'I', 'I;16' 三种模式
    if img.mode == 'F':
        return np.array(img).astype(np.float32)
    elif img.mode == 'I;16':
        # I;16 需要先转成 np.uint16，再转 float32
        return np.array(img, dtype=np.uint16).astype(np.float32)
    else:
        raise ValueError(f"Input image mode is {img.mode}, not float!")

def read_h5_to_double(file_path, dataset_name=None):
    with h5py.File(file_path, 'r') as f:
        # If dataset_name is not provided, try to get the first dataset
        if dataset_name is None:
            keys = list(f.keys())
            if not keys:
                raise ValueError(f"HDF5 file {file_path} is empty, no datasets available")
            dataset_name = keys[0]
        
        # Read the dataset
        if dataset_name not in f:
            raise ValueError(f"Dataset '{dataset_name}' not found in HDF5 file")
        data = np.array(f[dataset_name], dtype=np.float64)
        
        # Ensure data is 3D
        if len(data.shape) != 3:
            raise ValueError(f"Data is not 3D. Actual dimensions: {data.shape}")
        
        return data
    
def read_h5_to_float(file_path, dataset_name=None):
    with h5py.File(file_path, 'r') as f:
        if dataset_name is None:
            keys = list(f.keys())
            if not keys:
                raise ValueError(f"HDF5 file {file_path} is empty, no datasets available")
            dataset_name = keys[0]
        
        data = np.array(f[dataset_name], dtype=np.float32)
        return data

def scale_display_data(data, max_size=1024):
    """
    If the length or width of the 2D data is greater than the threshold 1024, it is downsampled to 1024
    """
    if len(data.shape) != 2:
        raise ValueError(f"Data is not 2D. Actual dimensions: {data.shape}")
    h, w = data.shape
    if h > max_size or w > max_size:
        # Calculate the scaling factor
        scale_h = max_size / h if h > max_size else 1.0
        scale_w = max_size / w if w > max_size else 1.0
        scale = min(scale_h, scale_w)
        new_h = int(h * scale)
        new_w = int(w * scale)
        # Use SimpleITK for downsampling
        img = sitk.GetImageFromArray(data.astype(np.float32))
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize([new_w, new_h])
        orig_spacing = img.GetSpacing()
        orig_size = img.GetSize()
        new_spacing = [orig_spacing[0] * orig_size[0] / new_w, orig_spacing[1] * orig_size[1] / new_h]
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetInterpolator(sitk.sitkLinear)
        img_resampled = resampler.Execute(img)
        data_resampled = sitk.GetArrayFromImage(img_resampled)
        # SimpleITK outputs shape as (height, width), consistent with numpy
        return data_resampled
    else:
        return data

def create_h5_file_dataset(file_path, dataset_name, shape):
    with h5py.File(file_path, 'w') as f:
        f.create_dataset(dataset_name, shape=shape, dtype=np.float32)

def save_h5_from_float(file_path, dataset_name, data):
    with h5py.File(file_path, 'w') as f:
        f.create_dataset(dataset_name, data=data, dtype=np.float32)

def append_h5_from_float(file_path, dataset_name, data):
    with h5py.File(file_path, 'a') as f:
        f.create_dataset(dataset_name, data=data, dtype=np.float32)

def save_3d_batch_data(file_path, dataset_name, data, offset):
    with h5py.File(file_path, 'a') as f:
        if dataset_name not in f:
            raise ValueError(f"Dataset '{dataset_name}' not found in file '{file_path}'")
        dataset = f[dataset_name]
        if dataset.ndim != 3 or data.ndim != 3:
            raise ValueError(f"Target dataset or input data is not 3D!")
        dataset[offset:offset+data.shape[0], :, :] = data

# Append a 4D batch (data) into an existing 4D dataset along the first dimension
def save_4d_batch_data(file_path, dataset_name, data, offset):
    with h5py.File(file_path, 'a') as f:
        if dataset_name not in f:
            raise ValueError(f"Dataset '{dataset_name}' not found in file '{file_path}'")
        dataset = f[dataset_name]
        if dataset.ndim != 4 or data.ndim != 4:
            raise ValueError(f"Target dataset or input data is not 4D!")
        
        data_to_write = data.astype(dataset.dtype) if data.dtype != dataset.dtype else data
        dataset[offset:offset+data.shape[0], :, :, :] = data_to_write

def read_holodata_info(file_path, datasets):
    dataset_list = [ds.strip() for ds in datasets.split(',')]
    angle_list = []
    img_1 = None
    with h5py.File(file_path, 'r') as f:
        for i in range(len(dataset_list)):
            if dataset_list[i] not in f:
                raise ValueError(f"Dataset '{dataset_list[i]}' not found in HDF5 file")
            data = np.array(f[dataset_list[i]], dtype=np.float32)
            if i == 0:
                img_1 = data[0]
            angle_list.append(data.shape[0])

    # Check if the angles of all datasets is consistent
    if not all(a == angle_list[0] for a in angle_list):
        raise ValueError(f"Not all datasets have the same number of angles: {angle_list}")
    angles = angle_list[0]
                
    return angles, img_1

def read_3d_data_info(file_path, dataset, num_images):
    with h5py.File(file_path, 'r') as f:
        if dataset not in f:
            raise ValueError(f"Dataset '{dataset}' not found in HDF5 file")
        data = np.array(f[dataset], dtype=np.float32)
        if data.ndim != 3:
            raise ValueError(f"Data is not 3D. Actual dimensions: {data.shape}")
        if num_images != data.shape[0]:
            raise ValueError(f"Num_images {num_images} does not match the number of images {data.shape[0]}")
        return data[0]

def read_4d_data_first(file_path, dataset, num_angles, num_distances):
    with h5py.File(file_path, 'r') as f:
        if dataset not in f:
            raise ValueError(f"Dataset '{dataset}' not found in HDF5 file")
        data = np.array(f[dataset], dtype=np.float32)
        if data.ndim != 4:
            raise ValueError(f"Data is not 4D. Actual dimensions: {data.shape}")
        if num_angles != data.shape[0]:
            raise ValueError(f"Num_angles {num_angles} does not match the number of angles {data.shape[0]}")
        if num_distances != data.shape[1]:
            raise ValueError(f"Num_distances {num_distances} does not match the number of distances {data.shape[1]}")
        return data[0][0]

def read_holodata_frame(file_path, datasets, distance, angle):
    dataset_list = [ds.strip() for ds in datasets.split(',')]
    dataset = dataset_list[distance]
    with h5py.File(file_path, 'r') as f:
        if dataset not in f:
            raise ValueError(f"Dataset '{dataset}' not found in HDF5 file")
        data = np.array(f[dataset], dtype=np.float32)
        return data[angle]

def read_3d_data_frame(file_path, dataset, index):
    with h5py.File(file_path, 'r') as f:
        if dataset not in f:
            raise ValueError(f"Dataset '{dataset}' not found in HDF5 file")
        data = np.array(f[dataset], dtype=np.float32)
        if data.ndim != 3:
            raise ValueError(f"Data is not 3D. Actual dimensions: {data.shape}")
        return data[index]

def read_4d_data_frame(file_path, dataset, angle, distance):
    with h5py.File(file_path, 'r') as f:
        if dataset not in f:
            raise ValueError(f"Dataset '{dataset}' not found in HDF5 file")
        data = np.array(f[dataset], dtype=np.float32)
        if data.ndim != 4:
            raise ValueError(f"Data is not 4D. Actual dimensions: {data.shape}")
        return data[angle][distance]

def read_dark_data(file_path, dataset):
    return read_h5_to_float(file_path, dataset)[0]

def get_angle_data(file_path, datasets, angle):
    dataset_list = [ds.strip() for ds in datasets.split(',')]
    data_angle = []
    with h5py.File(file_path, 'r') as f:
        for i in range(len(dataset_list)):
            if dataset_list[i] not in f:
                raise ValueError(f"Dataset '{dataset_list[i]}' not found in HDF5 file")
            data = np.array(f[dataset_list[i]], dtype=np.float32)
            data_angle.append(data[angle])

    data_angle = np.stack(data_angle, axis=0)
    return data_angle

def get_batch_raw_data(file_path, datasets, start, batch_size):
    dataset_list = [ds.strip() for ds in datasets.split(',')]
    data_batch = []
    with h5py.File(file_path, 'r') as f:
        for i in range(len(dataset_list)):
            if dataset_list[i] not in f:
                raise ValueError(f"Dataset '{dataset_list[i]}' not found in HDF5 file")
            data = np.array(f[dataset_list[i]], dtype=np.float32)
            data_batch.append(data[start:start+batch_size])

    data_batch = np.stack(data_batch, axis=1)
    return data_batch

def get_batch_recon_data(file_path, dataset, start, batch_size):
    with h5py.File(file_path, 'r') as f:
        if dataset not in f:
            raise ValueError(f"Dataset '{dataset}' not found in HDF5 file")
        data = np.array(f[dataset], dtype=np.float32)
        return data[start:start+batch_size]

def remove_outliers(data, kernelSize=5, threshold=2.0):
    # Ensure data is 3D or 4D
    if len(data.shape) == 3:
        processed_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            processed_data[i] = hiholo.removeOutliers(data[i], kernelSize, threshold)
        return processed_data
    elif len(data.shape) == 4:
        processed_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                processed_data[i, j] = hiholo.removeOutliers(data[i, j], kernelSize, threshold)
        return processed_data
    else:
        raise ValueError(f"Data must be 3D or 4D. Actual dimensions: {data.shape}")

def remove_stripes(data, rangeRows=0, rangeCols=0, windowSize=5, method="mul"):
    # Ensure data is 3D or 4D
    if len(data.shape) == 3:
        processed_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            processed_data[i] = hiholo.removeStripes(data[i], rangeRows, rangeCols, windowSize, method)
        return processed_data
    elif len(data.shape) == 4:
        processed_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                processed_data[i, j] = hiholo.removeStripes(data[i, j], rangeRows, rangeCols, windowSize, method)
        return processed_data
    else:
        raise ValueError(f"Data must be 3D or 4D. Actual dimensions: {data.shape}")

def dark_flat_correction(data, dark, flat, isAPWP=False):
    if dark.shape[0] != data.shape[0]:
        dark = np.repeat(dark, data.shape[0], axis=0)
    if isAPWP:
        holo = data - dark
        probe = flat - dark
    else:
        holo = (data - dark) / (flat - dark)
        probe = np.array([])
    
    return holo, probe

def numpy_to_sitk_image(array):
    """Convert numpy array to SimpleITK image"""
    # SimpleITK expects (width, height) order, numpy uses (height, width)
    if len(array.shape) == 2:
        array = array.T
    
    image = sitk.GetImageFromArray(array.astype(np.float32))
    return image

def sitk_image_to_numpy(image):
    """Convert SimpleITK image to numpy array"""
    array = sitk.GetArrayFromImage(image)
    # Convert back to (height, width) for 2D
    if len(array.shape) == 2:
        array = array.T
    return array

def register_image(fixed_image, moving_image):
    """
    Register moving image to fixed image using SimpleITK with optimized parameters
    
    Args:
        fixed_image: SimpleITK Image or numpy array (reference image)
        moving_image: SimpleITK Image or numpy array (image to be registered)
        
    Returns:
        tuple: (registered_moving_image, translation_parameters)
               translation_parameters is [dx, dy] as floats (sub-pixel precision)
    """
    try:
        # Convert numpy arrays to SimpleITK images if needed
        if isinstance(fixed_image, np.ndarray):
            fixed_image = numpy_to_sitk_image(fixed_image)
        if isinstance(moving_image, np.ndarray):
            moving_image = numpy_to_sitk_image(moving_image)
        
        return _register_precise(fixed_image, moving_image)
        
    except Exception as e:
        print(f"Error registering image: {e}")
        return moving_image, [0.0, 0.0]

def _register_precise(fixed_image, moving_image):
    """Precise registration optimized for small displacements in hologram images"""
    
    # Preprocess images for better registration
    fixed_processed = _preprocess_for_registration(fixed_image)
    moving_processed = _preprocess_for_registration(moving_image)
    
    # Create registration method
    registration = sitk.ImageRegistrationMethod()
    
    # Use Mean Squares metric which is more sensitive to small changes
    registration.SetMetricAsMeanSquares()
    
    # Use Regular Step Gradient Descent for more precise control
    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=0.5,
        minStep=1e-4,
        numberOfIterations=300,
        gradientMagnitudeTolerance=1e-6
    )
    
    # Set optimizer scales for translation (equal weight for both directions)
    registration.SetOptimizerScalesFromPhysicalShift()
    
    # Set initial translation transform
    initial_transform = sitk.TranslationTransform(fixed_image.GetDimension())
    registration.SetInitialTransform(initial_transform, inPlace=False)
    
    # Use B-spline interpolation for sub-pixel accuracy
    registration.SetInterpolator(sitk.sitkBSpline)
    
    # Multi-level framework for better convergence
    registration.SetShrinkFactorsPerLevel([2, 1])
    registration.SetSmoothingSigmasPerLevel([1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    
    transform = registration.Execute(fixed_processed, moving_processed)
    parameters = list(transform.GetParameters())
    
    # Apply transform with high-quality interpolation
    registered_moving = _apply_transform_with_padding(moving_image, transform, fixed_image)
    
    return registered_moving, parameters

def _preprocess_for_registration(image):
    """Preprocess image to enhance registration quality"""
    # Apply slight Gaussian smoothing to reduce noise
    smoothed = sitk.SmoothingRecursiveGaussian(image, sigma=0.5)
    
    # Enhance edges using gradient magnitude
    gradient = sitk.GradientMagnitude(smoothed)
    
    # Combine original and gradient information
    enhanced = sitk.Add(sitk.Multiply(smoothed, 0.7), sitk.Multiply(gradient, 0.3))
    
    return enhanced

def register_images(data):
    """
    Register multiple images with the first image as reference
    
    Args:
        data: numpy array of shape (num_images, rows, cols)
        
    Returns:
        tuple: (registered_data, translations)
               registered_data: numpy array with registered images
               translations: list of [dx, dy] translation parameters for each image
    """
    try:
        num_images = data.shape[0]
                
        # Convert to list of SimpleITK images
        holo_images = []
        for i in range(num_images):
            image = numpy_to_sitk_image(data[i])
            holo_images.append(image)
        
        # Initialize translations array
        translations = [[0.0, 0.0] for _ in range(num_images)]
        
        # Register each image to the first one
        for i in range(1, num_images):
            registered_img, translation = register_image(holo_images[0], holo_images[i])
            holo_images[i] = registered_img
            translations[i] = translation
        
        # Convert back to numpy array
        registered_data = np.zeros_like(data)
        for i in range(num_images):
            registered_data[i] = sitk_image_to_numpy(holo_images[i])
        
        return registered_data, translations
        
    except Exception as e:
        print(f"Error registering images: {e}")
        return data, [[0.0, 0.0] for _ in range(num_images)]

def _apply_transform_with_padding(moving_image, transform, reference_image):
    """Apply transform with proper padding and extraction
    Uses integer pixel shifts to preserve original pixel values
    """
    
    # Get original sub-pixel parameters for padding calculation
    parameters = transform.GetParameters()
    
    pad_bound = [max(1, int(np.ceil(abs(parameters[0])))), 
                 max(1, int(np.ceil(abs(parameters[1]))))]    
    padded_moving = sitk.ZeroFluxNeumannPad(moving_image, pad_bound, pad_bound)
    
    # Apply transform with nearest neighbor to preserve pixel values
    resampled = sitk.Resample(
        padded_moving,
        transform,
        sitk.sitkBSpline,
        0.0,
        padded_moving.GetPixelID()
    )
    
    # Extract the region of interest
    registered_moving = sitk.Extract(resampled, reference_image.GetSize(), pad_bound)
    
    return registered_moving

def apply_fixed_translations(data, translations):
    """
    Apply fixed translation values to images with sub-pixel precision
    Each image uses its own padding based on its translation
    
    Args:
        data: numpy array of shape (num_images, rows, cols)
        translations: list of [dx, dy] translation parameters for each image
                      first image translation should be [0, 0]
        
    Returns:
        numpy array: shifted images with same shape as input
    """
    try:
        if len(data.shape) != 3:
            raise ValueError(f"Data must be 3D array, got shape: {data.shape}")
        
        num_images, rows, cols = data.shape
        
        if len(translations) != num_images:
            raise ValueError(f"Number of translations ({len(translations)}) must match number of images ({num_images})")
        
        # Convert to SimpleITK images and process
        result_data = np.copy(data)
        original_size = [cols, rows]  # SimpleITK uses [width, height]
        
        for i in range(1, num_images):  # Skip first image (reference)
            dx, dy = translations[i]            
            # Skip if no movement needed
            if dx == 0.0 and dy == 0.0:
                print(f"    No movement needed, keeping original")
                continue
            
            # Convert to SimpleITK image
            sitk_image = numpy_to_sitk_image(data[i])
            
            # Calculate padding needed for displacement (ensure enough padding)
            pad_bound = [max(1, int(np.ceil(abs(dx)))), 
                         max(1, int(np.ceil(abs(dy))))]
            padded_image = sitk.ZeroFluxNeumannPad(sitk_image, pad_bound, pad_bound)
            
            # Create translation transform with sub-pixel parameters
            transform = sitk.TranslationTransform(2)
            transform.SetParameters([dx, dy])
            
            # Apply transform with B-spline interpolation for sub-pixel accuracy
            resampled = sitk.Resample(
                padded_image, 
                transform, 
                sitk.sitkBSpline,
                0.0,
                padded_image.GetPixelID()
            )
            
            # Extract original size region
            extracted = sitk.Extract(resampled, original_size, pad_bound)
            
            # Convert back to numpy
            result_data[i] = sitk_image_to_numpy(extracted)
        
        return result_data
        
    except Exception as e:
        print(f"Error applying fixed translations: {e}")
        import traceback
        traceback.print_exc()
        return data