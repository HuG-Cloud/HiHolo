import mytools
import numpy as np
import time

def test_image_registration():
    """Test image registration functions using h5 files"""
    
    # Define file paths (following test_sitk.cpp pattern)
    input_path = "/home/hug/Downloads/HoloTomo_Data/holo_purephase_shift.h5"
    output_path = "/home/hug/Downloads/HoloTomo_Data/holo_purephase_regist.h5"
    dataset_name = "holodata"
    
    # Try to read from the main data path first
    holo_data = mytools.read_h5_to_float(input_path, dataset_name)
    
    print(f"Original holo_data shape: {holo_data.shape}")
    
    # Ensure data is in the right format (num_images, rows, cols)
    if len(holo_data.shape) == 3:
        num_images, rows, cols = holo_data.shape
        print(f"Data dimensions: {num_images} images of {rows}x{cols} pixels")
    else:
        print(f"Unexpected data shape: {holo_data.shape}")
        return
    
    # Test batch image registration
    print("\n=== Testing batch image registration ===")
    try:
        registered_data, translations = mytools.register_images(holo_data)
        
        print(f"Registered data shape: {registered_data.shape}")
        print("Translation parameters for each image:")
        for i, trans in enumerate(translations):
            print(f"  Image {i}: [{trans[0]}, {trans[1]}]")
        
        # Save registered results
        print(f"\nSaving registered data to: {output_path}")
        mytools.save_h5_from_float(output_path, dataset_name, registered_data)
        
    except Exception as e:
        print(f"Batch registration failed: {e}")
        import traceback
        traceback.print_exc()

def test_with_artificial_shifts():
    """Test registration with artificially shifted images using fixed translations"""
    print("\n=== Testing with artificial shifts ===")
    
    try:
        # Load base data
        input_path = "/home/hug/Downloads/HoloTomo_Data/holo_purephase.h5"
        output_path_shift = "/home/hug/Downloads/HoloTomo_Data/holo_shift_new.h5"
        output_path_regist = "/home/hug/Downloads/HoloTomo_Data/holo_regist_new.h5"
        dataset_name = "holodata"
        base_data = mytools.read_h5_to_float(input_path, dataset_name)
        
        if len(base_data.shape) == 3 and base_data.shape[0] >= 3:
            # Use only first 4 images for testing (like test_sitk.cpp)
            test_images = base_data[:4].copy()
            
            # Define known translations (same as in test_sitk.cpp)
            known_translations = [[0, 0], [7, 8], [8, 7], [-7, 7]]
            
            print("Applying known translations:")
            for i, trans in enumerate(known_translations):
                print(f"  Image {i}: [{trans[0]}, {trans[1]}]")
            print()
            
            # Apply fixed translations to create shifted test data
            shifted_data = mytools.apply_fixed_translations(test_images, known_translations)
            
            # Save shifted data for comparison
            mytools.save_h5_from_float(output_path_shift, dataset_name, shifted_data)
            
            # Test different registration modes
            start_time = time.time()
        
            # Perform registration
            registered_data, detected_translations = mytools.register_images(shifted_data)
            for i in range(len(detected_translations)):
                print(f"  Image {i}: [{detected_translations[i][0]:.3f}, {detected_translations[i][1]:.3f}]")
        
            elapsed_time = time.time() - start_time
            print(f"  Processing time: {elapsed_time:.2f} seconds")
    
            # Save registered results
            mytools.save_h5_from_float(output_path_regist, dataset_name, registered_data)
        
    except Exception as e:
        print(f"Artificial shift test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Starting image registration...")
    # test_image_registration()
    test_with_artificial_shifts()