import numpy as np
import sys
import os
import h5py
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mytools

def display_phase(phase, title="Phase"):
    """Display phase image"""
    plt.figure(figsize=(8, 8))
    plt.imshow(phase, cmap='gray_r')
    plt.colorbar()
    plt.title(title)
    plt.pause(3)
    plt.close()

input_file = "/home/hug/Downloads/HoloTomo_Data/star.h5"
input_dataset = "wfr_in20"

output_file = "/home/hug/Downloads/HoloTomo_Data/star_gr.h5"
dataset_name = "phasedata"

with h5py.File(input_file, 'r') as f:
    # 直接读取为numpy数组，保持原始维度
    image_data = np.array(f[input_dataset], dtype=np.complex128)
print(f"Loaded image of size {image_data.shape}")

# 计算相位
phase = np.angle(image_data)
# Center-crop to 1000×1000
h, w = phase.shape
top = (h - 927) // 2
left = (w - 927) // 2
phase = phase[top:top+927, left:left+927]


display_phase(phase, "Phase")
plt.imsave("star_phase.png", phase, cmap='gray_r')

# with h5py.File(output_file, 'w') as f:
#     f.create_dataset(dataset_name, data=phase, dtype=np.float32)