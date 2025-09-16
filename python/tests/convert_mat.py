import numpy as np
import h5py

# 输入输出文件
input_file1 = "/home/hug/Downloads/HoloTomo_Data/purephase_result.h5"
input_file2 = "/home/hug/Downloads/HoloTomo_Data/holo_200angles_phase.h5"
dataset = "phasedata"
angles = 200  # 期望的角度数

# 读取2D数据
with h5py.File(input_file1, 'r') as f:
    phase_data = np.array(f[dataset], dtype=np.float32)  # 假设shape为(H, W)

# 检查原始数据shape
if phase_data.ndim != 2:
    raise ValueError(f"原始数据不是2D，实际shape为{phase_data.shape}")

# 将2D数据堆叠成3D，shape[0]=angles
phasedata_3d = np.stack([phase_data.copy() for _ in range(angles)], axis=0)  # shape=(angles, H, W)

# 写入新的h5文件
with h5py.File(input_file2, 'w') as f:
    f.create_dataset(dataset, data=phasedata_3d)

print(f"已在{input_file2}中写入数据集：{dataset}，形状为{phasedata_3d.shape}")
