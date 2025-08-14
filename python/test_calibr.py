import numpy as np
import h5py
import matplotlib.pyplot as plt
import hiholo

# Read holograms
input_file = "calibr_data.h5"
input_dataset = "holodata"
with h5py.File(input_file, 'r') as f:
    holo_data = np.array(f[input_dataset], dtype=np.float32)
direction = 0

# 调用函数
maxPSDs, frequencies, profiles = hiholo.computePSDs(holo_data, direction)
for i in range(len(maxPSDs)):
    print("{:.3e}".format(maxPSDs[i]), end=" ")
print()
print(np.array2string(np.array(profiles[0]), formatter={'float_kind':lambda x: "%.3e" % x}))
print(frequencies[0])
# maxPSDs: [val1, val2, ...]
# profiles: [[profile1_data], [profile2_data], ...]
# frequencies: [[freq1_data], [freq2_data], ...]