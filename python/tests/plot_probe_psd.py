import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mytools
import numpy as np
import matplotlib.pyplot as plt

# 定义需要绘制的多个数据集信息，使用与 obj 类似的配置，但文件和标签替换为 probe
datasets = [
    {"file": "../recons_data_1/probe_apwp.h5", "label": "APWP", "color": "tab:blue", "ls": "-", "lw": 2.0, "zorder": 1, "alpha": 0.8}, # 使用柔和的蓝色
    # PASNet 的 Probe
    {"file": "../recons_data_1/probe_pnet.h5", "label": "PASNet", "color": "#e63946", "ls": "-", "lw": 2.5, "zorder": 2, "alpha": 1.0}, # PASNet 线宽加粗
    {"file": "../recons_data_1/probe_gr.h5", "label": "Probe", "color": "#000000", "ls": "-", "lw": 2.0, "zorder": 3, "alpha": 0.9} # 使用黑色虚线作为基准
]
dataset_name = "phasedata"

# 绘制平均 PSD 折线图
plt.figure(figsize=(8, 6))

for data in datasets:
    try:
        # 注意: probe 数据如果 dataset 名字不同，需要在此修改。
        img = mytools.read_h5_to_float(data["file"], dataset_name)
        img = mytools.scale_display_data(img, 500)
    except Exception as e:
        print(f"Warning: Failed to read {data['file']}: {e}")
        continue

    # 计算 1D PSD
    mean_1d, sum_1d = mytools.calculate_1d_psd(img)

    # 频率坐标 (假设频率从 0 到 max_radius)
    # 在数字图像中，最高频率对应于 0.5 cycles/pixel (Nyquist frequency)
    max_radius = len(mean_1d)
    freqs = np.linspace(0, 0.5, max_radius)

    # 使用双对数坐标 (log-log) 进行绘制，这是标准的光学/信号 PSD 画法
    # 忽略索引为 0 的直流分量 (0 频率)，避免对数运算报错
    plt.loglog(
        freqs[1:], mean_1d[1:], 
        color=data.get("color", "black"), 
        linestyle=data.get("ls", "-"), 
        linewidth=data.get("lw", 2), 
        alpha=data.get("alpha", 1.0),
        zorder=data.get("zorder", 1),
        label=data["label"]
    )

# 设置坐标轴范围与参考图类似 (可选)
# plt.xlim(1e-3, 1e0)
# plt.ylim(1e-3, 1e10)

plt.xlabel(r'$\nu/(1/pixel)$', fontsize=18, ha='right')
plt.ylabel('PSDs', fontsize=18, rotation=0, ha='right', va='center')
ax = plt.gca()
ax.xaxis.set_label_coords(1.0, -0.06)
ax.yaxis.set_label_coords(-0.01, 0.95)  # 将PSDs标签向右移动，显得更紧凑
ax.set_xlim(8e-4, 7e-1)  # 增加左侧留白，避免曲线紧挨y轴
ax.set_xticks([1e-2, 1e-1])  # 移除10^-3的刻度，和最开始一致

# 动态调整 Y 轴，兼容 Probe 可能不同的量级
y_min, y_max = ax.get_ylim()
ax.set_ylim(y_min, y_max * 100)  # 保持纵坐标压缩留白
y_min, y_max = ax.get_ylim()
# 计算 Y 轴刻度的对数指数
y_exp_min = int(np.floor(np.log10(max(y_min, 1e-20))))
y_exp_max = int(np.ceil(np.log10(y_max)))
y_exp = np.arange(y_exp_min, y_exp_max + 1, 2)
# 强制移除可能与 "PSDs" 标签冲突的最上面的刻度
# 如果最后一个刻度比较接近顶端，就把它切掉
if len(y_exp) > 1 and (10.0 ** y_exp[-1]) > (y_max / 100):
    y_exp = y_exp[:-1]
ax.set_yticks(10.0 ** y_exp)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True, which="major", ls="--", alpha=0.35)
plt.legend(fontsize=14)

# 调整边距并保存
plt.tight_layout()
plt.savefig('psd_probe_plot.png', dpi=300)
print("Plot saved to psd_probe_plot.png")
plt.show()
