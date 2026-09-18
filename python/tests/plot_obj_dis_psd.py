import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mytools
import numpy as np
import matplotlib.pyplot as plt

# 定义需要绘制的多个数据集信息，加入高级配色、线型(ls)、线宽(lw)和图层顺序(zorder)
# 根据新的文件命名规则：obj_0, obj_1, obj_2, obj_pnet, obj_gr
# 使用 LaTeX 格式渲染 Fresnel number，增加学术感
datasets = [
    {"file": "../recons_data_dist/obj_0.h5", "label": r"$F = 1.26 \times 10^{-2}$", "color": "tab:blue", "ls": "-", "lw": 2.0, "zorder": 1, "alpha": 0.8},
    {"file": "../recons_data_dist/obj_1.h5", "label": r"$F = 7.25 \times 10^{-3}$", "color": "tab:orange", "ls": "-", "lw": 2.0, "zorder": 2, "alpha": 0.8},
    {"file": "../recons_data_dist/obj_2.h5", "label": r"$F = 4.26 \times 10^{-3}$", "color": "tab:green", "ls": "-", "lw": 2.0, "zorder": 3, "alpha": 0.8},
    # PNet 是重点展示的方法，使用醒目的红色，线宽加粗，zorder 提高以显示在上方
    {"file": "../recons_data_dist/obj_3.h5", "label": r"$F = 8.86 \times 10^{-4}$", "color": "#e63946", "ls": "-", "lw": 2.0, "zorder": 4, "alpha": 1.0},
    # 原图作为基准线 (Ground Truth)
    {"file": "../recons_data_dist/obj_gr.h5", "label": "Probe", "color": "#000000", "ls": "-", "lw": 2.0, "zorder": 5, "alpha": 0.9}
]
dataset_name = "phasedata"

# 绘制平均 PSD 折线图
plt.figure(figsize=(8, 6))

for data in datasets:
    try:
        img = mytools.read_h5_to_float(data["file"], dataset_name)
        img = mytools.scale_display_data(img, 512)
    except Exception as e:
        print(f"Warning: Failed to read {data['file']}: {e}")
        continue

    # 计算 1D PSD
    mean_1d, _ = mytools.calculate_1d_psd(img)

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

# 动态调整 Y 轴
y_min, y_max = ax.get_ylim()
ax.set_ylim(y_min, y_max * 100)  # 保持纵坐标压缩留白
y_min, y_max = ax.get_ylim()
# 计算 Y 轴刻度的对数指数
y_exp_min = int(np.floor(np.log10(max(y_min, 1e-20))))
y_exp_max = int(np.ceil(np.log10(y_max)))
y_exp = np.arange(y_exp_min, y_exp_max + 1, 2)
# 强制移除可能与 "PSDs" 标签冲突的最上面的刻度
if len(y_exp) > 1 and (10.0 ** y_exp[-1]) > (y_max / 100):
    y_exp = y_exp[:-1]
ax.set_yticks(10.0 ** y_exp)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True, which="major", ls="--", alpha=0.35)
plt.legend(fontsize=14)

# 调整边距并保存
plt.tight_layout()
plt.savefig('psd_obj_dis_plot.png', dpi=300)
print("Plot saved to psd_obj_dis_plot.png")
plt.show()