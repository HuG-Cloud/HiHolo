import cv2
import h5py
import numpy as np
import argparse
import os
import glob
from pathlib import Path

def convert_bmp_to_h5(input_path, output_path=None, dataset_name="holodata"):
    """将BMP图像转换为灰度图并保存为H5文件
    
    Args:
        input_path: 输入的BMP文件路径或包含BMP文件的文件夹
        output_path: 输出的H5文件路径或文件夹，若为None则使用输入路径的基础名
        dataset_name: H5文件中的数据集名称
    """
    # 检查输入路径是文件还是目录
    if os.path.isfile(input_path):
        # 单个文件处理
        file_paths = [input_path]
    elif os.path.isdir(input_path):
        # 目录中的所有BMP文件
        file_paths = glob.glob(os.path.join(input_path, "*.bmp"))
    else:
        raise ValueError(f"输入路径不存在: {input_path}")
    
    if not file_paths:
        print(f"未找到BMP文件: {input_path}")
        return
    
    for bmp_path in file_paths:
        # 确定输出路径
        if output_path is None:
            # 使用相同的文件名但扩展名为.h5
            h5_path = os.path.splitext(bmp_path)[0] + ".h5"
        elif os.path.isdir(output_path):
            # 输出到指定目录
            base_name = os.path.basename(os.path.splitext(bmp_path)[0])
            h5_path = os.path.join(output_path, base_name + ".h5")
        else:
            # 使用指定的输出路径
            h5_path = output_path
        
        # 读取BMP图像并转换为灰度图
        img = cv2.imread(bmp_path)
        if img is None:
            print(f"无法读取图像: {bmp_path}")
            continue
        
        # 转换为灰度图
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 转换为float32类型(可选)
        gray_img = gray_img.astype(np.float32)
        
        # 保存为H5文件
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset(dataset_name, data=gray_img)
        
        print(f"已转换: {bmp_path} -> {h5_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将BMP图像转换为灰度图并保存为H5文件")
    parser.add_argument("input", help="输入的BMP文件或包含BMP文件的文件夹")
    parser.add_argument("-o", "--output", help="输出的H5文件或文件夹")
    parser.add_argument("-d", "--dataset", default="holodata", help="H5文件中的数据集名称")
    args = parser.parse_args()
    
    # 确保输出目录存在
    if args.output and os.path.isdir(args.output):
        os.makedirs(args.output, exist_ok=True)
    
    convert_bmp_to_h5(args.input, args.output, args.dataset)
