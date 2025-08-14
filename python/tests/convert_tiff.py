#!/usr/bin/env python3
"""
将4张tif文件合成一个3D的h5数据文件
"""

import numpy as np
import h5py
from PIL import Image
import glob
import os
import argparse


def load_tiff_files(pattern="result_*.tif"):
    """
    加载匹配模式的tif文件并转换为灰度图
    
    Args:
        pattern: 文件匹配模式
        
    Returns:
        list: 按文件名排序的2D灰度图像数据列表
    """
    # 查找匹配的文件
    tiff_files = glob.glob(pattern)
    tiff_files.sort()  # 按文件名排序
    
    if len(tiff_files) == 0:
        raise FileNotFoundError(f"未找到匹配模式 '{pattern}' 的文件")
    
    print(f"找到 {len(tiff_files)} 个tif文件:")
    for file in tiff_files:
        print(f"  - {file}")
    
    images = []
    for file in tiff_files:
        # 使用PIL读取tif文件
        img = Image.open(file)
        
        # 转换为灰度图
        if img.mode != 'L':
            print(f"  转换 {file} 从 {img.mode} 模式到灰度模式")
            img = img.convert('L')
        
        # 转换为numpy数组并确保是float类型
        img_array = np.array(img, dtype=np.float32)
        
        # 确保是2D数组（如果是3D但最后一维是1，则压缩）
        if img_array.ndim == 3 and img_array.shape[2] == 1:
            img_array = img_array.squeeze(axis=2)
        elif img_array.ndim > 2:
            raise ValueError(f"无法将 {file} 转换为2D灰度图，当前形状: {img_array.shape}")
        
        images.append(img_array)
        print(f"加载 {file}: 形状 {img_array.shape}, 数据类型 {img_array.dtype}")
    
    return images


def create_3d_array(images):
    """
    将2D灰度图像列表合并为3D数组
    
    Args:
        images: 2D灰度图像数据列表，每个图像形状为 (height, width)
        
    Returns:
        numpy.ndarray: 3D数组，形状为 (num_images, height, width)
    """
    # 检查所有图像是否具有相同的形状
    shapes = [img.shape for img in images]
    if not all(shape == shapes[0] for shape in shapes):
        raise ValueError(f"所有图像必须具有相同的形状，但得到: {shapes}")
    
    # 确保所有图像都是2D的
    for i, img in enumerate(images):
        if img.ndim != 2:
            raise ValueError(f"图像 {i} 不是2D数组，形状: {img.shape}")
    
    # 将图像堆叠成3D数组
    data_3d = np.stack(images, axis=0)
    print(f"创建3D数组: 形状 {data_3d.shape}, 数据类型 {data_3d.dtype}")
    
    # 验证结果确实是3D的
    if data_3d.ndim != 3:
        raise ValueError(f"最终数组不是3D的，实际维度: {data_3d.ndim}, 形状: {data_3d.shape}")
    
    return data_3d


def save_to_h5(data_3d, output_file="result_3d.h5", dataset_name="images"):
    """
    将3D数组保存为h5文件
    
    Args:
        data_3d: 3D numpy数组
        output_file: 输出文件名
        dataset_name: 数据集名称
    """
    with h5py.File(output_file, 'w') as f:
        # 创建数据集
        dataset = f.create_dataset(dataset_name, data=data_3d, 
                                 compression='gzip', compression_opts=9)
        
        # 添加属性信息
        dataset.attrs['description'] = '由4张tif文件合成的3D数据'
        dataset.attrs['shape'] = data_3d.shape
        dataset.attrs['dtype'] = str(data_3d.dtype)
        dataset.attrs['num_images'] = data_3d.shape[0]
        dataset.attrs['height'] = data_3d.shape[1]
        dataset.attrs['width'] = data_3d.shape[2]
    
    print(f"成功保存到 {output_file}")
    print(f"数据集名称: {dataset_name}")
    print(f"数据形状: {data_3d.shape}")
    print(f"数据类型: {data_3d.dtype}")


def main():
    parser = argparse.ArgumentParser(description='将tif文件合成3D h5数据')
    parser.add_argument('--pattern', default='result_*.tif', 
                       help='tif文件匹配模式 (默认: result_*.tif)')
    parser.add_argument('--output', default='result_3d.h5', 
                       help='输出h5文件名 (默认: result_3d.h5)')
    parser.add_argument('--dataset', default='images', 
                       help='h5数据集名称 (默认: images)')
    
    args = parser.parse_args()
    
    try:
        # 加载tif文件
        images = load_tiff_files(args.pattern)
        
        # 创建3D数组
        data_3d = create_3d_array(images)
        
        # 保存为h5文件
        save_to_h5(data_3d, args.output, args.dataset)
        
        print("\n转换完成!")
        
    except Exception as e:
        print(f"错误: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
