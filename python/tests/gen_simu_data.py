#!/usr/bin/env python3
"""
数据格式转换工具
将4D holodata数据集转换为多个3D数据集
"""

import h5py
import os
from pathlib import Path


def transform_holo_data(input_file, output_file=None):
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"输入文件不存在: {input_file}")
    
    # 如果没有指定输出文件，则自动生成文件名
    if output_file is None:
        input_path = Path(input_file)
        output_file = input_path.parent / f"{input_path.stem}_transformed.h5"
    
    print(f"读取输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    
    # 读取原始数据
    with h5py.File(input_file, 'r') as f_in:
        # 检查holodata数据集是否存在
        if 'holodata' not in f_in:
            raise KeyError("输入文件中没有找到 'holodata' 数据集")
        
        holodata = f_in['holodata']
        print(f"原始数据形状: {holodata.shape}")
        print(f"原始数据类型: {holodata.dtype}")
        
        # 确保是4D数据
        if len(holodata.shape) != 4:
            raise ValueError(f"期望4D数据，但得到{len(holodata.shape)}D数据")
        
        # 获取数据形状
        dim0, dim1, dim2, dim3 = holodata.shape
        print(f"数据维度: dim0={dim0}, dim1={dim1}, dim2={dim2}, dim3={dim3}")
        
        # 创建输出文件
        with h5py.File(output_file, 'w') as f_out:
            # 复制原始文件的属性
            for attr_name, attr_value in f_in.attrs.items():
                f_out.attrs[attr_name] = attr_value
            
            # 沿第二维(dim1)分割数据，创建n个数据集
            # 每个数据集包含第一维和最后两维: (dim0, dim2, dim3)
            print(f"创建 {dim1} 个数据集，每个形状为 ({dim0}, {dim2}, {dim3})")
            
            for i in range(dim1):
                # 提取数据: 第i个切片，保留第一维和最后两维
                data_slice = holodata[:, i, :, :]
                
                # 创建数据集名称
                dataset_name = f"holodata_distance_{i}"
                
                # 保存数据集
                f_out.create_dataset(
                    dataset_name, 
                    data=data_slice,
                    compression='gzip',
                    compression_opts=9
                )
                
                # 添加元数据
                f_out[dataset_name].attrs['original_index'] = i
                f_out[dataset_name].attrs['original_shape'] = holodata.shape
                f_out[dataset_name].attrs['slice_info'] = f"[:, {i}, :, :]"
                
                print(f"  创建数据集 '{dataset_name}': 形状 {data_slice.shape}")
            
            # 添加转换信息到文件属性
            f_out.attrs['transformation_info'] = "4D holodata split into multiple 3D datasets"
            f_out.attrs['num_slices'] = dim1
            f_out.attrs['slice_shape'] = [dim0, dim2, dim3]
            f_out.attrs['original_file'] = str(input_file)
    
    print(f"数据转换完成! 输出文件: {output_file}")
    return str(output_file)


def verify_transformation(output_file):
    """
    验证转换结果
    
    Args:
        output_file (str): 输出HDF5文件路径
    """
    print(f"\n验证转换结果: {output_file}")
    
    with h5py.File(output_file, 'r') as f:
        print(f"文件属性:")
        for attr_name, attr_value in f.attrs.items():
            print(f"  {attr_name}: {attr_value}")
        
        print(f"\n数据集列表:")
        for dataset_name in f.keys():
            dataset = f[dataset_name]
            print(f"  {dataset_name}: 形状 {dataset.shape}, 类型 {dataset.dtype}")
            
            # 显示数据集属性
            for attr_name, attr_value in dataset.attrs.items():
                print(f"    {attr_name}: {attr_value}")


def main():
    """主函数"""
    # 默认输入文件路径
    input_file = os.path.expanduser("~/Downloads/HoloTomo_Data/holo_purephase_200angles.h5")
    
    try:
        # 执行数据转换
        output_file = transform_holo_data(input_file)
        
        # 验证结果
        verify_transformation(output_file)
        
    except Exception as e:
        print(f"错误: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
