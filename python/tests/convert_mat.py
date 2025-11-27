#!/usr/bin/env python3
"""
MAT到HDF5格式转换工具
将MATLAB格式文件(.mat)转换为HDF5格式(.h5)
数据格式转换: (h, w, c) -> (c, h, w)
"""

import h5py
import scipy.io
import numpy as np
import os
from pathlib import Path
import argparse


def convert_matlab_type_to_hdf5(data):
    """
    将MATLAB数据类型转换为适合HDF5的numpy数据类型
    保持原始精度和数据结构
    
    Args:
        data: 从MAT文件读取的数据
        
    Returns:
        转换后的numpy数组，保持原始数据类型
    """
    if isinstance(data, np.ndarray):
        # 如果是2D数据，直接返回
        if data.ndim == 2:
            return data
        # 保持原始数据类型
        return data
    elif isinstance(data, (int, float, complex)):
        # 标量数据转换为numpy数组
        return np.array(data)
    elif isinstance(data, str):
        # 字符串数据
        return data
    else:
        # 其他类型尝试转换为numpy数组
        try:
            return np.array(data)
        except:
            return data

def move_third_axis_to_first(arr):
    """
    将数组从 (h, w, c) 格式转换为 (c, h, w) 格式
    如果数组是3维或以上，把第三个维（axis=2）移到最前面，其余维度顺序向后。
    例如: (h, w, c) -> (c, h, w)
    """
    if not isinstance(arr, np.ndarray):
        return arr
    if arr.ndim < 3:
        return arr  # 无需变化
    # 移动第三维(c)到第一维，原第一维(h)和第二维(w)顺序后移
    axes = list(range(arr.ndim))
    third_idx = 2
    new_axes = [third_idx] + axes[:third_idx] + axes[third_idx+1:]
    arr_new = np.transpose(arr, new_axes)
    return arr_new

def convert_mat_to_hdf5(mat_file_path, hdf5_file_path=None, compression='gzip', compression_level=9):
    """
    将MAT文件转换为HDF5文件，数据格式从 (h, w, c) 转换为 (c, h, w)
    
    Args:
        mat_file_path (str): 输入MAT文件路径
        hdf5_file_path (str, optional): 输出HDF5文件路径，默认为None则自动生成
        compression (str): 压缩方法，默认'gzip'
        compression_level (int): 压缩级别，默认9
        
    Returns:
        str: 输出HDF5文件路径
    """
    
    # 检查输入文件
    if not os.path.exists(mat_file_path):
        raise FileNotFoundError(f"输入MAT文件不存在: {mat_file_path}")
    
    # 自动生成输出文件名
    if hdf5_file_path is None:
        input_path = Path(mat_file_path)
        hdf5_file_path = input_path.parent / f"{input_path.stem}.h5"
    
    print(f"读取MAT文件: {mat_file_path}")
    print(f"输出HDF5文件: {hdf5_file_path}")
    
    try:
        # 读取MAT文件
        mat_data = scipy.io.loadmat(mat_file_path)
        
        # 创建HDF5文件
        with h5py.File(hdf5_file_path, 'w') as h5f:
            
            # 遍历MAT文件中的所有变量
            for var_name, var_data in mat_data.items():
                
                # 跳过MATLAB内部变量
                if var_name.startswith('__'):
                    continue
                
                print(f"转换变量: {var_name}")
                
                # 转换数据类型
                converted_data = convert_matlab_type_to_hdf5(var_data)
                
                # 如为高维数据，转维: (h, w, c) -> (c, h, w)
                if isinstance(converted_data, np.ndarray) and converted_data.ndim >= 3:
                    # 打印原shape
                    print(f"  原始数据形状 (h,w,c): {converted_data.shape}")
                    converted_data = move_third_axis_to_first(converted_data)
                    print(f"  转换后形状 (c,h,w): {converted_data.shape}")
                elif isinstance(converted_data, np.ndarray):
                    print(f"  数据形状: {converted_data.shape}")

                # 获取数据信息
                if isinstance(converted_data, np.ndarray):
                    print(f"  数据类型: {converted_data.dtype}")
                    
                    # 保存到HDF5，保持原始数据类型
                    if compression and converted_data.size > 1000:  # 只对大数据应用压缩
                        h5f.create_dataset(
                            var_name, 
                            data=converted_data,
                            dtype=converted_data.dtype,
                            compression=compression,
                            compression_opts=compression_level
                        )
                    else:
                        h5f.create_dataset(
                            var_name, 
                            data=converted_data,
                            dtype=converted_data.dtype
                        )
                else:
                    print(f"  数据类型: {type(converted_data)}")
                    
                    # 处理非数组数据（字符串等）
                    if isinstance(converted_data, str):
                        h5f.create_dataset(var_name, data=converted_data, dtype=h5py.string_dtype())
                    else:
                        h5f.create_dataset(var_name, data=converted_data)
        
        print(f"转换完成! 输出文件: {hdf5_file_path}")
        return str(hdf5_file_path)
        
    except Exception as e:
        print(f"转换过程中出现错误: {e}")
        # 如果转换失败，删除可能创建的不完整文件
        if os.path.exists(hdf5_file_path):
            os.remove(hdf5_file_path)
        raise


def verify_conversion(mat_file_path, hdf5_file_path):
    """
    验证转换结果的正确性
    
    Args:
        mat_file_path (str): 原始MAT文件路径
        hdf5_file_path (str): 转换后的HDF5文件路径
    """
    print(f"\n验证转换结果...")
    print(f"原始MAT文件: {mat_file_path}")
    print(f"转换HDF5文件: {hdf5_file_path}")
    
    try:
        # 读取原始MAT文件
        mat_data = scipy.io.loadmat(mat_file_path)
        
        # 读取转换后的HDF5文件
        with h5py.File(hdf5_file_path, 'r') as h5f:
            
            print(f"\n数据集对比:")
            
            # 验证每个变量
            for var_name, mat_var in mat_data.items():
                
                # 跳过MATLAB内部变量
                if var_name.startswith('__'):
                    continue
                
                if var_name in h5f:
                    h5_var = np.array(h5f[var_name])
                    
                    print(f"\n变量: {var_name}")

                    # 对比之前尝试转维使shape一致: (h,w,c) -> (c,h,w)
                    mat_var_compare = mat_var
                    if isinstance(mat_var, np.ndarray) and mat_var.ndim >= 3:
                        # 将matlab的数据从(h,w,c)转换为(c,h,w)
                        mat_var_compare = move_third_axis_to_first(mat_var)

                    print(f"  MAT - 形状: {mat_var_compare.shape if hasattr(mat_var_compare, 'shape') else 'N/A'}, "
                          f"类型: {mat_var_compare.dtype if hasattr(mat_var_compare, 'dtype') else type(mat_var_compare)}")
                    print(f"  HDF5 - 形状: {h5_var.shape if hasattr(h5_var, 'shape') else 'N/A'}, "
                          f"类型: {h5_var.dtype if hasattr(h5_var, 'dtype') else type(h5_var)}")
                    
                    # 数据一致性检查
                    if isinstance(mat_var_compare, np.ndarray) and isinstance(h5_var, np.ndarray):
                        if mat_var_compare.shape == h5_var.shape:
                            # 检查数值是否相等（考虑浮点精度）
                            if np.allclose(mat_var_compare, h5_var, rtol=1e-15, atol=1e-15):
                                print(f"  ✓ 数据一致")
                            else:
                                max_diff = np.max(np.abs(mat_var_compare - h5_var))
                                print(f"  ⚠ 数据存在差异，最大差值: {max_diff}")
                        else:
                            print(f"  ✗ 形状不匹配")
                    
                else:
                    print(f"  ✗ 变量 {var_name} 在HDF5文件中未找到")
            
            # 检查HDF5中是否有额外的数据集
            h5_vars = set(h5f.keys())
            mat_vars = set(name for name in mat_data.keys() if not name.startswith('__'))
            
            extra_h5_vars = h5_vars - mat_vars
            if extra_h5_vars:
                print(f"\nHDF5文件中的额外变量: {extra_h5_vars}")
        
        print(f"\n验证完成!")
        
    except Exception as e:
        print(f"验证过程中出现错误: {e}")


def list_mat_variables(mat_file_path):
    """
    列出MAT文件中的所有变量信息
    
    Args:
        mat_file_path (str): MAT文件路径
    """
    print(f"MAT文件变量信息: {mat_file_path}")
    
    try:
        mat_data = scipy.io.loadmat(mat_file_path)
        
        print(f"\n找到 {len([k for k in mat_data.keys() if not k.startswith('__')])} 个用户变量:")
        
        for var_name, var_data in mat_data.items():
            if not var_name.startswith('__'):
                if isinstance(var_data, np.ndarray):
                    print(f"  {var_name}: 形状 {var_data.shape}, 类型 {var_data.dtype}, "
                          f"大小 {var_data.nbytes / 1024 / 1024:.2f} MB")
                else:
                    print(f"  {var_name}: 类型 {type(var_data)}, 值 {var_data}")
        
        # 显示MATLAB文件信息
        if '__header__' in mat_data:
            print(f"\nMAT文件头信息: {mat_data['__header__']}")
        if '__version__' in mat_data:
            print(f"MATLAB版本: {mat_data['__version__']}")
            
    except Exception as e:
        print(f"读取MAT文件时出现错误: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='将MAT文件转换为HDF5格式，数据从(h,w,c)转换为(c,h,w)')
    parser.add_argument('input_file', help='输入MAT文件路径')
    parser.add_argument('-o', '--output', help='输出HDF5文件路径（可选）')
    parser.add_argument('-c', '--compression', default='gzip', 
                       choices=['gzip', 'lzf', 'szip', None],
                       help='压缩方法 (默认: gzip)')
    parser.add_argument('-l', '--compression-level', type=int, default=9,
                       help='压缩级别 (默认: 9)')
    parser.add_argument('--list-vars', action='store_true',
                       help='只列出MAT文件中的变量信息，不进行转换')
    parser.add_argument('--verify', action='store_true',
                       help='转换后验证数据一致性')
    
    args = parser.parse_args()
    
    try:
        # 检查输入文件
        if not os.path.exists(args.input_file):
            print(f"错误: 输入文件不存在 - {args.input_file}")
            return 1
        
        # 只列出变量信息
        if args.list_vars:
            list_mat_variables(args.input_file)
            return 0
        
        # 执行转换
        output_file = convert_mat_to_hdf5(
            args.input_file, 
            args.output,
            compression=args.compression,
            compression_level=args.compression_level
        )
        
        # 验证转换结果
        if args.verify:
            verify_conversion(args.input_file, output_file)
        
        return 0
        
    except Exception as e:
        print(f"错误: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
