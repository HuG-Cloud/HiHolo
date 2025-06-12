#!/usr/bin/env python3
"""
测试fastholo的Python绑定
"""

import numpy as np
import sys

# 尝试导入绑定模块
try:
    import fastholo
    print("✓ 成功导入fastholo模块")
except ImportError as e:
    print(f"✗ 导入失败: {e}")
    sys.exit(1)

def test_enums():
    """测试枚举类型"""
    print("\n测试枚举类型:")
    
    padding_types = [
        fastholo.PaddingType.Constant,
        fastholo.PaddingType.Replicate,
        fastholo.PaddingType.Fadeout
    ]
    
    for pad_type in padding_types:
        print(f"PaddingType: {pad_type}")

def test_ctf_function():
    """测试CTF重建函数（只测试调用接口，不测试实际计算）"""
    print("\n测试CTF重建函数:")
    
    # 创建模拟数据
    rows, cols = 64, 64
    num_images = 2
    
    # 创建模拟全息图数据
    holograms = np.random.random(num_images * rows * cols).astype(np.float32).tolist()
    
    # 图像尺寸
    im_size = [rows, cols]
    
    # 菲涅尔数 (每个图像一组，使用嵌套列表)
    fresnel_numbers = [[0.001], [0.002]]
    
    low_freq_lim = 1e-3
    high_freq_lim = 1e-1
    beta_delta_ratio = 0.0
    
    pad_size = [16, 16]
    pad_type = fastholo.PaddingType.Replicate
    pad_value = 0.0
    
    try:
        print("调用reconstruct_ctf函数...")
        result = fastholo.reconstruct_ctf(
            holograms, num_images, im_size, fresnel_numbers,
            low_freq_lim, high_freq_lim, beta_delta_ratio,
            pad_size, pad_type, pad_value
        )
        print(f"✓ CTF重建成功, 结果长度: {len(result)}")
        
    except Exception as e:
        print(f"✗ CTF重建失败: {e}")
        return False
    
    return True

def test_ctf_reconstructor_class():
    """测试CTFReconstructor类"""
    print("\n测试CTFReconstructor类:")
    
    # 参数设置
    angles = 4
    batch_size = 2
    num_images = 2
    rows, cols = 64, 64
    im_size = [rows, cols]
    fresnel_numbers = [[0.001], [0.002]]
    low_freq_lim = 1e-3
    high_freq_lim = 1e-1
    beta_delta_ratio = 0.0
    pad_size = [16, 16]
    pad_type = fastholo.PaddingType.Replicate
    pad_value = 0.0
    
    try:
        # 创建重建器对象
        print("创建CTFReconstructor对象...")
        reconstructor = fastholo.CTFReconstructor(
            batch_size, num_images, im_size, fresnel_numbers,
            low_freq_lim, high_freq_lim, beta_delta_ratio,
            pad_size, pad_type, pad_value
        )
        print("✓ CTFReconstructor创建成功")
        
        # 测试批量重建
        for i in range(angles // batch_size):
            holograms = np.random.random(batch_size * num_images * rows * cols).astype(np.float32).tolist()
            print(f"正在处理第{i}个batch")
            result = reconstructor.reconsBatch(holograms)

        print(f"✓ 批量重建成功，结果长度: {len(result)}")
        
    except Exception as e:
        print(f"✗ CTFReconstructor测试失败: {e}")
        return False
    
    return True

def main():
    """主测试函数"""
    print("开始测试fastholo Python绑定...")
    
    # 测试基本功能
    test_enums()
    
    # 测试核心功能（需要GPU）
    print("\n注意：以下测试需要CUDA GPU支持")
    try:
        success = True
        success &= test_ctf_function()
        success &= test_ctf_reconstructor_class()
        
        if success:
            print("\n🎉 所有测试通过! Python绑定工作正常。")
        else:
            print("\n⚠️  部分测试失败, 请检查CUDA和GPU设置。")
            
    except Exception as e:
        print(f"\n❌ GPU相关测试失败: {e}")
        print("请确保：")
        print("1. 系统安装了CUDA")
        print("2. 有可用的GPU设备")
        print("3. 相关库正确编译和链接")

if __name__ == "__main__":
    main() 