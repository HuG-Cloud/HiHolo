import cv2
import numpy as np

def convert_to_grayscale(input_path, output_path):
    """
    将PNG图像转换为灰度图并保存
    
    Args:
        input_path: 输入PNG图像路径
        output_path: 输出灰度图像路径
    """
    # 读取图像
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"无法读取图像文件: {input_path}")
    
    # 转换为灰度图
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 保存灰度图
    cv2.imwrite(output_path, gray_img)
    print(f"灰度图已保存为: {output_path}")
    print(f"图像尺寸: {gray_img.shape}")

def main():
    # 输入和输出文件路径
    input_file = "amplitude.png"  # 可以根据需要修改输入文件名
    output_file = "output_grayscale.png"
    
    try:
        convert_to_grayscale(input_file, output_file)
    except Exception as e:
        print(f"转换失败: {e}")

if __name__ == "__main__":
    main()
