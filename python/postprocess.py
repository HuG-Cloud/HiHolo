import cv2
import os
import glob

def crop_image(image_path, x, y, width, height):
    """
    裁剪图像
    
    Args:
        image_path: 图像文件路径
        x: 起始x坐标
        y: 起始y坐标  
        width: 裁剪宽度
        height: 裁剪高度
    
    Returns:
        裁剪后的图像
    """
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像文件: {image_path}")
    
    # 裁剪图像
    cropped_img = img[y:y+height, x:x+width]
    
    return cropped_img

def main():
    # 查找当前目录下的jpg文件
    jpg_files = glob.glob("*.png")
    
    if not jpg_files:
        print("当前目录下没有找到png文件")
        return
    
    # 使用第一个找到的jpg文件
    image_path = jpg_files[0]
    print(f"找到图像文件: {image_path}")
    
    # 读取并显示原图信息
    img = cv2.imread(image_path)
    print(f"原图尺寸: {img.shape}")
    
    # 定义裁剪参数 (可以根据需要修改)
    x, y = 250, 250  # 起始坐标
    width, height = 2048, 2048  # 裁剪尺寸
    
    # 裁剪图像
    cropped_img = crop_image(image_path, x, y, width, height)
    
    # 保存裁剪后的图像
    output_path = "cropped_image.jpg"
    cv2.imwrite(output_path, cropped_img)
    print(f"裁剪后的图像已保存为: {output_path}")
    print(f"裁剪后尺寸: {cropped_img.shape}")

if __name__ == "__main__":
    main()

