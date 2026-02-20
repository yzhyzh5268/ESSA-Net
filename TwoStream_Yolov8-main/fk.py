import os
import random
import shutil

def split_images(source_folder, folder1, folder2, ratio=0.2):
    # 确保目标文件夹存在
    os.makedirs(folder1, exist_ok=True)
    os.makedirs(folder2, exist_ok=True)
    
    # 获取所有jpg文件
    images = [f for f in os.listdir(source_folder) if f.lower().endswith('.jpg')]
    random.shuffle(images)  # 随机打乱顺序
    
    # 计算分割点
    split_index = int(len(images) * ratio)
    
    # 分割文件
    group1 = images[:split_index]
    group2 = images[split_index:]
    
    # 复制文件到第一个文件夹
    for file in group1:
        src = os.path.join(source_folder, file)
        dst = os.path.join(folder1, file)
        shutil.copy2(src, dst)
    
    # 复制文件到第二个文件夹
    for file in group2:
        src = os.path.join(source_folder, file)
        dst = os.path.join(folder2, file)
        shutil.copy2(src, dst)
    
    print(f"总共 {len(images)} 张图片")
    print(f"文件夹 {folder1} 中存放了 {len(group1)} 张图片 (20%)")
    print(f"文件夹 {folder2} 中存放了 {len(group2)} 张图片 (80%)")

# 使用示例
source_folder = "D:\\v8sj\\test\\testimg"  # 替换为你的源文件夹路径
folder1 = "D:\\v8sj\\test\\test1"    # 替换为第一个目标文件夹路径
folder2 = "D:\\v8sj\\test\\test2"     # 替换为第二个目标文件夹路径

split_images(source_folder, folder1, folder2)