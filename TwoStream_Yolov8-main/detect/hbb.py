import os
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from tqdm import tqdm

def load_model(weights_path, device):
    if not os.path.exists(weights_path):
        print("Model weights not found!")
        exit()
    model = YOLO(weights_path).to(device)
    model.fuse()
    model.info(verbose=False)
    return model

def process_images(rgb_image_paths, ir_root_dir, model):
    for rgb_path in tqdm(rgb_image_paths):
        filename = os.path.basename(rgb_path)
        ir_path = os.path.join(ir_root_dir, filename)

        if not os.path.exists(ir_path):
            print(f"IR image not found for {filename}")
            continue

        img_rgb = cv2.imread(rgb_path)
        img_ir = cv2.imread(ir_path)
        if img_rgb is None or img_ir is None:
            print(f"Failed to load image pair: {filename}")
            continue

        maskrgb = img_rgb.copy()
        maskir = img_ir.copy()
        imgs = np.concatenate((img_rgb, img_ir), axis=2)

        colors = [
            [165, 0, 255],    # car
            [0, 255, 0],      # truck
            [102, 255, 255],  # bus
            [255, 165, 0],    # van
            [255, 255, 0]     # freight
        ]

        result = model.predict(imgs, save=False, imgsz=640, visualize=False, obb=True)
        cls, xywh = result[0].boxes.cls, result[0].boxes.xywh
        class_conf = result[0].boxes.conf
        cls_, xywh_, class_conf_ = cls.cpu().numpy(), xywh.cpu().numpy(), class_conf.cpu().numpy()

        class_names = ["car", "truck", "bus", "van", "freight"]
        x_coord_record = []

        for pos, cls_value, conf_value in zip(xywh_, cls_, class_conf_):
            pt1 = (int(pos[0] - pos[2] / 2), int(pos[1] - pos[3] / 2))
            pt2 = (int(pos[0] + pos[2] / 2), int(pos[1] + pos[3] / 2))
            color = colors[int(cls_value)]
            thickness = 12 if pt1[0] == 510 and pt1[0] not in x_coord_record else 4

            cv2.rectangle(maskrgb, pt1, pt2, color, thickness)
            cv2.rectangle(maskir, pt1, pt2, color, thickness)

            text_x, text_y = pt1[0], pt1[1] - 3
            if text_y < 0: text_y = pt2[1] + 10
            if text_x in x_coord_record and text_x == 510:
                text_x -= 90
                text_y += 40

            x_coord_record.append(text_x)

            class_name = class_names[int(cls_value)] if int(cls_value) < len(class_names) else "未知类别"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.6
            thickness_text = 4

            text_x = max(text_x, 0)
            text_y = max(text_y, 0)

            cv2.putText(maskrgb, class_name, (text_x, text_y), font, font_scale, color, thickness_text)
            cv2.putText(maskir, class_name, (text_x, text_y), font, font_scale, color, thickness_text)

        cv2.imwrite(f"D:/results1/rgb_{filename}", maskrgb)
        cv2.imwrite(f"D:/results1/ir_{filename}", maskir)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if not os.path.exists('./detect'):
        os.makedirs('./detect')

    model = load_model(r"D:\TwoStream_Yolov8-main\runs\detect\train28\weights\best.pt", device)

    # ✅ 只检测以下几张 RGB 图片（请自行修改路径）
    rgb_image_paths = [
        r"D:\TwoStream_Yolov8-main\datasets\test\testimg\000200.jpg",
        r"D:\TwoStream_Yolov8-main\datasets\test\testimg\000201.jpg",
        r"D:\TwoStream_Yolov8-main\datasets\test\testimg\000202.jpg"
    ]

    # ✅ IR 图像所在目录（文件名需与RGB相同）
    ir_dir = r"D:\TwoStream_Yolov8-main\datasets\test\testimgr"

    process_images(rgb_image_paths, ir_dir, model)

if __name__ == "__main__":
    main()
