#训练
from multiprocessing import freeze_support
from ultralytics import YOLO

import ultralytics.nn.tasks
def main():
    model = YOLO(r'TwoStream_Yolov8-main\yaml\PC2f_MPF_yolov8n.yaml')
    results = model.train(data=r'TwoStream_Yolov8-main/TwoStream_Yolov8-main/data/drone2.yaml',batch=12,epochs=100,copy_paste=0.1,close_mosaic=10)


if __name__ == "__main__":
    freeze_support()  # 加上这一句,防止windows环境下的多进程报错
    main()