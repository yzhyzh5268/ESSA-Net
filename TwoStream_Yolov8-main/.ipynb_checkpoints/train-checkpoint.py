#训练
from ultralytics import YOLO
import ultralytics.nn.tasks
model = YOLO('/root/autodl-tmp/TwoStream_Yolov8-main/TwoStream_Yolov8-main/yaml/PC2f_MPF_yolov8n.yaml')
results = model.train(data='/root/autodl-tmp/TwoStream_Yolov8-main/TwoStream_Yolov8-main/data/drone2.yaml',batch=12,epochs=200,copy_paste=0.1,close_mosaic=5)
