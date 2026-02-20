# 旋转框数据处理(DroneVehicle数据集)
# 转换Dota数据集格式为YOLO OBB格式
# 旋转框数据处理(DroneVehicle数据集)
# 转换Dota数据集格式为YOLO OBB格式
import sys 
from ultralytics.data.converter import convert_dota_to_yolo_obb
import inspect


# 直接指定路径
convert_dota_to_yolo_obb(r"D:\TwoStream_Yolov8-main\datasets\zh\kf")
