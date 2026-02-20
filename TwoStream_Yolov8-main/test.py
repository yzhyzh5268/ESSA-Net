# # 测试
# from ultralytics import YOLO 
# model = YOLO('/root/autodl-tmp/TwoStream_Yolov8-main/TwoStream_Yolov8-main/runs/detect/train171/weights/best.pt') 
# metrics = model.val(data='/root/autodl-tmp/TwoStream_Yolov8-main/TwoStream_Yolov8-main/data/drone2.yaml',split='val',imgsz=640,batch=12)
# test_simple.py
from ultralytics import YOLO
from codecarbon import EmissionsTracker

# 初始化追踪器
tracker = EmissionsTracker(project_name="yolov8_val", output_dir="./")
tracker.start()

# --- 你的原有代码 ---
model = YOLO('/root/autodl-tmp/TwoStream_Yolov8-main/TwoStream_Yolov8-main/runs/detect/train171/weights/best.pt')

# 运行验证 (这部分是高负载，会被记录)
metrics = model.val(
    data='/root/autodl-tmp/TwoStream_Yolov8-main/TwoStream_Yolov8-main/data/drone2.yaml',
    split='val',
    imgsz=640,
    batch=32
)
# -------------------

# 停止追踪并保存结果
emissions: float = tracker.stop()
print(f"本次运行能耗: {tracker.final_emissions_data.energy_consumed} kWh")