# print_model_indices.py
from ultralytics import YOLO

model_path = r"D:\TwoStream_Yolov8-main\yaml\PC2f_MPF_yolov8n.yaml"
y = YOLO(model_path)
m = y.model  # Ultralytics Model (nn.Module)
print("Total modules in m.model:", len(m.model))
print("Index : ClassName | out_ch_hint | repr (short)")
for i, layer in enumerate(m.model):
    out_ch = getattr(layer, 'out_channels', None) or getattr(layer, 'c2', None) or getattr(layer, 'ch', None) or getattr(layer, 'c', None)
    print(f"{i:03d}: {layer.__class__.__name__:20s} | out_ch: {out_ch!s:6s} | {repr(layer)[:240]}")
