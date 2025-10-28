from ultralytics import YOLO
import torch
torch.use_deterministic_algorithms(False)
import warnings
warnings.filterwarnings("ignore", message=".*does not have a deterministic implementation.*")
import os
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"  # 屏蔽 INFO/WARNING 日志

model = YOLO("./checkpoints/DroneVehicle-best.pt",)  
# validate
model.val(data="./ultralytics/cfg/datasets/DroneVehicle.yaml",  name='./run/val/exp')  # validate   