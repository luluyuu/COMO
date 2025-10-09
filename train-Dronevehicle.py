from ultralytics import YOLO
import torch
torch.use_deterministic_algorithms(False)
import warnings
warnings.filterwarnings("ignore", message=".*does not have a deterministic implementation.*")
import os
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"  # 屏蔽 INFO/WARNING 日志

model = YOLO("./ultralytics/cfg/models/COMO/yolov8s_como.yaml",)  # load a pretrained model (recommended for training)
model.train(data="./ultralytics/cfg/datasets/DroneVehicle.yaml", epochs=100, imgsz=640, name='./run/train/yolov8s-one-mamba', batch=128, device="0,1,2,3")  # trai
# validate
# model.val(data="./ultralytics/cfg/datasets/DroneVehicle.yaml", name='./run/val/exp')  # validate
