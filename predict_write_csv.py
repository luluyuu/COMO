from ultralytics import YOLO
from ultralytics_single import YOLO as YOLO_single
import csv
import os
import glob
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

import torch

from ultralytics.utils.ops import non_max_suppression
from ultralytics.engine.results import Results, Boxes
IMG_FORMATS = {"bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm", "heic"}  # image suffixes
# 禁用wandb日志

 
# 加载训练好的模型
model = YOLO("./checkpoints/COMO_best.pt")  # 修改为你的模型路径


# 设置图像路径（需替换为实际路径）
image_dir = ["/RGB/*.jpg","/TIR/*.jpg"]  # 修改为你的图像路径
model_name = "COMO"  # 模型名称
# 获取所有图像文件

files_rgb = glob.glob(image_dir[0])
files_ir = glob.glob(image_dir[1])

# hyperparameters
conf=0.45
iou=0.5
now = datetime.now()
month_day = now.strftime("%m%d_%H%M")


# Define files as images or videos
images_rgb, images_ir, videos = [], [], []
for f in files_rgb:
    suffix = f.split(".")[-1].lower()  # Get file extension without the dot and lowercase
    if suffix in IMG_FORMATS:
        images_rgb.append(f)

for f in files_ir:
    suffix = f.split(".")[-1].lower()
    if suffix in IMG_FORMATS:
        images_ir.append(f)


valid_rgb, valid_ir = [], []
missing_pairs = []

# 构建文件名到路径的映射字典（考虑大小写敏感需求）
rgb_dict = {Path(p).stem.lower(): p for p in images_rgb}
ir_dict = {Path(p).stem.lower(): p for p in images_ir}

# 获取所有可能的文件名基（统一小写处理）
all_stems = set(rgb_dict.keys()).union(ir_dict.keys())

with tqdm(all_stems, total=len(all_stems), 
        desc="\033[1mValidating pairs\033[0m",
        bar_format="{l_bar}\033[32m{bar:20}\033[0m| {n_fmt}/{total_fmt} | Elapsed: {elapsed}") as pbar:
    
    for stem in pbar:
        rgb_path = rgb_dict.get(stem)
        ir_path = ir_dict.get(stem)
        
        exists_check = {
            'rgb': rgb_path and os.path.isfile(rgb_path),
            'ir': ir_path and os.path.isfile(ir_path)
        }
        
        # 三种有效情况判断
        if all(exists_check.values()):
            valid_rgb.append(rgb_path)
            valid_ir.append(ir_path)
        else:
            error_details = []
            # 记录缺失类型
            if not exists_check['rgb']:
                error_details.append(f"RGB missing: {stem}")
                if stem in ir_dict:  # 补充实际找到的IR文件名
                    error_details[-1] += f" (IR has: {Path(ir_dict[stem]).name})"
            if not exists_check['ir']:
                error_details.append(f"IR missing: {stem}")
                if stem in rgb_dict:  # 补充实际找到的RGB文件名
                    error_details[-1] += f" (RGB has: {Path(rgb_dict[stem]).name})"
            
            # 记录实际路径对（可能为None）
            missing_pairs.append((
                rgb_path if exists_check['rgb'] else f"NOT_FOUND:{stem}_rgb",
                ir_path if exists_check['ir'] else f"NOT_FOUND:{stem}_ir",
                " | ".join(error_details)
            ))

# 最终错误报告
if missing_pairs:
    print(f"\n\033[1;31mERROR: 发现 {len(missing_pairs)} 个不匹配的数据对\033[0m")
    for idx, (rgb, ir, reason) in enumerate(missing_pairs, 1):
        print(f"  {idx}. {Path(rgb).name}")
        print(f"     \033[33mMissing -> {reason}\033[0m")
        print(f"     RGB路径: {rgb}")
        print(f"     IR路径:  {ir}\n")


# 创建预测结果CSV
with open(f"./test_result/{month_day}-{model_name}-conf{conf}-iou{iou}.csv", 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    # 写入表头
    csv_writer.writerow(['id', 'image_id', 'category_id', 'bbox', 'score'])
    
    entry_id = 0  # 自增的行ID
    
    for img_path_rgb, img_path_ir in zip(valid_rgb, valid_ir):
        # 执行预测
        print(f"Processing {img_path_rgb}{img_path_ir}...")
        results1 = model.predict([img_path_rgb, img_path_ir], conf=conf, iou=iou, device='cuda:0', verbose=False)
        result1 = results1[0]  # 获取单张图像的预测结果

        
        # 从文件名提取image_id（假设文件名为纯数字）
        image_id = os.path.splitext(os.path.basename(img_path_rgb))[0]
        
        # 初始化存储容器
        categories, bboxes, scores = [], [], []

        for box in result1.boxes:
            # 提取数据的代码（原有逻辑）
            x, y, w, h = box.xywh[0].tolist()
            bboxes.append(f"[{x},{y},{w},{h}]")
            categories.append(str(int(box.cls.item())))
            scores.append(str(box.conf.item()))

        # 处理无检测框的情况
        if not categories:  # 如果没有检测到任何目标
            categories = ["0"]          # 填充默认类别（根据比赛规则调整）
            bboxes = ["[0,0,0,0]"]      # 填充无效框或空框（根据规则调整）
            scores = ["0.0"]            # 填充最低置信度

        csv_writer.writerow([
            entry_id,
            image_id,
            ",".join(categories),
            ",".join(bboxes),
            ",".join(scores)
        ])
        entry_id += 1