import torch
from ultralytics import YOLO
import thop
from thop import profile
from prettytable import PrettyTable

if __name__ == '__main__':
    batch_size, height, width = 1, 640, 640

    # 自动检测设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载 YOLO 模型
    model = YOLO(r'./checkpoints/DroneVehicle-best.pt').model
    model.fuse()
    model = model.to(device).eval()  # ✅ 移动模型到 GPU 并设置 eval 模式

    # 构造输入
    input1 = torch.randn(batch_size, 3, height, width).to(device)
    input2 = torch.randn(batch_size, 3, height, width).to(device)
    inputs = torch.cat([input1, input2], dim=1)

    # 计算总 FLOPs 和 Params（不逐层）
    with torch.no_grad():
        total_flops, total_params = profile(model, inputs=(inputs,), verbose=False)

    print(f"\nTotal Params: {total_params/1e6:.4f} M")
    print(f"Total FLOPs : {total_flops/1e9:.2f} G\n")

    # 计算逐层 FLOPs / Params
    with torch.no_grad():
        total_flops, total_params, layer_info = profile(
            model,
            inputs=(inputs,),
            verbose=False,
            ret_layer_info=True
        )

    # 打印逐层信息表格
    if layer_info and 'model' in layer_info and isinstance(layer_info['model'], (list, tuple)):
        layers = layer_info['model'][2] if len(layer_info['model']) > 2 else layer_info['model']
    else:
        print("⚠️ 无法解析逐层信息（可能是 thop 版本不兼容 ret_layer_info）")
        layers = {}

    table = PrettyTable()
    FLOPs_total = total_flops / batch_size
    Params_total = total_params
    FLOPs_str, Params_str = thop.clever_format([FLOPs_total, Params_total], "%.3f")
    table.title = f'Model FLOPs: {FLOPs_str} | Params: {Params_str}'
    table.field_names = ['Layer ID', "FLOPs", "Params"]

    if isinstance(layers, dict):
        for layer_id in layers:
            data = layers[layer_id]
            FLOPs_layer, Params_layer = thop.clever_format([data[0] / batch_size, data[1]], "%.3f")
            table.add_row([layer_id, FLOPs_layer, Params_layer])
        print(table)
    else:
        print("⚠️ 没有逐层统计结果。")

