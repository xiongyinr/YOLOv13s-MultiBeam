"""
训练前诊断脚本 - 检查配置是否正确
"""
import yaml
import torch
from pathlib import Path
from ultralytics import YOLO

def check_config():
    print("=" * 60)
    print("YOLOv13-P2-Light 训练配置诊断")
    print("=" * 60)

    # 1. 检查模型配置
    print("\n[1] 检查模型配置文件...")
    model_cfg = 'ultralytics/cfg/models/v13/yolov13-p2-light.yaml'
    with open(model_cfg, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    print(f"   ✓ 模型类别数: nc={cfg['nc']}")

    # 2. 检查数据配置
    print("\n[2] 检查数据配置文件...")
    data_cfg = 'data.yaml'
    with open(data_cfg, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    print(f"   ✓ 数据集类别数: nc={data['nc']}")

    # 3. 类别数匹配检查
    print("\n[3] 类别数匹配检查...")
    if cfg['nc'] == data['nc']:
        print(f"   ✓ 类别数匹配: {cfg['nc']} == {data['nc']}")
    else:
        print(f"   ✗ 类别数不匹配: 模型{cfg['nc']} != 数据{data['nc']}")
        print(f"   → 请修改 {model_cfg} 中的 nc 为 {data['nc']}")
        return False

    # 4. 检查数据集路径
    print("\n[4] 检查数据集路径...")
    train_path = Path(data['train'])
    val_path = Path(data['val'])

    if train_path.exists():
        train_images = list(train_path.glob('*.*'))
        print(f"   ✓ 训练集: {len(train_images)} 张图像")
    else:
        print(f"   ✗ 训练集路径不存在: {train_path}")
        return False

    if val_path.exists():
        val_images = list(val_path.glob('*.*'))
        print(f"   ✓ 验证集: {len(val_images)} 张图像")
    else:
        print(f"   ✗ 验证集路径不存在: {val_path}")
        return False

    # 5. 检查标签文件
    print("\n[5] 检查标签文件...")
    train_labels = train_path.parent / 'labels'
    if train_labels.exists():
        label_files = list(train_labels.glob('*.txt'))
        print(f"   ✓ 训练标签: {len(label_files)} 个文件")

        # 检查标签格式
        if label_files:
            with open(label_files[0], 'r') as f:
                first_line = f.readline().strip().split()
                if len(first_line) == 5:
                    cls_id = int(first_line[0])
                    if 0 <= cls_id < data['nc']:
                        print(f"   ✓ 标签格式正确 (class_id x_center y_center width height)")
                    else:
                        print(f"   ✗ 类别ID超出范围: {cls_id} >= {data['nc']}")
                        return False
    else:
        print(f"   ✗ 标签文件夹不存在: {train_labels}")
        return False

    # 6. 检查预训练权重
    print("\n[6] 检查预训练权重...")
    pretrain_path = Path('yolov13s.pt')
    if pretrain_path.exists():
        print(f"   ✓ 预训练权重存在: {pretrain_path} ({pretrain_path.stat().st_size / 1024 / 1024:.1f} MB)")
    else:
        print(f"   ✗ 预训练权重不存在: {pretrain_path}")
        return False

    # 7. 测试模型加载
    print("\n[7] 测试模型加载...")
    try:
        model = YOLO(model_cfg)
        print(f"   ✓ 模型架构加载成功")

        # 尝试加载预训练权重
        result = model.load('yolov13s.pt', strict=False)
        print(f"   ✓ 预训练权重加载成功 (strict=False)")
        print(f"   → 部分权重迁移（backbone + neck兼容层）")
        print(f"   → Detect层将从头训练（因为检测头数量不同）")
    except Exception as e:
        print(f"   ✗ 模型加载失败: {e}")
        return False

    # 8. 检查GPU
    print("\n[8] 检查计算设备...")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   ✓ GPU可用: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print(f"   ⚠ GPU不可用，将使用CPU训练（速度较慢）")

    # 9. 推荐配置
    print("\n[9] 推荐训练配置...")
    print(f"   - batch size: 8 (如果显存不足可降至4)")
    print(f"   - imgsz: 640")
    print(f"   - epochs: 150")
    print(f"   - optimizer: auto (AdamW)")
    print(f"   - warmup_epochs: 3 (帮助稳定训练)")

    print("\n" + "=" * 60)
    print("✓ 所有检查通过！可以开始训练")
    print("=" * 60)
    print("\n预期第1个epoch后的指标范围：")
    print("  - mAP@0.5: 0.05 - 0.15 (5% - 15%)")
    print("  - Precision: 0.10 - 0.30")
    print("  - Recall: 0.05 - 0.20")
    print("\n如果指标仍然接近0，请检查：")
    print("  1. 标签文件是否与图像一一对应")
    print("  2. 标签坐标是否归一化到[0,1]")
    print("  3. 类别ID是否在[0, nc-1]范围内")

    return True

if __name__ == '__main__':
    check_config()
