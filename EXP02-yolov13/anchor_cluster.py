"""
YOLOv13 anchor clustering for small object detection.
Uses K-Means on dataset bounding boxes to find optimal anchors.
Usage: python anchor_cluster.py --data data.yaml --n 12 --img 640 --model ultralytics/cfg/models/v13/yolov13s-p2.yaml
"""
import argparse
import numpy as np
import yaml
from pathlib import Path


def load_boxes(data_yaml: str, split: str = "train") -> np.ndarray:
    """Load all normalized [w, h] from YOLO-format labels."""
    with open(data_yaml, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    # 稳健的路径查找
    if isinstance(cfg.get(split), str):
        img_dir = Path(cfg[split])
    elif isinstance(cfg.get('path'), str):
        base_path = Path(cfg['path'])
        img_dir = base_path / cfg[split]
    else:
        raise ValueError("Cannot find image path in data.yaml")

    if 'images' in img_dir.parts:
        label_dir = Path(str(img_dir).replace('images', 'labels'))
    else:
        label_dir = img_dir.parent / "labels" / img_dir.name

    print(f"Looking for labels in: {label_dir}")
    
    boxes = []
    for lf in label_dir.glob("*.txt"):
        try:
            with open(lf, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            with open(lf, 'r', encoding='gbk') as f:
                lines = f.readlines()
                
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                boxes.append([float(parts[3]), float(parts[4])])
    
    if len(boxes) == 0:
        raise FileNotFoundError(f"No labels found! Checked path: {label_dir}")
        
    return np.array(boxes, dtype=np.float32)


def iou_wh(boxes: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """
    修复版：计算 boxes (N,2) 和 anchors (K,2) 的 IoU
    """
    N = boxes.shape[0]
    K = anchors.shape[0]
    
    # 把 boxes 变成 (N, K, 2)，把 anchors 变成 (N, K, 2)
    # 利用广播机制
    b = boxes[:, np.newaxis, :]  # (N, 1, 2)
    a = anchors[np.newaxis, :, :] # (1, K, 2)
    
    # 计算交集
    wh_min = np.minimum(b, a)
    intersection = wh_min[:, :, 0] * wh_min[:, :, 1] # (N, K)
    
    # 计算面积
    area_boxes = boxes[:, 0] * boxes[:, 1] # (N,)
    area_anchors = anchors[:, 0] * anchors[:, 1] # (K,)
    
    # 计算并集：利用广播 (N,) + (K,) -> (N, K)
    union = area_boxes[:, np.newaxis] + area_anchors[np.newaxis, :] - intersection
    
    return intersection / (union + 1e-7)


def kmeans_anchors(boxes: np.ndarray, n: int, img_size: int = 640,
                   thr: float = 4.0, iters: int = 300) -> tuple:
    """K-Means clustering with IoU distance metric. Returns (anchors, yaml_list)."""
    boxes = boxes * img_size  # scale to pixel space
    # Filter tiny boxes (< 2px)
    boxes = boxes[(boxes > 2).all(1)]

    print(f"Clustering {len(boxes)} valid boxes into {n} anchors...")

    np.random.seed(42)
    indices = np.random.choice(len(boxes), n, replace=False)
    anchors = boxes[indices]

    for i in range(iters):
        # 计算距离
        iou = iou_wh(boxes, anchors)
        dist = 1 - iou
        
        # 分配簇
        assign = dist.argmin(axis=1)
        
        # 更新锚框
        new_anchors = np.zeros_like(anchors)
        for k in range(n):
            cluster_boxes = boxes[assign == k]
            if len(cluster_boxes) > 0:
                new_anchors[k] = cluster_boxes.mean(axis=0)
            else:
                new_anchors[k] = anchors[k] # 如果簇为空，保持原样
        
        # 判断收敛
        if np.allclose(new_anchors, anchors):
            print(f"Converged at iteration {i}")
            break
        anchors = new_anchors

    # 按面积从小到大排序
    areas = anchors[:, 0] * anchors[:, 1]
    sorted_indices = np.argsort(areas)
    anchors = anchors[sorted_indices]

    # 计算召回率
    best_iou = iou_wh(boxes, anchors).max(axis=1)
    recall = (best_iou > 1 / thr).mean()
    
    print(f"\n📊 结果统计:")
    print(f"  Best Possible Recall (thr={thr}): {recall:.4f}")
    print(f"\n🔢 生成的锚框 (像素单位, img_size={img_size}):")
    for i, (w, h) in enumerate(anchors):
        print(f"  [{i}] {w:.1f} x {h:.1f}")
    
    print(f"\n✅ 直接复制到 yolov13s-p2.yaml 的 anchors 字段:")
    a = anchors.round().astype(int).tolist()

    # 生成 YAML 格式
    yaml_list = []
    for i in range(0, n, 3):
        group = a[i:i+3]
        # 展平: [[w1,h1],[w2,h2]] -> [w1,h1,w2,h2]
        flat = [coord for wh in group for coord in wh]
        yaml_list.append(flat)
        print(f"  - {flat}")

    return anchors, yaml_list


def update_yaml_anchors(yaml_path: str, anchors_list: list):
    """Update anchors in YAML config file."""
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        print(f"❌ YAML file not found: {yaml_path}")
        return False

    # Read YAML
    with open(yaml_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Find anchors section
    anchor_start = None
    anchor_end = None
    for i, line in enumerate(lines):
        if line.strip().startswith('anchors:'):
            anchor_start = i
        elif anchor_start is not None and line.strip() and not line.strip().startswith('-') and not line.strip().startswith('#'):
            anchor_end = i
            break

    if anchor_start is None:
        # No anchors section, insert after nc
        for i, line in enumerate(lines):
            if line.strip().startswith('nc:'):
                insert_pos = i + 1
                # Skip comments
                while insert_pos < len(lines) and lines[insert_pos].strip().startswith('#'):
                    insert_pos += 1

                new_lines = ['\n', '# Anchors for small object detection\n',
                            '# Auto-generated by anchor_cluster.py\n',
                            'anchors:\n']
                for anchor_group in anchors_list:
                    new_lines.append(f'  - {anchor_group}\n')

                lines = lines[:insert_pos] + new_lines + lines[insert_pos:]
                break
    else:
        # Replace existing anchors
        if anchor_end is None:
            anchor_end = anchor_start + 1
            while anchor_end < len(lines) and (lines[anchor_end].strip().startswith('-') or
                                               lines[anchor_end].strip().startswith('#') or
                                               not lines[anchor_end].strip()):
                anchor_end += 1

        new_anchor_lines = ['anchors:\n']
        for anchor_group in anchors_list:
            new_anchor_lines.append(f'  - {anchor_group}\n')

        lines = lines[:anchor_start] + new_anchor_lines + lines[anchor_end:]

    # Write back
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    print(f"\n✅ Anchors已自动写入: {yaml_path}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data.yaml", help="Dataset YAML path")
    parser.add_argument("--n", type=int, default=12, help="Number of anchors (12 for P2 layer)")
    parser.add_argument("--img", type=int, default=640, help="Image size")
    parser.add_argument("--thr", type=float, default=4.0, help="IoU threshold")
    parser.add_argument("--model", default="ultralytics/cfg/models/v13/yolov13s-p2.yaml",
                       help="Model YAML to update")
    args = parser.parse_args()

    try:
        boxes = load_boxes(args.data)
        print(f"✅ 成功加载 {len(boxes)} 个标注框")
        anchors, yaml_list = kmeans_anchors(boxes, args.n, args.img, args.thr)

        # Auto-update YAML file
        update_yaml_anchors(args.model, yaml_list)

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()