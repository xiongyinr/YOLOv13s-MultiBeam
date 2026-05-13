import os
import yaml
from pathlib import Path
from typing import Tuple, List, Optional


class Validators:
    """输入验证工具类"""

    @staticmethod
    def validate_directory(path: str, must_exist: bool = False) -> Tuple[bool, str]:
        """验证目录路径"""
        if not path or not path.strip():
            return False, "路径不能为空"

        path_obj = Path(path)

        if must_exist and not path_obj.exists():
            return False, f"目录不存在: {path}"

        if must_exist and not path_obj.is_dir():
            return False, f"路径不是目录: {path}"

        return True, ""

    @staticmethod
    def validate_file(path: str, must_exist: bool = True, extensions: Optional[List[str]] = None) -> Tuple[bool, str]:
        """验证文件路径"""
        if not path or not path.strip():
            return False, "文件路径不能为空"

        path_obj = Path(path)

        if must_exist and not path_obj.exists():
            return False, f"文件不存在: {path}"

        if must_exist and not path_obj.is_file():
            return False, f"路径不是文件: {path}"

        if extensions:
            if path_obj.suffix.lower() not in [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' for ext in extensions]:
                return False, f"文件扩展名必须是: {', '.join(extensions)}"

        return True, ""

    @staticmethod
    def validate_yolo_dataset(dataset_path: str) -> Tuple[bool, str]:
        """验证YOLO格式数据集"""
        path_obj = Path(dataset_path)

        if not path_obj.exists():
            return False, f"数据集路径不存在: {dataset_path}"

        if not path_obj.is_dir():
            return False, f"数据集路径必须是目录: {dataset_path}"

        yaml_files = list(path_obj.glob('*.yaml')) + list(path_obj.glob('*.yml'))
        if not yaml_files:
            return False, "未找到数据集配置文件 (*.yaml 或 *.yml)"

        data_yaml = yaml_files[0]
        try:
            with open(data_yaml, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            if 'train' not in data:
                return False, "数据集配置文件缺少 'train' 字段"

            if 'val' not in data:
                return False, "数据集配置文件缺少 'val' 字段"

            if 'nc' not in data:
                return False, "数据集配置文件缺少 'nc' (类别数量) 字段"

            if 'names' not in data:
                return False, "数据集配置文件缺少 'names' (类别名称) 字段"

            train_path = Path(data['train']) if os.path.isabs(data['train']) else path_obj / data['train']
            if not train_path.exists():
                return False, f"训练集路径不存在: {train_path}"

            val_path = Path(data['val']) if os.path.isabs(data['val']) else path_obj / data['val']
            if not val_path.exists():
                return False, f"验证集路径不存在: {val_path}"

        except Exception as e:
            return False, f"解析数据集配置文件失败: {str(e)}"

        return True, ""

    @staticmethod
    def validate_model_file(path: str) -> Tuple[bool, str]:
        """验证模型文件"""
        valid, msg = Validators.validate_file(path, must_exist=True, extensions=['.pt', '.yaml', '.yml'])
        if not valid:
            return False, msg

        return True, ""

    @staticmethod
    def validate_positive_int(value: any, name: str = "值", min_value: int = 1) -> Tuple[bool, str]:
        """验证正整数"""
        try:
            int_value = int(value)
            if int_value < min_value:
                return False, f"{name}必须大于等于 {min_value}"
            return True, ""
        except (ValueError, TypeError):
            return False, f"{name}必须是整数"

    @staticmethod
    def validate_float_range(value: any, name: str = "值", min_value: float = 0.0, max_value: float = 1.0) -> Tuple[bool, str]:
        """验证浮点数范围"""
        try:
            float_value = float(value)
            if float_value < min_value or float_value > max_value:
                return False, f"{name}必须在 {min_value} 到 {max_value} 之间"
            return True, ""
        except (ValueError, TypeError):
            return False, f"{name}必须是数字"

    @staticmethod
    def validate_image_file(path: str) -> Tuple[bool, str]:
        """验证图片文件"""
        return Validators.validate_file(
            path,
            must_exist=True,
            extensions=['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        )

    @staticmethod
    def validate_video_file(path: str) -> Tuple[bool, str]:
        """验证视频文件"""
        return Validators.validate_file(
            path,
            must_exist=True,
            extensions=['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
        )
