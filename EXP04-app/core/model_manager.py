import os
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from utils.validators import Validators
from utils.logger import get_logger


class ModelManager:
    """模型管理器"""

    def __init__(self, models_dir: str):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger()

    def list_models(self) -> List[Dict]:
        """列出所有模型"""
        models = []

        for model_file in self.models_dir.glob('*.pt'):
            try:
                file_stat = model_file.stat()
                models.append({
                    'name': model_file.stem,
                    'path': str(model_file),
                    'size': file_stat.st_size,
                    'size_mb': round(file_stat.st_size / (1024 * 1024), 2),
                    'created_time': datetime.fromtimestamp(file_stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S'),
                    'modified_time': datetime.fromtimestamp(file_stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                })
            except Exception as e:
                self.logger.warning(f"读取模型文件信息失败: {model_file.name}, 错误: {str(e)}")
                continue

        return sorted(models, key=lambda x: x['modified_time'], reverse=True)

    def import_model(self, source_path: str, model_name: Optional[str] = None) -> Tuple[bool, str, Optional[str]]:
        """导入模型到管理目录"""
        valid, msg = Validators.validate_model_file(source_path)
        if not valid:
            return False, msg, None

        try:
            source_path_obj = Path(source_path)

            if model_name is None:
                model_name = source_path_obj.stem

            # 添加时间戳避免重名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            target_name = f"{model_name}_{timestamp}.pt"
            target_path = self.models_dir / target_name

            if target_path.exists():
                return False, f"目标路径已存在: {target_path}", None

            shutil.copy2(source_path, target_path)

            self.logger.info(f"模型导入成功: {target_name}")
            return True, "模型导入成功", str(target_path)

        except Exception as e:
            self.logger.error(f"导入模型时出错: {str(e)}", exc_info=True)
            return False, f"导入模型时出错: {str(e)}", None

    def export_model(self, model_path: str, target_dir: str) -> Tuple[bool, str]:
        """导出模型到指定目录"""
        try:
            model_path_obj = Path(model_path)
            if not model_path_obj.exists():
                return False, "模型文件不存在"

            target_dir_obj = Path(target_dir)
            target_dir_obj.mkdir(parents=True, exist_ok=True)

            target_path = target_dir_obj / model_path_obj.name

            if target_path.exists():
                return False, f"目标文件已存在: {target_path}"

            shutil.copy2(model_path, target_path)

            self.logger.info(f"模型导出成功: {target_path}")
            return True, f"模型导出成功: {target_path}"

        except Exception as e:
            self.logger.error(f"导出模型时出错: {str(e)}", exc_info=True)
            return False, f"导出模型时出错: {str(e)}"

    def delete_model(self, model_path: str) -> Tuple[bool, str]:
        """删除模型"""
        try:
            model_path_obj = Path(model_path)

            if not model_path_obj.exists():
                return False, "模型文件不存在"

            if not str(model_path_obj).startswith(str(self.models_dir)):
                return False, "只能删除管理目录中的模型"

            model_path_obj.unlink()

            self.logger.info(f"模型删除成功: {model_path_obj.name}")
            return True, "模型删除成功"

        except Exception as e:
            self.logger.error(f"删除模型时出错: {str(e)}", exc_info=True)
            return False, f"删除模型时出错: {str(e)}"

    def rename_model(self, model_path: str, new_name: str) -> Tuple[bool, str, Optional[str]]:
        """重命名模型"""
        try:
            model_path_obj = Path(model_path)

            if not model_path_obj.exists():
                return False, "模型文件不存在", None

            if not str(model_path_obj).startswith(str(self.models_dir)):
                return False, "只能重命名管理目录中的模型", None

            # 确保新名称有.pt扩展名
            if not new_name.endswith('.pt'):
                new_name += '.pt'

            new_path = model_path_obj.parent / new_name

            if new_path.exists():
                return False, f"目标文件名已存在: {new_name}", None

            model_path_obj.rename(new_path)

            self.logger.info(f"模型重命名成功: {model_path_obj.name} -> {new_name}")
            return True, "模型重命名成功", str(new_path)

        except Exception as e:
            self.logger.error(f"重命名模型时出错: {str(e)}", exc_info=True)
            return False, f"重命名模型时出错: {str(e)}", None

    def get_model_info(self, model_path: str) -> Optional[Dict]:
        """获取模型详细信息"""
        try:
            model_path_obj = Path(model_path)

            if not model_path_obj.exists():
                return None

            file_stat = model_path_obj.stat()

            return {
                'name': model_path_obj.stem,
                'path': str(model_path_obj),
                'size': file_stat.st_size,
                'size_mb': round(file_stat.st_size / (1024 * 1024), 2),
                'created_time': datetime.fromtimestamp(file_stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S'),
                'modified_time': datetime.fromtimestamp(file_stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
            }

        except Exception as e:
            self.logger.error(f"获取模型信息时出错: {str(e)}", exc_info=True)
            return None

    def get_pretrained_models(self) -> List[str]:
        """获取预训练模型列表"""
        # YOLOv13 官方预训练模型
        return [
            'yolov13n.pt',
            'yolov13s.pt',
            'yolov13m.pt',
            'yolov13l.pt',
            'yolov13x.pt',
        ]

    def search_models(self, keyword: str) -> List[Dict]:
        """搜索模型"""
        all_models = self.list_models()
        keyword_lower = keyword.lower()

        return [
            model for model in all_models
            if keyword_lower in model['name'].lower()
        ]
