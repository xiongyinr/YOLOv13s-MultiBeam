import os
import yaml
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from utils.validators import Validators
from utils.logger import get_logger


class DatasetManager:
    """数据集管理器"""

    def __init__(self, datasets_dir: str = 'app_data/datasets'):
        self.datasets_dir = Path(datasets_dir)
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger()

    def validate_dataset(self, dataset_path: str) -> Tuple[bool, str, Optional[Dict]]:
        """验证数据集并返回配置信息"""
        valid, msg = Validators.validate_yolo_dataset(dataset_path)
        if not valid:
            return False, msg, None

        try:
            dataset_path_obj = Path(dataset_path)
            yaml_files = list(dataset_path_obj.glob('*.yaml')) + list(dataset_path_obj.glob('*.yml'))
            data_yaml = yaml_files[0]

            with open(data_yaml, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            train_path = Path(config['train']) if os.path.isabs(config['train']) else dataset_path_obj / config['train']
            val_path = Path(config['val']) if os.path.isabs(config['val']) else dataset_path_obj / config['val']

            train_images = list(train_path.glob('*.jpg')) + list(train_path.glob('*.png')) + list(train_path.glob('*.jpeg'))
            val_images = list(val_path.glob('*.jpg')) + list(val_path.glob('*.png')) + list(val_path.glob('*.jpeg'))

            dataset_info = {
                'name': dataset_path_obj.name,
                'path': str(dataset_path_obj),
                'config_file': str(data_yaml),
                'nc': config['nc'],
                'names': config['names'],
                'train_path': str(train_path),
                'val_path': str(val_path),
                'train_count': len(train_images),
                'val_count': len(val_images),
                'test_path': str(Path(config['test']) if 'test' in config and os.path.isabs(config['test']) else dataset_path_obj / config.get('test', '')) if 'test' in config else None
            }

            self.logger.info(f"数据集验证成功: {dataset_info['name']}, 训练集: {dataset_info['train_count']}张, 验证集: {dataset_info['val_count']}张")
            return True, "数据集验证成功", dataset_info

        except Exception as e:
            self.logger.error(f"验证数据集时出错: {str(e)}", exc_info=True)
            return False, f"验证数据集时出错: {str(e)}", None

    def import_dataset(self, source_path: str, dataset_name: Optional[str] = None) -> Tuple[bool, str, Optional[str]]:
        """导入数据集到管理目录"""
        valid, msg, dataset_info = self.validate_dataset(source_path)
        if not valid:
            return False, msg, None

        try:
            if dataset_name is None:
                dataset_name = dataset_info['name']

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            target_name = f"{dataset_name}_{timestamp}"
            target_path = self.datasets_dir / target_name

            if target_path.exists():
                return False, f"目标路径已存在: {target_path}", None

            shutil.copytree(source_path, target_path)

            config_file = target_path / Path(dataset_info['config_file']).name
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            config['train'] = str(target_path / Path(dataset_info['train_path']).name)
            config['val'] = str(target_path / Path(dataset_info['val_path']).name)
            if dataset_info['test_path']:
                config['test'] = str(target_path / Path(dataset_info['test_path']).name)

            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

            self.logger.info(f"数据集导入成功: {target_name}")
            return True, "数据集导入成功", str(config_file)

        except Exception as e:
            self.logger.error(f"导入数据集时出错: {str(e)}", exc_info=True)
            return False, f"导入数据集时出错: {str(e)}", None

    def list_datasets(self) -> List[Dict]:
        """列出所有已导入的数据集"""
        datasets = []

        for dataset_dir in self.datasets_dir.iterdir():
            if not dataset_dir.is_dir():
                continue

            yaml_files = list(dataset_dir.glob('*.yaml')) + list(dataset_dir.glob('*.yml'))
            if not yaml_files:
                continue

            try:
                with open(yaml_files[0], 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)

                # 处理 names 字段，兼容字典和列表格式
                names = config.get('names', [])
                if isinstance(names, dict):
                    # 如果是字典格式 {0: 'cube', 1: 'ball'}，转换为列表
                    names = [names[i] for i in sorted(names.keys())]

                datasets.append({
                    'name': dataset_dir.name,
                    'path': str(dataset_dir),
                    'config_file': str(yaml_files[0]),
                    'nc': config.get('nc', 0),
                    'names': names,
                    'train': config.get('train', ''),
                    'val': config.get('val', ''),
                    'created_time': datetime.fromtimestamp(dataset_dir.stat().st_ctime).strftime('%Y-%m-%d %H:%M:%S')
                })

            except Exception as e:
                self.logger.warning(f"读取数据集配置失败: {dataset_dir.name}, 错误: {str(e)}")
                continue

        return sorted(datasets, key=lambda x: x['created_time'], reverse=True)

    def delete_dataset(self, dataset_path: str) -> Tuple[bool, str]:
        """删除数据集"""
        try:
            dataset_path_obj = Path(dataset_path)

            if not dataset_path_obj.exists():
                return False, "数据集不存在"

            if not str(dataset_path_obj).startswith(str(self.datasets_dir)):
                return False, "只能删除管理目录中的数据集"

            shutil.rmtree(dataset_path_obj)
            self.logger.info(f"数据集删除成功: {dataset_path_obj.name}")
            return True, "数据集删除成功"

        except Exception as e:
            self.logger.error(f"删除数据集时出错: {str(e)}", exc_info=True)
            return False, f"删除数据集时出错: {str(e)}"

    def create_dataset_config(self, train_path: str, val_path: str, class_names: List[str],
                             test_path: Optional[str] = None, output_path: Optional[str] = None) -> Tuple[bool, str, Optional[str]]:
        """创建数据集配置文件"""
        try:
            valid, msg = Validators.validate_directory(train_path, must_exist=True)
            if not valid:
                return False, f"训练集路径无效: {msg}", None

            valid, msg = Validators.validate_directory(val_path, must_exist=True)
            if not valid:
                return False, f"验证集路径无效: {msg}", None

            if test_path:
                valid, msg = Validators.validate_directory(test_path, must_exist=True)
                if not valid:
                    return False, f"测试集路径无效: {msg}", None

            config = {
                'train': train_path,
                'val': val_path,
                'nc': len(class_names),
                'names': {i: name for i, name in enumerate(class_names)}
            }

            if test_path:
                config['test'] = test_path

            if output_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = self.datasets_dir / f"dataset_{timestamp}" / "data.yaml"
                output_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

            self.logger.info(f"数据集配置文件创建成功: {output_path}")
            return True, "配置文件创建成功", str(output_path)

        except Exception as e:
            self.logger.error(f"创建配置文件时出错: {str(e)}", exc_info=True)
            return False, f"创建配置文件时出错: {str(e)}", None

    def get_dataset_config_path(self, dataset_name: str) -> Optional[str]:
        """获取数据集配置文件路径"""
        datasets = self.list_datasets()
        dataset = next((d for d in datasets if d['name'] == dataset_name), None)
        if dataset:
            return dataset['config_file']
        return None
