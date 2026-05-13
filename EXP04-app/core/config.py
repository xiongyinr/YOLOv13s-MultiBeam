import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """应用配置管理类"""

    def __init__(self, config_dir: Optional[str] = None):
        if config_dir is None:
            config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'app_data')

        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)

        self.config_file = self.config_dir / 'config.yaml'
        self.default_config = self._get_default_config()
        self.config = self._load_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'app': {
                'name': 'YOLO训练助手',
                'version': '1.0.0',
                'language': 'zh_CN'
            },
            'paths': {
                'models_dir': str(self.config_dir / 'models'),
                'datasets_dir': str(self.config_dir / 'datasets'),
                'logs_dir': str(self.config_dir / 'logs'),
                'temp_dir': str(self.config_dir / 'temp')
            },
            'training': {
                'default_epochs': 100,
                'default_batch': 16,
                'default_imgsz': 640,
                'default_device': 'auto',
                'default_optimizer': 'auto',
                'save_period': 10
            },
            'inference': {
                'default_conf': 0.25,
                'default_iou': 0.7,
                'default_max_det': 300,
                'save_results': True
            },
            'ui': {
                'theme': 'light',
                'window_width': 1200,
                'window_height': 800
            }
        }

    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                return self._merge_config(self.default_config, config)
            except Exception as e:
                print(f"加载配置文件失败: {e}, 使用默认配置")
                return self.default_config.copy()
        else:
            self.save_config(self.default_config)
            return self.default_config.copy()

    def _merge_config(self, default: Dict, loaded: Dict) -> Dict:
        """合并配置，确保所有默认键都存在"""
        result = default.copy()
        for key, value in loaded.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_config(result[key], value)
            else:
                result[key] = value
        return result

    def save_config(self, config: Optional[Dict[str, Any]] = None):
        """保存配置到文件"""
        if config is None:
            config = self.config

        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
        except Exception as e:
            print(f"保存配置文件失败: {e}")

    def get(self, key_path: str, default: Any = None) -> Any:
        """获取配置值，支持点号分隔的路径"""
        keys = key_path.split('.')
        value = self.config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def set(self, key_path: str, value: Any, save: bool = True):
        """设置配置值，支持点号分隔的路径"""
        keys = key_path.split('.')
        config = self.config

        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        config[keys[-1]] = value

        if save:
            self.save_config()

    def ensure_directories(self):
        """确保所有必需的目录存在"""
        for path_key in ['models_dir', 'datasets_dir', 'logs_dir', 'temp_dir']:
            path = self.get(f'paths.{path_key}')
            if path:
                Path(path).mkdir(parents=True, exist_ok=True)
