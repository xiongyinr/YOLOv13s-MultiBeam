import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Optional


class Logger:
    """日志管理类"""

    _instance = None
    _logger = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, log_dir: Optional[str] = None, log_level: int = logging.INFO):
        if self._logger is not None:
            return

        if log_dir is None:
            log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'app_data', 'logs')

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._logger = logging.getLogger('YOLOApp')
        self._logger.setLevel(log_level)

        if not self._logger.handlers:
            log_file = self.log_dir / f"app_{datetime.now().strftime('%Y%m%d')}.log"

            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(log_level)

            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)

            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )

            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            self._logger.addHandler(file_handler)
            self._logger.addHandler(console_handler)

    def debug(self, message: str):
        """记录调试信息"""
        self._logger.debug(message)

    def info(self, message: str):
        """记录一般信息"""
        self._logger.info(message)

    def warning(self, message: str):
        """记录警告信息"""
        self._logger.warning(message)

    def error(self, message: str, exc_info: bool = False):
        """记录错误信息"""
        self._logger.error(message, exc_info=exc_info)

    def critical(self, message: str, exc_info: bool = False):
        """记录严重错误信息"""
        self._logger.critical(message, exc_info=exc_info)


def get_logger() -> Logger:
    """获取日志实例"""
    return Logger()
