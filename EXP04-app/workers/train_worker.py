from PyQt5.QtCore import QThread, pyqtSignal
from ultralytics import YOLO
from pathlib import Path
from typing import Dict, Optional
from utils.logger import get_logger


class TrainWorker(QThread):
    """训练工作线程"""

    # 信号定义
    progress_updated = pyqtSignal(int, str)  # 进度百分比, 状态消息
    epoch_completed = pyqtSignal(int, dict)  # epoch编号, 指标字典
    training_completed = pyqtSignal(bool, str, str)  # 成功/失败, 消息, 最佳模型路径
    error_occurred = pyqtSignal(str)  # 错误消息

    def __init__(self):
        super().__init__()
        self.logger = get_logger()
        self.model = None
        self.is_running = False
        self.should_stop = False

        # 训练参数
        self.model_path: Optional[str] = None
        self.data_yaml: Optional[str] = None
        self.epochs: int = 100
        self.batch_size: int = 16
        self.img_size: int = 640
        self.project: str = 'runs/train'
        self.name: str = 'exp'
        self.device: str = '0'
        self.workers: int = 8
        self.optimizer: str = 'SGD'
        self.lr0: float = 0.01
        self.lrf: float = 0.01
        self.momentum: float = 0.937
        self.weight_decay: float = 0.0005
        self.warmup_epochs: int = 3
        self.warmup_momentum: float = 0.8
        self.warmup_bias_lr: float = 0.1
        self.box: float = 7.5
        self.cls: float = 0.5
        self.dfl: float = 1.5
        self.hsv_h: float = 0.015
        self.hsv_s: float = 0.7
        self.hsv_v: float = 0.4
        self.degrees: float = 0.0
        self.translate: float = 0.1
        self.scale: float = 0.5
        self.shear: float = 0.0
        self.perspective: float = 0.0
        self.flipud: float = 0.0
        self.fliplr: float = 0.5
        self.mosaic: float = 1.0
        self.mixup: float = 0.0
        self.copy_paste: float = 0.0
        self.resume: bool = False
        self.pretrained: bool = True
        self.save_period: int = -1

    def set_parameters(self, params: Dict):
        """设置训练参数"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def run(self):
        """执行训练"""
        self.is_running = True
        self.should_stop = False

        try:
            self.progress_updated.emit(0, "初始化模型...")
            self.logger.info(f"开始训练: 模型={self.model_path}, 数据={self.data_yaml}")

            # 加载模型
            self.model = YOLO(self.model_path)

            self.progress_updated.emit(5, "准备训练数据...")

            # 构建训练参数
            train_args = {
                'data': self.data_yaml,
                'epochs': self.epochs,
                'batch': self.batch_size,
                'imgsz': self.img_size,
                'project': self.project,
                'name': self.name,
                'device': self.device,
                'workers': self.workers,
                'optimizer': self.optimizer,
                'lr0': self.lr0,
                'lrf': self.lrf,
                'momentum': self.momentum,
                'weight_decay': self.weight_decay,
                'warmup_epochs': self.warmup_epochs,
                'warmup_momentum': self.warmup_momentum,
                'warmup_bias_lr': self.warmup_bias_lr,
                'box': self.box,
                'cls': self.cls,
                'dfl': self.dfl,
                'hsv_h': self.hsv_h,
                'hsv_s': self.hsv_s,
                'hsv_v': self.hsv_v,
                'degrees': self.degrees,
                'translate': self.translate,
                'scale': self.scale,
                'shear': self.shear,
                'perspective': self.perspective,
                'flipud': self.flipud,
                'fliplr': self.fliplr,
                'mosaic': self.mosaic,
                'mixup': self.mixup,
                'copy_paste': self.copy_paste,
                'resume': self.resume,
                'pretrained': self.pretrained,
                'save_period': self.save_period,
                'verbose': True,
            }

            self.progress_updated.emit(10, f"开始训练 (共{self.epochs}轮)...")

            # 添加回调来监控训练进度
            def on_train_epoch_end(trainer):
                if self.should_stop:
                    trainer.stop = True
                    return

                epoch = trainer.epoch + 1
                progress = int((epoch / self.epochs) * 90) + 10

                # 提取指标
                metrics = {}
                if hasattr(trainer, 'metrics'):
                    # trainer.metrics 可能是字典或对象
                    metrics_dict = trainer.metrics if isinstance(trainer.metrics, dict) else getattr(trainer.metrics, 'results_dict', {})
                    metrics = {
                        'box_loss': float(trainer.loss_items[0]) if hasattr(trainer, 'loss_items') else 0.0,
                        'cls_loss': float(trainer.loss_items[1]) if hasattr(trainer, 'loss_items') else 0.0,
                        'dfl_loss': float(trainer.loss_items[2]) if hasattr(trainer, 'loss_items') else 0.0,
                        'precision': float(metrics_dict.get('metrics/precision(B)', 0.0)),
                        'recall': float(metrics_dict.get('metrics/recall(B)', 0.0)),
                        'mAP50': float(metrics_dict.get('metrics/mAP50(B)', 0.0)),
                        'mAP50-95': float(metrics_dict.get('metrics/mAP50-95(B)', 0.0)),
                    }

                self.progress_updated.emit(progress, f"训练中: Epoch {epoch}/{self.epochs}")
                self.epoch_completed.emit(epoch, metrics)

            # 添加回调
            self.model.add_callback('on_train_epoch_end', on_train_epoch_end)

            # 开始训练
            results = self.model.train(**train_args)

            if self.should_stop:
                self.progress_updated.emit(100, "训练已停止")
                self.training_completed.emit(False, "训练被用户停止", "")
                self.logger.info("训练被用户停止")
            else:
                # 获取最佳模型路径
                best_model_path = str(Path(self.project) / self.name / 'weights' / 'best.pt')

                self.progress_updated.emit(100, "训练完成")
                self.training_completed.emit(True, "训练成功完成", best_model_path)
                self.logger.info(f"训练完成: 最佳模型={best_model_path}")

        except Exception as e:
            error_msg = f"训练过程中出错: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self.error_occurred.emit(error_msg)
            self.training_completed.emit(False, error_msg, "")

        finally:
            self.is_running = False
            self.model = None

    def stop(self):
        """停止训练"""
        if self.is_running:
            self.should_stop = True
            self.logger.info("请求停止训练")
