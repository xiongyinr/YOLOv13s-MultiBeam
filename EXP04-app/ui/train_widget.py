"""
训练配置和监控界面
"""
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox, QComboBox,
    QLineEdit, QCheckBox, QTextEdit, QProgressBar, QSplitter,
    QMessageBox, QFileDialog
)
from PyQt5.QtCore import Qt, pyqtSlot
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from workers.train_worker import TrainWorker
from core.dataset_manager import DatasetManager
from core.model_manager import ModelManager
from utils.logger import get_logger

logger = get_logger()


class TrainingPlot(FigureCanvas):
    """训练曲线图"""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(10, 6))
        super().__init__(self.fig)
        self.setParent(parent)

        # 创建子图
        self.ax_loss = self.fig.add_subplot(2, 2, 1)
        self.ax_map = self.fig.add_subplot(2, 2, 2)
        self.ax_precision = self.fig.add_subplot(2, 2, 3)
        self.ax_recall = self.fig.add_subplot(2, 2, 4)

        # 设置标题
        self.ax_loss.set_title('Loss')
        self.ax_map.set_title('mAP')
        self.ax_precision.set_title('Precision')
        self.ax_recall.set_title('Recall')

        # 设置标签
        for ax in [self.ax_loss, self.ax_map, self.ax_precision, self.ax_recall]:
            ax.set_xlabel('Epoch')
            ax.grid(True, alpha=0.3)

        self.fig.tight_layout()

        # 数据存储
        self.epochs = []
        self.train_loss = []
        self.val_loss = []
        self.map50 = []
        self.map50_95 = []
        self.precision = []
        self.recall = []

    def update_plot(self, metrics: dict):
        """更新图表"""
        epoch = metrics.get('epoch', 0)

        # 添加数据
        if epoch not in self.epochs:
            self.epochs.append(epoch)
            self.train_loss.append(metrics.get('train_loss', 0))
            self.val_loss.append(metrics.get('val_loss', 0))
            self.map50.append(metrics.get('map50', 0))
            self.map50_95.append(metrics.get('map50_95', 0))
            self.precision.append(metrics.get('precision', 0))
            self.recall.append(metrics.get('recall', 0))

        # 清空并重绘
        self.ax_loss.clear()
        self.ax_map.clear()
        self.ax_precision.clear()
        self.ax_recall.clear()

        # Loss
        self.ax_loss.plot(self.epochs, self.train_loss, 'b-', label='Train Loss')
        self.ax_loss.plot(self.epochs, self.val_loss, 'r-', label='Val Loss')
        self.ax_loss.set_title('Loss')
        self.ax_loss.set_xlabel('Epoch')
        self.ax_loss.legend()
        self.ax_loss.grid(True, alpha=0.3)

        # mAP
        self.ax_map.plot(self.epochs, self.map50, 'g-', label='mAP@0.5')
        self.ax_map.plot(self.epochs, self.map50_95, 'orange', label='mAP@0.5:0.95')
        self.ax_map.set_title('mAP')
        self.ax_map.set_xlabel('Epoch')
        self.ax_map.legend()
        self.ax_map.grid(True, alpha=0.3)

        # Precision
        self.ax_precision.plot(self.epochs, self.precision, 'm-')
        self.ax_precision.set_title('Precision')
        self.ax_precision.set_xlabel('Epoch')
        self.ax_precision.grid(True, alpha=0.3)

        # Recall
        self.ax_recall.plot(self.epochs, self.recall, 'c-')
        self.ax_recall.set_title('Recall')
        self.ax_recall.set_xlabel('Epoch')
        self.ax_recall.grid(True, alpha=0.3)

        self.fig.tight_layout()
        self.draw()

    def clear_plot(self):
        """清空图表"""
        self.epochs.clear()
        self.train_loss.clear()
        self.val_loss.clear()
        self.map50.clear()
        self.map50_95.clear()
        self.precision.clear()
        self.recall.clear()

        for ax in [self.ax_loss, self.ax_map, self.ax_precision, self.ax_recall]:
            ax.clear()
            ax.grid(True, alpha=0.3)

        self.draw()


class TrainWidget(QWidget):
    """训练配置和监控界面"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.dataset_manager = DatasetManager()
        self.model_manager = ModelManager('models')
        self.train_worker = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # 标题
        title = QLabel("🚀 模型训练")
        title.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            color: #e94560;
            padding: 10px;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                       stop:0 rgba(233, 69, 96, 0.1),
                                       stop:1 rgba(233, 69, 96, 0.05));
            border-left: 4px solid #e94560;
            border-radius: 4px;
        """)
        layout.addWidget(title)

        # 分割器
        splitter = QSplitter(Qt.Horizontal)

        # 左侧：配置面板
        config_widget = QWidget()
        config_layout = QVBoxLayout(config_widget)

        # 基础配置
        basic_group = QGroupBox("基础配置")
        basic_layout = QFormLayout()

        # 数据集选择
        self.dataset_combo = QComboBox()
        self.refresh_datasets()
        basic_layout.addRow("数据集:", self.dataset_combo)

        # 模型选择
        model_layout = QHBoxLayout()
        self.model_combo = QComboBox()
        self.refresh_models()
        model_layout.addWidget(self.model_combo)
        browse_model_btn = QPushButton("浏览...")
        browse_model_btn.clicked.connect(self.browse_model)
        model_layout.addWidget(browse_model_btn)
        basic_layout.addRow("预训练模型:", model_layout)

        # Epochs
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(100)
        basic_layout.addRow("训练轮数:", self.epochs_spin)

        # Batch size
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 128)
        self.batch_spin.setValue(16)
        basic_layout.addRow("批次大小:", self.batch_spin)

        # Image size
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(320, 1280)
        self.imgsz_spin.setValue(640)
        self.imgsz_spin.setSingleStep(32)
        basic_layout.addRow("图像尺寸:", self.imgsz_spin)

        # Device
        self.device_combo = QComboBox()
        self.device_combo.addItems(['0', 'cpu'])
        basic_layout.addRow("设备:", self.device_combo)

        basic_group.setLayout(basic_layout)
        config_layout.addWidget(basic_group)

        # 高级配置
        advanced_group = QGroupBox("高级配置")
        advanced_layout = QFormLayout()

        # Optimizer
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(['SGD', 'Adam', 'AdamW'])
        advanced_layout.addRow("优化器:", self.optimizer_combo)

        # Learning rate
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 1.0)
        self.lr_spin.setValue(0.01)
        self.lr_spin.setDecimals(4)
        self.lr_spin.setSingleStep(0.001)
        advanced_layout.addRow("学习率:", self.lr_spin)

        # Momentum
        self.momentum_spin = QDoubleSpinBox()
        self.momentum_spin.setRange(0.0, 1.0)
        self.momentum_spin.setValue(0.937)
        self.momentum_spin.setDecimals(3)
        self.momentum_spin.setSingleStep(0.01)
        advanced_layout.addRow("动量:", self.momentum_spin)

        # Weight decay
        self.weight_decay_spin = QDoubleSpinBox()
        self.weight_decay_spin.setRange(0.0, 0.1)
        self.weight_decay_spin.setValue(0.0005)
        self.weight_decay_spin.setDecimals(4)
        self.weight_decay_spin.setSingleStep(0.0001)
        advanced_layout.addRow("权重衰减:", self.weight_decay_spin)

        # Workers
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(0, 16)
        self.workers_spin.setValue(8)
        advanced_layout.addRow("数据加载线程:", self.workers_spin)

        # 数据增强
        self.augment_check = QCheckBox("启用数据增强")
        self.augment_check.setChecked(True)
        advanced_layout.addRow("", self.augment_check)

        advanced_group.setLayout(advanced_layout)
        config_layout.addWidget(advanced_group)

        # 控制按钮
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始训练")
        self.start_btn.clicked.connect(self.start_training)
        btn_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("停止训练")
        self.stop_btn.clicked.connect(self.stop_training)
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.stop_btn)

        config_layout.addLayout(btn_layout)
        config_layout.addStretch()

        splitter.addWidget(config_widget)

        # 右侧：监控面板
        monitor_widget = QWidget()
        monitor_layout = QVBoxLayout(monitor_widget)

        # 进度条
        self.progress_bar = QProgressBar()
        monitor_layout.addWidget(self.progress_bar)

        # 状态标签
        self.status_label = QLabel("就绪")
        monitor_layout.addWidget(self.status_label)

        # 训练曲线
        self.plot = TrainingPlot()
        monitor_layout.addWidget(self.plot)

        # 日志输出
        log_group = QGroupBox("训练日志")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        monitor_layout.addWidget(log_group)

        splitter.addWidget(monitor_widget)

        # 设置分割比例
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        layout.addWidget(splitter)

    def refresh_datasets(self):
        """刷新数据集列表"""
        self.dataset_combo.clear()
        datasets = self.dataset_manager.list_datasets()
        for dataset in datasets:
            self.dataset_combo.addItem(dataset['name'])

    def refresh_models(self):
        """刷新模型列表"""
        self.model_combo.clear()
        # 添加预训练模型
        pretrained = self.model_manager.get_pretrained_models()
        for model in pretrained:
            self.model_combo.addItem(f"[预训练] {model}", model)

        # 添加本地模型
        local_models = self.model_manager.list_models()
        for model in local_models:
            self.model_combo.addItem(f"[本地] {model['name']}", model['path'])

    def browse_model(self):
        """浏览模型文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "", "PyTorch模型 (*.pt)"
        )
        if file_path:
            self.model_combo.addItem(f"[自定义] {file_path}", file_path)
            self.model_combo.setCurrentIndex(self.model_combo.count() - 1)

    def start_training(self):
        """开始训练"""
        # 验证配置
        if self.dataset_combo.currentText() == "":
            QMessageBox.warning(self, "警告", "请选择数据集")
            return

        if self.model_combo.currentText() == "":
            QMessageBox.warning(self, "警告", "请选择模型")
            return

        # 获取数据集配置路径
        dataset_name = self.dataset_combo.currentText()
        dataset_config = self.dataset_manager.get_dataset_config_path(dataset_name)

        if not dataset_config:
            QMessageBox.critical(self, "错误", f"找不到数据集配置: {dataset_name}")
            return

        # 获取模型路径
        model_path = self.model_combo.currentData()

        # 验证模型路径
        if not model_path:
            QMessageBox.warning(self, "警告", "请选择模型")
            return

        logger.info(f"训练参数 - 数据集配置: {dataset_config}, 模型路径: {model_path}")

        # 准备训练参数
        params = {
            'data_yaml': dataset_config,
            'model_path': model_path,
            'epochs': self.epochs_spin.value(),
            'batch_size': self.batch_spin.value(),
            'img_size': self.imgsz_spin.value(),
            'device': self.device_combo.currentText(),
            'optimizer': self.optimizer_combo.currentText(),
            'lr0': self.lr_spin.value(),
            'momentum': self.momentum_spin.value(),
            'weight_decay': self.weight_decay_spin.value(),
            'workers': self.workers_spin.value(),
            'project': 'runs/train',
            'name': f'{dataset_name}_{model_path.split("/")[-1].replace(".pt", "")}',
        }

        # 创建训练线程
        self.train_worker = TrainWorker()
        self.train_worker.set_parameters(params)

        # 连接信号
        self.train_worker.progress_updated.connect(self.on_progress_updated)
        self.train_worker.epoch_completed.connect(self.on_epoch_completed)
        self.train_worker.training_completed.connect(self.on_training_completed)
        self.train_worker.error_occurred.connect(self.on_error_occurred)

        # 清空日志和图表
        self.log_text.clear()
        self.plot.clear_plot()

        # 启动训练
        self.train_worker.start()

        # 更新UI状态
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("训练中...")

        logger.info(f"开始训练: 数据集={dataset_name}, 模型={model_path}")

    def stop_training(self):
        """停止训练"""
        if self.train_worker and self.train_worker.isRunning():
            self.train_worker.stop()
            self.status_label.setText("正在停止训练...")
            self.stop_btn.setEnabled(False)

    @pyqtSlot(int, str)
    def on_progress_updated(self, progress: int, message: str):
        """进度更新"""
        self.progress_bar.setValue(progress)
        self.log_text.append(message)

    @pyqtSlot(int, dict)
    def on_epoch_completed(self, epoch: int, metrics: dict):
        """epoch完成，更新指标"""
        self.plot.update_plot(metrics)

        # 更新状态标签
        total_epochs = metrics.get('total_epochs', self.epochs_spin.value())
        self.status_label.setText(f"训练中: Epoch {epoch}/{total_epochs}")

    @pyqtSlot(bool, str, str)
    def on_training_completed(self, success: bool, message: str, model_path: str):
        """训练完成"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(100 if success else 0)

        if success:
            self.status_label.setText("训练完成")
            QMessageBox.information(
                self, "训练完成",
                f"{message}\n\n模型保存路径:\n{model_path}"
            )
        else:
            self.status_label.setText("训练失败")
            QMessageBox.warning(self, "训练失败", message)

        logger.info(f"训练完成: success={success}, message={message}")

    @pyqtSlot(str)
    def on_error_occurred(self, error_message: str):
        """错误发生"""
        self.log_text.append(f"<span style='color: red;'>错误: {error_message}</span>")
        logger.error(f"训练错误: {error_message}")
