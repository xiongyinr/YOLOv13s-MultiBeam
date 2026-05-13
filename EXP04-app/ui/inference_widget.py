"""
推理预测界面
"""
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QGroupBox, QFormLayout, QComboBox, QDoubleSpinBox, QSpinBox,
    QFileDialog, QMessageBox, QTextEdit, QProgressBar, QSplitter,
    QListWidget, QListWidgetItem
)
from PyQt5.QtCore import Qt, pyqtSlot, QTimer
from PyQt5.QtGui import QPixmap, QImage
from workers.inference_worker import InferenceWorker
from core.model_manager import ModelManager
from utils.logger import get_logger
import cv2
import os

logger = get_logger()


class InferenceWidget(QWidget):
    """推理预测界面"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.model_manager = ModelManager('models')
        self.inference_worker = None
        self.current_source = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # 标题
        title = QLabel("🎯 推理预测")
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

        # 模型配置
        model_group = QGroupBox("模型配置")
        model_layout = QFormLayout()

        # 模型选择
        model_select_layout = QHBoxLayout()
        self.model_combo = QComboBox()
        self.refresh_models()
        model_select_layout.addWidget(self.model_combo)
        browse_model_btn = QPushButton("浏览...")
        browse_model_btn.clicked.connect(self.browse_model)
        model_select_layout.addWidget(browse_model_btn)
        model_layout.addRow("模型:", model_select_layout)

        # 置信度阈值
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.0, 1.0)
        self.conf_spin.setValue(0.25)
        self.conf_spin.setDecimals(2)
        self.conf_spin.setSingleStep(0.05)
        model_layout.addRow("置信度阈值:", self.conf_spin)

        # IOU阈值
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.0, 1.0)
        self.iou_spin.setValue(0.45)
        self.iou_spin.setDecimals(2)
        self.iou_spin.setSingleStep(0.05)
        model_layout.addRow("IOU阈值:", self.iou_spin)

        # 图像尺寸
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(320, 1280)
        self.imgsz_spin.setValue(640)
        self.imgsz_spin.setSingleStep(32)
        model_layout.addRow("图像尺寸:", self.imgsz_spin)

        # 设备
        self.device_combo = QComboBox()
        self.device_combo.addItems(['0', 'cpu'])
        model_layout.addRow("设备:", self.device_combo)

        model_group.setLayout(model_layout)
        config_layout.addWidget(model_group)

        # 输入源配置
        source_group = QGroupBox("输入源")
        source_layout = QVBoxLayout()

        # 输入类型选择
        type_layout = QHBoxLayout()
        self.image_btn = QPushButton("选择图片")
        self.image_btn.clicked.connect(self.select_images)
        type_layout.addWidget(self.image_btn)

        self.video_btn = QPushButton("选择视频")
        self.video_btn.clicked.connect(self.select_video)
        type_layout.addWidget(self.video_btn)

        source_layout.addLayout(type_layout)

        # 输入文件列表
        self.source_list = QListWidget()
        self.source_list.setMaximumHeight(150)
        source_layout.addWidget(self.source_list)

        # 清空按钮
        clear_btn = QPushButton("清空列表")
        clear_btn.clicked.connect(self.clear_sources)
        source_layout.addWidget(clear_btn)

        source_group.setLayout(source_layout)
        config_layout.addWidget(source_group)

        # 控制按钮
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始推理")
        self.start_btn.clicked.connect(self.start_inference)
        btn_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("停止推理")
        self.stop_btn.clicked.connect(self.stop_inference)
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.stop_btn)

        config_layout.addLayout(btn_layout)
        config_layout.addStretch()

        splitter.addWidget(config_widget)

        # 右侧：结果显示
        result_widget = QWidget()
        result_layout = QVBoxLayout(result_widget)

        # 进度条
        self.progress_bar = QProgressBar()
        result_layout.addWidget(self.progress_bar)

        # 状态标签
        self.status_label = QLabel("就绪")
        result_layout.addWidget(self.status_label)

        # 结果预览
        preview_group = QGroupBox("结果预览")
        preview_layout = QVBoxLayout()
        self.preview_label = QLabel("暂无预览")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(640, 480)
        self.preview_label.setStyleSheet("border: 1px solid #ccc; background-color: #f0f0f0;")
        self.preview_label.setScaledContents(False)
        preview_layout.addWidget(self.preview_label)
        preview_group.setLayout(preview_layout)
        result_layout.addWidget(preview_group)

        # 检测结果
        detection_group = QGroupBox("检测结果")
        detection_layout = QVBoxLayout()
        self.detection_text = QTextEdit()
        self.detection_text.setReadOnly(True)
        self.detection_text.setMaximumHeight(150)
        detection_layout.addWidget(self.detection_text)
        detection_group.setLayout(detection_layout)
        result_layout.addWidget(detection_group)

        splitter.addWidget(result_widget)

        # 设置分割比例
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        layout.addWidget(splitter)

    def refresh_models(self):
        """刷新模型列表"""
        self.model_combo.clear()

        # 添加本地模型
        local_models = self.model_manager.list_models()
        for model in local_models:
            self.model_combo.addItem(f"{model['name']}", model['path'])

        # 添加预训练模型
        pretrained = self.model_manager.get_pretrained_models()
        for model in pretrained:
            self.model_combo.addItem(f"[预训练] {model}", model)

    def browse_model(self):
        """浏览模型文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "", "PyTorch模型 (*.pt)"
        )
        if file_path:
            self.model_combo.addItem(f"[自定义] {file_path}", file_path)
            self.model_combo.setCurrentIndex(self.model_combo.count() - 1)

    def select_images(self):
        """选择图片"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "选择图片", "",
            "图片文件 (*.jpg *.jpeg *.png *.bmp);;所有文件 (*.*)"
        )
        if file_paths:
            for path in file_paths:
                self.source_list.addItem(path)

    def select_video(self):
        """选择视频"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频", "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*.*)"
        )
        if file_path:
            self.source_list.addItem(file_path)

    def clear_sources(self):
        """清空输入源列表"""
        self.source_list.clear()

    def start_inference(self):
        """开始推理"""
        # 验证配置
        if self.model_combo.currentText() == "":
            QMessageBox.warning(self, "警告", "请选择模型")
            return

        if self.source_list.count() == 0:
            QMessageBox.warning(self, "警告", "请选择输入源")
            return

        # 获取模型路径
        model_path = self.model_combo.currentData()

        # 获取输入源列表
        sources = []
        for i in range(self.source_list.count()):
            sources.append(self.source_list.item(i).text())

        # 准备推理参数
        params = {
            'model': model_path,
            'source': sources,
            'conf': self.conf_spin.value(),
            'iou': self.iou_spin.value(),
            'imgsz': self.imgsz_spin.value(),
            'device': self.device_combo.currentText(),
            'project': 'runs/predict',
            'name': 'exp',
            'save': True,
        }

        # 创建推理线程
        self.inference_worker = InferenceWorker()
        self.inference_worker.set_parameters(params)

        # 连接信号
        self.inference_worker.progress_updated.connect(self.on_progress_updated)
        self.inference_worker.frame_processed.connect(self.on_frame_processed)
        self.inference_worker.inference_completed.connect(self.on_inference_completed)
        self.inference_worker.error_occurred.connect(self.on_error_occurred)

        # 清空结果
        self.detection_text.clear()
        self.preview_label.setText("推理中...")

        # 启动推理
        self.inference_worker.start()

        # 更新UI状态
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("推理中...")

        logger.info(f"开始推理: 模型={model_path}, 输入源数量={len(sources)}")

    def stop_inference(self):
        """停止推理"""
        if self.inference_worker and self.inference_worker.isRunning():
            self.inference_worker.stop()
            self.status_label.setText("正在停止推理...")
            self.stop_btn.setEnabled(False)

    @pyqtSlot(int, str)
    def on_progress_updated(self, progress: int, message: str):
        """进度更新"""
        self.progress_bar.setValue(progress)
        self.status_label.setText(message)

    @pyqtSlot(object, list)
    def on_frame_processed(self, frame, detections: list):
        """帧处理完成"""
        # 显示检测结果
        if detections:
            result_text = f"检测到 {len(detections)} 个目标:\n"
            for det in detections:
                result_text += f"  - {det['class']}: {det['confidence']:.2f}\n"
            self.detection_text.append(result_text)

        # 显示预览图
        if frame is not None:
            self.display_frame(frame)

    @pyqtSlot(bool, str, str)
    def on_inference_completed(self, success: bool, message: str, output_dir: str):
        """推理完成"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(100 if success else 0)

        if success:
            self.status_label.setText("推理完成")
            QMessageBox.information(
                self, "推理完成",
                f"{message}\n\n结果保存路径:\n{output_dir}"
            )
        else:
            self.status_label.setText("推理失败")
            QMessageBox.warning(self, "推理失败", message)

        logger.info(f"推理完成: success={success}, message={message}")

    @pyqtSlot(str)
    def on_error_occurred(self, error_message: str):
        """错误发生"""
        self.detection_text.append(f"<span style='color: red;'>错误: {error_message}</span>")
        logger.error(f"推理错误: {error_message}")

    def display_frame(self, frame):
        """显示帧"""
        try:
            # 转换为RGB
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame

            # 转换为QImage
            h, w = frame_rgb.shape[:2]
            bytes_per_line = 3 * w
            q_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # 转换为QPixmap
            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(
                self.preview_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )

            self.preview_label.setPixmap(scaled_pixmap)

        except Exception as e:
            logger.error(f"显示帧失败: {e}")
