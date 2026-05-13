from PyQt5.QtCore import QThread, pyqtSignal
from ultralytics import YOLO
from pathlib import Path
import cv2
from typing import Optional
from utils.logger import get_logger


class InferenceWorker(QThread):
    """推理工作线程"""

    # 信号定义
    progress_updated = pyqtSignal(int, str)  # 进度百分比, 状态消息
    frame_processed = pyqtSignal(object, int, int)  # 处理后的帧, 当前帧号, 总帧数
    inference_completed = pyqtSignal(bool, str, str)  # 成功/失败, 消息, 输出路径
    error_occurred = pyqtSignal(str)  # 错误消息

    def __init__(self):
        super().__init__()
        self.logger = get_logger()
        self.model = None
        self.is_running = False
        self.should_stop = False

        # 推理参数
        self.model_path: Optional[str] = None
        self.source: Optional[str] = None  # 图片或视频路径
        self.conf_threshold: float = 0.25
        self.iou_threshold: float = 0.45
        self.img_size: int = 640
        self.device: str = '0'
        self.save_output: bool = True
        self.output_dir: str = 'runs/predict'
        self.show_labels: bool = True
        self.show_conf: bool = True
        self.line_width: Optional[int] = None

    def set_parameters(self, params: dict):
        """设置推理参数"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def run(self):
        """执行推理"""
        self.is_running = True
        self.should_stop = False

        try:
            self.progress_updated.emit(0, "加载模型...")
            self.logger.info(f"开始推理: 模型={self.model_path}, 源={self.source}")

            # 加载模型
            self.model = YOLO(self.model_path)

            source_path = Path(self.source)
            if not source_path.exists():
                raise FileNotFoundError(f"源文件不存在: {self.source}")

            # 判断是图片还是视频
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
            video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}

            if source_path.suffix.lower() in image_extensions:
                self._process_image()
            elif source_path.suffix.lower() in video_extensions:
                self._process_video()
            else:
                raise ValueError(f"不支持的文件格式: {source_path.suffix}")

        except Exception as e:
            error_msg = f"推理过程中出错: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self.error_occurred.emit(error_msg)
            self.inference_completed.emit(False, error_msg, "")

        finally:
            self.is_running = False
            self.model = None

    def _process_image(self):
        """处理单张图片"""
        self.progress_updated.emit(10, "处理图片...")

        results = self.model.predict(
            source=self.source,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.img_size,
            device=self.device,
            save=self.save_output,
            project=self.output_dir,
            name='image',
            exist_ok=True,
            show_labels=self.show_labels,
            show_conf=self.show_conf,
            line_width=self.line_width,
        )

        if self.should_stop:
            self.progress_updated.emit(100, "推理已停止")
            self.inference_completed.emit(False, "推理被用户停止", "")
            return

        # 获取结果
        result = results[0]
        annotated_frame = result.plot()

        # 发送处理后的帧
        self.frame_processed.emit(annotated_frame, 1, 1)

        # 获取输出路径
        output_path = ""
        if self.save_output:
            output_path = str(Path(self.output_dir) / 'image' / Path(self.source).name)

        self.progress_updated.emit(100, "推理完成")
        self.inference_completed.emit(True, f"检测到 {len(result.boxes)} 个目标", output_path)
        self.logger.info(f"图片推理完成: 检测到 {len(result.boxes)} 个目标")

    def _process_video(self):
        """处理视频"""
        self.progress_updated.emit(10, "打开视频文件...")

        # 打开视频
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {self.source}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.logger.info(f"视频信息: {total_frames}帧, {fps}fps, {width}x{height}")

        # 准备输出视频
        output_path = ""
        writer = None
        if self.save_output:
            output_dir = Path(self.output_dir) / 'video'
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(output_dir / Path(self.source).name)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        try:
            frame_count = 0
            detection_count = 0

            while cap.isOpened() and not self.should_stop:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # 推理
                results = self.model.predict(
                    source=frame,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    imgsz=self.img_size,
                    device=self.device,
                    verbose=False,
                    show_labels=self.show_labels,
                    show_conf=self.show_conf,
                    line_width=self.line_width,
                )

                result = results[0]
                annotated_frame = result.plot()
                detection_count += len(result.boxes)

                # 保存帧
                if writer is not None:
                    writer.write(annotated_frame)

                # 发送处理后的帧（每10帧发送一次以减少UI更新频率）
                if frame_count % 10 == 0 or frame_count == total_frames:
                    self.frame_processed.emit(annotated_frame, frame_count, total_frames)

                # 更新进度
                progress = int((frame_count / total_frames) * 90) + 10
                self.progress_updated.emit(progress, f"处理中: {frame_count}/{total_frames} 帧")

            if self.should_stop:
                self.progress_updated.emit(100, "推理已停止")
                self.inference_completed.emit(False, "推理被用户停止", "")
                self.logger.info("视频推理被用户停止")
            else:
                self.progress_updated.emit(100, "推理完成")
                avg_detections = detection_count / frame_count if frame_count > 0 else 0
                self.inference_completed.emit(
                    True,
                    f"处理完成: {frame_count}帧, 平均每帧检测 {avg_detections:.1f} 个目标",
                    output_path
                )
                self.logger.info(f"视频推理完成: {frame_count}帧, 总检测 {detection_count} 个目标")

        finally:
            cap.release()
            if writer is not None:
                writer.release()

    def stop(self):
        """停止推理"""
        if self.is_running:
            self.should_stop = True
            self.logger.info("请求停止推理")
