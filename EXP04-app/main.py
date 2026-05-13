import sys
from pathlib import Path
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QAction, QMessageBox
from PyQt5.QtGui import QIcon
from core.config import Config
from utils.logger import get_logger
from ui.dataset_widget import DatasetWidget
from ui.train_widget import TrainWidget
from ui.inference_widget import InferenceWidget
from ui.model_widget import ModelWidget


class MainWindow(QMainWindow):
    """主窗口"""

    def __init__(self):
        super().__init__()
        self.config = Config()
        self.logger = get_logger()
        self.init_ui()

    def init_ui(self):
        """初始化UI"""
        self.setWindowTitle('YOLO-MBS')

        # 设置应用图标
        logo_path = Path(__file__).parent / 'logo.png'
        if logo_path.exists():
            self.setWindowIcon(QIcon(str(logo_path)))

        self.setGeometry(100, 100,
                        self.config.get('ui.window_width', 1200),
                        self.config.get('ui.window_height', 800))

        # 创建菜单栏
        self.create_menu_bar()

        # 创建标签页
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # 添加各个功能模块
        self.dataset_widget = DatasetWidget()
        self.tabs.addTab(self.dataset_widget, "数据集管理")

        self.train_widget = TrainWidget()
        self.tabs.addTab(self.train_widget, "模型训练")

        self.inference_widget = InferenceWidget()
        self.tabs.addTab(self.inference_widget, "推理预测")

        self.model_widget = ModelWidget()
        self.tabs.addTab(self.model_widget, "模型管理")

        # 连接信号：数据集变化时刷新训练页面
        self.dataset_widget.dataset_imported.connect(self.train_widget.refresh_datasets)
        self.dataset_widget.dataset_deleted.connect(self.train_widget.refresh_datasets)

        # 创建状态栏
        self.statusBar().showMessage('就绪')

        self.logger.info('主窗口初始化完成')

    def create_menu_bar(self):
        """创建菜单栏"""
        menubar = self.menuBar()

        # 文件菜单
        file_menu = menubar.addMenu('文件')

        exit_action = QAction('退出', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # 帮助菜单
        help_menu = menubar.addMenu('帮助')

        about_action = QAction('关于', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def show_about(self):
        """显示关于对话框"""
        QMessageBox.about(
            self,
            "关于",
            "<h3>YOLO-MBS</h3>"
            f"<p>版本: {self.config.get('app.version', '1.0.0')}</p>"
            "<p>基于 YOLOv13 的目标检测模型训练和推理工具</p>"
            "<p>支持自定义数据集、训练参数配置、实时监控和模型推理</p>"
        )

    def closeEvent(self, event):
        """关闭事件"""
        reply = QMessageBox.question(
            self, '确认退出',
            '确定要退出应用吗?',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.logger.info('应用关闭')
            event.accept()
        else:
            event.ignore()


def main():
    """应用入口"""
    logger = get_logger()
    logger.info('应用启动')

    # 加载配置
    config = Config()
    config.ensure_directories()
    logger.info('配置加载完成')

    # 创建应用
    app = QApplication(sys.argv)

    # 设置应用图标
    logo_path = Path(__file__).parent / 'logo.png'
    if logo_path.exists():
        app.setWindowIcon(QIcon(str(logo_path)))

    # 应用现代科技风格样式表
    app.setStyleSheet("""
        /* 全局样式 */
        QWidget {
            background-color: #1a1a2e;
            color: #eaeaea;
            font-family: "Microsoft YaHei UI", "Segoe UI", Arial;
            font-size: 10pt;
        }

        /* 主窗口 */
        QMainWindow {
            background-color: #16213e;
        }

        /* 标签页 */
        QTabWidget::pane {
            border: 1px solid #0f3460;
            background-color: #1a1a2e;
            border-radius: 4px;
        }

        QTabBar::tab {
            background-color: #0f3460;
            color: #eaeaea;
            padding: 10px 20px;
            margin-right: 2px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }

        QTabBar::tab:selected {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                       stop:0 #e94560, stop:1 #c23a4f);
            color: white;
            font-weight: bold;
        }

        QTabBar::tab:hover:!selected {
            background-color: #16213e;
        }

        /* 按钮 */
        QPushButton {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                       stop:0 #0f3460, stop:1 #16213e);
            color: #eaeaea;
            border: 1px solid #e94560;
            border-radius: 4px;
            padding: 8px 16px;
            font-weight: bold;
        }

        QPushButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                       stop:0 #e94560, stop:1 #c23a4f);
            border: 1px solid #ff6b81;
        }

        QPushButton:pressed {
            background-color: #c23a4f;
        }

        QPushButton:disabled {
            background-color: #2a2a3e;
            color: #666;
            border: 1px solid #444;
        }

        /* 输入框 */
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
            background-color: #0f3460;
            color: #eaeaea;
            border: 1px solid #16213e;
            border-radius: 4px;
            padding: 6px;
            selection-background-color: #e94560;
        }

        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
            border: 1px solid #e94560;
        }

        /* 下拉框 */
        QComboBox::drop-down {
            border: none;
            width: 20px;
        }

        QComboBox::down-arrow {
            image: none;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 5px solid #e94560;
            margin-right: 5px;
        }

        QComboBox QAbstractItemView {
            background-color: #0f3460;
            color: #eaeaea;
            selection-background-color: #e94560;
            border: 1px solid #e94560;
        }

        /* 表格 */
        QTableWidget {
            background-color: #0f3460;
            alternate-background-color: #16213e;
            gridline-color: #1a1a2e;
            border: 1px solid #16213e;
            border-radius: 4px;
        }

        QTableWidget::item {
            padding: 5px;
            color: #eaeaea;
        }

        QTableWidget::item:selected {
            background-color: #e94560;
            color: white;
        }

        QHeaderView::section {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                       stop:0 #e94560, stop:1 #c23a4f);
            color: white;
            padding: 8px;
            border: none;
            font-weight: bold;
        }

        /* 文本编辑器 */
        QTextEdit {
            background-color: #0f3460;
            color: #eaeaea;
            border: 1px solid #16213e;
            border-radius: 4px;
            selection-background-color: #e94560;
        }

        /* 复选框 */
        QCheckBox {
            spacing: 8px;
            color: #eaeaea;
        }

        QCheckBox::indicator {
            width: 18px;
            height: 18px;
            border: 2px solid #e94560;
            border-radius: 3px;
            background-color: #0f3460;
        }

        QCheckBox::indicator:checked {
            background-color: #e94560;
            image: none;
        }

        /* 分组框 */
        QGroupBox {
            border: 2px solid #0f3460;
            border-radius: 6px;
            margin-top: 12px;
            padding-top: 10px;
            font-weight: bold;
            color: #e94560;
        }

        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 10px;
            padding: 0 5px;
            background-color: #1a1a2e;
        }

        /* 进度条 */
        QProgressBar {
            border: 1px solid #0f3460;
            border-radius: 4px;
            text-align: center;
            background-color: #0f3460;
            color: white;
            font-weight: bold;
        }

        QProgressBar::chunk {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                       stop:0 #e94560, stop:1 #ff6b81);
            border-radius: 3px;
        }

        /* 标签 */
        QLabel {
            color: #eaeaea;
            background-color: transparent;
        }

        /* 滚动条 */
        QScrollBar:vertical {
            background-color: #0f3460;
            width: 12px;
            border-radius: 6px;
        }

        QScrollBar::handle:vertical {
            background-color: #e94560;
            border-radius: 6px;
            min-height: 20px;
        }

        QScrollBar::handle:vertical:hover {
            background-color: #ff6b81;
        }

        QScrollBar:horizontal {
            background-color: #0f3460;
            height: 12px;
            border-radius: 6px;
        }

        QScrollBar::handle:horizontal {
            background-color: #e94560;
            border-radius: 6px;
            min-width: 20px;
        }

        QScrollBar::handle:horizontal:hover {
            background-color: #ff6b81;
        }

        QScrollBar::add-line, QScrollBar::sub-line {
            border: none;
            background: none;
        }

        /* 菜单栏 */
        QMenuBar {
            background-color: #16213e;
            color: #eaeaea;
            border-bottom: 2px solid #e94560;
        }

        QMenuBar::item {
            padding: 8px 12px;
            background-color: transparent;
        }

        QMenuBar::item:selected {
            background-color: #e94560;
        }

        QMenu {
            background-color: #0f3460;
            color: #eaeaea;
            border: 1px solid #e94560;
        }

        QMenu::item:selected {
            background-color: #e94560;
        }

        /* 状态栏 */
        QStatusBar {
            background-color: #16213e;
            color: #eaeaea;
            border-top: 1px solid #e94560;
        }

        /* 分割器 */
        QSplitter::handle {
            background-color: #e94560;
        }

        QSplitter::handle:horizontal {
            width: 2px;
        }

        QSplitter::handle:vertical {
            height: 2px;
        }
    """)

    # 创建主窗口
    window = MainWindow()
    window.show()

    logger.info('主窗口显示')
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
