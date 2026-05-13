"""
数据集管理界面
"""
import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QTableWidget, QTableWidgetItem, QLabel, QFileDialog,
    QMessageBox, QDialog, QFormLayout, QLineEdit, QSpinBox,
    QTextEdit, QGroupBox
)
from PyQt5.QtCore import Qt, pyqtSignal
from core.dataset_manager import DatasetManager
from utils.logger import get_logger

logger = get_logger()


class DatasetDialog(QDialog):
    """数据集导入对话框"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("导入数据集")
        self.setModal(True)
        self.resize(500, 400)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # 表单
        form_layout = QFormLayout()

        # 数据集名称
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("例如: coco_subset")
        form_layout.addRow("数据集名称:", self.name_input)

        # 数据集根目录
        root_layout = QHBoxLayout()
        self.root_input = QLineEdit()
        self.root_input.setPlaceholderText("选择包含 images/ 和 labels/ 的目录")
        browse_btn = QPushButton("浏览...")
        browse_btn.clicked.connect(self.browse_root)
        root_layout.addWidget(self.root_input)
        root_layout.addWidget(browse_btn)
        form_layout.addRow("数据集根目录:", root_layout)

        # 类别数量
        self.nc_input = QSpinBox()
        self.nc_input.setRange(1, 1000)
        self.nc_input.setValue(80)
        form_layout.addRow("类别数量:", self.nc_input)

        # 类别名称
        self.names_input = QTextEdit()
        self.names_input.setPlaceholderText("每行一个类别名称，例如:\nperson\ncar\ndog")
        self.names_input.setMaximumHeight(150)
        form_layout.addRow("类别名称:", self.names_input)

        layout.addLayout(form_layout)

        # 提示信息
        info_group = QGroupBox("数据集格式要求")
        info_layout = QVBoxLayout()
        info_text = QLabel(
            "• 目录结构:\n"
            "  dataset_root/\n"
            "    ├── images/\n"
            "    │   ├── train/\n"
            "    │   └── val/\n"
            "    └── labels/\n"
            "        ├── train/\n"
            "        └── val/\n\n"
            "• 标注格式: YOLO格式 (class x_center y_center width height)"
        )
        info_text.setWordWrap(True)
        info_layout.addWidget(info_text)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        # 按钮
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        ok_btn = QPushButton("导入")
        ok_btn.clicked.connect(self.accept)
        ok_btn.setDefault(True)
        btn_layout.addWidget(ok_btn)

        layout.addLayout(btn_layout)

    def browse_root(self):
        """浏览数据集根目录"""
        directory = QFileDialog.getExistingDirectory(
            self, "选择数据集根目录"
        )
        if directory:
            self.root_input.setText(directory)

    def get_data(self):
        """获取表单数据"""
        names_text = self.names_input.toPlainText().strip()
        names = [name.strip() for name in names_text.split('\n') if name.strip()]

        return {
            'name': self.name_input.text().strip(),
            'root': self.root_input.text().strip(),
            'nc': self.nc_input.value(),
            'names': names
        }


class DatasetWidget(QWidget):
    """数据集管理界面"""

    dataset_selected = pyqtSignal(str)  # 数据集被选中信号
    dataset_imported = pyqtSignal()  # 数据集导入成功信号
    dataset_deleted = pyqtSignal()  # 数据集删除成功信号

    def __init__(self, parent=None):
        super().__init__(parent)
        self.dataset_manager = DatasetManager()
        self.init_ui()
        self.refresh_table()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # 标题
        title = QLabel("📊 数据集管理")
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

        # 工具栏
        toolbar = QHBoxLayout()

        import_btn = QPushButton("导入数据集")
        import_btn.clicked.connect(self.import_dataset)
        toolbar.addWidget(import_btn)

        refresh_btn = QPushButton("刷新")
        refresh_btn.clicked.connect(self.refresh_table)
        toolbar.addWidget(refresh_btn)

        delete_btn = QPushButton("删除")
        delete_btn.clicked.connect(self.delete_dataset)
        toolbar.addWidget(delete_btn)

        toolbar.addStretch()
        layout.addLayout(toolbar)

        # 数据集表格
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["数据集名称", "类别数", "训练集", "验证集"])
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self.on_selection_changed)
        layout.addWidget(self.table)

        # 详细信息
        info_group = QGroupBox("数据集详情")
        info_layout = QVBoxLayout()
        self.info_label = QLabel("请选择一个数据集查看详情")
        self.info_label.setWordWrap(True)
        info_layout.addWidget(self.info_label)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

    def refresh_table(self):
        """刷新数据集表格"""
        try:
            datasets = self.dataset_manager.list_datasets()
            self.table.setRowCount(len(datasets))

            for i, dataset in enumerate(datasets):
                self.table.setItem(i, 0, QTableWidgetItem(dataset['name']))
                self.table.setItem(i, 1, QTableWidgetItem(str(dataset['nc'])))
                self.table.setItem(i, 2, QTableWidgetItem(dataset.get('train', 'N/A')))
                self.table.setItem(i, 3, QTableWidgetItem(dataset.get('val', 'N/A')))

            logger.info(f"刷新数据集列表，共 {len(datasets)} 个数据集")

        except Exception as e:
            logger.error(f"刷新数据集列表失败: {e}")
            QMessageBox.critical(self, "错误", f"刷新数据集列表失败:\n{str(e)}")

    def import_dataset(self):
        """导入数据集"""
        dialog = DatasetDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            data = dialog.get_data()

            # 验证输入
            if not data['name']:
                QMessageBox.warning(self, "警告", "请输入数据集名称")
                return

            if not data['root']:
                QMessageBox.warning(self, "警告", "请选择数据集根目录")
                return

            if not data['names']:
                QMessageBox.warning(self, "警告", "请输入类别名称")
                return

            if len(data['names']) != data['nc']:
                QMessageBox.warning(
                    self, "警告",
                    f"类别名称数量 ({len(data['names'])}) 与类别数量 ({data['nc']}) 不匹配"
                )
                return

            try:
                # 使用 create_dataset_config 创建配置
                train_path = os.path.join(data['root'], 'images', 'train')
                val_path = os.path.join(data['root'], 'images', 'val')

                success, message, config_path = self.dataset_manager.create_dataset_config(
                    train_path=train_path,
                    val_path=val_path,
                    class_names=data['names']
                )

                if success:
                    QMessageBox.information(
                        self, "成功",
                        f"数据集导入成功!\n配置文件: {config_path}"
                    )
                    self.refresh_table()
                    self.dataset_imported.emit()  # 发射导入成功信号
                else:
                    QMessageBox.warning(self, "失败", message)

            except Exception as e:
                logger.error(f"导入数据集失败: {e}")
                QMessageBox.critical(self, "错误", f"导入数据集失败:\n{str(e)}")

    def delete_dataset(self):
        """删除数据集"""
        selected_items = self.table.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "警告", "请先选择要删除的数据集")
            return

        dataset_name = self.table.item(selected_items[0].row(), 0).text()

        reply = QMessageBox.question(
            self, "确认删除",
            f"确定要删除数据集 '{dataset_name}' 吗?\n注意: 这只会删除配置文件，不会删除实际数据。",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            # 获取数据集路径
            datasets = self.dataset_manager.list_datasets()
            dataset = next((d for d in datasets if d['name'] == dataset_name), None)

            if not dataset:
                QMessageBox.warning(self, "错误", "找不到数据集")
                return

            try:
                success, message = self.dataset_manager.delete_dataset(dataset['path'])

                if success:
                    QMessageBox.information(self, "成功", message)
                    self.refresh_table()
                    self.dataset_deleted.emit()  # 发射删除成功信号
                    self.info_label.setText("请选择一个数据集查看详情")
                else:
                    QMessageBox.warning(self, "失败", message)

            except Exception as e:
                logger.error(f"删除数据集失败: {e}")
                QMessageBox.critical(self, "错误", f"删除数据集失败:\n{str(e)}")

    def on_selection_changed(self):
        """选中项改变"""
        selected_items = self.table.selectedItems()
        if not selected_items:
            self.info_label.setText("请选择一个数据集查看详情")
            return

        dataset_name = self.table.item(selected_items[0].row(), 0).text()

        try:
            datasets = self.dataset_manager.list_datasets()
            dataset = next((d for d in datasets if d['name'] == dataset_name), None)

            if dataset:
                info_text = f"<b>数据集名称:</b> {dataset['name']}<br>"
                info_text += f"<b>类别数量:</b> {dataset['nc']}<br>"
                info_text += f"<b>训练集:</b> {dataset.get('train', 'N/A')}<br>"
                info_text += f"<b>验证集:</b> {dataset.get('val', 'N/A')}<br>"

                if 'names' in dataset and dataset['names']:
                    # names 可能是字典或列表
                    names = dataset['names']
                    if isinstance(names, dict):
                        names_list = [names[i] for i in sorted(names.keys())]
                    else:
                        names_list = names

                    names_str = ', '.join(names_list[:10])
                    if len(names_list) > 10:
                        names_str += f" ... (共 {len(names_list)} 个)"
                    info_text += f"<b>类别名称:</b> {names_str}"

                self.info_label.setText(info_text)
                self.dataset_selected.emit(dataset_name)

        except Exception as e:
            logger.error(f"获取数据集详情失败: {e}")
            self.info_label.setText(f"获取详情失败: {str(e)}")
