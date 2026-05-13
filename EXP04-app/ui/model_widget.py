"""
模型管理界面
"""
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTableWidget, QTableWidgetItem, QGroupBox, QMessageBox,
    QFileDialog, QInputDialog
)
from PyQt5.QtCore import Qt
from core.model_manager import ModelManager
from utils.logger import get_logger

logger = get_logger()


class ModelWidget(QWidget):
    """模型管理界面"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.model_manager = ModelManager('models')
        self.init_ui()
        self.refresh_table()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # 标题
        title = QLabel("🔧 模型管理")
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

        import_btn = QPushButton("导入模型")
        import_btn.clicked.connect(self.import_model)
        toolbar.addWidget(import_btn)

        export_btn = QPushButton("导出模型")
        export_btn.clicked.connect(self.export_model)
        toolbar.addWidget(export_btn)

        rename_btn = QPushButton("重命名")
        rename_btn.clicked.connect(self.rename_model)
        toolbar.addWidget(rename_btn)

        delete_btn = QPushButton("删除")
        delete_btn.clicked.connect(self.delete_model)
        toolbar.addWidget(delete_btn)

        refresh_btn = QPushButton("刷新")
        refresh_btn.clicked.connect(self.refresh_table)
        toolbar.addWidget(refresh_btn)

        toolbar.addStretch()
        layout.addLayout(toolbar)

        # 模型表格
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["模型名称", "大小(MB)", "创建时间", "修改时间"])
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self.on_selection_changed)
        layout.addWidget(self.table)

        # 详细信息
        info_group = QGroupBox("模型详情")
        info_layout = QVBoxLayout()
        self.info_label = QLabel("请选择一个模型查看详情")
        self.info_label.setWordWrap(True)
        info_layout.addWidget(self.info_label)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

    def refresh_table(self):
        """刷新模型表格"""
        try:
            models = self.model_manager.list_models()
            self.table.setRowCount(len(models))

            for i, model in enumerate(models):
                self.table.setItem(i, 0, QTableWidgetItem(model['name']))
                self.table.setItem(i, 1, QTableWidgetItem(str(model['size_mb'])))
                self.table.setItem(i, 2, QTableWidgetItem(model['created_time']))
                self.table.setItem(i, 3, QTableWidgetItem(model['modified_time']))

            logger.info(f"刷新模型列表，共 {len(models)} 个模型")

        except Exception as e:
            logger.error(f"刷新模型列表失败: {e}")
            QMessageBox.critical(self, "错误", f"刷新模型列表失败:\n{str(e)}")

    def import_model(self):
        """导入模型"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "", "PyTorch模型 (*.pt);;所有文件 (*.*)"
        )

        if not file_path:
            return

        # 询问模型名称
        name, ok = QInputDialog.getText(
            self, "输入模型名称", "模型名称:",
            text=file_path.split('/')[-1].replace('.pt', '')
        )

        if not ok or not name:
            return

        try:
            success, message, model_path = self.model_manager.import_model(file_path, name)

            if success:
                QMessageBox.information(self, "成功", f"模型导入成功!\n路径: {model_path}")
                self.refresh_table()
            else:
                QMessageBox.warning(self, "失败", message)

        except Exception as e:
            logger.error(f"导入模型失败: {e}")
            QMessageBox.critical(self, "错误", f"导入模型失败:\n{str(e)}")

    def export_model(self):
        """导出模型"""
        selected_items = self.table.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "警告", "请先选择要导出的模型")
            return

        model_name = self.table.item(selected_items[0].row(), 0).text()

        # 获取模型路径
        models = self.model_manager.list_models()
        model = next((m for m in models if m['name'] == model_name), None)

        if not model:
            QMessageBox.warning(self, "错误", "找不到模型")
            return

        # 选择导出目录
        target_dir = QFileDialog.getExistingDirectory(self, "选择导出目录")

        if not target_dir:
            return

        try:
            success, message = self.model_manager.export_model(model['path'], target_dir)

            if success:
                QMessageBox.information(self, "成功", message)
            else:
                QMessageBox.warning(self, "失败", message)

        except Exception as e:
            logger.error(f"导出模型失败: {e}")
            QMessageBox.critical(self, "错误", f"导出模型失败:\n{str(e)}")

    def rename_model(self):
        """重命名模型"""
        selected_items = self.table.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "警告", "请先选择要重命名的模型")
            return

        model_name = self.table.item(selected_items[0].row(), 0).text()

        # 获取模型路径
        models = self.model_manager.list_models()
        model = next((m for m in models if m['name'] == model_name), None)

        if not model:
            QMessageBox.warning(self, "错误", "找不到模型")
            return

        # 询问新名称
        new_name, ok = QInputDialog.getText(
            self, "重命名模型", "新名称:", text=model_name
        )

        if not ok or not new_name:
            return

        try:
            success, message, new_path = self.model_manager.rename_model(model['path'], new_name)

            if success:
                QMessageBox.information(self, "成功", message)
                self.refresh_table()
            else:
                QMessageBox.warning(self, "失败", message)

        except Exception as e:
            logger.error(f"重命名模型失败: {e}")
            QMessageBox.critical(self, "错误", f"重命名模型失败:\n{str(e)}")

    def delete_model(self):
        """删除模型"""
        selected_items = self.table.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "警告", "请先选择要删除的模型")
            return

        model_name = self.table.item(selected_items[0].row(), 0).text()

        reply = QMessageBox.question(
            self, "确认删除",
            f"确定要删除模型 '{model_name}' 吗?\n此操作不可恢复!",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            # 获取模型路径
            models = self.model_manager.list_models()
            model = next((m for m in models if m['name'] == model_name), None)

            if not model:
                QMessageBox.warning(self, "错误", "找不到模型")
                return

            try:
                success, message = self.model_manager.delete_model(model['path'])

                if success:
                    QMessageBox.information(self, "成功", message)
                    self.refresh_table()
                    self.info_label.setText("请选择一个模型查看详情")
                else:
                    QMessageBox.warning(self, "失败", message)

            except Exception as e:
                logger.error(f"删除模型失败: {e}")
                QMessageBox.critical(self, "错误", f"删除模型失败:\n{str(e)}")

    def on_selection_changed(self):
        """选中项改变"""
        selected_items = self.table.selectedItems()
        if not selected_items:
            self.info_label.setText("请选择一个模型查看详情")
            return

        model_name = self.table.item(selected_items[0].row(), 0).text()

        try:
            models = self.model_manager.list_models()
            model = next((m for m in models if m['name'] == model_name), None)

            if model:
                info_text = f"<b>模型名称:</b> {model['name']}<br>"
                info_text += f"<b>文件路径:</b> {model['path']}<br>"
                info_text += f"<b>文件大小:</b> {model['size_mb']} MB<br>"
                info_text += f"<b>创建时间:</b> {model['created_time']}<br>"
                info_text += f"<b>修改时间:</b> {model['modified_time']}"

                self.info_label.setText(info_text)

        except Exception as e:
            logger.error(f"获取模型详情失败: {e}")
            self.info_label.setText(f"获取详情失败: {str(e)}")
