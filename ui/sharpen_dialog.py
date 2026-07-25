from typing import Callable, Optional
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QMessageBox,
    QProgressBar,
    QVBoxLayout,
)

from processing.sharpen import sharpen_image
from ui.processing_worker import ProcessingWorker


class SharpenDialog(QDialog):
    """Parameter dialog for the selective unsharp-mask tool."""

    def __init__(self, image_data=None, on_apply: Optional[Callable] = None, parent=None):
        super().__init__(parent)
        self.image_data = image_data
        self.on_apply = on_apply
        self.result_data = None
        self.worker = None

        self.setWindowTitle("Sharpen Image")
        self.setMinimumWidth(360)
        layout = QVBoxLayout(self)

        note = QLabel(
            "Uses fine-detail contrast to sharpen. Start conservatively; excessive "
            "strength can create dark halos around stars."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #aaaaaa;")
        layout.addWidget(note)

        self.group = QGroupBox("Sharpening Parameters")
        form = QFormLayout(self.group)
        form.setSpacing(10)

        self.spin_strength = QDoubleSpinBox()
        self.spin_strength.setRange(0.0, 5.0)
        self.spin_strength.setSingleStep(0.05)
        self.spin_strength.setDecimals(2)
        self.spin_strength.setValue(0.75)
        form.addRow("Strength:", self.spin_strength)

        self.spin_radius = QDoubleSpinBox()
        self.spin_radius.setRange(0.1, 20.0)
        self.spin_radius.setSingleStep(0.1)
        self.spin_radius.setDecimals(1)
        self.spin_radius.setValue(1.2)
        self.spin_radius.setSuffix(" px")
        form.addRow("Detail radius:", self.spin_radius)

        self.combo_target = QComboBox()
        self.combo_target.addItem("Both stellar and non-stellar", "both")
        self.combo_target.addItem("Stellar detail only", "stellar")
        self.combo_target.addItem("Non-stellar detail only", "non_stellar")
        form.addRow("Sharpen:", self.combo_target)

        self.spin_threshold = QDoubleSpinBox()
        self.spin_threshold.setRange(0.5, 20.0)
        self.spin_threshold.setSingleStep(0.5)
        self.spin_threshold.setDecimals(1)
        self.spin_threshold.setValue(3.0)
        self.spin_threshold.setSuffix(" σ")
        self.spin_threshold.setToolTip("Higher values make the stellar-only selection more restrictive.")
        form.addRow("Stellar threshold:", self.spin_threshold)
        layout.addWidget(self.group)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.accepted.connect(self._on_ok_clicked)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

    def get_params(self) -> dict:
        return {
            "strength": self.spin_strength.value(),
            "radius": self.spin_radius.value(),
            "target": self.combo_target.currentData(),
            "stellar_threshold": self.spin_threshold.value(),
        }

    def get_result(self):
        return self.result_data

    def _on_ok_clicked(self):
        if self.image_data is None:
            self.accept()
            return

        self.group.setEnabled(False)
        self.buttons.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.adjustSize()

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

        self.worker = ProcessingWorker(
            target_func=sharpen_image,
            image_data=self.image_data,
            params=self.get_params(),
            parent=self,
        )
        self.worker.progress_changed.connect(self.progress_bar.setValue)
        self.worker.finished_with_result.connect(self._on_processing_finished)
        self.worker.failed_with_error.connect(self._on_processing_failed)
        self.worker.start()

    def _on_processing_finished(self, result):
        self.result_data = result
        self.progress_bar.setValue(85)
        QApplication.processEvents()

        if self.on_apply:
            try:
                self.on_apply(result)
                QApplication.processEvents()
            except Exception as err:
                QApplication.restoreOverrideCursor()
                self.progress_bar.setVisible(False)
                self.group.setEnabled(True)
                self.buttons.setEnabled(True)
                QMessageBox.critical(self, "Display Error", f"Failed to update main interface:\n\n{err}")
                return

        self.progress_bar.setValue(100)
        QApplication.processEvents()
        QApplication.restoreOverrideCursor()
        self.accept()

    def _on_processing_failed(self, error_msg):
        QApplication.restoreOverrideCursor()
        self.progress_bar.setVisible(False)
        self.group.setEnabled(True)
        self.buttons.setEnabled(True)
        QMessageBox.critical(self, "Sharpen Error", f"Failed to sharpen image:\n\n{error_msg}")
