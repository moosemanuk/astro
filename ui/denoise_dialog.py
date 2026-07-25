from typing import Callable, Optional
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
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

from processing.denoise import denoise_image
from ui.processing_worker import ProcessingWorker


class DenoiseDialog(QDialog):
    """Collects parameters for the detail-preserving denoise operation."""

    def __init__(self, image_data=None, on_apply: Optional[Callable] = None, parent=None):
        super().__init__(parent)
        self.image_data = image_data
        self.on_apply = on_apply
        self.result_data = None
        self.worker = None

        self.setWindowTitle("Denoise Image")
        self.setMinimumWidth(360)

        layout = QVBoxLayout(self)

        description = QLabel(
            "Reduces fine grain while protecting stronger stars and nebula detail. "
            "Start with the defaults and increase strength gradually."
        )
        description.setWordWrap(True)
        description.setStyleSheet("color: #aaaaaa;")
        layout.addWidget(description)

        self.group = QGroupBox("Denoise Parameters")
        form = QFormLayout(self.group)
        form.setSpacing(10)

        self.spin_strength = QDoubleSpinBox()
        self.spin_strength.setRange(0.0, 1.0)
        self.spin_strength.setSingleStep(0.05)
        self.spin_strength.setDecimals(2)
        self.spin_strength.setValue(0.50)
        self.spin_strength.setToolTip("Higher values remove more fine-grain noise.")
        form.addRow("Strength:", self.spin_strength)

        self.spin_radius = QDoubleSpinBox()
        self.spin_radius.setRange(0.1, 10.0)
        self.spin_radius.setSingleStep(0.1)
        self.spin_radius.setDecimals(1)
        self.spin_radius.setValue(1.2)
        self.spin_radius.setSuffix(" px")
        self.spin_radius.setToolTip("The scale of noise to smooth, in pixels.")
        form.addRow("Smoothing radius:", self.spin_radius)

        self.spin_detail_threshold = QDoubleSpinBox()
        self.spin_detail_threshold.setRange(0.1, 10.0)
        self.spin_detail_threshold.setSingleStep(0.1)
        self.spin_detail_threshold.setDecimals(1)
        self.spin_detail_threshold.setValue(2.0)
        self.spin_detail_threshold.setSuffix(" σ")
        self.spin_detail_threshold.setToolTip("Higher values retain more stars and fine detail.")
        form.addRow("Detail protection:", self.spin_detail_threshold)

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
        """Returns the selected denoise settings."""
        return {
            "strength": self.spin_strength.value(),
            "smoothing_radius": self.spin_radius.value(),
            "detail_threshold": self.spin_detail_threshold.value(),
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
            target_func=denoise_image,
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
        QMessageBox.critical(self, "Denoise Error", f"Failed to denoise image:\n\n{error_msg}")
