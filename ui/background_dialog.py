from typing import Callable, Optional
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QMessageBox,
    QProgressBar,
    QSpinBox,
    QVBoxLayout,
)

from processing.background import extract_background_poly
from ui.processing_worker import ProcessingWorker


class BackgroundExtractionDialog(QDialog):

    def __init__(self, image_data=None, on_apply: Optional[Callable] = None, parent=None):
        super().__init__(parent)
        self.image_data = image_data
        self.on_apply = on_apply
        self.result_data = None
        self.worker = None

        self.setWindowTitle("Background Extraction Parameters")
        self.setMinimumWidth(320)

        layout = QVBoxLayout(self)

        self.group = QGroupBox("Model Parameters")
        form = QFormLayout(self.group)
        form.setSpacing(10)

        # 1. Number of Sample Points
        self.spin_points = QSpinBox()
        self.spin_points.setRange(9, 400)
        self.spin_points.setValue(64)
        self.spin_points.setSingleStep(5)
        form.addRow("Sample Points:", self.spin_points)

        # 2. Polynomial Degree
        self.combo_deg = QComboBox()
        self.combo_deg.addItem("Degree 1 (Linear Plane)", 1)
        self.combo_deg.addItem("Degree 2 (Parabolic / Vignetting)", 2)
        self.combo_deg.addItem("Degree 3 (Complex Gradient)", 3)
        self.combo_deg.addItem("Degree 4 (High Order Surface)", 4)
        self.combo_deg.setCurrentIndex(1)  # Default Degree 2
        form.addRow("Polynomial Degree:", self.combo_deg)

        # 3. Sample Radius
        self.spin_radius = QSpinBox()
        self.spin_radius.setRange(2, 50)
        self.spin_radius.setValue(8)
        self.spin_radius.setSuffix(" px")
        form.addRow("Sample Box Radius:", self.spin_radius)

        layout.addWidget(self.group)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Standard OK / Cancel Buttons
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.accepted.connect(self._on_ok_clicked)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

    def get_params(self) -> dict:
        """Returns the configured extraction parameters as a dictionary."""
        return {
            "num_points": self.spin_points.value(),
            "degree": self.combo_deg.currentData(),
            "sample_radius": self.spin_radius.value(),
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
            target_func=extract_background_poly,
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
        QMessageBox.critical(self, "Background Extraction Error", f"Failed to extract background:\n\n{error_msg}")