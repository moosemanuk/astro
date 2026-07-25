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
    QSpinBox,
    QVBoxLayout,
)

from processing.star_removal import remove_stars
from ui.processing_worker import ProcessingWorker


class StarRemovalDialog(QDialog):
    """Options dialog for local star removal."""

    def __init__(self, image_data=None, on_apply: Optional[Callable] = None, parent=None):
        super().__init__(parent)
        self.image_data = image_data
        self.on_apply = on_apply
        self.result_data = None
        self.worker = None

        self.setWindowTitle("Remove Stars")
        self.setMinimumWidth(360)
        layout = QVBoxLayout(self)

        note = QLabel(
            "Detects compact bright stars and fills them from their local surroundings. "
            "Higher detection thresholds remove fewer stars."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #aaaaaa;")
        layout.addWidget(note)

        self.group = QGroupBox("Star Removal Parameters")
        form = QFormLayout(self.group)
        form.setSpacing(10)

        self.spin_threshold = QDoubleSpinBox()
        self.spin_threshold.setRange(1.0, 20.0)
        self.spin_threshold.setSingleStep(0.5)
        self.spin_threshold.setValue(3.0)
        self.spin_threshold.setSuffix(" σ")
        form.addRow("Detection threshold:", self.spin_threshold)

        self.spin_star_radius = QSpinBox()
        self.spin_star_radius.setRange(1, 20)
        self.spin_star_radius.setValue(4)
        self.spin_star_radius.setSuffix(" px")
        form.addRow("Star radius:", self.spin_star_radius)

        self.spin_inpaint_radius = QSpinBox()
        self.spin_inpaint_radius.setRange(1, 15)
        self.spin_inpaint_radius.setValue(3)
        self.spin_inpaint_radius.setSuffix(" px")
        form.addRow("Inpaint radius:", self.spin_inpaint_radius)
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
            "detection_threshold": self.spin_threshold.value(),
            "star_radius": self.spin_star_radius.value(),
            "inpaint_radius": self.spin_inpaint_radius.value(),
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
            target_func=remove_stars,
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
        QMessageBox.critical(self, "Star Removal Error", f"Failed to remove stars:\n\n{error_msg}")