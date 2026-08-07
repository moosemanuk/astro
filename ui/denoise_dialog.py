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

from processing.wavelet_denoise import WaveletDenoising
from ui.processing_worker import ProcessingWorker


#def run_wavelet_denoise(image_data, params: dict, progress_callback=None):

def run_wavelet_denoise(image_data, strength: float = 1.0, progress_callback=None):
    """Worker function to execute WaveletDenoising off the UI thread."""
    wd = WaveletDenoising(
        normalize=False,
        wavelet='db3',
        level=2,
        thr_mode='soft',
        selected_level=2,
        method="universal",
        energy_perc=0.90
    )
    # Perform fit on input image/array
    denoised = wd.fit(image_data)
    
    # If strength < 1.0, blend back with the original image data
    if strength < 1.0:
        denoised = (image_data * (1.0 - strength)) + (denoised * strength)

    return denoised


class DenoiseDialog(QDialog):
    """Collects parameters and runs the Multiscale Wavelet denoising operation."""

    def __init__(self, image_data=None, on_apply: Optional[Callable] = None, parent=None):
        super().__init__(parent)
        self.image_data = image_data
        self.on_apply = on_apply
        self.result_data = None
        self.worker = None

        self.setWindowTitle("Multiscale Wavelet Denoise")
        self.setMinimumWidth(360)

        layout = QVBoxLayout(self)

        description = QLabel(
            "Applies Multiscale Starlet (À Trous) wavelet noise reduction. "
            "Attenuates high-frequency background grain while leaving stars and "
            "nebula structures intact."
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
        self.spin_strength.setValue(1.00)
        self.spin_strength.setToolTip(
            "1.00 applies full wavelet noise suppression. Lower values blend back original fine details."
        )
        form.addRow("Strength:", self.spin_strength)

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
        """Returns setting dictionary matching the denoise function signature."""
        return {
            "strength": self.spin_strength.value(),
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

        # Offload execution to ProcessingWorker thread
        self.worker = ProcessingWorker(
            target_func=run_wavelet_denoise,
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