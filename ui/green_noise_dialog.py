from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSlider, 
    QDoubleSpinBox, QComboBox, QGroupBox, QProgressBar, QDialogButtonBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from processing.green_noise import remove_green_noise


class GreenNoiseWorker(QThread):
    finished_signal = pyqtSignal(object)
    progress_signal = pyqtSignal(int)

    def __init__(self, image_data, strength, method):
        super().__init__()
        self.image_data = image_data
        self.strength = strength
        self.method = method

    def run(self):
        self.progress_signal.emit(30)
        # Execute SCNR processing
        result = remove_green_noise(
            self.image_data, 
            strength=self.strength, 
            method=self.method
        )
        self.progress_signal.emit(100)
        self.finished_signal.emit(result)


class GreenNoiseDialog(QDialog):
    def __init__(self, image_data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Green Noise Removal (SCNR)")
        self.setMinimumWidth(400)

        self.image_data = image_data
        self.parent_editor = parent
        self.processed_data = None
        self.worker = None

        self.init_ui()
        self.apply_preview()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(12)

        # --- Explainer Text (Clean layout matching Sharpen dialog) ---
        explainer = QLabel(
            "Subtractive Chromatic Noise Reduction (SCNR) removes unnatural green "
            "casts caused by Bayer matrix filters while preserving astronomical detail."
        )
        explainer.setWordWrap(True)
        # Inherit standard text styling without a container box background
        main_layout.addWidget(explainer)

        # --- Group Box Controls ---
        param_group = QGroupBox("Parameters")
        group_layout = QVBoxLayout(param_group)
        group_layout.setSpacing(10)

        # Method Selection
        method_layout = QHBoxLayout()
        method_label = QLabel("Method:")
        method_label.setFixedWidth(70)
        self.combo_method = QComboBox()
        self.combo_method.addItems(["Average Neutral", "Maximum Neutral"])
        self.combo_method.currentIndexChanged.connect(self.apply_preview)
        
        method_layout.addWidget(method_label)
        method_layout.addWidget(self.combo_method, stretch=1)
        group_layout.addLayout(method_layout)

        # Strength Slider & Spinbox
        strength_layout = QHBoxLayout()
        strength_label = QLabel("Strength:")
        strength_label.setFixedWidth(70)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(100)  # Default 1.0 (100%)
        self.slider.valueChanged.connect(self._on_slider_change)

        self.spinbox = QDoubleSpinBox()
        self.spinbox.setRange(0.0, 1.0)
        self.spinbox.setSingleStep(0.05)
        self.spinbox.setValue(1.0)
        self.spinbox.setDecimals(2)
        self.spinbox.setFixedWidth(70)
        self.spinbox.valueChanged.connect(self._on_spinbox_change)

        strength_layout.addWidget(strength_label)
        strength_layout.addWidget(self.slider, stretch=1)
        strength_layout.addWidget(self.spinbox)
        group_layout.addLayout(strength_layout)

        main_layout.addWidget(param_group)

        # --- Progress Bar (Hidden by default) ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(12)
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        # --- Dialog Buttons ---
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        main_layout.addWidget(self.button_box)

    def _on_slider_change(self, val):
        self.spinbox.blockSignals(True)
        self.spinbox.setValue(val / 100.0)
        self.spinbox.blockSignals(False)
        self.apply_preview()

    def _on_spinbox_change(self, val):
        self.slider.blockSignals(True)
        self.slider.setValue(int(val * 100))
        self.slider.blockSignals(False)
        self.apply_preview()

    def apply_preview(self):
        if self.image_data is None:
            return

        strength = self.spinbox.value()
        method = self.combo_method.currentText()

        self.processed_data = remove_green_noise(
            self.image_data, 
            strength=strength, 
            method=method
        )

        if self.parent_editor and hasattr(self.parent_editor, "current_image_data"):
            self.parent_editor.current_image_data = self.processed_data
            self.parent_editor.update_display(autoRange=False)

    def accept(self):
        """Show progress bar and start worker execution when OK is pressed."""
        self.button_box.setEnabled(False)
        self.combo_method.setEnabled(False)
        self.slider.setEnabled(False)
        self.spinbox.setEnabled(False)

        # Unhide and initialize progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        strength = self.spinbox.value()
        method = self.combo_method.currentText()

        # Run background thread execution
        self.worker = GreenNoiseWorker(self.image_data, strength, method)
        self.worker.progress_signal.connect(self.progress_bar.setValue)
        self.worker.finished_signal.connect(self._on_worker_finished)
        self.worker.start()

    def _on_worker_finished(self, result):
        self.processed_data = result
        super().accept()  # Close dialog when task completes

    def get_result(self):
        return self.processed_data