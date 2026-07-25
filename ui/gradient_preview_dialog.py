# ui/dialogs/gradient_preview_dialog.py
import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QVBoxLayout


class GradientPreviewDialog(QDialog):
    """Dialog to display the extracted background gradient surface model in grayscale."""

    def __init__(self, bg_model: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Extracted Background Gradient Model")
        self.resize(600, 750)

        layout = QVBoxLayout(self)

        # Title Label
        lbl_info = QLabel("Extracted Background Model Preview")
        lbl_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_info.setStyleSheet("font-weight: bold; font-size: 13px; color: #4da6ff;")
        layout.addWidget(lbl_info)

        # Use ImageView to perfectly mirror the main editor's orientation handling
        self.image_view = pg.ImageView()

        # Hide built-in histogram, ROI button, and Menu button for a clean look
        self.image_view.ui.histogram.hide()
        self.image_view.ui.roiBtn.hide()
        self.image_view.ui.menuBtn.hide()

        layout.addWidget(self.image_view)

        # 1. Convert to 2D grayscale if a 3D RGB array was passed
        preview_data = bg_model
        if preview_data.ndim == 3 and preview_data.shape[2] in (3, 4):
            # Standard RGB to luminance conversion
            preview_data = np.mean(preview_data[:, :, :3], axis=2)

        # 2. Pass data to ImageView
        self.image_view.setImage(preview_data, autoLevels=True)

        # 3. Explicitly set a smooth Linear Grayscale colormap
        try:
            # Linear 0 (black) to 1 (white) color stops
            pos = np.array([0.0, 1.0])
            color = np.array([[0, 0, 0, 255], [255, 255, 255, 255]], dtype=np.ubyte)
            gray_map = pg.ColorMap(pos, color)
            self.image_view.setColorMap(gray_map)
        except Exception:
            pass

        # OK Button
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)