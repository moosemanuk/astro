# ui/dialogs/curves_dialog.py
import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)
from scipy.interpolate import CubicSpline, PchipInterpolator


class InteractiveCurveWidget(pg.PlotWidget):
    """Plot widget providing interactive curve adjustment via draggable control points."""

    curve_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackground("#1e1e1e")
        self.showGrid(x=True, y=True, alpha=0.3)
        self.setRange(xRange=[0, 1], yRange=[0, 1], padding=0.02)
        self.setLimits(xMin=0, xMax=1, yMin=0, yMax=1)

        # 1. Plot Items
        self.hist_item = pg.PlotCurveItem(
            pen=pg.mkPen(color=(100, 100, 100, 150), width=1),
            fillLevel=0,
            fillBrush=(80, 80, 80, 50),
        )
        self.addItem(self.hist_item)

        # Baseline diagonal (identity line)
        self.identity_line = pg.PlotCurveItem(
            [0, 1], [0, 1], pen=pg.mkPen(color=(80, 80, 80), style=Qt.PenStyle.DashLine)
        )
        self.addItem(self.identity_line)

        # Active Spline Curve
        self.spline_curve = pg.PlotCurveItem(
            pen=pg.mkPen(color="#4da6ff", width=2)
        )
        self.addItem(self.spline_curve)

        # Scatter Item for Control Points
        self.nodes_scatter = pg.ScatterPlotItem(
            size=10, pen=pg.mkPen("#ffffff"), brush=pg.mkBrush("#4da6ff")
        )
        self.addItem(self.nodes_scatter)

        # Default Control Points: (0,0) and (1,1)
        self.control_points = [(0.0, 0.0), (1.0, 1.0)]
        self.selected_point_idx = None

        self._update_spline()

    def _sanitize_points(self):
        """Ensures points are strictly increasing in X without pushing endpoints out of range."""
        pts = sorted(self.control_points, key=lambda p: p[0])
        clean_pts = []
        last_x = -1.0
        eps = 1e-4

        for x, y in pts:
            if x <= last_x:
                x = last_x + eps
            clean_pts.append((min(1.0, x), y))
            last_x = x

        return clean_pts

    def set_histogram(self, image_data: np.ndarray):
        """Calculates and renders background histogram overlay."""
        flat_data = image_data.ravel()
        if flat_data.max() > 1.0:
            flat_data = flat_data / flat_data.max()

        counts, bin_edges = np.histogram(flat_data, bins=256, range=(0, 1))
        
        counts_log = np.log1p(counts)
        max_c = counts_log.max()
        counts_norm = counts_log / max_c if max_c > 0 else counts_log

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        self.hist_item.setData(bin_centers, counts_norm)

    def _update_spline(self):
        """Re-evaluates curve using PCHIP (prevents wild overshoots/oscillations)."""
        pts = self._sanitize_points()
        x_pts, y_pts = zip(*pts)

        x_dense = np.linspace(0, 1, 256)

        if len(pts) <= 2:
            y_dense = np.interp(x_dense, x_pts, y_pts)
        else:
            try:
                interp = PchipInterpolator(x_pts, y_pts)
                y_dense = np.clip(interp(x_dense), 0, 1)
            except Exception:
                y_dense = np.interp(x_dense, x_pts, y_pts)

        self.spline_curve.setData(x_dense, y_dense)
        self.nodes_scatter.setData([p[0] for p in pts], [p[1] for p in pts])
        self.curve_changed.emit()

    def evaluate_lut(self, num_points: int = 65536) -> np.ndarray:
        """Evaluates LUT safely with PCHIP interpolation."""
        pts = self._sanitize_points()
        x_pts, y_pts = zip(*pts)

        x_dense = np.linspace(0, 1, num_points)

        if len(pts) <= 2:
            return np.interp(x_dense, x_pts, y_pts)

        try:
            interp = PchipInterpolator(x_pts, y_pts)
            return np.clip(interp(x_dense), 0, 1)
        except Exception:
            return np.interp(x_dense, x_pts, y_pts)

    def reset_curve(self):
        """Resets curve back to identity line."""
        self.control_points = [(0.0, 0.0), (1.0, 1.0)]
        self._update_spline()

    # --- Mouse Interaction Events ---
    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            pos = self.plotItem.vb.mapSceneToView(ev.position())
            x, y = float(pos.x()), float(pos.y())

            threshold = 0.04
            for idx, (px, py) in enumerate(self.control_points):
                if abs(px - x) < threshold and abs(py - y) < threshold:
                    self.selected_point_idx = idx
                    ev.accept()
                    return

            if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                inserted_idx = 0
                for idx, (px, _) in enumerate(self.control_points):
                    if x > px:
                        inserted_idx = idx + 1
                
                self.control_points.insert(inserted_idx, (x, y))
                self.selected_point_idx = inserted_idx
                self._update_spline()
                ev.accept()
                return

        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self.selected_point_idx is not None and ev.buttons() == Qt.MouseButton.LeftButton:
            pos = self.plotItem.vb.mapSceneToView(ev.position())
            y = float(np.clip(pos.y(), 0.0, 1.0))

            if self.selected_point_idx == 0:
                x = 0.0
            elif self.selected_point_idx == len(self.control_points) - 1:
                x = 1.0
            else:
                prev_x = self.control_points[self.selected_point_idx - 1][0]
                next_x = self.control_points[self.selected_point_idx + 1][0]
                min_x = prev_x + 0.005
                max_x = next_x - 0.005
                
                if min_x < max_x:
                    x = float(np.clip(pos.x(), min_x, max_x))
                else:
                    x = self.control_points[self.selected_point_idx][0]

            self.control_points[self.selected_point_idx] = (x, y)
            self._update_spline()
            ev.accept()
            return

        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        self.selected_point_idx = None
        super().mouseReleaseEvent(ev)


class CurvesTransformationDialog(QDialog):
    """Dialog providing interactive curves adjustment with embedded live preview window."""

    def __init__(self, image_data: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Curves Transformation")
        self.resize(950, 500)

        self.original_image = image_data
        self.parent_app = parent
        self._preview_initialized = False

        # 1. Generate lightweight preview downsample (max side ~768px)
        self.preview_image = self._create_preview_subsample(image_data, max_dim=768)

        # Main horizontal layout
        main_layout = QHBoxLayout(self)

        # --- LEFT PANEL: Curves & Controls ---
        left_layout = QVBoxLayout()

        lbl_info = QLabel("Click plot to add points. Drag nodes to shape stretch curve.")
        lbl_info.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        left_layout.addWidget(lbl_info)

        self.curve_widget = InteractiveCurveWidget(self)
        self.curve_widget.set_histogram(self.original_image)
        left_layout.addWidget(self.curve_widget)

        btn_layout = QHBoxLayout()
        btn_reset = QPushButton("Reset Curve")
        btn_reset.clicked.connect(self.curve_widget.reset_curve)
        btn_layout.addWidget(btn_reset)
        btn_layout.addStretch()

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        btn_layout.addWidget(self.button_box)

        left_layout.addLayout(btn_layout)
        main_layout.addLayout(left_layout, stretch=1)

        # --- RIGHT PANEL: Embedded Live Preview Window ---
        preview_box = QVBoxLayout()
        lbl_preview = QLabel("Live Preview")
        lbl_preview.setStyleSheet("color: #aaaaaa; font-weight: bold; font-size: 11px;")
        preview_box.addWidget(lbl_preview)

        self.preview_view = pg.ImageView()
        # Clean up image view controls
        self.preview_view.ui.histogram.hide()
        self.preview_view.ui.roiBtn.hide()
        self.preview_view.ui.menuBtn.hide()
        preview_box.addWidget(self.preview_view)

        main_layout.addLayout(preview_box, stretch=1)

        # Connect live curve signal
        self.curve_widget.curve_changed.connect(self._apply_preview)
        
        # Initial render
        self._apply_preview()

    def _create_preview_subsample(self, img: np.ndarray, max_dim: int = 768) -> np.ndarray:
        """Fast downsampling preserving original aspect ratio and orientation."""
        h, w = img.shape[:2]
        max_side = max(h, w)
        if max_side <= max_dim:
            return img.copy()
        
        step = int(np.ceil(max_side / max_dim))
        return img[::step, ::step].copy()

    def _apply_preview(self):
        """Applies transformation to downsampled image and displays with native orientation."""
        transformed_preview = self.get_transformed_image(source_image=self.preview_image)
        
        # Swap row/col axes so NumPy (Height, Width) maps to PyQtGraph (X, Y)
        oriented_img = np.swapaxes(transformed_preview, 0, 1)

        # Auto-fit the view frame on first load; keep zoom stable during curve dragging
        first_render = not self._preview_initialized
        self.preview_view.setImage(
            oriented_img,
            autoRange=first_render,
            autoLevels=first_render
        )

        if first_render:
            self._preview_initialized = True

    def get_transformed_image(self, source_image: np.ndarray = None) -> np.ndarray:
        """Applies current curve LUT to target image."""
        if source_image is None:
            source_image = self.original_image

        lut = self.curve_widget.evaluate_lut(num_points=65536)
        
        data = source_image.copy().astype(np.float32)
        d_min, d_max = data.min(), data.max()
        
        if d_max > d_min:
            norm_data = (data - d_min) / (d_max - d_min)
        else:
            norm_data = data

        indices = (norm_data * (len(lut) - 1)).astype(np.int32)
        transformed = lut[indices]

        return (transformed * (d_max - d_min) + d_min).astype(source_image.dtype)

    def accept(self):
        """Commit transformation to main parent app window when OK is pressed."""
        if self.parent_app and hasattr(self.parent_app, "current_image_data"):
            transformed_full = self.get_transformed_image(self.original_image)
            self.parent_app.current_image_data = transformed_full
            self.parent_app.autostretched_cache = None
            self.parent_app.update_display(autoRange=False)
        super().accept()