# ui/dialogs/stretch_workbench.py
import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from processing import stretch


class StretchWorkbenchDialog(QDialog):
    """Multi-algorithm stretching workbench delegating processing to processing.stretch."""

    # UI Styling
    CONTROL_STYLE = """
        QComboBox, QDoubleSpinBox {
            background-color: #2b2b2b;
            color: #ffffff;
            border: 1px solid #555555;
            border-radius: 4px;
            min-height: 32px;
            font-size: 13px;
            padding-left: 8px;
            padding-right: 4px;
        }
        QComboBox:hover, QDoubleSpinBox:hover {
            border: 1px solid #777777;
        }
        QComboBox:focus, QDoubleSpinBox:focus {
            border: 1px solid #4da6ff;
        }
        QComboBox::drop-down {
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 24px;
            border-left: 1px solid #555555;
        }
        QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
            width: 20px;
            background-color: #383838;
            border-left: 1px solid #555555;
        }
        QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {
            background-color: #4a4a4a;
        }
        QLabel {
            font-size: 13px;
            color: #dddddd;
        }
        QPushButton {
            min-height: 36px;
            font-size: 13px;
            font-weight: bold;
            border-radius: 4px;
        }
    """

    ALGO_DESCRIPTIONS = {
        0: "<b>GHS (Generalised Hyperbolic Stretch):</b> Ideal for targeted stretching. Highly effective at opening up subtle background nebulosity while preventing bright star cores from blowing out.",
        1: "<b>Arcsinh Stretch:</b> Preserves color saturation during aggressive midtone stretching. Best used on star clusters, galaxies, and rich star fields where star colors tend to wash out.",
        2: "<b>Midtone Transfer Function (MTF):</b> Classical statistical curve stretch. Great for general-purpose midtone lifting and controlled shadow clipping on linear space images.",
        3: "<b>Logarithmic Stretch:</b> Extremely aggressive initial stretch. Best for revealing very faint outer halos (e.g., planetary nebulae) or extreme high-dynamic-range scenes.",
        4: "<b>Root Stretch (Power Law):</b> A smooth gamma-style curve. Excellent for soft, natural transitions in large emission nebulae and faint gas dust clouds.",
        5: "<b>Exponential Stretch:</b> Strong non-linear contrast booster. Best suited for late-stage adjustments or pulling dim signal out of high signal-to-noise images."
    }

    def __init__(self, image_data: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Stretch Workbench")
        self.resize(850, 840)
        self.setStyleSheet(self.CONTROL_STYLE)

        self.original_image = image_data.copy()
        self.current_stacked_image = image_data.copy()
        self.parent_app = parent
        self._preview_initialized = False

        self.preview_base = self._create_preview_subsample(self.current_stacked_image, max_dim=1024)
        self.preview_active = self.preview_base.copy()

        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(12)

        # 1. HEADER: Algorithm Selector & Status
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        
        lbl_algo = QLabel("<b>Stretch Algorithm:</b>")
        header_layout.addWidget(lbl_algo, alignment=Qt.AlignmentFlag.AlignVCenter)

        self.combo_algo = QComboBox()
        self.combo_algo.addItems([
            "Generalised Hyperbolic (GHS)",
            "Arcsinh Stretch",
            "Midtone Transfer Function (MTF)",
            "Logarithmic Stretch",
            "Root Stretch (Power Law)",
            "Exponential Stretch",
        ])
        self.combo_algo.currentIndexChanged.connect(self._on_algorithm_changed)
        header_layout.addWidget(self.combo_algo, stretch=1, alignment=Qt.AlignmentFlag.AlignVCenter)

        header_layout.addSpacing(20)

        self.lbl_info = QLabel("<b>Stack Count:</b> 0 Stretches Applied")
        self.lbl_info.setStyleSheet("color: #4da6ff;")
        header_layout.addWidget(self.lbl_info, alignment=Qt.AlignmentFlag.AlignVCenter)

        main_layout.addLayout(header_layout)

        # 2. DYNAMIC DESCRIPTION BOX
        self.lbl_description = QLabel()
        self.lbl_description.setWordWrap(True)
        self.lbl_description.setStyleSheet("""
            background-color: #222222;
            color: #cccccc;
            border: 1px solid #3d3d3d;
            border-radius: 4px;
            padding: 8px 10px;
            font-size: 12px;
            line-height: 1.3;
        """)
        main_layout.addWidget(self.lbl_description)

        # 3. DYNAMIC PARAMETER FORM
        self.param_container = QWidget()
        self.param_layout = QFormLayout(self.param_container)
        self.param_layout.setContentsMargins(0, 4, 0, 4)
        self.param_layout.setVerticalSpacing(10)
        self.param_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.param_layout.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        main_layout.addWidget(self.param_container)

        # Manual execution button for running dynamic changes
        self.btn_apply_preview = QPushButton("Apply Stretch Parameters")
        self.btn_apply_preview.setStyleSheet("background-color: #1f4e79; color: white;")
        self.btn_apply_preview.clicked.connect(self._recalculate_preview)
        main_layout.addWidget(self.btn_apply_preview)

        # 4. LIVE PREVIEW WINDOW
        preview_header = QLabel("<b>Preview:</b>")
        main_layout.addWidget(preview_header)

        self.preview_view = pg.ImageView()
        self.preview_view.ui.histogram.hide()
        self.preview_view.ui.roiBtn.hide()
        self.preview_view.ui.menuBtn.hide()
        main_layout.addWidget(self.preview_view, stretch=1)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        main_layout.addWidget(line)

        # 5. STACKING & HISTORY CONTROLS
        stack_lbl = QLabel("<b>Stacking & History Controls:</b>")
        main_layout.addWidget(stack_lbl)

        stack_btn_layout = QHBoxLayout()

        btn_stack = QPushButton("Apply to Stack ➔")
        btn_stack.setStyleSheet("background-color: #2b5c2b; color: white;")
        btn_stack.clicked.connect(self._stack_current_stretch)
        stack_btn_layout.addWidget(btn_stack)

        btn_undo = QPushButton("Undo Stack")
        btn_undo.clicked.connect(self._undo_last_stack)
        stack_btn_layout.addWidget(btn_undo)

        btn_reset = QPushButton("Reset All")
        btn_reset.clicked.connect(self._reset_all_stretches)
        stack_btn_layout.addWidget(btn_reset)

        main_layout.addLayout(stack_btn_layout)

        # 6. OK / CANCEL
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        main_layout.addWidget(self.button_box)

        self.stack_history = []
        self._on_algorithm_changed(0)
        self._update_preview_display(self.preview_base)

    def _create_preview_subsample(self, img: np.ndarray, max_dim: int = 1024) -> np.ndarray:
        h, w = img.shape[:2]
        max_side = max(h, w)
        
        # Subsample if large
        if max_side > max_dim:
            step = int(np.ceil(max_side / max_dim))
            img = img[::step, ::step].copy()
        else:
            img = img.copy()

        # Convert to float32 and normalize safely to [0.0, 1.0]
        img = img.astype(np.float32)
        d_min, d_max = float(np.min(img)), float(np.max(img))
        if d_max > d_min:
            img = (img - d_min) / (d_max - d_min)
        else:
            img = np.zeros_like(img)

        return img

    def _clear_param_layout(self):
        while self.param_layout.count():
            item = self.param_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

    def _add_double_spin_control(self, name: str, val: float, vmin: float, vmax: float, step: float, decimals: int = 3):
        spin = QDoubleSpinBox()
        spin.setRange(vmin, vmax)
        spin.setSingleStep(step)
        spin.setDecimals(decimals)
        spin.setValue(val)
        # Note: valueChanged signal connection removed to require manual apply click
        
        lbl = QLabel(name)
        self.param_layout.addRow(lbl, spin)
        return spin

    def _on_algorithm_changed(self, index: int):
        # Update dynamic description box text
        self.lbl_description.setText(self.ALGO_DESCRIPTIONS.get(index, ""))

        # Clear existing controls
        self._clear_param_layout()

        if index == 0:  # GHS
            self.sp_ghs_b = self._add_double_spin_control("Stretch Factor:", 0.15, 0.0, 1.0, 0.01, 3)
            self.sp_ghs_x0 = self._add_double_spin_control("Symmetry Point:", 0.0015, 0.0001, 0.0500, 0.0005, 4)

        elif index == 1:  # Arcsinh
            self.sp_asinh_s = self._add_double_spin_control("Stretch Factor:", 0.25, 0.0, 1.0, 0.01, 3)
            self.sp_asinh_bp = self._add_double_spin_control("Black Point:", 0.0, 0.0, 1.0, 0.01, 3)

        elif index == 2:  # MTF
            self.sp_mtf_m = self._add_double_spin_control("Midtone Balance:", 0.10, 0.0, 1.0, 0.01, 3)
            self.sp_mtf_b = self._add_double_spin_control("Shadow Clip:", 0.0, 0.0, 1.0, 0.01, 3)

        elif index == 3:  # Logarithmic
            self.sp_log_a = self._add_double_spin_control("Scaling Factor:", 0.20, 0.0, 1.0, 0.01, 3)

        elif index == 4:  # Root
            self.sp_root_p = self._add_double_spin_control("Root Power:", 0.20, 0.0, 1.0, 0.01, 3)

        elif index == 5:  # Exponential
            self.sp_exp_b = self._add_double_spin_control("Exponent Slope:", 0.15, 0.0, 1.0, 0.01, 3)

        # Show initial un-stretched baseline image without calling stretch execution
        self._update_preview_display(self.preview_base)

    def _apply_algorithm(self, data: np.ndarray) -> np.ndarray:
        idx = self.combo_algo.currentIndex()

        if idx == 0:
            return stretch.ghs_stretch(data, self.sp_ghs_b.value(), self.sp_ghs_x0.value())
        elif idx == 1:
            return stretch.arcsinh_stretch(data, self.sp_asinh_s.value(), self.sp_asinh_bp.value())
        elif idx == 2:
            return stretch.midtone_transfer_function(data, self.sp_mtf_m.value(), self.sp_mtf_b.value())
        elif idx == 3:
            return stretch.log_stretch(data, self.sp_log_a.value())
        elif idx == 4:
            return stretch.root_stretch(data, self.sp_root_p.value())
        elif idx == 5:
            return stretch.exp_stretch(data, self.sp_exp_b.value())
        return data

    def _recalculate_preview(self):
        self.preview_active = self._apply_algorithm(self.preview_base)
        self._update_preview_display(self.preview_active)

    def _update_preview_display(self, img_data: np.ndarray):
        oriented = np.swapaxes(img_data, 0, 1)

        # 1. Guarantee array is float32 normalized [0.0, 1.0]
        d_min, d_max = img_data.min(), img_data.max()
        if d_max > 1.0 or d_min < 0.0:
            if d_max > d_min:
                oriented = (oriented - d_min) / (d_max - d_min)

        # 2. Force autoLevels or set fixed bounds [0.0, 1.0]
        first_render = not self._preview_initialized
        self.preview_view.setImage(
            oriented,
            autoRange=first_render,
            autoLevels=True,       # Let pyqtgraph map min/max black-to-white
            levels=[0.0, 1.0]      # Enforce standard normalized display window
        )
        
        if first_render:
            self._preview_initialized = True

    def _stack_current_stretch(self):
        self.stack_history.append(self.current_stacked_image.copy())
        self.current_stacked_image = self._apply_algorithm(self.current_stacked_image)

        self.preview_base = self._create_preview_subsample(self.current_stacked_image, max_dim=1024)
        self.lbl_info.setText(f"<b>Stack Count:</b> {len(self.stack_history)} Stretches Applied")
        self._update_preview_display(self.preview_base)

    def _undo_last_stack(self):
        if self.stack_history:
            self.current_stacked_image = self.stack_history.pop()
            self.preview_base = self._create_preview_subsample(self.current_stacked_image, max_dim=1024)
            self.lbl_info.setText(f"<b>Stack Count:</b> {len(self.stack_history)} Stretches Applied")
            self._update_preview_display(self.preview_base)

    def _reset_all_stretches(self):
        self.stack_history.clear()
        self.current_stacked_image = self.original_image.copy()
        self.preview_base = self._create_preview_subsample(self.current_stacked_image, max_dim=1024)
        self.lbl_info.setText("<b>Stack Count:</b> 0 Stretches Applied")
        self._update_preview_display(self.preview_base)

    def accept(self):
        if self.parent_app and hasattr(self.parent_app, "current_image_data"):
            # Calculate final high-res result
            final_full_image = self._apply_algorithm(self.current_stacked_image)

            # DEBUG
            print(
                f"[DIALOG DEBUG] final_full_image -> dtype: {final_full_image.dtype}, min:"
                f" {final_full_image.min():.5f}, max: {final_full_image.max():.5f},"
                f" median: {np.median(final_full_image):.5f}"
            )

            # Update parent data
            self.parent_app.current_image_data = final_full_image
            self.parent_app.autostretched_cache = None

            # Turn OFF auto-stretch mode on main window since data is now non-linear
            if hasattr(self.parent_app, "is_autostretch_active"):
                self.parent_app.is_autostretch_active = False
            if hasattr(self.parent_app, "btn_autostretch"):
                self.parent_app.btn_autostretch.setChecked(False)

            # Update display with explicit STF bypass
            self.parent_app.update_display(autoRange=False, disable_stf=True)

        super().accept()