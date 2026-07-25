# ui/stretch_dialog.py
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
)


class StretchDialog(QDialog):

  def __init__(self, parent=None, initial_target=0.25):
    super().__init__(parent)
    self.setWindowTitle("Stretch Parameters")

    # Dialog dimensions
    self.setMinimumSize(380, 280)
    self.resize(380, 280)

    # Main Layout
    layout = QVBoxLayout(self)
    layout.setContentsMargins(16, 16, 16, 16)
    layout.setSpacing(10)

    # 1. Dropdown to choose stretch type
    layout.addWidget(QLabel("Choose the stretch method to apply:"))
    self.combo_method = QComboBox()
    self.combo_method.addItems(
        ["Midtone Transfer Function (MTF)", "Arcsinh Stretch"]
    )
    self.combo_method.setMinimumHeight(32)
    layout.addWidget(self.combo_method)

    # 2. Slider & DoubleSpinBox Control Row
    layout.addWidget(QLabel("Set target median value (0.0 to 1.0):"))
    control_layout = QHBoxLayout()
    control_layout.setSpacing(10)

    # Slider (0 to 100 integer range internally)
    self.slider = QSlider(Qt.Orientation.Horizontal)
    self.slider.setRange(0, 100)
    self.slider.setValue(int(initial_target * 100))
    self.slider.setMinimumHeight(24)

    # SpinBox for decimal precision
    self.spin_box = QDoubleSpinBox()
    self.spin_box.setRange(0.0, 1.0)
    self.spin_box.setSingleStep(0.01)
    self.spin_box.setDecimals(2)
    self.spin_box.setValue(initial_target)
    self.spin_box.setMinimumHeight(32)
    self.spin_box.setFixedWidth(75)

    # Bidirectional Signals
    self.slider.valueChanged.connect(self._on_slider_changed)
    self.spin_box.valueChanged.connect(self._on_spin_box_changed)

    control_layout.addWidget(self.slider)
    control_layout.addWidget(self.spin_box)
    layout.addLayout(control_layout)

    # Auto black point checkbox
    self.auto_black_checkbox = QCheckBox("Auto black point correction")
    self.auto_black_checkbox.setChecked(True)
    self.auto_black_checkbox.setToolTip(
        "Automatically adjust the black point based on image data."
    )
    self.auto_black_checkbox.setMinimumHeight(24)
    layout.addWidget(self.auto_black_checkbox)

    # Push buttons to bottom
    layout.addStretch()

    # 3. Action Buttons Row
    button_layout = QHBoxLayout()
    button_layout.setSpacing(10)
    button_layout.addStretch()

    self.btn_cancel = QPushButton("Cancel")
    self.btn_cancel.setMinimumHeight(32)
    self.btn_cancel.setMinimumWidth(85)
    self.btn_cancel.clicked.connect(self.reject)

    self.btn_apply = QPushButton("Apply")
    self.btn_apply.setDefault(True)
    self.btn_apply.setMinimumHeight(32)
    self.btn_apply.setMinimumWidth(85)
    self.btn_apply.clicked.connect(self._on_apply_clicked)

    button_layout.addWidget(self.btn_cancel)
    button_layout.addWidget(self.btn_apply)
    layout.addLayout(button_layout)

  def _on_apply_clicked(self):
    """Provides immediate visual feedback when Apply is pressed."""
    self.btn_apply.setText("Processing...")
    self.btn_apply.setEnabled(False)
    self.btn_cancel.setEnabled(False)
    self.setCursor(Qt.CursorShape.WaitCursor)

    QApplication.processEvents()  # Force GUI refresh before dialog closes
    self.accept()

  def _on_slider_changed(self, value):
    self.spin_box.blockSignals(True)
    self.spin_box.setValue(value / 100.0)
    self.spin_box.blockSignals(False)

  def _on_spin_box_changed(self, value):
    self.slider.blockSignals(True)
    self.slider.setValue(int(value * 100))
    self.slider.blockSignals(False)

  def get_target_median(self):
    """Helper to fetch chosen decimal value when dialog closes."""
    return self.spin_box.value()

  def is_auto_black_point_enabled(self):
    """Helper to fetch auto black point checkbox state."""
    return self.auto_black_checkbox.isChecked()