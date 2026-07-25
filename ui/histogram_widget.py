# ui/histogram_widget.py
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QWidget, QVBoxLayout


class HistogramWidget(QWidget):

  def __init__(self, parent=None):
    super().__init__(parent)
    self.setMinimumHeight(65)
    self.setMaximumHeight(80)

    layout = QVBoxLayout(self)
    layout.setContentsMargins(0, 0, 0, 0)

    # 1. Create a minimalist PyQtGraph PlotWidget
    self.plot = pg.PlotWidget()
    self.plot.setBackground(None)  # Transparent so it matches QGroupBox QSS

    # Hide unnecessary interactive elements and axes to keep it clean & simple
    self.plot.hideAxis("left")
    self.plot.showAxis("bottom")
    self.plot.getAxis("bottom").setStyle(showValues=False)
    self.plot.setMenuEnabled(False)
    self.plot.setMouseEnabled(x=False, y=False)  # Read-only

    # Style bottom axis line
    self.plot.getAxis("bottom").setPen(pg.mkPen(color=(100, 100, 100), width=1))

    # Enable log scale for Y axis to better visualize faint signals alongside bright peaks
    self.plot.setLogMode(x=False, y=True)

    layout.addWidget(self.plot)

  def update_histogram(self, data, bins=128):
    """Calculates and displays the image pixel distribution safely."""
    self.plot.clear()

    if data is None or data.size == 0:
      return

    # 1. Flatten and downsample for UI speed
    flat_data = data.ravel()
    non_zero = flat_data[flat_data > 0]  # Exclude zero values for log scale
    if len(non_zero) > 0:
      flat_data = non_zero

    if len(flat_data) > 500_000:
      flat_data = np.random.choice(flat_data, size=500_000, replace=False)

    # 2. Calculate histogram
    y_vals, x_edges = np.histogram(flat_data, bins=bins)

    # 3. FIX: Clamp zero counts to 1 to prevent log10(0) = -inf crash
    y_vals = np.maximum(y_vals, 1)

    # 4. Plot step graph
    self.plot.plot(
        x_edges,
        y_vals,
        stepMode="center",
        fillLevel=0,  # Set fill level to 0 for log scale (since log10(1) = 0)
        brush=(77, 166, 255, 60),
        pen=pg.mkPen(color=(77, 166, 255), width=1.5),
    )

    self.plot.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)  # Auto-scale Y axis for log view