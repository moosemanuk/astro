# ui/header_panel.py
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from ui.histogram_widget import HistogramWidget


class ImageHeaderPanel(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(280)
        self.setMaximumWidth(420)

        # Main layout for side panel
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Scroll area to comfortably accommodate collapsible or expanding cards
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        # Hide the horizontal scrollbar; vertical scrollbar appears only when needed
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(12, 8, 12, 8)
        container_layout.setSpacing(12)

        # -------------------------------------------------------------------------
        # Card 1: Target & Session Summary
        # -------------------------------------------------------------------------
        grp_file = QGroupBox("Target & Session")
        layout_file = QFormLayout(grp_file)
        layout_file.setSpacing(6)

        self.lbl_target = QLabel("—")
        self.lbl_date_obs = QLabel("—")
        self.lbl_exposure = QLabel("—")

        layout_file.addRow("Object:", self.lbl_target)
        layout_file.addRow("Date (UTC):", self.lbl_date_obs)
        layout_file.addRow("Exposure:", self.lbl_exposure)
        container_layout.addWidget(grp_file)

        # -------------------------------------------------------------------------
        # Card 2: Camera & Optics
        # -------------------------------------------------------------------------
        grp_cam = QGroupBox("Camera & Optics")
        layout_cam = QFormLayout(grp_cam)
        layout_cam.setSpacing(6)

        self.lbl_camera = QLabel("—")
        self.lbl_telescope = QLabel("—")
        self.lbl_gain_offset = QLabel("—")
        self.lbl_temp = QLabel("—")
        self.lbl_filter = QLabel("—")
        self.lbl_bayer = QLabel("—")

        layout_cam.addRow("Camera:", self.lbl_camera)
        layout_cam.addRow("Scope / Lens:", self.lbl_telescope)
        layout_cam.addRow("Gain / Offset:", self.lbl_gain_offset)
        layout_cam.addRow("Sensor Temp:", self.lbl_temp)
        layout_cam.addRow("Filter:", self.lbl_filter)
        layout_cam.addRow("Bayer Matrix:", self.lbl_bayer)
        container_layout.addWidget(grp_cam)

        # -------------------------------------------------------------------------
        # Card 3: Frame Metrics & Dimensions
        # -------------------------------------------------------------------------
        grp_frame = QGroupBox("Frame & Dimensions")
        layout_frame = QFormLayout(grp_frame)
        layout_frame.setSpacing(6)

        self.lbl_frame_type = QLabel("—")
        self.lbl_dimensions = QLabel("—")
        self.lbl_pixel_size = QLabel("—")
        self.lbl_fwhm_hfr = QLabel("—")

        layout_frame.addRow("Frame Type:", self.lbl_frame_type)
        layout_frame.addRow("Dimensions:", self.lbl_dimensions)
        layout_frame.addRow("Pixel Size:", self.lbl_pixel_size)
        layout_frame.addRow("HFR / FWHM:", self.lbl_fwhm_hfr)
        container_layout.addWidget(grp_frame)

        # -------------------------------------------------------------------------
        # Card 4: Image Histogram
        # -------------------------------------------------------------------------
        grp_hist = QGroupBox("Histogram")
        layout_hist = QVBoxLayout(grp_hist)
        layout_hist.setContentsMargins(6, 12, 6, 6)

        self.hist_widget = HistogramWidget()
        layout_hist.addWidget(self.hist_widget)

        # Add all cards to container layout
        container_layout.addWidget(grp_file)
        container_layout.addWidget(grp_cam)
        container_layout.addWidget(grp_frame)
        container_layout.addWidget(grp_hist)

        # Apply standard styling and selection properties to all labels
        all_labels = [
            self.lbl_target,
            self.lbl_date_obs,
            self.lbl_exposure,
            self.lbl_camera,
            self.lbl_telescope,
            self.lbl_gain_offset,
            self.lbl_temp,
            self.lbl_filter,
            self.lbl_bayer,
            self.lbl_frame_type,
            self.lbl_dimensions,
            self.lbl_pixel_size,
            self.lbl_fwhm_hfr,
        ]
        for lbl in all_labels:
            lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            lbl.setStyleSheet("color: #4da6ff; font-weight: bold;")

        container_layout.addStretch()  # Keep cards neatly docked at top
        scroll.setWidget(container)
        main_layout.addWidget(scroll)

    def clear(self):
        """Resets all header labels to default states and clears the histogram plot."""
        all_labels = [
            self.lbl_target,
            self.lbl_date_obs,
            self.lbl_exposure,
            self.lbl_camera,
            self.lbl_telescope,
            self.lbl_gain_offset,
            self.lbl_temp,
            self.lbl_filter,
            self.lbl_bayer,
            self.lbl_frame_type,
            self.lbl_dimensions,
            self.lbl_pixel_size,
            self.lbl_fwhm_hfr,
        ]
        for lbl in all_labels:
            lbl.setText("—")

        if hasattr(self.hist_widget, "clear"):
            self.hist_widget.clear()
        elif hasattr(self.hist_widget, "update_histogram"):
            self.hist_widget.update_histogram(None)

    def _get_fits_val(self, header, keys, default="N/A"):
        """Safely searches a list of possible FITS header keys."""
        if not header:
            return default
        for k in keys:
            if k in header and header[k] not in [None, ""]:
                return header[k]
        return default

    def update_histogram_only(self, data):
        """Updates only the histogram without changing other metadata."""
        if data is not None:
            self.hist_widget.update_histogram(data)

    def update_header_info(self, file_path, data, fits_header=None):
        """Populates side panel cards with detailed metadata from FITS header."""
        # 1. Dimensions
        if data is not None:
            shape_str = " × ".join(str(d) for d in data.shape)
            self.lbl_dimensions.setText(shape_str)
        else:
            self.lbl_dimensions.setText("N/A")

        if not fits_header:
            # Clear fields if no header exists
            for lbl in [
                self.lbl_target,
                self.lbl_date_obs,
                self.lbl_exposure,
                self.lbl_camera,
                self.lbl_telescope,
                self.lbl_gain_offset,
                self.lbl_temp,
                self.lbl_filter,
                self.lbl_bayer,
                self.lbl_frame_type,
                self.lbl_pixel_size,
                self.lbl_fwhm_hfr,
            ]:
                lbl.setText("N/A")
            return

        # 2. Target & Timing
        target = self._get_fits_val(fits_header, ["OBJECT", "TARGET"])
        date_obs = self._get_fits_val(fits_header, ["DATE-OBS", "DATE"])
        exp_time = self._get_fits_val(fits_header, ["EXPOSURE", "EXPTIME"])
        if isinstance(exp_time, (int, float)):
            exp_time = f"{exp_time:.2f} s"

        self.lbl_target.setText(str(target))
        self.lbl_date_obs.setText(str(date_obs))
        self.lbl_exposure.setText(str(exp_time))

        # 3. Camera & Optics
        camera = self._get_fits_val(
            fits_header, ["INSTRUME", "CAMERA", "DETECTOR"]
        )
        telescope = self._get_fits_val(fits_header, ["TELESCOP", "TELESCOPE"])

        gain = self._get_fits_val(fits_header, ["GAIN", "ISO"])
        offset = self._get_fits_val(fits_header, ["OFFSET", "BLKLEVEL"])
        gain_offset_str = f"{gain} / {offset}"

        temp = self._get_fits_val(fits_header, ["CCD-TEMP", "TEMP"])
        if isinstance(temp, (int, float)):
            temp = f"{temp:.1f} °C"

        filter_name = self._get_fits_val(fits_header, ["FILTER", "FILTERNAME"])
        bayer = self._get_fits_val(
            fits_header, ["BAYERPAT", "COLORTYP", "CROSSOVER"]
        )

        self.lbl_camera.setText(str(camera))
        self.lbl_telescope.setText(str(telescope))
        self.lbl_gain_offset.setText(gain_offset_str)
        self.lbl_temp.setText(str(temp))
        self.lbl_filter.setText(str(filter_name))
        self.lbl_bayer.setText(str(bayer))

        # 4. Frame Metrics
        frame_type = self._get_fits_val(fits_header, ["IMAGETYP", "FRAME"])
        pix_size = self._get_fits_val(
            fits_header, ["PIXSIZE1", "XPIXSZ", "PIXSIZE"]
        )
        if isinstance(pix_size, (int, float)):
            pix_size = f"{pix_size:.2f} µm"

        hfr = self._get_fits_val(fits_header, ["HFR", "FWHM"])
        if isinstance(hfr, (int, float)):
            hfr = f"{hfr:.2f} px"

        self.lbl_frame_type.setText(str(frame_type))
        self.lbl_pixel_size.setText(str(pix_size))
        self.lbl_fwhm_hfr.setText(str(hfr))

        # 5. Update Histogram
        self.update_histogram_only(data)