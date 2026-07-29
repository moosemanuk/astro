import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QCheckBox, QFileDialog, QMessageBox, QDialog, QLabel, QSplitter
)
from PyQt6.QtGui import QAction
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import Qt
from ui.header_panel import ImageHeaderPanel
from astropy.io import fits

import pyqtgraph as pg
import numpy as np
import os

from processing.stretch import midtone_transfer_function, arcsinh_stretch, auto_midtone_stretch
from ui.stretch_dialog import StretchDialog
from processing.pedestal import remove_pedestal
from processing.geometry import flip_horizontal, flip_vertical
from ui.background_dialog import BackgroundExtractionDialog
from ui.gradient_preview_dialog import GradientPreviewDialog
from ui.curves_dialog import CurvesTransformationDialog
from ui.stretch_workbench import StretchWorkbenchDialog
from processing.background import extract_background_poly
from processing.denoise import denoise_image
from ui.denoise_dialog import DenoiseDialog
from processing.sharpen import sharpen_image
from ui.sharpen_dialog import SharpenDialog
from processing.star_removal import remove_stars
from ui.star_removal_dialog import StarRemovalDialog


# AutoStretch is a display-only preview. Robust normalization in the stretch
# keeps saturated stars from hiding faint nebula structure.
AUTO_STRETCH_TARGET_BACKGROUND = 0.20


class AstroImageEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Astro Image Editor")
        self.setGeometry(100, 100, 1450, 900)

        # State tracking
        self.current_image_data = None
        self.previous_image_data = None
        self.current_fits_header = None
        
        # Screen Stretch Preview State
        self.is_autostretched = False
        self.autostretched_cache = None

        # Crop Tool State
        self.crop_roi = None

        # Build UI components
        self.init_actions()
        self.init_menu_bar()
        self.init_ui()
        self.load_stylesheet()

    def init_actions(self):
        """Initializes QAction objects so they can be added to menus or shortcuts."""

        # --- File Menu Actions
        self.open_action = QAction("&Open Image...", self)
        self.open_action.setShortcut("Ctrl+O")
        self.open_action.triggered.connect(self.open_file)

        self.close_image_action = QAction("&Close Image", self)
        self.close_image_action.setShortcut("Ctrl+W")
        self.close_image_action.setEnabled(False)
        self.close_image_action.triggered.connect(self.close_image)

        self.close_action = QAction("&Exit", self)
        self.close_action.setShortcut("Ctrl+Q")
        self.close_action.triggered.connect(self.close)

        # --- Edit Menu Actions ---
        self.undo_action = QAction("&Undo", self)
        self.undo_action.setShortcut("Ctrl+Z")
        self.undo_action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        self.undo_action.setEnabled(False)
        self.undo_action.triggered.connect(self.undo_last_operation)
        self.addAction(self.undo_action)

        # --- Image Menu Actions ---
        self.pedestal_action = QAction("Pedestal Removal", self)
        self.pedestal_action.setEnabled(False)
        self.pedestal_action.triggered.connect(self.apply_pedestal_removal)

        self.background_removal = QAction("Background Extraction ...", self)
        self.background_removal.setEnabled(False)
        self.background_removal.triggered.connect(self.apply_background_removal)

        self.sharpen_action = QAction("Sharpen ...", self)
        self.sharpen_action.setEnabled(False)
        self.sharpen_action.triggered.connect(self.apply_sharpen)

        self.denoise_action = QAction("Denoise ...", self)
        self.denoise_action.setEnabled(False)
        self.denoise_action.triggered.connect(self.apply_denoise)

        self.star_removal_action = QAction("Star Removal ...", self)
        self.star_removal_action.setEnabled(False)
        self.star_removal_action.triggered.connect(self.apply_star_removal)

        self.stretch_workbench = QAction("Stretch Workbench ...", self)
        self.stretch_workbench.setEnabled(False)
        self.stretch_workbench.triggered.connect(self.open_stretch_workbench)

        self.apply_curves = QAction("Apply Curves ...", self)
        self.apply_curves.setEnabled(False)
        self.apply_curves.triggered.connect(self.apply_curves_transformation)

        self.crop_action = QAction("Crop Region", self)
        self.crop_action.setCheckable(True)
        self.crop_action.setEnabled(False)
        self.crop_action.triggered.connect(self.toggle_crop_tool)

        # --- Flip Actions ---
        self.flip_h_action = QAction("Flip Horizontal", self)
        self.flip_h_action.setEnabled(False)
        self.flip_h_action.triggered.connect(self.apply_flip_horizontal)

        self.flip_v_action = QAction("Flip Vertical", self)
        self.flip_v_action.setEnabled(False)
        self.flip_v_action.triggered.connect(self.apply_flip_vertical)

        # --- Help Menu Actions ---
        self.about_action = QAction("&About", self)
        self.about_action.triggered.connect(self.show_about)

    def load_stylesheet(self, filename="style.qss"):
        filepath = os.path.join("ui", filename)
        try:
            with open(filepath, "r") as f:
                self.setStyleSheet(f.read())
        except FileNotFoundError:
            print(f"Stylesheet file '{filepath}' not found. Using default styles.")

    def init_menu_bar(self):
        """Constructs the menu bar strictly in File -> Edit -> Help order."""
        menu_bar = self.menuBar()

        # 1. File Menu
        file_menu = menu_bar.addMenu("&File")
        file_menu.addAction(self.open_action)
        file_menu.addAction(self.close_image_action)
        file_menu.addSeparator()
        file_menu.addAction(self.close_action)

        # 2. Edit Menu
        edit_menu = menu_bar.addMenu("&Edit")
        edit_menu.addAction(self.undo_action)
        
        # 2.5 Image Editing Menu
        image_menu = menu_bar.addMenu("&Image")
        image_menu.addAction(self.pedestal_action)
        image_menu.addAction(self.background_removal)
        image_menu.addAction(self.sharpen_action)
        image_menu.addAction(self.denoise_action)
        image_menu.addAction(self.star_removal_action)
        image_menu.addSeparator()

        image_menu.addAction(self.stretch_workbench)
        image_menu.addAction(self.apply_curves)
        image_menu.addSeparator()

        image_menu.addAction(self.flip_h_action)
        image_menu.addAction(self.flip_v_action)
        image_menu.addAction(self.crop_action)

        # 3. Help Menu
        help_menu = menu_bar.addMenu("&Help")
        help_menu.addAction(self.about_action)

    def init_ui(self):
        # Central Main Widget
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)

        # Top Control Bar (Buttons, controls)
        control_layout = QHBoxLayout()        
        

        # AutoStretch Checkbox
        self.chk_autostretch = QCheckBox("AutoStretch")
        self.chk_autostretch.setEnabled(False)
        self.chk_autostretch.toggled.connect(self.toggle_autostretch)
        control_layout.addWidget(self.chk_autostretch)

        # Undo Button
        self.undo_btn = QPushButton("Undo")
        self.undo_btn.setEnabled(False)
        undo_icon = QIcon.fromTheme("edit-undo")
        if not undo_icon.isNull():
            self.undo_btn.setIcon(undo_icon)
        self.undo_btn.setToolTip("Undo last action")
        self.undo_btn.clicked.connect(self.undo_last_operation)
        control_layout.addWidget(self.undo_btn)

        # Stretch Button
        self.btn_stretch = QPushButton("Stretch...")
        self.btn_stretch.setEnabled(False)
        self.btn_stretch.clicked.connect(self.open_stretch_workbench)
        control_layout.addWidget(self.btn_stretch)

        # Crop Tool Controls
        self.btn_crop_tool = QPushButton("Toggle Crop")
        self.btn_crop_tool.setCheckable(True)
        self.btn_crop_tool.setEnabled(False)
        self.btn_crop_tool.clicked.connect(self.toggle_crop_tool)
        control_layout.addWidget(self.btn_crop_tool)

        self.btn_apply_crop = QPushButton("Apply Crop")
        self.btn_apply_crop.setEnabled(False)
        self.btn_apply_crop.setVisible(False)
        self.btn_apply_crop.clicked.connect(self.apply_crop)
        control_layout.addWidget(self.btn_apply_crop)

        # Keep all tools left-aligned
        control_layout.addStretch()  
        main_layout.addLayout(control_layout)

        # Splitter for Image View and Metadata Panel
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(False)

        # Image Viewer Area
        self.image_view = pg.ImageView()
        self.image_view.ui.histogram.hide()
        self.image_view.ui.roiBtn.hide()
        self.image_view.ui.menuBtn.hide()

        # Side Panel for Metadata
        self.header_panel = ImageHeaderPanel()
        self.header_panel.setMinimumWidth(260)

        # Splitter assembly
        self.splitter.addWidget(self.image_view)
        self.splitter.addWidget(self.header_panel)
        self.splitter.setStretchFactor(0, 4)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([800, 200])
        main_layout.addWidget(self.splitter)

    # --- FILE & DISPLAY OPERATIONS ---

    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Astronomical Image",
            "",
            "Astro Images (*.fits *.fit *.fits.gz);;All Files (*)"
        )

        if file_path:
            self.load_fits_data(file_path)

    def load_fits_data(self, path):
        print(f"Loading FITS data from: {path}")
        try:
            with fits.open(path) as hdul:
                data = None
                for hdu in hdul:
                    if hdu.data is not None and hdu.data.ndim in (2, 3):
                        data = hdu.data.astype(np.float32)
                        header = hdu.header
                        break
                if data is None:
                    raise ValueError("No valid 2D or 3D image data found in FITS extensions.")

                if data.dtype.byteorder not in ('=', '|'):
                    data = data.astype(data.dtype.newbyteorder('='))

                if data.ndim == 3 and data.shape[0] == 3:
                    display_data = np.transpose(data, (2, 1, 0))
                elif data.ndim == 3 and data.shape[2] == 3:
                    display_data = np.transpose(data, (1, 0, 2))
                else:
                    display_data = np.transpose(data)

                self.current_image_data = display_data
                self.current_fits_header = header

                # Reset UI & Tool States
                self.remove_crop_roi()
                self.is_autostretched = False
                self.autostretched_cache = None
                self.chk_autostretch.setChecked(False)

                self.update_display(autoRange=True)
                
                # Enable Toolbar Actions
                self.btn_stretch.setEnabled(True)
                self.chk_autostretch.setEnabled(True)
                self.btn_crop_tool.setEnabled(True)
                self.background_removal.setEnabled(True)
                self.denoise_action.setEnabled(True)
                self.sharpen_action.setEnabled(True)
                self.star_removal_action.setEnabled(True)
                self.close_image_action.setEnabled(True)
                self.pedestal_action.setEnabled(True)
                self.apply_curves.setEnabled(True)
                self.stretch_workbench.setEnabled(True)
                self.flip_h_action.setEnabled(True)
                self.flip_v_action.setEnabled(True)

                self.header_panel.update_header_info(path, display_data, header)

                self.setWindowTitle(f"Astro Image Editor - {path.split('/')[-1]}")
                self.statusBar().showMessage(f"Loaded: {path}", 5000)

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Failed to load FITS file: {path}\n\nDetails: {str(e)}")
            return

    def close_image(self):
        """Clears image state, cleans up controls, and resets view."""
        if self.current_image_data is None:
            return

        self.remove_crop_roi()

        self.current_image_data = None
        self.previous_image_data = None
        self.current_fits_header = None
        self.autostretched_cache = None
        self.is_autostretched = False

        self.chk_autostretch.setChecked(False)
        self.chk_autostretch.setEnabled(False)
        self.btn_stretch.setEnabled(False)
        self.btn_crop_tool.setEnabled(False)
        self.background_removal.setEnabled(False)
        self.denoise_action.setEnabled(False)
        self.sharpen_action.setEnabled(False)
        self.star_removal_action.setEnabled(False)
        self.apply_curves.setEnabled(False)
        self.stretch_workbench.setEnabled(False)
        self.undo_action.setEnabled(False)
        self.undo_btn.setEnabled(False)
        self.close_image_action.setEnabled(False)
        self.flip_h_action.setEnabled(False)
        self.flip_v_action.setEnabled(False)

        self.image_view.clear()
        self.header_panel.clear()

        self.setWindowTitle("Astro Image Editor")
        self.statusBar().showMessage("Image closed.", 4000)

    def update_display(self, autoRange=False, disable_stf=False):
        """Helper to refresh viewport based on current image data & stretch preview state."""
        if self.current_image_data is None:
            return

        # 1. Reset autostretch state when applying permanent stretch
        if disable_stf:
            self.is_autostretched = False
            self.autostretched_cache = None

        # 2. Render path
        if self.is_autostretched:
            if self.autostretched_cache is None:
                self.autostretched_cache = auto_midtone_stretch(
                    self.current_image_data,
                    target_background=AUTO_STRETCH_TARGET_BACKGROUND,
                )
            self.image_view.setImage(
                self.autostretched_cache,
                autoRange=bool(autoRange),
                autoLevels=False,
                levels=(0.0, 1.0),
            )
        else:
            if disable_stf:
                # Permanent non-linear stretch output is already normalized in [0.0, 1.0]
                data_to_show = np.clip(self.current_image_data, 0.0, 1.0)
                self.image_view.setImage(
                    data_to_show,
                    autoRange=bool(autoRange),
                    autoLevels=False,
                    levels=(0.0, 1.0),
                )
            else:
                # Raw/Linear FITS Data view: Let pyqtgraph compute autoLevels based on 
                # actual min/max data range so linear files don't blow out to white.
                d_min = float(np.min(self.current_image_data))
                d_max = float(np.max(self.current_image_data))
                
                # Prevent min == max collapse
                if d_min == d_max:
                    d_max = d_min + 1.0

                self.image_view.setImage(
                    self.current_image_data,
                    autoRange=bool(autoRange),
                    autoLevels=False,
                    levels=(d_min, d_max),
                )
    # --- STRETCH OPERATIONS ---

    def toggle_autostretch(self, checked: bool):
        if self.current_image_data is None:
            return

        self.is_autostretched = checked

        if self.is_autostretched and self.autostretched_cache is None:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            self.statusBar().showMessage("Calculating screen AutoStretch...")
            QApplication.processEvents()
            try:
                self.autostretched_cache = auto_midtone_stretch(
                    self.current_image_data,
                    target_background=AUTO_STRETCH_TARGET_BACKGROUND,
                )
            finally:
                QApplication.restoreOverrideCursor()

        self.update_display()
        status_msg = "Screen AutoStretch: Enabled (Preview Only)" if self.is_autostretched else "Screen AutoStretch: Disabled"
        self.statusBar().showMessage(status_msg, 3000)

    def apply_stretch(self):
        if self.current_image_data is None:
            return

        dialog = StretchDialog(self, initial_target=0.25)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            target_val = dialog.spin_box.value()
            stretch_method = dialog.combo_method.currentText()
            use_black_point = dialog.is_auto_black_point_enabled()

            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            self.statusBar().showMessage(f"Applying {stretch_method}... Please wait.")
            QApplication.processEvents()

            try:
                self.previous_image_data = self.current_image_data.copy()

                if stretch_method == "Midtone Transfer Function (MTF)":
                    stretched_data = midtone_transfer_function(
                        self.current_image_data,
                        midtone=target_val,
                        shadow_clip=0.01 if use_black_point else 0.0,
                    )
                elif stretch_method == "Arcsinh Stretch":
                    stretched_data = arcsinh_stretch(
                        self.current_image_data,
                        factor=target_val,
                        black_point=0.02 if use_black_point else 0.0,
                    )

                self.current_image_data = stretched_data
                self.autostretched_cache = None
                
                if self.is_autostretched:
                    self.is_autostretched = False
                    self.chk_autostretch.setChecked(False)

                self.update_display(autoRange=False)

                self.statusBar().showMessage("Updating histogram...")
                QApplication.processEvents()

                self.header_panel.update_histogram_only(self.current_image_data)
                self.undo_action.setEnabled(True)
                self.undo_btn.setEnabled(True)

                bp_str = " (Auto BP)" if use_black_point else ""
                self.statusBar().showMessage(
                    f"Applied {stretch_method}{bp_str}. Press Ctrl+Z to undo.", 5000
                )

            finally:
                QApplication.restoreOverrideCursor()

    # --- CROP TOOL OPERATIONS ---

    def toggle_crop_tool(self, checked: bool):
        """Shows or hides the interactive selection box over the image."""
        if self.current_image_data is None:
            return

        if checked:
            h, w = self.current_image_data.shape[0], self.current_image_data.shape[1]

            crop_w, crop_h = w // 2, h // 2
            start_x, start_y = w // 4, h // 4

            self.crop_roi = pg.RectROI(
                [start_x, start_y],
                [crop_w, crop_h],
                pen=pg.mkPen(color='yellow', width=2),
                sideScalers=True
            )
            self.crop_roi.addScaleHandle([1, 1], [0, 0])
            self.crop_roi.addScaleHandle([0, 0], [1, 1])
            
            self.image_view.addItem(self.crop_roi)
            self.btn_apply_crop.setVisible(True)
            self.btn_apply_crop.setEnabled(True)
            self.statusBar().showMessage("Drag and scale yellow box, then click 'Apply Crop'.", 5000)
        else:
            self.remove_crop_roi()

    def remove_crop_roi(self):
        """Removes the ROI box and hides the apply button."""
        if self.crop_roi is not None:
            self.image_view.removeItem(self.crop_roi)
            self.crop_roi = None
        self.btn_crop_tool.setChecked(False)
        self.btn_apply_crop.setVisible(False)
        self.btn_apply_crop.setEnabled(False)

    def apply_crop(self):
        """Slices self.current_image_data to match the selected ROI area."""
        if self.current_image_data is None or self.crop_roi is None:
            return

        self.previous_image_data = self.current_image_data.copy()

        pos = self.crop_roi.pos()
        size = self.crop_roi.size()

        x1 = max(0, int(round(pos.x())))
        y1 = max(0, int(round(pos.y())))
        x2 = min(self.current_image_data.shape[0], int(round(pos.x() + size.x())))
        y2 = min(self.current_image_data.shape[1], int(round(pos.y() + size.y())))

        if (x2 - x1) <= 2 or (y2 - y1) <= 2:
            QMessageBox.warning(self, "Invalid Crop Area", "Selected region is too small to crop.")
            return

        cropped_data = self.current_image_data[x1:x2, y1:y2]

        self.remove_crop_roi()

        self.current_image_data = cropped_data
        self.autostretched_cache = None

        self.update_display(autoRange=True)
        
        self.header_panel.update_header_info(
            self.windowTitle(), self.current_image_data, self.current_fits_header
        )
        self.undo_action.setEnabled(True)
        self.undo_btn.setEnabled(True)
        self.statusBar().showMessage(f"Cropped image to {x2 - x1} × {y2 - y1} px. Press Ctrl+Z to undo.", 5000)

    def apply_pedestal_removal(self):
        """Calculates and subtracts baseline pedestal offset across channels with Undo support."""
        if self.current_image_data is None:
            return

        # DEBUG
        print(f"[Pedestal] Min before: {np.min(self.current_image_data):.6f}, Mean before: {np.mean(self.current_image_data):.6f}")

        # 1. Provide immediate UI feedback for heavy numpy math
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self.statusBar().showMessage("Calculating and removing pedestal offset...")
        QApplication.processEvents()

        try:
            # Save previous state for Ctrl+Z Undo
            self.previous_image_data = self.current_image_data.copy()

            # Run pedestal removal computation (0.1% percentile floor)
            corrected_data, ped_vals = remove_pedestal(self.current_image_data, percentile=0.1)

            # DEBUG
            print(f"[Pedestal] Subtracted: {ped_vals}")
            print(f"[Pedestal] Min after: {np.min(corrected_data):.6f}, Mean after: {np.mean(corrected_data):.6f}")

            # Update state
            self.current_image_data = corrected_data
            self.autostretched_cache = None  # Invalidate screen stretch cache

            # Refresh display and histogram
            self.update_display(autoRange=False)
            self.header_panel.update_histogram_only(self.current_image_data)

            # Enable Undo action
            self.undo_action.setEnabled(True)
            self.undo_btn.setEnabled(True)

            # Format subtracted values for status message
            if len(ped_vals) > 1:
                ped_str = ", ".join([f"Ch{i+1}: {v:.4f}" for i, v in enumerate(ped_vals)])
            else:
                ped_str = f"{ped_vals[0]:.4f}"

            self.statusBar().showMessage(f"Removed Pedestal ({ped_str}). Press Ctrl+Z to undo.", 5000)

        finally:
            QApplication.restoreOverrideCursor()

    def _commit_processed_result(self, result, status_msg: str):
        if result is None:
            return
        self.previous_image_data = self.current_image_data.copy()
        self.current_image_data = result
        self.autostretched_cache = None
        self.update_display(autoRange=False)
        # Guarded call in case update_histogram_only is deprecated/removed
        if hasattr(self, "header_panel") and hasattr(self.header_panel, "update_histogram_only"):
            try:
                self.header_panel.update_histogram_only(self.current_image_data)
            except Exception:
                pass
        self.undo_action.setEnabled(True)
        self.undo_btn.setEnabled(True)
        self.statusBar().showMessage(status_msg, 5000)

    def _commit_background_result(self, result):
        if result is None:
            return
        corrected_img, _ = result
        self.previous_image_data = self.current_image_data.copy()
        self.current_image_data = corrected_img
        self.autostretched_cache = None
        self.update_display(autoRange=False)
        if hasattr(self, "header_panel"):
            self.header_panel.update_histogram_only(self.current_image_data)
        self.undo_action.setEnabled(True)
        self.undo_btn.setEnabled(True)
        self.statusBar().showMessage("Removed fitted background gradient. Press Ctrl+Z to undo.", 5000)

    def apply_background_removal(self):
        if self.current_image_data is None:
            return

        dialog = BackgroundExtractionDialog(
            image_data=self.current_image_data,
            parent=self,
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            result = dialog.get_result()
            if result is not None:
                # result is a tuple: (corrected_image, background_model)
                corrected_img, bg_model = result
                self._commit_processed_result(
                    corrected_img, "Removed fitted background gradient. Press Ctrl+Z to undo."
                )
                
                # Pop up the extracted gradient surface preview dialog
                preview_dialog = GradientPreviewDialog(bg_model, parent=self)
                preview_dialog.exec()

    def apply_denoise(self):
        """Applies detail-preserving multiscale wavelet denoising with one-step Undo support."""
        if self.current_image_data is None:
            return

        dialog = DenoiseDialog(
            image_data=self.current_image_data,
            parent=self,
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            result = dialog.get_result()
            if result is not None:
                self._commit_processed_result(
                    result, "Denoised image (Wavelet). Press Ctrl+Z to undo."
                )

    def apply_sharpen(self):
        """Applies selective unsharp masking with one-step Undo support."""
        if self.current_image_data is None:
            return

        dialog = SharpenDialog(
            image_data=self.current_image_data,
            parent=self,
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            result = dialog.get_result()
            if result is not None:
                self._commit_processed_result(
                    result, "Sharpened image. Press Ctrl+Z to undo."
                )

    def apply_star_removal(self):
        """Removes compact bright stars with one-step Undo support."""
        if self.current_image_data is None:
            return

        dialog = StarRemovalDialog(
            image_data=self.current_image_data,
            parent=self,
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            result = dialog.get_result()
            if result is not None:
                self._commit_processed_result(
                    result, "Removed stars. Press Ctrl+Z to undo."
                )

    def apply_flip_horizontal(self):
        """Flips the current image horizontally with Undo support."""
        if self.current_image_data is None:
            return

        self.previous_image_data = self.current_image_data.copy()
        self.current_image_data = flip_horizontal(self.current_image_data)
        
        self.autostretched_cache = None  # Invalidate stretch preview cache
        self.update_display(autoRange=False)
        
        self.undo_action.setEnabled(True)
        self.undo_btn.setEnabled(True)
        self.statusBar().showMessage("Flipped image horizontally. Press Ctrl+Z to undo.", 4000)

    def apply_flip_vertical(self):
        """Flips the current image vertically with Undo support."""
        if self.current_image_data is None:
            return

        self.previous_image_data = self.current_image_data.copy()
        self.current_image_data = flip_vertical(self.current_image_data)
        
        self.autostretched_cache = None  # Invalidate stretch preview cache
        self.update_display(autoRange=False)
        
        self.undo_action.setEnabled(True)
        self.undo_btn.setEnabled(True)
        self.statusBar().showMessage("Flipped image vertically. Press Ctrl+Z to undo.", 4000)

    def apply_curves_transformation(self):
        if self.current_image_data is None:
            return

        # Keep backup of original image in case user cancels, and as an Undo point
        # if the transformation is accepted.
        backup_data = self.current_image_data.copy()

        # Disable preview auto-stretch during curve transformation so user sees linear response
        self.is_autostretched = False

        dialog = CurvesTransformationDialog(self.current_image_data, parent=self)
        if dialog.exec() == CurvesTransformationDialog.DialogCode.Accepted:
            # The dialog commits its full-resolution result when it is accepted.
            self.previous_image_data = backup_data
            self.autostretched_cache = None
            self.update_display(autoRange=False)
            self.header_panel.update_histogram_only(self.current_image_data)
            self.undo_action.setEnabled(True)
            self.undo_btn.setEnabled(True)
            self.statusBar().showMessage("Applied curves transformation. Press Ctrl+Z to undo.", 5000)
        else:
            # User pressed Cancel: Restore original state
            self.current_image_data = backup_data
            self.autostretched_cache = None
            self.update_display(autoRange=False)

    def open_stretch_workbench(self):
        if self.current_image_data is None:
            return

        backup_data = self.current_image_data.copy()
        dialog = StretchWorkbenchDialog(self.current_image_data, parent=self)

        if dialog.exec() == StretchWorkbenchDialog.DialogCode.Accepted:
            # Data and display are already updated by dialog.accept()
            self.previous_image_data = backup_data
            self.undo_action.setEnabled(True)
            self.undo_btn.setEnabled(True)
            self.statusBar().showMessage(
                "Applied stretch workbench result. Press Ctrl+Z to undo.", 5000
            )
        # --- UNDO & HELP ---

    def undo_last_operation(self):
        if self.previous_image_data is None:
            return

        self.current_image_data, self.previous_image_data = (
            self.previous_image_data,
            self.current_image_data,
        )

        self.autostretched_cache = None
        self.update_display(autoRange=False)
        self.header_panel.update_header_info(
            self.windowTitle(), self.current_image_data, self.current_fits_header
        )
        self.undo_btn.setEnabled(True)
        self.statusBar().showMessage("Reverted image state (Ctrl+Z)", 4000)

    def show_about(self):
        QMessageBox.about(
            self,
            "About",
            "### Silly Astro Suite\n\n"
            "A custom Python tool for parsing, stretching, and visualising "
            "astronomical FITS data.\n\n"
            "Built with PyQt6 and PyQtGraph. And AI. A ton of AI"
        )


if __name__ == "__main__":
    pg.setConfigOptions(
        antialias=True,
        useOpenGL=True,
        background=(30, 30, 30),
        foreground=(200, 200, 200)
    )
    app = QApplication(sys.argv)
    window = AstroImageEditor()
    window.showMaximized()
    sys.exit(app.exec())
