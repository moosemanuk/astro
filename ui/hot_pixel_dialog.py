from PyQt6.QtWidgets import QProgressDialog
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from processing.hot_pixels import remove_hot_pixels


class HotPixelWorker(QThread):
    """Background worker thread to run hot pixel removal without freezing the UI."""
    progress_changed = pyqtSignal(int)
    result_ready = pyqtSignal(object)

    def __init__(self, image_data, threshold=3.0, detect_chroma=True, parent=None):
        super().__init__(parent)
        self.image_data = image_data
        self.threshold = threshold
        self.detect_chroma = detect_chroma

    def run(self):
        def report_progress(val):
            self.progress_changed.emit(int(val))

        cleaned = remove_hot_pixels(
            self.image_data,
            threshold=self.threshold,
            detect_chroma=self.detect_chroma,
            progress_callback=report_progress
        )
        self.result_ready.emit(cleaned)


class HotPixelProgressDialog(QProgressDialog):
    """Progress dialog showing worker status and completion bar."""
    result_ready = pyqtSignal(object)

    def __init__(self, image_data, threshold=3.0, detect_chroma=True, parent=None):
        super().__init__("Removing hot pixels and impulse noise...", "Cancel", 0, 100, parent)
        self.setWindowTitle("Processing Image")
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.setMinimumDuration(0)
        self.setValue(0)

        self.cleaned_data = None
        self.worker = HotPixelWorker(image_data, threshold, detect_chroma, parent=self)
        self.worker.progress_changed.connect(self.setValue)
        self.worker.result_ready.connect(self._on_worker_finished)
        self.canceled.connect(self._on_canceled)

    def start(self):
        self.show()
        self.worker.start()

    def _on_worker_finished(self, cleaned_data):
        self.cleaned_data = cleaned_data
        self.setValue(100)
        self.accept()  # Triggers QDialog.Accepted cleanly

    def _on_canceled(self):
        if self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()