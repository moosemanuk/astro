from typing import Any, Callable
from PyQt6.QtCore import QThread, pyqtSignal


class ProcessingWorker(QThread):
    """Executes an image processing function in a background thread and reports progress."""

    progress_changed = pyqtSignal(int)
    finished_with_result = pyqtSignal(object)
    failed_with_error = pyqtSignal(str)

    def __init__(
        self,
        target_func: Callable,
        image_data: Any,
        params: dict,
        parent=None,
    ):
        super().__init__(parent)
        self.target_func = target_func
        self.image_data = image_data
        self.params = params

    def run(self):
        try:
            def callback(percent: int):
                self.progress_changed.emit(percent)

            result = self.target_func(
                self.image_data, **self.params, progress_callback=callback
            )
            self.finished_with_result.emit(result)
        except Exception as err:
            self.failed_with_error.emit(str(err))
