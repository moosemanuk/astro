# Astro Image Processor

A Python-based astronomical image processing application featuring a PyQt interface, specialized image manipulation algorithms, custom background gradient extraction, and star processing pipeline tools.

## Features

- **Interactive UI**: Custom PyQt interface designed for astronomical image display and analysis.
- **Background Gradient Extraction**: Surface model extraction with grayscale visualization for background noise and light pollution reduction.
- **Image Processing Engine**: Core image processing pipeline supporting custom stretching, sharpening, and color adjustments.
- **Extensible Architecture**: Modular structure designed for fast integration of new processing workflows.

## Directory Structure

```text
astro/
├── models/                  # AI model weights and binaries (ignored in git)
├── processing/              # Image processing algorithms and computational logic
├── ui/                      # PyQt window components, dialogs, and worker threads
│   └── dialogs/             # Processing configuration and preview dialogs
├── .gitignore               # Ignored binary files, data formats, and environments
├── main.py                  # Main application entry point
└── README.md                # Project documentation

Setup & Installation
1. Prerequisites
Python 3.9+ installed on your system.

2. Clone the Repository
Bash
git clone [https://github.com/moosemanuk/astro.git](https://github.com/moosemanuk/astro.git)
cd astro
3. Create a Virtual Environment
Bash
# On Windows
python -m venv astro_env
astro_env\Scripts\activate

# On macOS/Linux
python3 -m venv astro_env
source astro_env/bin/activate
4. Install Dependencies
Install the required packages within your active virtual environment:

Bash
pip install numpy pyqt6 pyqtgraph scipy astropy
Note: If you plan to experiment with ONNX-based neural network models, you can also install:

Bash
pip install onnxruntime
# For CUDA-enabled NVIDIA GPUs:
# pip install onnxruntime-gpu
Running the Application
Launch the application directly using Python:

Bash
python main.py
License
Distributed under the MIT License. See LICENSE for more information.


---

### How to push it to GitHub:
Once you've saved the `README.md` file in `E:\c\astro`, push it up with these two quick steps in VS Code terminal:

```bash
git add README.md
git commit -m "Add README documentation"
git push