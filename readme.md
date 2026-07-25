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