# Lightweight Video Processing System with OCR

A real-time video processing system with OCR capabilities that allows drawing on video frames and replacing detected text with custom content.

## Features

- Real-time video processing (30-60 FPS)
- Text detection using lightweight OCR
- Text replacement/removal
- Custom drawing on video frames
- Modular and extensible architecture

## Requirements

```
python>=3.6
opencv-python>=4.5.0
numpy>=1.19.0
easyocr>=1.4.1
torch>=1.7.0
torchvision>=0.8.0
pillow>=8.0.0
```

## Project Structure

```
├── src/                    # Source code
│   ├── capture/            # Video capture module
│   ├── processing/         # Video processing pipeline
│   ├── ocr/                # OCR detection and text replacement
│   ├── drawing/            # Drawing utilities
│   └── utils/              # Helper functions
├── examples/               # Example scripts
├── tests/                  # Unit tests
├── requirements.txt        # Dependencies
└── README.md               # Documentation
```

## Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`

## Usage

Run the example script:

```python
python examples/demo.py
```

## License

MIT