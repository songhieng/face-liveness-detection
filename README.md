# Face Liveness Detection System

A robust face liveness detection system that can identify real (live) faces from fake (spoof) ones, such as photos, videos, or masks. This system is built on the CelebA-Spoof dataset model and provides both a command-line interface and a REST API.

## Features

- **High Accuracy**: Uses a deep learning model trained on the CelebA-Spoof dataset
- **Multiple Interfaces**:
  - Command-line tool for processing images or webcam feed
  - REST API for integration with other systems
- **Flexible Deployment**: Run locally or deploy as a Docker container
- **Multiple Input Methods**: Process single images, directories of images, or webcam feed
- **Batch Processing**: Process multiple images at once via the API

## Project Structure

```
.
├── config/                  # Configuration files
├── models/                  # Model storage
│   └── ckpt_iter.pth.tar    # CelebA-Spoof model checkpoint
├── src/                     # Source code
│   ├── api/                 # API functionality
│   │   ├── app.py           # FastAPI application
│   │   └── routes.py        # API routes
│   ├── core/                # Core functionality
│   │   ├── app.py           # CLI application
│   │   ├── cli.py           # CLI entry point
│   │   └── detector.py      # Liveness detector
│   └── models/              # Model definitions
│       ├── aenet.py         # AENet model architecture
│       └── detector.py      # Detector abstract class
├── utils/                   # Utility functions
├── run.py                   # Main runner script
├── Dockerfile               # Docker configuration
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Requirements

- Python 3.7+
- PyTorch 1.7.0+
- OpenCV 4.5.0+
- FastAPI (for API mode)
- See `requirements.txt` for full dependencies

## Installation

### Option 1: Local Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/liveness-detection.git
   cd liveness-detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the model file is in the correct location:
   ```
   CelebA-Spoof/intra_dataset_code/ckpt_iter.pth.tar
   ```

### Option 2: Docker Installation

1. Build the Docker image:
   ```bash
   docker build -t liveness-detection .
   ```

2. Run the container:
   ```bash
   docker run -p 8000:8000 liveness-detection
   ```

## Usage

### Command-line Interface

The system can be used directly from the command line:

```bash
# Process a single image
python run.py detect path/to/image.jpg

# Process all images in a directory
python run.py detect path/to/directory

# Start webcam mode
python run.py webcam

# Enable verbose logging
python run.py detect -v path/to/image.jpg
```

### Direct Integration

If you want to integrate the liveness detection system directly into your project without using the API, you can use the integration module:

```python
# Import the integration functions
from src.integration import detect_from_image, detect_from_file

# Process an image file
result = detect_from_file("path/to/image.jpg")
print(f"Is Live: {result['is_live']}")
print(f"Live Probability: {result['live_probability']}")
print(f"Status: {result['status']}")  # "LIVE", "SPOOF", or "ERROR"

# Process a numpy image array (RGB format)
import cv2
image = cv2.imread("path/to/image.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
result = detect_from_image(image_rgb)

# For more examples, see examples/integration_example.py
```

### REST API

Start the API server:

```bash
python run.py api
```

Or with custom host and port:

```bash
python run.py api --host 127.0.0.1 --port 5000
```

The API documentation will be available at http://localhost:8000/docs

#### API Endpoints

- `GET /health` - Health check endpoint
- `POST /detect/image` - Detect liveness from an uploaded image file
- `POST /detect/base64` - Detect liveness from a base64 encoded image
- `POST /detect/batch` - Batch process multiple images

Example curl commands:

```bash
# Health check
curl -X GET http://localhost:8000/health

# Upload and process a single image
curl -X POST \
  -F "file=@path/to/image.jpg" \
  -F "threshold=0.5" \
  http://localhost:8000/detect/image

# Process a base64 encoded image
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD/2wBD...", "threshold": 0.6}' \
  http://localhost:8000/detect/base64

# Batch process multiple images
curl -X POST \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "threshold=0.5" \
  http://localhost:8000/detect/batch
```

Example Python client:

```python
import requests

# Upload an image file
with open('image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/detect/image',
        files={'file': f},
        data={'threshold': 0.6}
    )
    result = response.json()
    print(f"Is Live: {result['is_live']}")
    print(f"Live Probability: {result['live_probability']}")
    print(f"Status: {result['status']}")  # "LIVE", "SPOOF", or "ERROR"
```

### API Response Format

The API returns responses in the following format:

```json
{
  "is_live": true,
  "live_probability": 0.9998519420623779,
  "spoof_probability": 0.00014802678197156638,
  "message": "Real face detected",
  "status": "LIVE"
}
```

The `status` field contains a machine-readable identifier:
- `"LIVE"`: Real face detected
- `"SPOOF"`: Fake face detected
- `"ERROR"`: Processing error occurred

## Development

To set up a development environment:

1. Clone the repository
2. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the API server with hot reload:
   ```bash
   python run.py api --reload
   ```

## Model Information

This system uses the AENet model from the CelebA-Spoof paper. The model architecture has been integrated directly into the project in `src/models/aenet.py` and the model checkpoint is stored in the `models` directory.

```
@inproceedings{CelebA-Spoof,
  title={CelebA-Spoof: Large-Scale Face Anti-Spoofing Dataset with Rich Annotations},
  author={Zhang, Yuanhan and Yin, Zhenfei and Li, Yidong and Yin, Guojun and Yan, Junjie and Shao, Jing and Liu, Ziwei},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2020}
}
```

## License

This project is available for non-commercial research purposes only, as per the CelebA-Spoof dataset license. See the CelebA-Spoof LICENSE file for details.