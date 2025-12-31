# ZALO AI Challenge 2025 - Object Detection Solution

##  Overview

This solution implements an object detection system for drone videos using YOLOv8 with color-based filtering. The system detects and tracks specific objects across video frames by combining deep learning object detection with color similarity matching against reference images.

##  Project Structure

```
.
├── annotate_video.py          # Video annotation visualization tool
├── predict.py                 # Main prediction script
├── predict.sh                 # Shell script to run
├── predict_notebook.ipynb     # Jupyter notebook for experimentation
├── Dockerfile                 # Docker container configuration
├── Makefile                   # Build and run commands
├── requirements.txt           # Python dependencies
├── .env / .docker.env        # Environment configuration
├── data/                      # Input data directory
│   └── public_test/
│       └── samples/           # Test cases
├── result/                    # Output directory
├── saved_models/              # Trained models
│   └── yolov8l_1e.pt         # YOLOv8 Large model
|── zalo-ai-train-yolo-2.ipynb  # Training notebook
└── start_jupyter.sh        # Script to start Jupyter Notebook
```

##  Quick Start

### Prerequisites

- Docker with GPU support (recommended)
- Python 3.8+ (for local execution)

1. **Build the Docker image:**

```bash
docker build -t zalo-ai-2025 .
```

2. **Run prediction:**

```bash
docker run --gpus all --rm -it \
		-v "$(PWD)/data/public_test/public_test:/data" \
		-v "$(PWD)/result:/result" \
		--env-file .docker.env \
		zac2025:v1 \
```

Or use the Makefile:

```bash
make run-docker
```

Replace the paths as needed.

3. **Visualize results:**

```bash
python3 annotate_video.py \
  --video_dir {path to video directory} \
  --output_dir {path to output directory} \
  --json_path submission.json
```


