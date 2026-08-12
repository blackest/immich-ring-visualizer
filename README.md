# Immich Ring Visualizer

A local tool to visualize face and CLIP embeddings from Immich in concentric similarity rings, with video frame analysis via InsightFace.

## Features
- **Immich Visualizer**: Interactively explore nearest face or CLIP matches around a seed asset.
- **Video Clip Analyzer**: Frame-by-frame face detection, blur scoring, and similarity filtering on local MP4s using InsightFace.
- **Fisheye Focus**: Interactive radial magnification on hover.

## Requirements
- Python 3.10+
- `ffmpeg` (installed on system PATH for playback generation)
- Read access to your Immich PostgreSQL database and API key

## Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/immich-ring-visualizer.git](https://github.com/your-username/immich-ring-visualizer.git)
   cd immich-ring-visualizer
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

## Configuration

Edit the configuration block near the top of `ring_viz.py` to match your local setup:

```python
PG_HOST = "localhost"
PG_PORT = 5432
PG_USER = "postgres"
PG_PASSWORD = "your_db_password"
PG_DB = "immich"

IMMICH_BASE_URL = "http://localhost:2283"
IMMICH_API_KEY = "your_api_key_here"
```

> **Warning**: Do not commit your actual API keys or database passwords to GitHub.

## Usage

Run the web app:

```bash
python3 ring_viz.py
```

Open `http://localhost:5050` in your browser.
