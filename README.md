# Immich Ring Visualizer

A local tool to visualize face and CLIP embeddings from Immich in concentric similarity rings, with video frame analysis via InsightFace.

Thats not the best description for this tool. 
The immich half that works with the immich database to take an initial image with a face from your collection and find the closest matches
immich was written to deal with collections of real people and real people tend to wear the same face everyday. Immich Ring Visualizer was written
with AI generated faces in mind, which tend to drift into other faces. immich will find pictures of aunt mary but doesnt give the confidence that its 
actually aunt mary. With AI for a repeatable charactor you need a lora and that requires a certain number of good quality images of a charactor. 

The ring Visualizer was created to graphically show the confidence that your other images match the reference image which is at the centre. 
other images orbit this reference image the nearer they are to the reference the closer they are to being that reference face. 
with the data from your immich database any particular face file can be the reference infact clicking on an orbiting image makes that become the reference and then 
the best matches to that face are shown. it also has a list of images from most similar to least similar. 
immich takes the video thumbnail as the representative image for an mp4 unfortunately that might not be very representative of the frames inside the mp4

This is where the second side of the visualizer comes into play you can drag in a mp4 select a frame as reference and set simularity and blur parameters 
this will then process the mp4 file and choose which frames are above the threshold set. It does a blur check initially because a blurry face is not whats
needed for a lora , and then extracts the frames reaching your criteria. it can also create a video file with the 'junk' frames removed. or you can save the individual 
frames. these can be added to immich which will do its facial recognition scans and ideally will match other faces in your collection. 

The main thing is to generate enough good quality frames of the same charactor so the lora knows who that character is , you need 15 at least to train the lora. 
once you have the lora you should be able to create more images of that charactor and if its still not generating closely enough you can use the images it does produce to 
create better training material.  


## Features
- **Immich Visualizer**: Interactively explore nearest face or CLIP matches around a seed asset.
- **Video Clip Analyzer**: Frame-by-frame face detection, blur scoring, and similarity filtering on local MP4s using InsightFace.
- **Fisheye Focus**: Interactive radial magnification on hover.

## Requirements
- Python 3.10+
- `ffmpeg` (installed on system PATH for playback generation)
- Read access to your Immich PostgreSQL database and API key (optional if you don't run immich)

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
