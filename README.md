Immich Ring Visualizer & LoRA Frame Curator
<img width="2156" height="1354" alt="image" src="https://github.com/user-attachments/assets/d7715ea6-880c-4c76-86cf-9ca644b7d129" />


A specialized dataset curation and visual quality-control tool designed for AI workflow developers.

While Immich excels at organizing real-world photo collections, AI-generated characters present a unique challenge: facial drift. Immich’s default face clustering finds similar faces, but it doesn’t give a granular, visual representation of model confidence or character fidelity.

This tool bridges that gap by turning your Immich database into an interactive similarity radar and providing an automated video frame extractor designed specifically for building clean, unblurred LoRA training datasets.

Why This Tool Exists

Training a consistent character LoRA requires at least 15–20 high-quality, sharp, subject-accurate images.

1. Immich Ring Visualizer: Places a reference face at the center. Other images orbit this reference—the closer they orbit, the higher the confidence that they match the target face. Clicking any orbiting node instantly turns it into the new reference, recalculating surrounding matches so you can traverse face clusters, spot character drift, and isolate bad renders.
2. Video Frame Extractor (InsightFace): Immich uses a single static thumbnail to represent an entire MP4. This side of the tool lets you drop in an MP4, set a reference frame, and configure similarity and blur thresholds. It filters out motion blur, discards low-confidence frames, and extracts clean dataset candidates. You can save individual frames or output a transcode with “junk” frames removed.

Once exported, these clean frames can be fed into Immich or directly into your trainer to refine your LoRA and eliminate facial drift in future generations.

Features

* Concentric Similarity Radar: Visual confidence map where distance from center reflects facial embedding proximity.
* Dynamic Recenter on Click: Click any face to promote it to the reference node and re-index the visual cluster.
* Ranked Similarity Sidebar: Side-by-side list showing exact match percentages from highest to lowest.
* MP4 Frame Extractor: Interactive reference frame picker for video files.
* Automated Blur & Drift Filtering: Dual-stage evaluation using InsightFace to discard blurry or off-model frames.
* Junk-Frame Removal: Export cleaned video files or raw frame sets directly for training pipelines.

Prerequisites

* Python 3.10+
* ffmpeg installed on your system PATH (required for video processing)
* Read access to an Immich PostgreSQL database and Immich API key (optional if only using the MP4 video extractor)

For instructions on exposing PostgreSQL in Docker and generating an API key, see IMMICH_SETUP.md.

Installation

1. Clone the Repository

git clone https://github.com/blackest/immich-ring-visualizer.git
cd immich-ring-visualizer

2. Create a Virtual Environment

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Configuration

Edit the configuration block near the top of ring_viz.py to match your environment:

PG_HOST = "localhost"
PG_PORT = 5432
PG_USER = "postgres"
PG_PASSWORD = "your_db_password"
PG_DB = "immich"
IMMICH_BASE_URL = "http://localhost:2283"
IMMICH_API_KEY = "your_api_key_here"

Security Note: Never commit actual database passwords or API keys to GitHub.

Usage

Start the local server:

python3 immichring.py

Then open:

http://localhost:5050

this repo works standalone without immich being installed on your system it will say loading initially as it cant find the immich postgres db
if you want to use it with immich you need to expose the postgres.db ports normally only accessible within docker 
there are 4 versions now probably sticky is the best as it both lets you compare existing images in the immich db 
and selective saving of both the frames and any existing immich images that corolate, still some judgement needed for side profiles
they will score lower but you need coverage for the lora. 

An Example prompt might be 

"A woman moves in to a quiet courtyard with natural, unselfconscious movement — checking something nearby, glancing over her shoulder, reaching for an object, pausing to look directly at camera, then turning away again. The camera doesn't hold still: it drifts and re-angles with her, circling partway around during her movement, pushing in close on her face during a still moment, pulling back wide as she crosses the space. Coverage includes clear profile, three-quarter, and face-on angles as she turns. Natural shifting light. No single held pose — her head and body continuously reorient through small, ordinary actions."
