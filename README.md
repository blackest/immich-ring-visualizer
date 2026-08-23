# Immich Ring Visualizer

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

## 1. Clone the Repository
```bash
git clone https://github.com/blackest/immich-ring-visualizer.git
cd immich-ring-visualizer
```

## 2. Create a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Configuration
Edit the configuration block near the top of `ring_viz.py` to match your environment:
```python
PG_HOST = "localhost"
PG_PORT = 5432
PG_USER = "postgres"
PG_PASSWORD = "your_db_password"
PG_DB = "immich"
IMMICH_BASE_URL = "http://localhost:2283"
IMMICH_API_KEY = "your_api_key_here"
```
**Security Note:** Never commit actual database passwords or API keys to GitHub.

## Usage
Start the local server:
```bash
python3 ring_viz.py
```
Then open `http://localhost:5050` in your browser.

### Standalone vs. Immich Mode
*   **Standalone:** This works without Immich installed on your system. It will initially say "loading" as it cannot find the Immich Postgres DB, but you can still easily analyze an MP4 file or a folder of images.
*   **Immich Integration:** If you want to use it with Immich, you need to expose the `postgres.db` ports (which are normally only accessible within Docker).

## Versions & Architecture
There are a few legacy versions of the original Python script in the repository, but **`ring_viz.py`** is the latest and most feature-complete version. It has no obvious regressions and refactors the project by separating the Python, JavaScript, HTML, and CSS into dedicated files. 

File handling is significantly improved: most data is dynamically held in RAM, and nothing gets written to your disk until you explicitly decide to save it. Immich images are now treated as first-class citizens and can be analyzed just as easily as an MP4 or local image folder.

## Features & LoRA Training Optimization
*   **Similarity Ring:** The first pass of the analyzer matches to whatever starting image is selected. The "Find Neutral" function attempts to locate a straight-on "passport photo" of your character. 
*   **Adjustable Thresholds:** Re-analyzing with this neutral reference finds the best similarity. You can squeeze the similarity index—for example, shifting from a 10% similarity cutoff up to 99% to cull images. Keep in mind that side profiles naturally score lower, so you will be hunting near the lower ends to get proper profile coverage.
*   **Pose Selectors:** Filter by roll (head rotation), yaw (left/right profile), and pitch (tilt up/down).
*   **Selective Saving:** Easily compare existing images in the Immich DB and selectively save both extracted frames and any correlating Immich images.

## What makes a good training profile?
Blurry images are the worst, so this is included as a primary filter. Good coverage and directional light also help the model significantly. 

Interestingly, if you have training images for LoRA A (which is effective) and LoRA B (which is weak), you can use the weak one as a guide to what is actually important for successful training. (Note: This tool does not handle captioning, as that is highly model-specific, and even the ideal image criteria may vary by model).

**Example Prompt for Generating Coverage:**
> "A woman moves in to a quiet courtyard with natural, unselfconscious movement — checking something nearby, glancing over her shoulder, reaching for an object, pausing to look directly at camera, then turning away again. The camera doesn't hold still: it drifts and re-angles with her, circling partway around during her movement, pushing in close on her face during a still moment, pulling back wide as she crosses the space. Coverage includes clear profile, three-quarter, and face-on angles as she turns. Natural shifting light. No single held pose — her head and body continuously reorient through small, ordinary actions."

## Ethics
**Please use this tool ethically.** This is a tool for creating more consistent fictional people, not for making someone's life a living hell. We welcome bug reports, discussion, and good prompts for getting better dataset coverage.
