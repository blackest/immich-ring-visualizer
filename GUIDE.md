# Immich Ring — the guide

What this app is, what each part does, how the parts fit together, and what to watch out for.
Written 2026-11-06 from the code and the docs in the repo; brought up to date on 2026-11-09 (Speech, Talk, Scene,
Soft finish, the v5.0.7 runner, RefMods as characters, Low memory). Most of it was read from source rather than
clicked through in a browser, so anything marked **(unverified)** is inferred rather than seen working. Features
added on 8 to 9 November were also proven with real renders through the app, and say so.

---

## 1. What it is

A local Flask app (`ring_viz.py`, port **5050**, bound to `0.0.0.0` so it is reachable over the LAN and
Tailscale) that began as a face-curation tool for building LoRA training sets and grew into a front end
for a whole local AI studio on the Mac Studio:

1. **Curate** faces from an Immich library, a video, or a folder/zip, ranked by similarity.
2. **Generate** a character sheet (a turnaround of poses) from one photo.
3. **Animate** a picture into video (LTX-2.5).
4. **H3** video (MiniMax-H3, with reference pictures, LoRAs, chaining, re-rolls).
5. **Music** (YuE2) and ABC-score editing.
6. **Speech** (Qwen3-TTS): a script and a voice in, a WAV out.
7. **Talk**: speech plus a character into a lip-synced 5 s clip.
8. **Scene**: a script of lines said by different characters, rendered as one joined clip with sound.
9. **Pictures** with FLUX 3 Klein, HiDream multi-ref edits, and arbitrary ComfyUI workflows.
10. **Chat** with a local model (Ollama) and with Rachel (a Hermes agent).

Open the current UI at **`/immichRingNG`** (not `/`). The old single-page app at `/` still exists and is
frozen. Everything new goes in the "NG" files.

## 2. The ground rules the code follows

These explain why the repo looks the way it does.

- **NG twins.** Every module the new UI needs is a copy with an `NG` suffix (`videoNG.py`, `detectionNG.py`).
  NG never imports from the originals, and the originals are read-only. This stops a change to the new UI
  breaking the old page.
- **Nothing fetches or generates by itself.** No Immich query or render starts except from an explicit click.
  A page refresh must not re-run anything.
- **No disk cache.** Only exports (and renders, which are expensive to redo) write to disk. Ingested video
  frames, downloads and thumbnails live in RAM only.
- **Small single-purpose engine files.** An engine is a file that turns one input into one output file and
  runs the model as a **subprocess in its own venv**, never in-process. Nothing heavy is imported into the app.
- **Job queues, not locks.** Each engine has an in-memory FIFO queue with its own worker thread. POST returns a
  job id at once and the browser polls. The in-memory list is a ring buffer of 20 jobs, so each engine also has a
  separate durable **job log** that is never pruned.
- **One GPU.** Heavy jobs wait for each other (H3, HiDream, Klein, LTX, Speech). Before a big render the app asks
  Ollama and the Gemma server to unload their models (`model_offloadNG.py`).
- **Machine addresses are settings, not constants.** See section 15.

## 3. The screen: four zones

| Zone | What lives there |
|---|---|
| **Top bar** | Character/project tabs. A tab is one character with its own state, video, ring and selection. |
| **Bottom bar** | The task (the "what"): Video, Immich, Folder / Zip, Generate, Animate, H3, Enhanced H3, Music, Music Edit, Speech, Talk, Scene, Chat, Rachel, Hd-Multi, Klein, ComfyUI. |
| **Left rail** | Tools for the current task (the "how"). The big generation tasks take over the whole rail. |
| **Main stage** | The working surface: the ring, the picker grids, a prompt box, a gallery. |

A settings cog opens addresses, Tailscale, model lists and a few curated launchers.

Projects are saved as characters (`/api/ng/characters`). Eviction of idle projects from RAM is silent: durable
state is saved, so nothing is lost, only re-decoded when you return.

## 4. Curation: Video, Immich, Folder / Zip

All three feed the same ring and sidebar.

**Video** (`routes/videoNG.py`, `video_analysisNG.py`)
- Load a clip (upload, or a URL through yt-dlp), scrub to pick a **reference frame**, then analyse. PyAV decodes
  from memory, InsightFace (buffalo_l) finds the face per frame, and each frame gets similarity to the reference,
  blur (Laplacian variance), yaw/pitch/roll and shot scale.
- Only frames that pass the similarity and blur thresholds are kept and exportable.
- **Build playback** renders a copy of the clip with rejected frames blanked, so you can audit a threshold against
  the real footage. It opens in a modal.
- **Video split** cuts a clip into parts.
- Face analysis is serialised with a lock because the shared InsightFace session is not thread-safe, while jobs
  from different tabs still run in parallel.

**Immich** (`routes/immichNG.py`)
- Reads Immich's Postgres/pgvector directly: random face, find by filename, nearest neighbours, person clusters,
  per-person assets, per-asset pose. Thumbnails and previews are proxied, never stored.
- Click any node to make it the new reference and the ring recomputes.
- Needs Postgres reachable (the Immich DB port exposed) and an API key. Without Immich the tool still works on
  videos and folders.

**Folder / Zip** (`routes/folderNG.py`)
- Upload images or a `.zip`; the same analysis runs over stills.

**Ring and rail controls**
- Concentric similarity ring: distance from the centre is face-embedding proximity.
- **Anchor**, ring scale, similarity squeeze, sharpness cut-off, **Yaw / Pitch** filters, a ranked sidebar with
  match percentages, a fisheye lens, hover previews with the matched-face box.
- **Pose picker** and **Shot-scale picker** are full-width workspaces on the main stage.
- **Reject vs deselect:** *reject* kicks a frame out of the ring and sheet for good (use it for embedding false
  positives); *deselect* just removes it from the export set.
- **Find neutral** looks for a straight-on passport-style face to use as a better reference.

**Export** (`routes/exportNG.py`, `exporterNG.py`, `image_opsNG.py`)
- Export Settings control crop and resize. Exports go to `exportsNG/<name>/`, from a video job or straight from a
  list of Immich assets. This is the step that actually writes a training set to disk.

**Dome** (`dome_engineNG.py`, `dome_pick.py`): picks one face per yaw/pitch cell for even pose coverage. The
engine and a command-line script exist; whether a panel route is finished is **(unverified)**.

## 5. Generate (character sheet)

`routes/generateNG.py`, `character_sheetNG.py`, `sheet_jobsNG.py`, `hidream_engineNG.py`, `shot_presetsNG.py`

- One reference photo (from disk, pasted, an Immich asset, or the current curation frame) becomes a **shot list**
  of turnaround views, rendered by HiDream-O1 as a subprocess.
- Presets: a `default` and an `extended` shot list; backgrounds `studio`, `interior`, `outdoor`, `minimal`; looks
  `none`, `cinematic`, `editorial`, `documentary`, `warm_film`, `dramatic`. Each shot can carry a free-prompt
  override.
- Shots are anchor-chained, seeded per shot, and joined into a composite strip. Re-roll a single shot. A truncated
  PNG from an aborted render is detected and discarded.
- The render queue is in the left rail; characters land in a picker grid; the chosen picture can be used as the
  reference elsewhere ("use Generate's current reference" in Animate and H3).
- A spent character avatar and sheet are saved under the character.

## 6. Animate (LTX-2.5 image or text to video)

`routes/videogenNG.py`, `ltx_engineNG.py`, `video_jobsNG.py`, `ltx_*`

- Reference picture (or none for text-to-video) plus a motion prompt and duration. Fields: width, height, frame
  rate, seed, optional LoRA and strength.
- **Enhance** rewrites your prompt with Gemma, and Gemma now *sees* the reference image.
- **Chat with Gemma** and **Discuss next scene** (proposes three next-shot options with durations) both take
  image attachments. Note Gemma only sees the image attached to the last turn.
- **Estimate duration** asks Gemma how long a shot needs.
- **Install now** (about 66 GB) sets up the engine; **Job Log** is the durable history with the reference and
  the video.
- **Output codec** (`ltx_codecNG.py`): the Animate output-quality setting defaults to **Web** (yuv420p, crf 23).
  Before commit `1481a9b` (21 Oct) clips were 4:4:4 near-lossless at 31 to 103 Mbps, which is why older clips in
  `exportsNG/_videogen` are huge. No new clip has been rendered since the fix, so the small size is **unverified**.
- 2x upscale uses LTX's spatial upscaler (`upscale_x2NG.py`).

## 7. H3 (MiniMax-H3 video)

`routes/h3NG.py`, `h3_engineNG.py`, `h3_jobsNG.py` and a family of small `h3_*NG.py` helpers. The runner is
**vendored** in `vendor/minimax_h3_mlx` (an MLX port of the model), run in Phosphene's venv. It was last merged with
upstream on 8 November, to the pin of Phosphene v5.0.7 (`59da1474b3`); `VENDORED.md` records each merge and every
flag that is ours.

**Models:** `h3` (bf16), `h3q8` (Q8, the default), `h3ref` (the official Ref2VA, needed for references).
Quality tiers draft / standard / high (1024x576) / native, or explicit width and height.

**Speed:** *Fast* (`tristep`, the TaoMate 3-step adapter), *Turbo* (4-step LoRA), *LightX2V v1.3*, *Flex* (Pika's
six-forward sampler). Turbo and v1.3 need the bf16 model.

**Memory and decode:**
- **Low memory** tickbox (runner `--stream-dit`): the DiT blocks are read from disk one at a time with the next one
  prefetched, so the denoise peak was about 3.75 GiB through the app instead of about 22 GiB, with identical output.
  Measured with warm files only; a cold read from the SSD has not been timed. It refuses cleanly on a runner without
  the flag.
- **Fast VAE** tickbox: decodes in plain float16. Unticked, the runner's default is a float16-mixed decoder that
  stays closest to float32 (`H3_VAE_DTYPE=float32` reproduces the old decode exactly).

**Inputs:** text only; a first-frame keyframe; keyframes at times; a last frame; extending an existing clip.

**Audio drive** (`h3_audio_driveNG.py`, runner `--audio-drive`): give a soundtrack and the picture is generated to it,
so a face mouths the words. Each 5 s window gets its own slice and the delivered audio is the file you gave, not a
decoder round trip. About 19 to 20 min per 5 s at 640x384, 16 steps. A RefMod or Ref2VA reference can drive a face
too (the old refusal was lifted on 8 November after a test: the mouth followed the words and the identity held with no
LoRA and no first frame). It does not combine with *extend*. Talk and Scene are built on this.

**Duration:** one window is 5 s. Longer clips are **chained windows**, each starting from the previous window's
last frame, then stitched. A different prompt per window is supported (`chain_prompts`).

**References (Ref2VA):**
- *Reference pictures/clips* (`h3_referencesNG.py`, `--ref2va-reference`): plain pictures or clips encoded at
  render time. Work on one 5 s window; a chained render carries none.
- *RefMods* (`h3_ref2vaNG.py`, `h3_refmod_*NG.py`): a saved latent of up to 24 pictures of one person. Create one
  from pictures in the H3 tools (`h3_refmod_makeNG.py` runs the encoder as a queued job). A RefMod is the whole
  character: it needs the `h3ref` model, takes no first frame and no LoRA, works on one 5 s window, and Fast is
  allowed. Talk and Scene can use one as a character, and the **identity kit** in the Scene view builds one from a
  single picture.
- *Scene batch* (`h3_batchNG.py`): several windows, each with its own references and optionally the previous
  window's last frame.
- **Look for reference pictures** (new): *Standard* uses the Ref2VA model; *Clean* uses the FL2VA Q8 base with
  Ref2VA AdaLN borrowed on blocks 45 to 49 (the "hybrid"). Findings: Clean gives less crushed blacks and softer
  contrast than Standard, but block 49 alone, and no overlay at all, measured the same. So most of the look comes
  from the FL2VA base. 45 to 49 is kept as the more experimental option. The dropdown is greyed out for Flex and
  for RefMods. It has **not been clicked through in a browser**.

**LoRAs:** style and character LoRAs from the shared Phosphene folder, with strength. Kohya-format files are listed
"unusable" until you use **Convert** (original kept in `_originals`).

**After a render:**
- **Windows panel:** re-roll one window (new seed, optional new prompt, optionally start on the previous last
  frame), then re-stitch.
- **Joins:** hard cut, dissolve, fade through black, or an audio lead (J-cut).
- **Ask Gemma** (`h3_critiqueNG.py`) critiques one window from a contact sheet.
- **Upscale 2x** as a queued job.
- **Job Log** with year/month archives; downloads; a film joined from several renders can be added to the log.
- **Install now** with a **guard**: if the engine checkout belongs to Phosphene the install buttons no longer
  re-pin or rewrite it.

**Enhanced H3** (`routes/enhancedh3NG.py`): a sentence plus up to 9 pictures, clips (2 to 15 s) and audio goes to
OpenH3-IR's `h3ir serve`, which writes a structured brief. The pane splits the brief into one prompt per 5 s
window, shows it for editing, then sends it through the ordinary H3 queue. Options include seconds, aspect, shots,
creativity, a Director profile (the compiler has seven built in; `DIRECTOR_PROFILES.md` has drafts for more),
dialogue lines with language, and silent. `h3_continueNG.py` fills in windows the brief left empty. Needs the
compiler running at the `h3ir_base_url` setting. **Not tested:** video clips in the tray, a 15 s three-window chain
from the pane, a spoken line through the pane.

Known behaviour worth remembering: legible text on props (name cards, signs) garbles in every H3 model, so prompt
around it. 9:16 faces look tall; 16:9 faces look more natural but darker.

## 8. Music

- **Music** (`music_*NG.py`): YuE2 turns a style and lyrics into a song. Fields: style, lyrics, duration, seed,
  precision, cfg scale, temperature. A **Cover** mode transcribes an existing track's melody and chords and
  renders your style over them (first use downloads extra models, a few GB). **Install now** is about 11 GB.
- **Music Edit:** load or paste an ABC score, change tempo and key, and render real audio from it. Includes an AI
  lyrics writer/reviser and a "discuss the song" chat, with a model picker and an editable system prompt.
- Same queue plus durable Job Log shape as the video tasks.

## 9. Speech

`routes/speechNG.py`, `speech_engineNG.py`, `speech_jobsNG.py`, `static/speechNG.js`,
`vendor/speech_runner/` (a copy of Phosphene's runner plus our VoiceDesign runner)

A script and a voice in, one WAV out. Qwen3-TTS 12 Hz 1.7B (Apache-2.0) through mlx-audio, run as a subprocess in
Phosphene's `speech-mlx` venv. Only the venv and the weights are shared with Phosphene; the runner is ours, so a
Phosphene update cannot change it. Peak about 6 GB, 1 to 1.6 times real time.

Four ways to choose a voice:
- **Preset:** nine named fictional speakers plus a delivery style (eight built in, or free text).
- **Describe a voice:** a voice made from a written description (Qwen3-TTS VoiceDesign). Optional: the option is
  hidden when its weights are not installed. A designed voice differs on every generation, so you can **keep a take
  as a saved voice**.
- **Character voice:** a clone from `<name>.voice.<ext>` saved beside a character LoRA in the shared loras folder.
- **Clip:** a clone from a short clip you give it.

A clone needs a per-request consent tick (the voice is yours, or you have the speaker's permission). It is written
into the output's `.json` record. Output is in `exportsNG/_speechgen/<job>/speech.wav`. The queue rebuilds from the
server after a reload, and trimming the in-memory list never deletes a finished take.

## 10. Talk

`routes/talkNG.py`, `talk_jobsNG.py`, `static/talkNG.js`

Script, voice and character in, one lip-synced 5 s clip out. One job chains a Speech job into an H3 audio-drive job.
The character is a **RefMod** (the whole character; no picture or LoRA), or a character **LoRA** with an optional
face picture as the first frame, or a face picture alone. Quality and the Fast adapter are the H3 ones, and an
optional **Soft finish** pass can be applied to the result.

Proven by real renders: a speech WAV drives an articulated mouth that stops when the voice stops, with a LoRA, with
none, and with a RefMod. Limits: the clip is a fixed 5 s window, the first second can stay closed on some clips, and
word-level timing is not verified. The app does not offer a downloadable library of real people's LoRAs; characters
come from your own pictures, saved voices or presets.

## 11. Scene

`routes/sceneNG.py`, `scene_jobsNG.py`, `scene_scriptNG.py`, `scene_voiceNG.py`, `scene_ambienceNG.py`,
`scene_composeNG.py`, `scene_looksNG.py`, `static/sceneNG.js`. Output in `exportsNG/_scenegen/<job>/`.

A script of lines said by different characters, rendered as one joined clip with sound:

```
KNIGHT [on the battlements]: The wall has stood a long time, and so have I.
WOMAN [in the great hall]: Then come in out of the wind.
KNIGHT: Some nights I count the stones.
```

- **One line is one 5 s clip** (about 12 words; at most 18 words and 12 lines). The `[setting]` after a name says
  where that clip happens; an untagged line stays where that speaker last was.
- **Voices:** one Speech pass per character over all of their lines, cut back into one piece per line at the pauses
  (`scene_voiceNG`), then brought to one speech level. Cloning each line separately gave three different voices, so
  this is deliberate.
- **Characters:** each is a RefMod, a LoRA, or one or more named **looks** (pictures), with a voice. A look whose
  name matches a line's setting becomes that clip's first frame. A RefMod character renders on the `h3ref` model with
  no first frame, so the place comes from the prompt text only and follows it loosely (a "torch-lit great hall" came
  out as battlements with torches).
- **Two speakers in one frame:** a line starting with `&` joins the line before it into one shot, as a vertical
  split-screen made from the two characters' pictures, with one audio track carrying both voices in turn. This works
  for clearly different people and is ambiguous for look-alikes. A shared shot needs a picture for both characters
  and cannot use a RefMod, and the two lines together must fit one 5 s window.
- **Ambience:** optional, one sound bed per setting (H3's own sound of the empty place, mixed 22 dB under the
  dialogue). A bed that fails leaves that setting dry; it never stops the scene.
- **Joining:** the clips are trimmed to their speech and cut together in script order. Each clip is held to its own
  starting exposure, because H3 clips brighten as they play and that stepped 16 to 20 levels at every cut. The full
  Soft finish look is optional, and a failed finish never loses a render.
- **Looks and the identity kit** (`scene_looksNG.py`, Klein): from one picture of a character, *Looks* makes the same
  person somewhere else (about 170 s a picture), and the **identity kit** makes eight pictures (four face angles, four
  body shots, all ending in the same "same person, identical face, hairstyle, skin tone and outfit" line), and can
  turn them straight into a RefMod (about 25 minutes). The first picture sets the clothes, so all eight keep them.

Scenes run one at a time. The first Scene render through the app with a RefMod character (two Knight lines,
Fast, draft) held identity and outfit across both shots.

## 12. Soft finish

`finish_engineNG.py`, `finish_jobsNG.py`, `routes/finishNG.py`, `vendor/soft_finish/soft_finish.py`

A finished mp4 in, a finished mp4 out; the original is never touched. A numpy/OpenCV port of two MIT-licensed
Orion4D_FXMax nodes, run as a subprocess in the app's own Python (no torch, no ComfyUI, about 2 s for a 5 s clip).
Per frame: an exposure hold (so a joined clip does not step in brightness), Clarity at -0.20 (softer skin texture),
then a gentle Cineon-style tone map at 0.25 with saturation 0.9. Measured on a lip-sync clip: crushed blacks
22.5% to 13.1%, brightness up about 9, contrast about the same. It cannot bring back shadows that were clipped during
generation, and the Filmic tone modes darken normal video, so the mode is fixed. Talk and Scene offer it; the H3
page's queue rows do not yet.

## 13. Pictures: Klein, Hd-Multi, ComfyUI

- **Klein** (`kleinNG.py`, `klein_*NG.py`): FLUX 3 Klein from text or up to three labelled references (subject,
  location, extra), several variations per job, LoRAs from Phosphene's `flux3` folder, **Outpaint** (pads with grey,
  pastes the original pixels back), and **Improve prompt** (a narrow Ollama call that rewrites a short request as
  the full caption Klein expects; you see it and choose whether to use it). Keeps a gallery of earlier jobs. Waits
  while an H3 render holds the GPU. The Scene view uses Klein to make *looks* and an identity kit from one picture
  (section 11).
- **Hd-Multi:** HiDream edit and multi-reference, one-off jobs: object removal, or composing from 1 to 3 pictures
  with a prompt.
- **ComfyUI:** pick a PNG that ComfyUI made, and the app extracts the embedded workflow and builds a generic form
  for its parameters, then runs it on your ComfyUI server and polls the result. You can browse ComfyUI's own
  input/output/temp images, and save workflows with thumbnails.

## 14. Chat and Rachel

- **Chat:** a thin proxy to local Ollama. Model picker, system prompt, temperature, streaming, one global
  conversation held in memory.
- **Rachel:** a Hermes agent chat through a LAN gateway. Same shape as Chat without the model picker. The server
  holds the API key, replies stream as SSE converted to NDJSON, and a session header gives her memory. She can
  take a minute or two to start replying.
- Rachel can also drive Phosphene from the other machine through her own MCP copy. `RING_VIZ_MCP.md` and the memory
  notes describe a curated immichring MCP (compile, render, Klein, status, fetch, recent). **That server has not
  been built.**

## 15. Settings, addresses and installs

- `ng_settings.json` stores editable addresses (ComfyUI, Hermes session, Tailscale hostname and so on). New
  machine-specific endpoints belong in `NG_ADDRESS_SETTINGS` in `configNG.py`.
- Settings actions are curated single endpoints, not a shell: update yt-dlp, launch Suno, launch ComfyUI, a
  Tailscale status/up check, and an address check.
- **Install framework** (`engine_installNG.py`): one background job runner with per-engine step lists for LTX,
  Music and H3 (clone and pin, venv, deps, weights), ported from Phosphene's MIT-licensed installers. Models are
  shared with Phosphene under the Pinokio folder.

## 16. How it relates to Phosphene

- Phosphene is a separate app. immichring reuses its **model folders and venvs**, and the engine checkouts that sit
  beside `mlx_ltx_panel.py` (`ltx-2-mlx`, `yue2-mlx`, `minimax-h3-mlx`).
- Pinokio's Update moves Phosphene's pins. immichring must not. Install buttons now skip anything Phosphene owns.
- immichring keeps **its own vendored H3 runner** because Phosphene's Update re-pins its copy and would wipe the
  flags added here (AdaLN overlay, Ref2VA reference, RefMod, audio drive, prompt chaining). It is merged with
  upstream by hand when Phosphene moves its pin (last done for v5.0.7).
- **Phosphene.app creates a new version folder on every self-update**, so the engine links it provides (`image-gen`,
  `speech-mlx`, `yue2-mlx`, and the H3 checkout) point at a stale folder afterwards. That is what breaks Music, Speech
  and Klein after an update. Re-create them with `~/bin/phosphene-relink.sh` and restart this app if one is still
  broken. `~/bin/phosphene-whats-new` is a read-only check of what a release changes (engine pin, licence, outbound
  calls, our patch). Setting `PHOSPHENE_DISABLE_VERSION_CHECK=1` in Pinokio's ENVIRONMENT stops the studio polling for
  notices.
- Phosphene carries one local patch for access from other machines: it binds `0.0.0.0` and allows the LAN IPs and
  the Tailscale name/IP in its host and origin checks. Pinokio's Update overwrites it, so re-apply it after
  updating.
- Known gap: `ltx_codecNG.apply_codec_patch_ng` runs before every Animate render and can edit Phosphene's
  `ltx-2-mlx` source (the ffmpeg line). The markers are compatible with Phosphene's own patch.

## 17. Running it

```bash
cd ~/immich-app/repo/immichring
source venv/bin/activate
python3 ring_viz.py            # http://localhost:5050/immichRingNG
```

- Remote use goes over Tailscale; the iPad often relays through DERP London, which can cause "Load failed".
  Check `tailscale ping` before blaming an engine.
- The app runs with the **werkzeug reloader**. Editing a tracked `.py` file restarts the server and **kills any
  running render**. Check the queues first. To make a template or static-only edit take effect, `touch` a tracked
  `.py` file. The reloader shows two processes; that is parent and worker, so do not kill one.
- Tests: `python3 -m unittest discover tests` (79 test files, 1,035 tests as of 2026-11-09). Browser behaviour is
  mostly untested.

## 18. Things to know or fix

1. **A real Immich API key sits in `configNG.py`** (and in the old config). The repo's own README says never to
   commit these. Move it to an environment variable or `ng_settings.json` and rotate it if the repo is ever public.
2. The server binds `0.0.0.0` with no authentication. Fine on a trusted LAN or tailnet; do not expose it.
3. Job state is in memory. A restart loses queues and running jobs (the durable Job Logs survive).
4. The old Animate clips (about 119 clips, around 5 GB including copies in `_videogen` and `_job_logsNG`) are
   still the huge lossless kind. A re-encode is possible but not done.
5. `/Volumes/AI` was at 96% full after the last clean-up (about 76 GiB free). APFS clones mean deleting one copy can
   free nothing.
6. Old docs in the repo (`README.md`, `PROJECT_STRUCTURE.md`, `QuickStart.md`) describe the original single-page
   app and are out of date for the NG UI. `ADDING_A_PANEL.md` is the checklist for a new bottom-bar task and is
   still accurate. Seven dead `immichring*.py` files need a keep/archive decision.
7. The hybrid dropdown, the Enhanced H3 video/audio paths and the Dome panel have not been exercised in a browser.
   Speech, Talk and Scene were proven with real renders through the app's own routes; their pages are less tested
   (a character voice through the Speech page, clip drag-and-drop and the iPad layout in particular).
8. **Audio RefMods** (a voice latent that transfers a voice) are not built. A scratch prototype gave a suggestive but
   unproven pitch transfer, and the pack author says voice transfer is not validated.
9. The Scene view cannot put a RefMod character in a shared split-screen frame, and a RefMod's setting follows the
   prompt only loosely. Low memory has only been timed with warm files.
10. Werkzeug's reloader also restarts the server when a vendored `.py` file changes, so do not edit one while a
    render is running.

## 19. Where to look

| Question | File |
|---|---|
| Add a new bottom-bar task | `ADDING_A_PANEL.md` |
| Why the layout and data model are what they are | `APP_ARCHITECTURE_NOTES.md` |
| Enhanced H3 design and findings | `ENHANCED_H3_PLAN.md` |
| Director style drafts | `DIRECTOR_PROFILES.md` |
| Per-engine setup | `H3_SETUP.md`, `LTX_SETUP.md`, `MUSIC_SETUP.md`, `HIDREAM_SETUP.md`, `IMMICH_SETUP.md` |
| Ref2VA pruned packs | `docs/BUILD_REF2VA_PACKS.md` |
| What was taken from upstream's H3 runner, and what is ours | `vendor/minimax_h3_mlx/VENDORED.md` |
| Speech runner provenance | `vendor/speech_runner/VENDORED.md` |
| Soft finish provenance | `vendor/soft_finish/VENDORED.md` |
| An MCP server for the app (idea only) | `RING_VIZ_MCP.md` |
| Front-end load order and wiring | `static/WIRING_ORDER.md` |
