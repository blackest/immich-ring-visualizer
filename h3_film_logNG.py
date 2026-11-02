"""immichRingNG -- add a finished film that was joined from several renders to the H3 job log.

h3_job_logsNG.record_job_ng() files one render. A scene made of several 5 s renders (h3_batchNG.py)
has no single job behind it, so this writes the same three files (`<id>.json`, `<id>.mp4`,
`<id>.log`) into the same day folder, which makes the film show up in the Job Log like any other.
"""

import json
import os
import shutil
import time
import uuid
from pathlib import Path

from configNG import H3_JOB_LOG_DIR


def record_film_ng(*, mp4_path, prompt: str, duration_s: float, model: str, width: int, height: int,
                   steps: int, notes: str, log_text: str, queued_at=None) -> dict:
    """Copy `mp4_path` into the job log as a finished entry and return {"job_id", "date"}."""
    now = time.time()
    date_str = time.strftime("%Y-%m-%d", time.localtime(now))
    day_dir = Path(H3_JOB_LOG_DIR) / date_str
    day_dir.mkdir(parents=True, exist_ok=True)
    job_id = uuid.uuid4().hex[:12]
    while (day_dir / f"{job_id}.json").exists():
        job_id = uuid.uuid4().hex[:12]
    shutil.copyfile(mp4_path, day_dir / f"{job_id}.mp4")
    (day_dir / f"{job_id}.log").write_text(log_text, encoding="utf-8")
    meta = {
        "schema": "ringviz/h3_job_log@1", "job_id": job_id, "status": "completed", "prompt": prompt,
        "duration_s": duration_s, "seed": None, "model": model, "width": width, "height": height,
        "steps": steps, "turbo": False, "v13": False, "loras": [], "error": None, "error_type": None,
        "queued_at": queued_at or now, "dispatched_at": queued_at or now, "finished_at": now,
        "has_video": True, "has_ref": False, "has_metrics": False, "notes": notes,
    }
    path = day_dir / f"{job_id}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)
    return {"job_id": job_id, "date": date_str}
