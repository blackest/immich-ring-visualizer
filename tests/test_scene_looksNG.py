"""Unit tests for character looks and the identity kit: the prompts, the Klein hand-off, the kit's run with Klein and the
RefMod maker replaced by fakes (cancel and a failing shot included), and the routes. No model runs.

Run with: python3 -m unittest tests.test_scene_looksNG -v
"""

import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask
from PIL import Image

import routes.sceneNG as scene_routes
import scene_looksNG as looks


def png(size=(256, 256), color=(120, 80, 60)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


class PromptTests(unittest.TestCase):
    def test_a_look_prompt_names_the_place_keeps_face_and_hair_and_leaves_clothes_to_the_setting(self):
        p = looks.look_prompt_ng("on stone castle battlements under a blue sky", who="older man with a bald head")
        self.assertIn("The same older man with a bald head", p)
        self.assertIn("now on stone castle battlements", p)
        self.assertIn("same face and same hair", p)
        self.assertNotIn("same clothes", p)                       # the setting may suggest other clothes
        self.assertNotIn("identical outfit", p)
        self.assertIn("Keep the same clothes.", looks.look_prompt_ng("in a hall", keep_outfit=True))
        self.assertIn("The same person", looks.look_prompt_ng("in a hall"))

    def test_look_prompt_refusals(self):
        for bad in ("", "   ", "x" * (looks.MAX_SETTING + 1)):
            with self.subTest(bad=bad[:10]), self.assertRaises(looks.LooksRequestError):
                looks.look_prompt_ng(bad)

    def test_the_kit_is_eight_shots_four_faces_four_bodies_each_ending_with_the_identity_line(self):
        kit = looks.kit_prompts_ng()
        self.assertEqual(len(kit), 8)
        self.assertEqual(sum(l.startswith("face") for l, _ in kit), 4)
        self.assertEqual(sum(l.startswith("body") for l, _ in kit), 4)
        self.assertTrue(all(p.endswith(looks.IDENTITY) for _, p in kit))
        self.assertEqual(len({p for _, p in kit}), 8)
        self.assertIn("identical face, hairstyle, skin tone, and outfit", looks.IDENTITY)
        self.assertTrue(any("from behind" in p for _, p in kit))


class _Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.t = Path(self.tmp.name)
        self.klein_submits = []
        self.n = {"i": 0}

        def new_dir():
            self.n["i"] += 1
            folder = self.t / f"k{self.n['i']}" / "refs"
            folder.mkdir(parents=True)
            return f"k{self.n['i']}", folder

        def start(job_id, prompt, n, width, height, seed, model, refs, options=None, outpaint_plan=None):
            self.klein_submits.append(dict(id=job_id, prompt=prompt, n=n, size=(width, height), seed=seed, refs=refs,
                                           ref_bytes=Path(refs[0]["path"]).read_bytes()))
            return {"id": job_id}

        for target, name, val in (
                (looks.klein_jobs, "new_job_dir_ng", new_dir),
                (looks.klein_jobs, "start_ng", start),
                (looks.klein_engine, "klein_health_ng", lambda: {"ready": True, "models": ["distilled-q8"]}),
                (looks, "KLEIN_DIR", str(self.t)),
                (looks, "_sleep", lambda: None),
                (looks.refmod_make, "refmods_dir_ng", lambda: self.t / "refmods")):
            p = mock.patch.object(target, name, val)
            p.start()
            self.addCleanup(p.stop)
        looks._KITS.clear()
        self.addCleanup(looks._KITS.clear)


class LookTests(_Base):
    def test_a_look_is_one_klein_job_with_the_picture_as_the_subject(self):
        pic = png()
        job_id = looks.start_look_ng(pic, ".PNG", "in a great hall", who="woman", n=2, seed=7)
        (sub,) = self.klein_submits
        self.assertEqual((job_id, sub["n"], sub["size"], sub["seed"]), ("k1", 2, looks.LOOK_SIZE, 7))
        self.assertEqual(sub["refs"][0]["slot"], "subject")
        self.assertEqual(sub["ref_bytes"], pic)
        self.assertTrue(sub["refs"][0]["path"].endswith("subject.png"))
        self.assertIn("The same woman", sub["prompt"])

    def test_bad_requests_queue_nothing(self):
        for args, kw in ((((b"", ".png", "in a hall")), {}), ((b"notapicture", ".png", "in a hall"), {}),
                         ((png((64, 64)), ".png", "in a hall"), {}), ((png(), ".png", ""), {}),
                         ((png(), ".png", "in a hall"), {"n": 9})):
            with self.subTest(args=args[0][:6], kw=kw), self.assertRaises(looks.LooksRequestError):
                looks.start_look_ng(*args, **kw)
        self.assertEqual(self.klein_submits, [])


def fake_klein_done(t, fail_at=None):
    """get_ng for the kit's jobs: each is 'done' with a picture on disk, unless it is the one told to fail."""
    def get(job_id):
        if job_id == fail_at:
            return {"status": "error", "error": "out of memory", "results": []}
        out = t / job_id / "out"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{job_id}.png").write_bytes(png(color=(10, 20, int(job_id[1:]) * 20)))
        return {"status": "done", "results": [{"file": f"{job_id}.png"}], "error": None}
    return get


class KitTests(_Base):
    def make(self, name="", seed=3):
        return looks.KitJobNG(job_id="kit1", picture=png(), ext=".png", name=name, seed=seed,
                              shots=[{"label": l, "status": "waiting", "klein_job": None, "file": None}
                                     for l, _ in looks.KIT_SHOTS])

    def test_eight_klein_jobs_one_per_shot_each_with_the_picture_and_its_own_seed(self):
        job = self.make()
        with mock.patch.object(looks.klein_jobs, "get_ng", fake_klein_done(self.t)):
            looks._run_kit(job)
        self.assertEqual(job.stage, "done")
        self.assertEqual(len(self.klein_submits), 8)
        self.assertEqual([s["seed"] for s in self.klein_submits], list(range(3, 11)))
        self.assertTrue(all(s["n"] == 1 and s["size"] == (looks.KIT_SIZE, looks.KIT_SIZE) for s in self.klein_submits))
        self.assertTrue(all(s["ref_bytes"] == job.picture and s["prompt"].endswith(looks.IDENTITY) for s in self.klein_submits))

    def test_with_a_name_the_original_and_the_eight_become_a_refmod(self):
        job = self.make(name="john_kit")
        made = {}

        def fake_refmod(name, pictures, canvas="square", **kw):
            made.update(name=name, pictures=pictures, canvas=canvas)
            return SimpleNamespace(job_id="r1", finished_at=1.0, error=None)
        with mock.patch.object(looks.klein_jobs, "get_ng", fake_klein_done(self.t)), \
                mock.patch.object(looks.refmod_make, "start_refmod_job_ng", fake_refmod):
            looks._run_kit(job)
        self.assertEqual((job.stage, job.refmod_job_id), ("done", "r1"))
        self.assertEqual((made["name"], made["canvas"], len(made["pictures"])), ("john_kit", "square", 9))
        self.assertEqual(made["pictures"][0][0], job.picture)                # the original first, then the eight in order
        self.assertTrue(all(p[1] == ".png" for p in made["pictures"]))

    def test_a_failed_shot_and_a_failed_refmod_stop_the_run_with_the_reason(self):
        job = self.make()
        with mock.patch.object(looks.klein_jobs, "get_ng", fake_klein_done(self.t, fail_at="k3")):
            looks._run_kit(job)
        self.assertEqual(job.stage, "failed")
        self.assertIn("out of memory", job.error)
        self.assertIn(looks.KIT_SHOTS[2][0], job.error)
        self.assertEqual([s["status"] for s in job.shots][:3], ["done", "done", "failed"])
        bad = self.make(name="x_kit")
        with mock.patch.object(looks.klein_jobs, "get_ng", fake_klein_done(self.t)), \
                mock.patch.object(looks.refmod_make, "start_refmod_job_ng",
                                  return_value=SimpleNamespace(job_id="r", finished_at=1.0, error="no VAE")):
            looks._run_kit(bad)
        self.assertEqual((bad.stage, bad.error), ("failed", "building the RefMod: no VAE"))

    def test_cancel_stops_it_and_tells_klein_to_stop_the_jobs(self):
        job = self.make()
        job.cancelled = True
        running = lambda jid: {"status": "running", "results": [], "error": None}
        with mock.patch.object(looks.klein_jobs, "get_ng", running), \
                mock.patch.object(looks.klein_jobs, "cancel_ng", return_value=True) as cancel:
            looks._run_kit(job)
        self.assertEqual((job.stage, job.error), ("cancelled", "cancelled"))
        self.assertEqual(cancel.call_count, 8)                        # every picture job, not just the one being waited on

    def test_start_validates_the_name_and_registers_a_pollable_job(self):
        with mock.patch.object(looks, "_run_kit"), mock.patch.object(looks.threading, "Thread") as th:
            job = looks.start_kit_ng(png(), ".png", name="john_kit", seed=3)
        th.return_value.start.assert_called_once()
        st = looks.kit_status_ng(job.job_id)
        self.assertEqual((st["stage"], st["total"], st["done"], st["name"]), ("queued", 8, 0, "john_kit"))
        (self.t / "refmods").mkdir()
        (self.t / "refmods" / "taken.safetensors").write_bytes(b"x")
        for kw, why in (({"name": "bad name!"}, "letters, digits"), ({"name": "taken"}, "already exists")):
            with self.subTest(why=why), self.assertRaisesRegex(looks.LooksRequestError, why):
                looks.start_kit_ng(png(), ".png", **kw)
        with self.assertRaises(looks.LooksRequestError):
            looks.start_kit_ng(b"", ".png")
        with mock.patch.object(looks.klein_engine, "klein_health_ng", lambda: {"ready": False, "models": []}):
            with self.assertRaisesRegex(looks.LooksRequestError, "Klein is not ready"):
                looks.start_kit_ng(png(), ".png")
        self.assertTrue(looks.cancel_kit_ng(job.job_id))
        job.finished_at = 1.0
        self.assertFalse(looks.cancel_kit_ng(job.job_id))
        with self.assertRaises(LookupError):
            looks.kit_status_ng("nope")

    def test_status_gives_picture_urls_as_they_land(self):
        job = self.make()
        job.shots[0].update(status="done", klein_job="k1", file="a.png")
        looks._KITS["kit1"] = job
        st = looks.kit_status_ng("kit1")
        self.assertEqual(st["done"], 1)
        self.assertEqual(st["shots"][0]["url"], "/api/ng/klein/image/k1/a.png")
        self.assertIsNone(st["shots"][1]["url"])


class RouteTests(_Base):
    def setUp(self):
        super().setUp()
        app = Flask(__name__)
        app.register_blueprint(scene_routes.sceneNG_bp)
        self.client = app.test_client()

    def test_look_and_kit_routes(self):
        r = self.client.post("/api/ng/scene/looks", data={"picture": (io.BytesIO(png()), "w.PNG"), "setting": "in a hall",
                                                           "who": "woman", "n": "2", "seed": "5"}, content_type="multipart/form-data")
        self.assertEqual((r.status_code, r.get_json()["klein_job"]), (202, "k1"))
        self.assertEqual(self.klein_submits[0]["seed"], 5)
        with mock.patch.object(looks, "_run_kit"), mock.patch.object(looks.threading, "Thread"):
            k = self.client.post("/api/ng/scene/kit", data={"picture": (io.BytesIO(png()), "w.png"), "name": "kitx"},
                                 content_type="multipart/form-data")
        self.assertEqual(k.status_code, 202)
        jid = k.get_json()["job_id"]
        self.assertEqual(self.client.get(f"/api/ng/scene/kit/{jid}").get_json()["total"], 8)
        self.assertTrue(self.client.delete(f"/api/ng/scene/kit/{jid}").get_json()["ok"])

    def test_errors_are_400_or_503_not_500(self):
        for path, data in (("/api/ng/scene/looks", {"setting": "in a hall"}),                       # no picture
                           ("/api/ng/scene/looks", {"picture": (io.BytesIO(png()), "w.png"), "setting": ""}),
                           ("/api/ng/scene/looks", {"picture": (io.BytesIO(png()), "w.png"), "setting": "x", "n": "abc"}),
                           ("/api/ng/scene/kit", {})):
            r = self.client.post(path, data=data, content_type="multipart/form-data")
            self.assertEqual(r.status_code, 400, (path, r.get_json()))
        with mock.patch.object(looks, "start_look_ng", side_effect=RuntimeError("Klein is not ready")):
            r = self.client.post("/api/ng/scene/looks", data={"picture": (io.BytesIO(png()), "w.png"), "setting": "x"},
                                 content_type="multipart/form-data")
            self.assertEqual(r.status_code, 503)
        self.assertEqual(self.client.get("/api/ng/scene/kit/nope").status_code, 404)


if __name__ == "__main__":
    unittest.main()
