"""Flask-test-client tests for routes/generateNG.py.

The character-sheet, job-queue and HiDream layers are mocked at the module
attribute level (their own logic is covered by test_character_sheetNG /
test_sheet_jobsNG / test_hidream_engineNG); this file checks the HTTP layer:
validation, status-code mapping, settings plumbing and the trigger-suffix
logic for photos that collide with an existing character.

Run with: python3 -m unittest tests.test_routes_generateNG -v
"""

import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask

import routes.generateNG as gen
from stateNG import _analysis_jobs_ng

cs = gen.character_sheet


def _client():
    app = Flask(__name__)
    app.register_blueprint(gen.generateNG_bp)
    app.testing = True
    return app.test_client()


def _job(job_id="job1", character_id="petra", shot_keys=("a",)):
    return SimpleNamespace(job_id=job_id, character_id=character_id, shot_keys=list(shot_keys))


class StatusAndPresetsTests(unittest.TestCase):
    def test_status_merges_health(self):
        with mock.patch.object(gen.hidream_engineNG, "hidream_health_ng",
                               return_value={"ready": True, "detail": "x"}):
            body = _client().get("/api/ng/generate/status").get_json()
        self.assertEqual(body, {"reachable": True, "ready": True, "detail": "x"})

    def test_presets_shape(self):
        body = _client().get("/api/ng/generate/presets").get_json()
        self.assertIn("default", body["presets"])
        self.assertTrue(body["styles"])
        self.assertEqual(body["render"]["default_width"], cs.DEFAULT_RENDER_W)
        self.assertTrue(all(isinstance(wh, list) for wh in body["render"]["trained_resolutions"]))
        for scene in body["scenes"]:
            self.assertEqual(scene["label"], scene["key"].replace("_", " ").title())


class PreviewPromptTests(unittest.TestCase):
    def test_passes_defaults(self):
        with mock.patch.object(cs, "preview_shot_prompt_ng", return_value={"prompt": "p"}) as f:
            r = _client().post("/api/ng/generate/preview-prompt", json={})
        self.assertEqual(r.get_json(), {"prompt": "p"})
        kw = f.call_args.kwargs
        self.assertEqual((kw["preset"], kw["style"], kw["identity_lock"]), ("default", "none", True))

    def test_value_error_is_400(self):
        with mock.patch.object(cs, "preview_shot_prompt_ng", side_effect=ValueError("bad")):
            r = _client().post("/api/ng/generate/preview-prompt", json={"preset": "zzz"})
        self.assertEqual((r.status_code, r.get_json()["error"]), (400, "bad"))


class CreateDraftTests(unittest.TestCase):
    def _post(self, body, **patch):
        with mock.patch.object(cs, "create_draft_character_ng", **patch) as f:
            return _client().post("/api/ng/generate/characters", json=body), f

    def test_requires_trigger_and_path(self):
        r, f = self._post({"trigger": "x"})
        self.assertEqual(r.status_code, 400)
        f.assert_not_called()

    def test_success(self):
        r, f = self._post({"trigger": " x ", "source_image_path": "/p", "pronoun": "they"},
                          return_value={"id": "x"})
        self.assertEqual(r.get_json(), {"id": "x"})
        f.assert_called_once_with("x", "/p", pronoun="they", subject_noun="")

    def test_error_mapping(self):
        cases = [(cs.DraftCharacterExistsError("dup"), 409),
                 (FileNotFoundError("gone"), 404),
                 (ValueError("bad"), 400)]
        for exc, code in cases:
            r, _ = self._post({"trigger": "x", "source_image_path": "/p"}, side_effect=exc)
            self.assertEqual(r.status_code, code, exc)


class SettingsFromBodyTests(unittest.TestCase):
    def test_defaults(self):
        with mock.patch.object(cs, "build_custom_shot_ng", return_value=None):
            s = gen._generation_settings_from_body_ng({})
        self.assertEqual(s["preset"], "default")
        self.assertIsNone(s["shots"])
        self.assertEqual((s["seed"], s["anchor_chain"], s["identity_lock"], s["style"]),
                         (-1, True, True, "none"))
        self.assertEqual(s["width"], cs.DEFAULT_RENDER_W)

    def test_custom_shot_replaces_views(self):
        with mock.patch.object(cs, "build_custom_shot_ng", return_value="SHOT"):
            s = gen._generation_settings_from_body_ng({"views": ["a"], "custom_prompt": "x"})
        self.assertEqual(s["shots"], ["SHOT"])
        self.assertIsNone(s["views"])


class MapJobStartErrorTests(unittest.TestCase):
    def test_codes(self):
        app = Flask(__name__)
        with app.app_context():
            codes = [gen._map_job_start_error(e)[1]
                     for e in (LookupError("a"), FileNotFoundError("b"), ValueError("c"), RuntimeError("d"))]
        self.assertEqual(codes, [404, 404, 400, 500])


class SameFileBytesTests(unittest.TestCase):
    def test_comparisons(self):
        with tempfile.TemporaryDirectory() as d:
            a, b, c, e = (Path(d, n) for n in "abce")
            a.write_bytes(b"hello")
            b.write_bytes(b"hello")
            c.write_bytes(b"hellp")
            e.write_bytes(b"hi")
            self.assertTrue(gen._same_file_bytes_ng(a, b))
            self.assertFalse(gen._same_file_bytes_ng(a, c))
            self.assertFalse(gen._same_file_bytes_ng(a, e))
            self.assertFalse(gen._same_file_bytes_ng(a, Path(d, "missing")))
            self.assertTrue(gen._same_file_bytes_ng(a, a, chunk_size=2))


class RegisterDraftTests(unittest.TestCase):
    def _run(self, trigger="petra", settings=None, create=None, avatar=None, same=False, start=None):
        app = Flask(__name__)
        with app.app_context(), \
             mock.patch.object(cs, "create_draft_character_ng", **(create or {})) as c, \
             mock.patch.object(cs, "character_avatar_ng", return_value=avatar), \
             mock.patch.object(gen, "_same_file_bytes_ng", return_value=same), \
             mock.patch.object(gen.sheet_jobs, "start_job_ng", **(start or {"return_value": _job()})) as s:
            resp = gen._register_draft_and_start_job_ng(trigger, "/tmp/x.jpg", settings or {"preset": "default"})
        return resp, c, s

    def test_fresh_trigger(self):
        resp, c, s = self._run()
        body, code = resp[0].get_json(), resp[1]
        self.assertEqual(code, 202)
        self.assertEqual(body["poll_url"], "/api/ng/generate/sheet-jobs/job1")
        s.assert_called_once_with("petra", preset="default")

    def test_collision_with_different_photo_gets_suffix(self):
        exists = cs.DraftCharacterExistsError("dup")
        _, c, s = self._run(create={"side_effect": [exists, exists, None]}, avatar="/a.jpg", same=False)
        self.assertEqual([x.args[0] for x in c.call_args_list], ["petra", "petra-2", "petra-3"])
        self.assertEqual(s.call_args.args[0], "petra-3")

    def test_collision_with_same_photo_reuses_character(self):
        exists = cs.DraftCharacterExistsError("dup")
        _, c, s = self._run(create={"side_effect": [exists]}, avatar="/a.jpg", same=True)
        self.assertEqual(c.call_count, 1)
        self.assertEqual(s.call_args.args[0], "petra")

    def test_gives_up_after_max_attempts(self):
        exists = cs.DraftCharacterExistsError("dup")
        resp, c, s = self._run(create={"side_effect": exists}, avatar="/a.jpg", same=False)
        self.assertEqual(resp[1], 500)
        s.assert_not_called()
        self.assertEqual(c.call_count, gen._MAX_TRIGGER_SUFFIX_ATTEMPTS)

    def test_create_errors_map_to_http(self):
        for exc, code in ((FileNotFoundError("x"), 404), (ValueError("x"), 400)):
            resp, _, s = self._run(create={"side_effect": exc})
            self.assertEqual(resp[1], code)
            s.assert_not_called()

    def test_job_start_failure_maps(self):
        resp, _, _ = self._run(start={"side_effect": LookupError("nope")})
        self.assertEqual(resp[1], 404)


class SheetFromAssetTests(unittest.TestCase):
    def _post(self, body, get=None):
        get = get or mock.Mock(return_value=SimpleNamespace(
            status_code=200, content=b"img", headers={"Content-Type": "image/png; charset=x"}))
        seen = {}

        def fake_register(trigger, tmp_path, settings):
            seen.update(trigger=trigger, ext=os.path.splitext(tmp_path)[1],
                        content=Path(tmp_path).read_bytes(), tmp=tmp_path)
            return {"ok": True}, 202

        with mock.patch.object(gen, "_requests_get_immich_asset_ng", get), \
             mock.patch.object(gen, "_register_draft_and_start_job_ng", side_effect=fake_register):
            r = _client().post("/api/ng/generate/sheet-from-asset", json=body)
        return r, seen, get

    def test_requires_asset_id(self):
        r, _, get = self._post({})
        self.assertEqual(r.status_code, 400)
        get.assert_not_called()

    def test_downloads_and_cleans_up_temp_file(self):
        r, seen, _ = self._post({"asset_id": "A1", "name": "Petra"})
        self.assertEqual(r.status_code, 202)
        self.assertEqual((seen["trigger"], seen["ext"], seen["content"]), ("Petra", ".png", b"img"))
        self.assertFalse(os.path.exists(seen["tmp"]))

    def test_trigger_falls_back_to_asset_id(self):
        _, seen, _ = self._post({"asset_id": "A1"})
        self.assertEqual(seen["trigger"], "A1")

    def test_unreachable_immich_is_502(self):
        r, _, _ = self._post({"asset_id": "A1"}, get=mock.Mock(side_effect=OSError("down")))
        self.assertEqual(r.status_code, 502)

    def test_immich_http_error_is_502(self):
        r, _, _ = self._post({"asset_id": "A1"},
                             get=mock.Mock(return_value=SimpleNamespace(status_code=404, headers={}, content=b"")))
        self.assertEqual(r.status_code, 502)
        self.assertIn("404", r.get_json()["error"])


class SheetFromUploadTests(unittest.TestCase):
    def _post(self, data, filename="face.png"):
        data = dict(data)
        if filename:
            data["file"] = (io.BytesIO(b"bytes"), filename)
        seen = {}

        def fake_register(trigger, tmp_path, settings):
            seen.update(trigger=trigger, ext=os.path.splitext(tmp_path)[1],
                        settings=settings, content=Path(tmp_path).read_bytes(), tmp=tmp_path)
            return {"ok": True}, 202

        with mock.patch.object(cs, "build_custom_shot_ng", return_value=None), \
             mock.patch.object(gen, "_register_draft_and_start_job_ng", side_effect=fake_register):
            r = _client().post("/api/ng/generate/sheet-from-upload", data=data,
                               content_type="multipart/form-data")
        return r, seen

    def test_missing_file_and_trigger(self):
        r, _ = self._post({"trigger": "x"}, filename=None)
        self.assertEqual(r.status_code, 400)
        r, _ = self._post({})
        self.assertEqual(r.status_code, 400)

    def test_settings_parsing(self):
        r, seen = self._post({"trigger": "x", "views": json.dumps(["a", "b"]), "seed": "7",
                              "anchor_chain": "false", "width": "640", "hair_color": " red "})
        self.assertEqual(r.status_code, 202)
        s = seen["settings"]
        self.assertEqual((s["views"], s["seed"], s["anchor_chain"], s["identity_lock"], s["width"], s["hair_color"]),
                         (["a", "b"], 7, False, True, 640, "red"))
        self.assertEqual((seen["ext"], seen["content"]), (".png", b"bytes"))
        self.assertFalse(os.path.exists(seen["tmp"]))

    def test_bad_views_rejected(self):
        for views in ("not json", json.dumps({"a": 1}), json.dumps([1, 2])):
            r, _ = self._post({"trigger": "x", "views": views})
            self.assertEqual(r.status_code, 400, views)

    def test_unknown_extension_falls_back_to_jpg(self):
        _, seen = self._post({"trigger": "x"}, filename="face.bin")
        self.assertEqual(seen["ext"], ".jpg")


class JobRoutesTests(unittest.TestCase):
    def test_generate_for_character(self):
        with mock.patch.object(cs, "build_custom_shot_ng", return_value=None), \
             mock.patch.object(gen.sheet_jobs, "start_job_ng", return_value=_job()) as s:
            r = _client().post("/api/ng/generate/characters/petra/sheet/generate", json={"preset": "p"})
        self.assertEqual(r.status_code, 202)
        self.assertEqual(s.call_args.args[0], "petra")
        self.assertEqual(s.call_args.kwargs["preset"], "p")

    def test_generate_error_mapped(self):
        with mock.patch.object(cs, "build_custom_shot_ng", return_value=None), \
             mock.patch.object(gen.sheet_jobs, "start_job_ng", side_effect=LookupError("none")):
            r = _client().post("/api/ng/generate/characters/petra/sheet/generate", json={})
        self.assertEqual(r.status_code, 404)

    def test_reroll_requires_shot_key(self):
        with mock.patch.object(gen.sheet_jobs, "start_reroll_ng") as f:
            r = _client().post("/api/ng/generate/characters/petra/sheet/reroll", json={})
        self.assertEqual(r.status_code, 400)
        f.assert_not_called()

    def test_reroll_passes_overrides(self):
        with mock.patch.object(gen.sheet_jobs, "start_reroll_ng", return_value=_job()) as f:
            r = _client().post("/api/ng/generate/characters/petra/sheet/reroll",
                               json={"shot_key": "a", "seed": 5, "steps": 20})
        self.assertEqual(r.status_code, 202)
        f.assert_called_once_with("petra", "a", seed=5, prompt=None, width=None, height=None, steps=20)

    def test_job_status_and_404(self):
        with mock.patch.object(gen.sheet_jobs, "job_status_ng", return_value={"status": "done"}):
            self.assertEqual(_client().get("/api/ng/generate/sheet-jobs/j").get_json(), {"status": "done"})
        with mock.patch.object(gen.sheet_jobs, "job_status_ng", side_effect=LookupError("no")):
            self.assertEqual(_client().get("/api/ng/generate/sheet-jobs/j").status_code, 404)

    def test_cancel(self):
        with mock.patch.object(gen.sheet_jobs, "cancel_job_ng", return_value=False):
            self.assertEqual(_client().delete("/api/ng/generate/sheet-jobs/j").get_json(), {"ok": False})


class ServeFilesTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def test_sheet_png(self):
        png = self.dir / "sheet.png"
        png.write_bytes(b"\x89PNG")
        c = _client()
        with mock.patch.object(cs, "character_sheet_png_ng", return_value=png):
            r = c.get("/api/ng/generate/characters/petra/sheet")
            self.assertEqual((r.status_code, r.mimetype), (200, "image/png"))
            r.close()
        with mock.patch.object(cs, "character_sheet_png_ng", return_value=None):
            self.assertEqual(c.get("/api/ng/generate/characters/petra/sheet").status_code, 404)
        with mock.patch.object(cs, "character_sheet_png_ng", side_effect=ValueError("bad")):
            self.assertEqual(c.get("/api/ng/generate/characters/petra/sheet").status_code, 400)

    def test_avatar(self):
        img = self.dir / "avatar.jpg"
        img.write_bytes(b"jpg")
        c = _client()
        with mock.patch.object(cs, "character_avatar_ng", return_value=img):
            r = c.get("/api/ng/generate/characters/petra/avatar")
            self.assertEqual(r.data, b"jpg")
            r.close()
        with mock.patch.object(cs, "character_avatar_ng", return_value=None):
            self.assertEqual(c.get("/api/ng/generate/characters/petra/avatar").status_code, 404)
        self.assertEqual(c.get("/api/ng/generate/characters/..%2Fx/avatar").status_code, 404)

    def test_sheet_meta(self):
        with mock.patch.object(cs, "character_sheet_meta_ng", return_value={"shots": []}):
            self.assertEqual(_client().get("/api/ng/generate/characters/petra/sheet-meta").get_json(),
                             {"shots": []})

    def test_shot_thumbnail_serves_newest_candidate(self):
        view = self.dir / "sheet_views" / "front"
        view.mkdir(parents=True)
        old, new = view / "cand_1.png", view / "cand_2.png"
        old.write_bytes(b"old")
        new.write_bytes(b"new")
        os.utime(old, (1, 1))
        os.utime(new, (2, 2))
        c = _client()
        with mock.patch.object(cs, "_character_dir_ng", return_value=self.dir):
            r = c.get("/api/ng/generate/characters/petra/shots/front")
            self.assertEqual(r.data, b"new")
            r.close()
            self.assertEqual(c.get("/api/ng/generate/characters/petra/shots/missing").status_code, 404)
            for p in view.glob("cand_*.png"):
                p.unlink()
            self.assertEqual(c.get("/api/ng/generate/characters/petra/shots/front").status_code, 404)


class AddToRingTests(unittest.TestCase):
    def _post(self, query="", avatar="/a.jpg", shots=("/s1.png", "/s2.png")):
        created = {}
        thread_cls = mock.Mock()
        with mock.patch.object(cs, "character_avatar_ng", return_value=avatar), \
             mock.patch.object(cs, "sheet_shot_image_paths_ng", return_value=list(shots)), \
             mock.patch.object(gen.threading, "Thread", thread_cls):
            r = _client().post(f"/api/ng/generate/characters/petra/sheet/add-to-ring{query}")
        return r, thread_cls

    def tearDown(self):
        for k in [k for k, v in _analysis_jobs_ng.items() if v.get("sourceName") == "petra (character sheet)"]:
            _analysis_jobs_ng.pop(k)

    def test_no_avatar_or_no_shots(self):
        self.assertEqual(self._post(avatar=None)[0].status_code, 404)
        self.assertEqual(self._post(shots=())[0].status_code, 404)

    def test_starts_analysis_thread_with_avatar_first(self):
        r, thread = self._post("?simThreshold=0.3&cacheFormat=png")
        body = r.get_json()
        self.assertEqual((r.status_code, body["imageCount"]), (202, 3))
        kw = thread.call_args.kwargs
        self.assertEqual(kw["args"][1], ["/a.jpg", "/s1.png", "/s2.png"])
        self.assertEqual(kw["args"][2], 0.3)
        self.assertEqual(kw["args"][5], "png")
        self.assertEqual(kw["kwargs"], {"always_cache": True})
        thread.return_value.start.assert_called_once()
        job = _analysis_jobs_ng[body["jobId"]]
        self.assertEqual((job["status"], job["sourceType"]), ("running", "folder"))


if __name__ == "__main__":
    unittest.main()
