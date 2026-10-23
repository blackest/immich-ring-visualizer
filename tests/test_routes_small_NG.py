"""Flask-test-client tests for the small NG blueprints: charactersNG,
mainNG and folderNG. Storage goes to a temp EXPORT_DIR; the folder
analysis thread target is mocked so no InsightFace runs.

Run with: python3 -m unittest tests.test_routes_small_NG -v
"""

import io
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask

import character_sheetNG as cs
import routes.charactersNG as characters_routes
import routes.folderNG as folder_routes
import routes.mainNG as main_routes
import stateNG


def _app(*bps):
    app = Flask(__name__, template_folder=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "templates"))
    for bp in bps:
        app.register_blueprint(bp)
    app.testing = True
    return app


class CharactersRouteTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        p = mock.patch.object(cs, "EXPORT_DIR", self.root)
        p.start()
        self.addCleanup(p.stop)
        self.c = _app(characters_routes.charactersNG_bp).test_client()

    def test_list_empty_then_after_save(self):
        self.assertEqual(self.c.get("/api/ng/characters").get_json(), {"characters": []})
        self.c.post("/api/ng/characters/mary/doc", json={"identity": {"name": "Mary"}})
        got = self.c.get("/api/ng/characters").get_json()["characters"]
        self.assertEqual([(x["id"], x["name"]) for x in got], [("mary", "Mary")])

    def test_doc_roundtrip_is_stored_as_is(self):
        doc = {"identity": {"name": "Mary"}, "anything": [1, 2, {"x": "é"}]}
        r = self.c.post("/api/ng/characters/mary/doc", json=doc)
        self.assertEqual((r.status_code, r.get_json()), (200, {"ok": True}))
        self.assertEqual(self.c.get("/api/ng/characters/mary/doc").get_json(), doc)

    def test_missing_doc_is_404(self):
        r = self.c.get("/api/ng/characters/ghost/doc")
        self.assertEqual(r.status_code, 404)
        self.assertIn("ghost", r.get_json()["error"])

    def test_bad_ids_are_400_on_every_verb(self):
        bad = "bad.name"
        for resp in (self.c.get(f"/api/ng/characters/{bad}/doc"),
                     self.c.post(f"/api/ng/characters/{bad}/doc", json={}),
                     self.c.delete(f"/api/ng/characters/{bad}")):
            self.assertEqual(resp.status_code, 400)

    def test_save_rejects_non_object_bodies(self):
        for kw in ({"json": [1, 2]}, {"json": "str"}, {"data": "not json", "content_type": "text/plain"}, {}):
            with self.subTest(kw=kw):
                self.assertEqual(self.c.post("/api/ng/characters/mary/doc", **kw).status_code, 400)
        self.assertFalse((self.root / "mary").exists())

    def test_delete_reports_whether_anything_was_there(self):
        self.c.post("/api/ng/characters/mary/doc", json={"a": 1})
        self.assertEqual(self.c.delete("/api/ng/characters/mary").get_json(), {"ok": True})
        self.assertEqual(self.c.delete("/api/ng/characters/mary").get_json(), {"ok": False})
        self.assertEqual(self.c.get("/api/ng/characters").get_json(), {"characters": []})


class MainRouteTests(unittest.TestCase):
    def test_index_renders_the_ng_template(self):
        with mock.patch.object(main_routes, "render_template", return_value="PAGE") as rt:
            r = _app(main_routes.mainNG_bp).test_client().get("/immichRingNG")
        self.assertEqual((r.status_code, r.get_data(as_text=True)), (200, "PAGE"))
        rt.assert_called_once_with("indexNG.html")

    def test_template_renders_when_the_original_main_page_is_also_registered(self):
        # indexNG.html links back to the original page via url_for('main.index'),
        # so the NG page only renders with the original main blueprint present.
        import routes.main as original_main
        r = _app(main_routes.mainNG_bp, original_main.main_bp).test_client().get("/immichRingNG")
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"<html", r.data.lower())

    def test_root_is_not_served_by_this_blueprint(self):
        self.assertEqual(_app(main_routes.mainNG_bp).test_client().get("/").status_code, 404)


def _zip(members):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    buf.seek(0)
    return buf


class FolderRouteTests(unittest.TestCase):
    def setUp(self):
        self.c = _app(folder_routes.folderNG_bp).test_client()
        self.started = []
        self.addCleanup(self._cleanup)

        class _Thread:
            def __init__(inner, target, args, daemon):
                self.started.append((target, args, daemon))

            def start(inner):
                pass
        p = mock.patch.object(folder_routes.threading, "Thread", _Thread)
        p.start()
        self.addCleanup(p.stop)

    def _cleanup(self):
        for jid in list(stateNG._analysis_jobs_ng):
            if any(a[0] == jid for _, a, _ in self.started):
                stateNG._analysis_jobs_ng.pop(jid, None)

    def _post(self, data):
        return self.c.post("/api/ng/analyze-folder", data=data, content_type="multipart/form-data")

    def test_multiple_images_sorted_case_insensitively_and_job_registered(self):
        r = self._post({"images": [(io.BytesIO(b"b"), "B.png"), (io.BytesIO(b"a"), "a.JPG"),
                                   (io.BytesIO(b"t"), "notes.txt")],
                        "sourceName": "holiday"})
        body = r.get_json()
        self.assertEqual(body["imageCount"], 2)
        target, args, daemon = self.started[-1]
        self.assertIs(target, folder_routes.run_folder_analysis_ng)
        self.assertTrue(daemon)
        self.assertEqual(args[0], body["jobId"])
        self.assertEqual([n for n, _ in args[1]], ["a.JPG", "B.png"])
        job = stateNG._analysis_jobs_ng[body["jobId"]]
        self.assertEqual((job["sourceName"], job["sourceType"], job["status"]), ("holiday", "folder", "running"))
        self.assertEqual(job["srcImages"], {"a.JPG": b"a", "B.png": b"b"})

    def test_defaults_and_threshold_parsing(self):
        r = self._post({"images": (io.BytesIO(b"a"), "a.png")})
        _, args, _ = self.started[-1]
        self.assertEqual(args[2:], (0.1, 1.0, 1, "jpg"))
        self.assertEqual(stateNG._analysis_jobs_ng[r.get_json()["jobId"]]["sourceName"], "folder_set")
        r = self._post({"images": (io.BytesIO(b"a"), "a.png"), "simThreshold": "0.4",
                        "blurThreshold": "50", "refIndex": "3", "cacheFormat": "png"})
        _, args, _ = self.started[-1]
        self.assertEqual(args[2:], (0.4, 50.0, 3, "png"))

    def test_unknown_cache_format_falls_back_to_jpg(self):
        self._post({"images": (io.BytesIO(b"a"), "a.png"), "cacheFormat": "bmp"})
        self.assertEqual(self.started[-1][1][5], "jpg")

    def test_zip_upload_skips_non_images_macosx_and_directories_and_flattens_paths(self):
        z = _zip({"pics/one.png": b"1", "__MACOSX/pics/._one.png": b"x", "readme.txt": b"t",
                  "pics/": b"", "deep/er/two.JPEG": b"2"})
        r = self._post({"zip": (z, "faces.zip")})
        body = r.get_json()
        self.assertEqual(body["imageCount"], 2)
        self.assertEqual([n for n, _ in self.started[-1][1][1]], ["one.png", "two.JPEG"])
        self.assertEqual(stateNG._analysis_jobs_ng[body["jobId"]]["sourceName"], "faces")

    def test_zip_wins_over_images_field(self):
        r = self._post({"zip": (_zip({"z.png": b"z"}), "s.zip"), "images": (io.BytesIO(b"a"), "a.png")})
        self.assertEqual(r.get_json()["imageCount"], 1)

    def test_error_responses(self):
        self.assertEqual(self._post({"zip": (io.BytesIO(b"not a zip"), "s.zip")}).status_code, 400)
        self.assertIn("provide", self._post({"sourceName": "x"}).get_json()["error"])
        r = self._post({"images": (io.BytesIO(b"t"), "notes.txt")})
        self.assertEqual((r.status_code, "no valid images" in r.get_json()["error"]), (400, True))
        r = self._post({"zip": (_zip({"readme.txt": b"t"}), "s.zip")})
        self.assertEqual(r.status_code, 400)
        self.assertEqual(self.started, [])


if __name__ == "__main__":
    unittest.main()
