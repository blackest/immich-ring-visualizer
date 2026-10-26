"""Unit tests for H3 RefMod (Ref2VA reference) support: h3_refmod_libraryNG
(listing / resolving), h3_ref2vaNG (model path + refusals + argv), the
h3_engineNG command line and health, h3_jobsNG enqueue validation, and the
/api/ng/h3/refmods + /generate `refmods` field. Nothing renders and no real
model files are read -- RefMods here are tiny synthetic safetensors.

Run with: python3 -m unittest tests.test_h3_refmodNG -v
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from flask import Flask

import h3_engineNG as eng
import h3_jobsNG as jobs
import h3_ref2vaNG as ref2va
import h3_refmod_libraryNG as lib
import routes.h3NG as h3_routes

REFMOD_RUNNER = "--refmod PATH[:STRENGTH]\n--lora-qkv-permute\nRepeat the flag to stack adapters\n"


def _write_st(path: Path, tensors: dict, metadata=None) -> None:
    header, blobs, pos = {}, [], 0
    for name, arr in tensors.items():
        blob = np.ascontiguousarray(arr, dtype="<f4").tobytes()
        header[name] = {"dtype": "F32", "shape": list(arr.shape), "data_offsets": [pos, pos + len(blob)]}
        blobs.append(blob)
        pos += len(blob)
    if metadata:
        header["__metadata__"] = metadata
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    path.write_bytes(len(raw).to_bytes(8, "little") + raw + b"".join(blobs))


def _refmod(path: Path, kind="video", t=4, h=16, w=16, extra_tensors=None, meta=None) -> Path:
    member = {"name": path.stem, "kind": kind, "latent_t": t, "latent_h": h, "latent_w": w}
    tensors = {"latent": np.zeros((1,), dtype=np.float32)}
    tensors.update(extra_tensors or {})
    _write_st(path, tensors, {"refmod_meta": json.dumps(meta or member)})
    return path


class _Tmp(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.dir = self.tmp / "refmods"
        self.dir.mkdir()
        p = mock.patch.dict(os.environ, {"LTX_H3_REFMODS": str(self.dir)})
        p.start()
        self.addCleanup(p.stop)
        self.runner = self.tmp / "generate_staged.py"
        self.runner.write_text(REFMOD_RUNNER)


class LibraryTests(_Tmp):
    def test_missing_folder_is_empty(self):
        with mock.patch.dict(os.environ, {"LTX_H3_REFMODS": str(self.tmp / "nope")}):
            self.assertEqual(lib.list_h3_refmods_ng(), [])

    def test_rows_are_frames_times_patches(self):
        _refmod(self.dir / "v.safetensors", "video", t=20, h=32, w=32)
        _refmod(self.dir / "i.safetensors", "image", t=1, h=64, w=48)
        e = {x["name"]: x for x in lib.list_h3_refmods_ng()}
        self.assertEqual(e["v"]["rows"], 20 * 16 * 16)      # the 5,120-row library file
        self.assertEqual(e["i"]["rows"], 1 * 32 * 24)
        self.assertEqual((e["v"]["videos"], e["v"]["images"]), (1, 0))
        self.assertEqual((e["i"]["videos"], e["i"]["images"]), (0, 1))

    def test_refused_files_say_why(self):
        _refmod(self.dir / "audio.safetensors", "audio", t=200, h=0, w=0)
        _refmod(self.dir / "odd.safetensors", "video", t=3, h=15, w=16)
        (self.dir / "lora.safetensors").write_bytes(b"x")
        _write_st(self.dir / "plain.safetensors", {"a": np.zeros(1, dtype=np.float32)})
        e = {x["name"]: x for x in lib.list_h3_refmods_ng()}
        self.assertFalse(any(x["ok"] for x in e.values()))
        self.assertIn("audio-only", e["audio"]["reason"])
        self.assertIn("even", e["odd"]["reason"])
        self.assertIn("not a readable", e["lora"]["reason"])
        self.assertIn("no RefMod metadata", e["plain"]["reason"])

    def test_reflora_bundle_counts_visual_members_only_and_flags_the_lora(self):
        meta = {"_format_version": 5, "kind": "bundle", "name": "x", "members": [
            {"kind": "video", "latent_t": 4, "latent_h": 16, "latent_w": 16},
            {"kind": "audio", "latent_t": 100},
            {"kind": "image", "latent_t": 1, "latent_h": 8, "latent_w": 8}]}
        _refmod(self.dir / "hybrid.safetensors", meta=meta, extra_tensors={
            "lora_unet_blocks_0_attn_qkv_proj.lora_down.weight": np.zeros((2, 2), dtype=np.float32)})
        (e,) = lib.list_h3_refmods_ng()
        self.assertTrue(e["ok"] and e["has_lora"])
        self.assertEqual((e["videos"], e["images"], e["rows"]), (1, 1, 4 * 64 + 16))

    def test_runnable_first_and_resolve_never_takes_a_path(self):
        _refmod(self.dir / "zz.safetensors")
        _refmod(self.dir / "bad.safetensors", "audio", t=10, h=0, w=0)
        self.assertEqual([x["name"] for x in lib.list_h3_refmods_ng()], ["zz", "bad"])
        self.assertEqual(lib.resolve_h3_refmod_ng("zz")["path"], str(self.dir / "zz.safetensors"))
        for bad in ("bad", "missing", "../refmods/zz", str(self.dir / "zz.safetensors"), "", None):
            self.assertIsNone(lib.resolve_h3_refmod_ng(bad))


class Ref2vaTests(_Tmp):
    def setUp(self):
        super().setUp()
        self.f = _refmod(self.dir / "a.safetensors")

    def _validate(self, refmods, **kw):
        args = dict(model="h3ref", image_path=None, turbo=False, v13=False, chain_windows=1, runner=self.runner)
        args.update(kw)
        return ref2va.validate_refmods_ng(refmods, **args)

    def _item(self, **kw):
        return {"path": str(self.f), "strength": 1.0, "images": 0, "videos": 1, **kw}

    def test_empty_is_always_fine(self):
        self.assertEqual(self._validate([], model="h3q8"), [])
        self.assertEqual(self._validate(None, model="h3", turbo=True), [])

    def test_refusals(self):
        for kw, pat in (({"model": "h3q8"}, "Ref2VA model"), ({"model": "h3"}, "Ref2VA model"),
                        ({"image_path": "x.png"}, "first-frame"), ({"turbo": True}, "Turbo"),
                        ({"v13": True}, "Turbo"), ({"chain_windows": 2}, "single 5 s window")):
            with self.assertRaisesRegex(ValueError, pat):
                self._validate([self._item()], **kw)
        self.runner.write_text("--lora\n")
        with self.assertRaisesRegex(ValueError, "no --refmod support"):
            self._validate([self._item()])

    def test_strength_and_files(self):
        for bad in (-0.1, 1.5, True, "1", None):
            with self.assertRaisesRegex(ValueError, "between 0 and 1"):
                self._validate([self._item(strength=bad)])
        with self.assertRaises(FileNotFoundError):
            self._validate([self._item(path=str(self.dir / "gone.safetensors"))])
        with self.assertRaisesRegex(ValueError, "strength 0"):
            self._validate([self._item(strength=0)])
        kept = self._validate([self._item(strength=0), self._item(strength=0.5)])
        self.assertEqual([k["strength"] for k in kept], [0.5])

    def test_reference_limits_are_the_models(self):
        with self.assertRaisesRegex(ValueError, "at most 3 video"):
            self._validate([self._item()] * 4)
        self._validate([self._item()] * 3)
        with self.assertRaisesRegex(ValueError, "at most 9 image"):
            self._validate([self._item(images=10, videos=0)])
        # the largest request the model allows: 9 images + 3 videos
        self.assertEqual(len(self._validate([self._item(images=9, videos=0)] + [self._item()] * 3)), 4)

    def test_argv(self):
        self.assertEqual(ref2va.refmod_argv_ng([]), [])
        self.assertEqual(ref2va.refmod_argv_ng([{"path": "/p/a.safetensors", "strength": 1.0},
                                                {"path": "/p/b.safetensors", "strength": 0.75}]),
                         ["--refmod", "/p/a.safetensors:1", "--refmod", "/p/b.safetensors:0.75"])

    def test_dit_resolution(self):
        q8 = self.tmp / "q8"
        q8.mkdir()
        self.assertIsNone(ref2va.resolve_ref2va_dit_ng(str(q8), self.tmp))
        for n in ("config.json", "quant_config.json", "model-00001.safetensors"):
            (q8 / n).write_text("{}")
        self.assertEqual(ref2va.resolve_ref2va_dit_ng(str(q8), self.tmp), str(q8))
        with mock.patch.dict(os.environ, {ref2va.REF2VA_DIT_ENV: str(q8)}):
            self.assertEqual(ref2va.resolve_ref2va_dit_ng("", self.tmp / "elsewhere"), str(q8))


class _Captured(Exception):
    pass


class EngineArgvTests(_Tmp):
    def setUp(self):
        super().setUp()
        self.f = _refmod(self.dir / "a.safetensors")
        for target, val in (
                (eng, ("_resolve_h3_python_ng", lambda c: "/py")),
                (eng, ("_resolve_h3_runner_ng", lambda c: str(self.runner))),
                (eng, ("_resolve_h3_dit_ng", lambda c: "/dit")),
                (eng, ("_resolve_h3_compact_root_ng", lambda c: "/compact")),
                (eng, ("_resolve_h3_text_config_ng", lambda c: "/text")),
        ):
            p = mock.patch.object(target, *val)
            p.start()
            self.addCleanup(p.stop)

    def _cmd(self, model="h3ref", **kw):
        args = dict(prompt="p", image_path=None, duration_s=5.0, output_dir=self.tmp / "o",
                    seed=1, config=eng.H3Config(model=model))
        args.update(kw)
        with mock.patch.object(eng.subprocess, "Popen",
                               side_effect=lambda cmd, **k: (_ for _ in ()).throw(_Captured(cmd))), \
             mock.patch("model_offloadNG.offload_resident_models_ng"):
            with self.assertRaises(_Captured) as cm:
                eng.generate_h3_video_ng(**args)
        return cm.exception.args[0]

    def _item(self, **kw):
        return {"path": str(self.f), "strength": 0.8, "images": 0, "videos": 1, **kw}

    def test_no_refmods_adds_no_flag_on_any_model(self):
        for model in ("h3", "h3q8", "h3ref"):
            self.assertNotIn("--refmod", self._cmd(model=model))

    def test_refmods_become_repeated_flags(self):
        g = _refmod(self.dir / "b.safetensors")
        cmd = self._cmd(refmods=[self._item(), self._item(path=str(g), strength=1.0)])
        flags = [cmd[i + 1] for i, a in enumerate(cmd) if a == "--refmod"]
        self.assertEqual(flags, [f"{self.f}:0.8", f"{g}:1"])

    def test_refusals_happen_before_launch(self):
        for kw in ({"model": "h3q8"}, {"turbo": True}, {"duration_s": 9.0}, {"image_path": str(self.f)}):
            with self.assertRaises((ValueError, FileNotFoundError)):
                self._cmd(refmods=[self._item()], **kw)

    def test_h3ref_model_resolves_its_own_dit(self):
        with mock.patch.object(eng, "_resolve_h3_dit_ng", wraps=None) as m:
            m.side_effect = lambda c: "/dit-" + c.model
            self.assertEqual(self._cmd(model="h3ref")[self._cmd(model="h3ref").index("--dit") + 1], "/dit-h3ref")


class HealthTests(_Tmp):
    def test_ready_needs_both_halves(self):
        with mock.patch.object(h3_routes.h3_engineNG, "_resolve_h3_runner_ng", return_value=str(self.runner)), \
             mock.patch.object(ref2va, "resolve_ref2va_dit_ng", return_value="/q8"):
            h = eng.h3_health_ng("h3ref")
            self.assertTrue(h["refmod_ready"] and h["ref2va_ok"] and h["refmod_supported"])
            self.runner.write_text("--lora\n")
            h = eng.h3_health_ng("h3ref")
            self.assertTrue(h["ref2va_ok"] and not h["refmod_supported"] and not h["refmod_ready"])
        with mock.patch.object(h3_routes.h3_engineNG, "_resolve_h3_runner_ng", return_value=str(self.runner)), \
             mock.patch.object(ref2va, "resolve_ref2va_dit_ng", return_value=None):
            self.runner.write_text(REFMOD_RUNNER)
            h = eng.h3_health_ng("h3ref")
            self.assertTrue(h["refmod_supported"] and not h["ref2va_ok"] and not h["refmod_ready"])


class RouteTests(_Tmp):
    def setUp(self):
        super().setUp()
        _refmod(self.dir / "good.safetensors", "video", t=4, h=16, w=16)
        _refmod(self.dir / "audio.safetensors", "audio", t=9, h=0, w=0)
        app = Flask(__name__)
        app.register_blueprint(h3_routes.h3NG_bp)
        self.client = app.test_client()

    def test_listing_hides_paths_and_reports_readiness(self):
        r = self.client.get("/api/ng/h3/refmods").get_json()
        self.assertEqual([e["name"] for e in r["refmods"]], ["good", "audio"])
        self.assertTrue(all("path" not in e for e in r["refmods"]))
        self.assertEqual((r["max_images"], r["max_videos"]), (9, 3))
        self.assertIn("ready", r)

    def test_form_field_parsing(self):
        def parse(raw):
            with Flask(__name__).test_request_context(method="POST", data={"refmods": raw}):
                return h3_routes._refmods_from_form()
        self.assertEqual(parse(""), [])
        (one,) = parse(json.dumps([{"name": "good", "strength": 0.5}]))
        self.assertEqual((one["strength"], one["videos"], one["images"]), (0.5, 1, 0))
        self.assertEqual(parse(json.dumps([{"name": "good"}]))[0]["strength"], 1.0)
        for raw, pat in (("{", "JSON array"), ("[1]", "JSON array"),
                         (json.dumps([{"name": "nope"}]), "unknown or unusable"),
                         (json.dumps([{"name": "audio"}]), "unknown or unusable"),
                         (json.dumps([{"name": "good"}, {"name": "good"}]), "twice"),
                         (json.dumps([{"name": "good", "strength": 2}]), "between 0 and 1"),
                         (json.dumps([{"name": "good", "strength": True}]), "between 0 and 1")):
            with self.assertRaisesRegex(ValueError, pat):
                parse(raw)

    def test_generate_rejects_a_bad_refmods_field_with_400(self):
        r = self.client.post("/api/ng/h3/generate", data={"prompt": "p", "refmods": "{"})
        self.assertEqual(r.status_code, 400)
        self.assertIn("JSON array", r.get_json()["error"])

    def test_status_accepts_the_new_model(self):
        self.assertEqual(self.client.get("/api/ng/h3/status?model=h3ref").status_code, 200)
        self.assertEqual(self.client.get("/api/ng/h3/status?model=bogus").status_code, 400)


class JobEnqueueTests(_Tmp):
    def test_bad_combinations_fail_the_post_synchronously(self):
        f = _refmod(self.dir / "a.safetensors")
        item = {"name": "a", "path": str(f), "strength": 1.0, "images": 0, "videos": 1}
        with mock.patch.object(jobs.h3, "_resolve_h3_runner_ng", return_value=str(self.runner)), \
             mock.patch.object(jobs, "_ensure_worker_started") as worker:
            for kw in ({"model": "h3q8"}, {"duration_s": 12.0}, {"turbo": True, "model": "h3"}):
                args = dict(image_bytes=None, image_ext=".jpg", prompt="p", duration_s=5.0, model="h3ref",
                            refmods=[item])
                args.update(kw)
                with self.assertRaises(ValueError):
                    jobs.start_h3_job_ng(**args)
            worker.assert_not_called()  # nothing was queued for any of them


if __name__ == "__main__":
    unittest.main()
