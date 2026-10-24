"""Unit tests for H3 user style/character LoRA support:
h3_loraNG (qkv layout detection + per-LoRA runner file), h3_lora_libraryNG
(listing / resolving), the h3_engineNG argv, h3_jobsNG plumbing and the
/api/ng/h3/loras + /generate `loras` field. Nothing renders and no real
model files are read -- LoRAs here are tiny synthetic safetensors.

Run with: python3 -m unittest tests.test_h3_user_lorasNG -v
"""

import io
import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from flask import Flask

import h3_engineNG as eng
import h3_jobsNG as jobs
import h3_lora_libraryNG as lib
import h3_loraNG as lora
import routes.h3NG as h3_routes

ROWS = 3 * lora.HEADS * lora.HEAD_DIM
PERMUTE_RUNNER = "--lora-qkv-permute\n--video-shift\nRepeat the flag to stack adapters\n"


def _write_st(path: Path, tensors: dict, metadata=None) -> None:
    """tensors: name -> float32 ndarray."""
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


def _qkv_lora(path: Path, order: str, metadata=None, modules=3, zero=False) -> None:
    """A LoRA whose qkv lora_B row norms differ by q/k/v group, with the group
    laid out contiguously ((3, heads, head_dim)) or per head."""
    idx = np.arange(ROWS)
    group = idx // (lora.HEADS * lora.HEAD_DIM) if order == "contiguous" else (idx // lora.HEAD_DIM) % 3
    tensors = {}
    for m in range(modules):
        b = np.zeros((ROWS, 2), dtype=np.float32)
        if not zero:
            b[:, 0] = (group + 1) * 1.0 + (idx % 7) * 0.01
        a = np.zeros((2, 4), dtype=np.float32)
        a[0, 0] = 1.0
        tensors[f"blocks.{m}.attn.qkv_proj.lora_A.weight"] = a
        tensors[f"blocks.{m}.attn.qkv_proj.lora_B.weight"] = b
    _write_st(path, tensors, metadata)


def _plain_lora(path: Path, keys=("blocks.0.attn.out_proj.lora_A.weight", "blocks.0.attn.out_proj.lora_B.weight")):
    _write_st(path, {k: np.zeros((2, 2), dtype=np.float32) for k in keys})


class _Tmp(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.runner = self.tmp / "generate_staged.py"
        self.runner.write_text(PERMUTE_RUNNER)


class QkvLayoutTests(_Tmp):
    def test_measures_contiguous_and_per_head(self):
        for order, expected in (("contiguous", "contiguous"), ("per-head", "per-head")):
            p = self.tmp / f"{order}.safetensors"
            _qkv_lora(p, order if order == "contiguous" else "per-head")
            info = lora.qkv_layout_ng(p)
            self.assertEqual(info["layout"], expected, info)
            self.assertEqual(info["modules"], 3)

    def test_declared_metadata_skips_the_vote(self):
        p = self.tmp / "declared.safetensors"
        _qkv_lora(p, "contiguous", metadata={"qkv_layout": "interleaved"})
        info = lora.qkv_layout_ng(p)
        self.assertEqual((info["layout"], info["modules"]), ("per-head", 0))
        self.assertIn("qkv_layout=interleaved", info["declared"])

    def test_untrained_and_missing_qkv_is_none(self):
        zero = self.tmp / "zero.safetensors"
        _qkv_lora(zero, "contiguous", zero=True)
        self.assertEqual(lora.qkv_layout_ng(zero)["layout"], "none")
        plain = self.tmp / "plain.safetensors"
        _plain_lora(plain)
        self.assertEqual(lora.qkv_layout_ng(plain)["layout"], "none")

    def test_result_is_cached_per_file_version(self):
        p = self.tmp / "c.safetensors"
        _qkv_lora(p, "contiguous")
        first = lora.qkv_layout_ng(p)
        self.assertIs(lora.qkv_layout_ng(p), first)
        _qkv_lora(p, "per-head")  # rewritten -> new mtime/size key
        os.utime(p, ns=(1, 1))
        self.assertEqual(lora.qkv_layout_ng(p)["layout"], "per-head")


class LoraForRunnerTests(_Tmp):
    def test_old_runner_gets_the_source_file(self):
        self.runner.write_text("--lora only")
        p = self.tmp / "a.safetensors"
        _qkv_lora(p, "contiguous")
        self.assertEqual(lora.lora_for_runner_ng(p, self.runner), (p, "runner"))

    def test_contiguous_is_baked_and_logged(self):
        p = self.tmp / "a.safetensors"
        _qkv_lora(p, "contiguous")
        log = []
        out, how = lora.lora_for_runner_ng(p, self.runner, log.append)
        self.assertEqual(how, "re-ordered")
        self.assertEqual(out, p.parent / "runner-layout" / "a.safetensors")
        self.assertTrue(out.is_file())
        self.assertTrue(any("re-ordered" in line for line in log))
        # second call finds the copy current and doesn't rewrite it
        mtime = out.stat().st_mtime_ns
        self.assertEqual(lora.lora_for_runner_ng(p, self.runner)[0], out)
        self.assertEqual(out.stat().st_mtime_ns, mtime)

    def test_per_head_and_mlp_only_are_left_alone(self):
        ph = self.tmp / "ph.safetensors"
        _qkv_lora(ph, "per-head")
        self.assertEqual(lora.lora_for_runner_ng(ph, self.runner), (ph, "as-trained"))
        mlp = self.tmp / "mlp.safetensors"
        _plain_lora(mlp)
        self.assertEqual(lora.lora_for_runner_ng(mlp, self.runner), (mlp, "none"))
        self.assertFalse((self.tmp / "runner-layout").exists())

    def test_unreadable_values_fall_back_to_baking(self):
        p = self.tmp / "odd.safetensors"
        _qkv_lora(p, "contiguous")
        with mock.patch.object(lora, "qkv_layout_ng", side_effect=ValueError("unsupported dtype I8")):
            out, how = lora.lora_for_runner_ng(p, self.runner)
        self.assertEqual(how, "re-ordered")
        self.assertTrue(out.is_file())


class BakeLockTests(_Tmp):
    def test_concurrent_bakes_of_one_file_both_succeed(self):
        p = self.tmp / "a.safetensors"
        _qkv_lora(p, "contiguous")
        results, errors = [], []

        def run():
            try:
                results.append(lora.bake_runner_layout_ng(p))
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=run) for _ in range(4)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertEqual(errors, [])
        self.assertEqual(len(set(results)), 1)
        self.assertEqual([f.name for f in (self.tmp / "runner-layout").iterdir()], ["a.safetensors"])

    def test_copy_of_a_removed_source_is_pruned(self):
        a, b = self.tmp / "a.safetensors", self.tmp / "b.safetensors"
        _qkv_lora(a, "contiguous")
        _qkv_lora(b, "contiguous")
        lora.bake_runner_layout_ng(a)
        a.unlink()
        lora.bake_runner_layout_ng(b)
        self.assertEqual([f.name for f in (self.tmp / "runner-layout").iterdir()], ["b.safetensors"])


class LibraryTests(_Tmp):
    def setUp(self):
        super().setUp()
        p = mock.patch.object(lib, "H3_LORAS_DIR", self.tmp / "loras")
        p.start()
        self.addCleanup(p.stop)
        (self.tmp / "loras").mkdir()
        self.dir = self.tmp / "loras"

    def test_missing_folder_is_empty(self):
        with mock.patch.object(lib, "H3_LORAS_DIR", self.tmp / "nope"):
            self.assertEqual(lib.list_h3_loras_ng(), [])

    def test_lists_runnable_and_refused_files(self):
        _plain_lora(self.dir / "good.safetensors")
        _plain_lora(self.dir / "kohya.safetensors", keys=("x.lora_down.weight", "x.lora_up.weight", "x.lora_A.weight"))
        _plain_lora(self.dir / "peft.safetensors", keys=("blocks.0.attn.to_q.lora_A.weight",))
        _plain_lora(self.dir / "empty.safetensors", keys=("blocks.0.weight",))
        (self.dir / "junk.safetensors").write_bytes(b"not a safetensors file")
        (self.dir / "notes.txt").write_text("ignored")
        entries = {e["name"]: e for e in lib.list_h3_loras_ng()}
        self.assertEqual(sorted(entries), ["empty", "good", "junk", "kohya", "peft"])
        self.assertEqual({n: e["ok"] for n, e in entries.items()},
                         {"good": True, "kohya": False, "peft": False, "empty": False, "junk": False})
        self.assertIn("kohya", entries["kohya"]["reason"])
        self.assertIn("diffusers", entries["peft"]["reason"])
        self.assertIsNone(entries["good"]["reason"])

    def test_sidecar_supplies_triggers_and_strength(self):
        _plain_lora(self.dir / "a.safetensors")
        (self.dir / "a.json").write_text(json.dumps(
            {"name": "A Title", "trigger_words": ["zed", 5, "qux"], "recommended_strength": 0.7}))
        _plain_lora(self.dir / "b.safetensors")
        (self.dir / "b.json").write_text("{broken")
        _plain_lora(self.dir / "c.safetensors")
        (self.dir / "c.json").write_text(json.dumps({"recommended_strength": True, "trigger_words": "x"}))
        e = {x["name"]: x for x in lib.list_h3_loras_ng()}
        self.assertEqual((e["a"]["title"], e["a"]["trigger_words"], e["a"]["recommended_strength"]),
                         ("A Title", ["zed", "qux"], 0.7))
        self.assertEqual((e["b"]["trigger_words"], e["b"]["recommended_strength"]), ([], 1.0))
        self.assertEqual((e["c"]["trigger_words"], e["c"]["recommended_strength"]), ([], 1.0))

    def test_resolve_only_returns_runnable_library_paths(self):
        _plain_lora(self.dir / "good.safetensors")
        _plain_lora(self.dir / "peft.safetensors", keys=("blocks.0.attn.to_q.lora_A.weight",))
        self.assertEqual(lib.resolve_h3_lora_path_ng("good"), str(self.dir / "good.safetensors"))
        self.assertIsNone(lib.resolve_h3_lora_path_ng("peft"))
        self.assertIsNone(lib.resolve_h3_lora_path_ng("missing"))
        self.assertIsNone(lib.resolve_h3_lora_path_ng("../loras/good"))
        self.assertIsNone(lib.resolve_h3_lora_path_ng(""))
        self.assertIsNone(lib.resolve_h3_lora_path_ng(None))


class _Captured(Exception):
    pass


class EngineArgvTests(_Tmp):
    def setUp(self):
        super().setUp()
        self.v13 = self.tmp / "v13.safetensors"
        self.v13.write_bytes(b"x")
        self.lora_a = self.tmp / "a.safetensors"
        self.lora_b = self.tmp / "b.safetensors"
        _qkv_lora(self.lora_a, "contiguous")
        _plain_lora(self.lora_b)
        for target, val in (
                (eng, ("H3_V13_LORA_PATH", self.v13)),
                (eng, ("_resolve_h3_python_ng", lambda c: "/py")),
                (eng, ("_resolve_h3_runner_ng", lambda c: str(self.runner))),
                (eng, ("_resolve_h3_dit_ng", lambda c: "/dit")),
                (eng, ("_resolve_h3_compact_root_ng", lambda c: "/compact")),
                (eng, ("_resolve_h3_text_config_ng", lambda c: "/text")),
        ):
            p = mock.patch.object(target, *val)
            p.start()
            self.addCleanup(p.stop)

    def _cmd(self, **kw):
        args = dict(prompt="p", image_path=None, duration_s=5.0, output_dir=self.tmp / "o",
                    seed=1, config=eng.H3Config(model="h3"))
        args.update(kw)
        with mock.patch.object(eng.subprocess, "Popen",
                               side_effect=lambda cmd, **k: (_ for _ in ()).throw(_Captured(cmd))), \
             mock.patch("model_offloadNG.offload_resident_models_ng"):
            with self.assertRaises(_Captured) as cm:
                eng.generate_h3_video_ng(**args)
        return cm.exception.args[0]

    @staticmethod
    def _lora_args(cmd):
        return [cmd[i + 1] for i, a in enumerate(cmd) if a == "--lora"]

    def test_no_loras_adds_no_lora_flag(self):
        self.assertEqual(self._lora_args(self._cmd()), [])

    def test_user_loras_follow_the_speed_adapter_in_order(self):
        cmd = self._cmd(v13=True, loras=[(str(self.lora_a), 0.8), (str(self.lora_b), 1.0)])
        args = self._lora_args(cmd)
        self.assertEqual(args[0], f"{self.v13}:1.0")
        # contiguous qkv file -> its per-head copy; MLP-only file as it is
        self.assertEqual(args[1], f"{self.tmp / 'runner-layout' / 'a.safetensors'}:0.8")
        self.assertEqual(args[2], f"{self.lora_b}:1")

    def test_old_runner_passes_source_files_untouched(self):
        self.runner.write_text("--lora\nRepeat the flag to stack adapters")
        cmd = self._cmd(loras=[(str(self.lora_a), 1.0)])
        self.assertEqual(self._lora_args(cmd), [f"{self.lora_a}:1"])
        self.assertFalse((self.tmp / "runner-layout").exists())

    def test_single_slot_runner_limits(self):
        self.runner.write_text("--lora-qkv-permute\n--lora only one slot")
        with self.assertRaisesRegex(ValueError, "at most 1"):
            self._cmd(loras=[(str(self.lora_a), 1.0), (str(self.lora_b), 1.0)])
        with self.assertRaisesRegex(ValueError, "at most 0"):
            self._cmd(v13=True, loras=[(str(self.lora_a), 1.0)])
        self.assertEqual(len(self._lora_args(self._cmd(loras=[(str(self.lora_a), 1.0)]))), 1)

    def test_stack_limit_on_stacking_runner(self):
        many = [(str(self.lora_b), 1.0)] * (eng.H3_LORA_STACK_LIMIT + 1)
        with self.assertRaisesRegex(ValueError, f"at most {eng.H3_LORA_STACK_LIMIT}"):
            self._cmd(loras=many)

    def test_validation(self):
        with self.assertRaisesRegex(ValueError, "between"):
            self._cmd(loras=[(str(self.lora_a), 3.0)])
        with self.assertRaisesRegex(ValueError, "numeric"):
            self._cmd(loras=[(str(self.lora_a), "x")])
        with self.assertRaisesRegex(ValueError, "path and"):
            self._cmd(loras=[str(self.lora_a)])
        with self.assertRaises(FileNotFoundError):
            self._cmd(loras=[(str(self.tmp / "gone.safetensors"), 1.0)])

    def test_launch_log_names_the_loras(self):
        log = []
        self._cmd(loras=[(str(self.lora_b), 0.5)], on_log=log.append)
        self.assertTrue(any("[b@0.5]" in line for line in log))


class JobTests(_Tmp):
    def setUp(self):
        super().setUp()
        for target, val in ((jobs, ("H3GEN_DIR", str(self.tmp))),
                            (jobs, ("_JOBS", {})),
                            (jobs, ("_ensure_worker_started", mock.Mock())),
                            (jobs, ("_QUEUE", mock.Mock())),
                            (eng, ("H3_RUNNER", self.runner))):
            p = mock.patch.object(target, *val)
            p.start()
            self.addCleanup(p.stop)

    def _start(self, loras, **kw):
        return jobs.start_h3_job_ng(None, ".png", "a prompt", 5.0, 1, model="h3", loras=loras, **kw)

    def test_job_carries_loras_into_status(self):
        job = self._start([{"name": "a", "path": "/x/a.safetensors", "strength": 0.8}])
        self.assertEqual(job.loras, [{"name": "a", "path": "/x/a.safetensors", "strength": 0.8}])
        self.assertEqual(jobs.job_status_ng(job.job_id)["loras"], [{"name": "a", "strength": 0.8}])

    def test_no_loras_default(self):
        self.assertEqual(self._start(None).loras, [])

    def test_validation_fails_synchronously(self):
        with self.assertRaisesRegex(ValueError, "between"):
            self._start([{"name": "a", "path": "/x", "strength": 9}])
        too_many = [{"name": str(i), "path": "/x", "strength": 1} for i in range(eng.H3_LORA_STACK_LIMIT + 1)]
        with self.assertRaisesRegex(ValueError, "at most"):
            self._start(too_many)
        self.assertEqual(jobs._JOBS, {})

    def test_worker_passes_paths_and_strengths_to_the_engine(self):
        job = self._start([{"name": "a", "path": "/x/a.safetensors", "strength": 0.8}])
        jobs._QUEUE.get = mock.Mock(side_effect=[job.job_id, KeyboardInterrupt])
        with mock.patch.object(jobs.h3, "generate_h3_video_ng", return_value={"mp4_path": "/o.mp4"}) as gen, \
             mock.patch.object(jobs.h3_job_logsNG, "record_job_ng"):
            with self.assertRaises(KeyboardInterrupt):
                jobs._worker_loop()
        self.assertEqual(gen.call_args.kwargs["loras"], [("/x/a.safetensors", 0.8)])


class RouteTests(_Tmp):
    def setUp(self):
        super().setUp()
        self.loras_dir = self.tmp / "loras"
        self.loras_dir.mkdir()
        _plain_lora(self.loras_dir / "good.safetensors")
        _plain_lora(self.loras_dir / "other.safetensors")
        _plain_lora(self.loras_dir / "peft.safetensors", keys=("blocks.0.attn.to_q.lora_A.weight",))
        for target, val in ((lib, ("H3_LORAS_DIR", self.loras_dir)),
                            (eng, ("H3_RUNNER", self.runner)),
                            (eng, ("_resolve_h3_runner_ng", lambda c: str(self.runner)))):
            p = mock.patch.object(target, *val)
            p.start()
            self.addCleanup(p.stop)
        app = Flask(__name__)
        app.register_blueprint(h3_routes.h3NG_bp)
        app.testing = True
        self.client = app.test_client()

    def test_listing_hides_server_paths_and_reports_stack_limits(self):
        body = self.client.get("/api/ng/h3/loras").get_json()
        self.assertEqual([l["name"] for l in body["loras"]], ["good", "other", "peft"])
        self.assertTrue(all("path" not in l for l in body["loras"]))
        self.assertEqual((body["max_stack"], body["max_stack_with_speed"]), (4, 4))

    def test_single_slot_runner_listing(self):
        self.runner.write_text("--lora-qkv-permute")
        body = self.client.get("/api/ng/h3/loras").get_json()
        self.assertEqual((body["max_stack"], body["max_stack_with_speed"]), (1, 0))

    def _generate(self, loras=None, **extra):
        data = {"prompt": "a shot", "model": "h3", **extra}
        if loras is not None:
            data["loras"] = loras if isinstance(loras, str) else json.dumps(loras)
        with mock.patch.object(h3_routes.h3_jobs, "start_h3_job_ng") as start:
            start.return_value = mock.Mock(job_id="j1")
            r = self.client.post("/api/ng/h3/generate", data=data, content_type="multipart/form-data")
        return r, start

    def test_names_are_resolved_to_library_paths(self):
        r, start = self._generate([{"name": "good", "strength": 0.6}, {"name": "other"}])
        self.assertEqual(r.status_code, 202)
        self.assertEqual(start.call_args.kwargs["loras"], [
            {"name": "good", "path": str(self.loras_dir / "good.safetensors"), "strength": 0.6},
            {"name": "other", "path": str(self.loras_dir / "other.safetensors"), "strength": 1.0},
        ])

    def test_no_loras_field_means_empty_list(self):
        r, start = self._generate()
        self.assertEqual(start.call_args.kwargs["loras"], [])

    def test_bad_requests_are_400_and_never_start_a_job(self):
        bad = [
            "not json",
            json.dumps({"name": "good"}),
            json.dumps(["good"]),
            [{"name": "missing"}],
            [{"name": "peft"}],
            [{"name": "../loras/good"}],
            [{"name": "good"}, {"name": "good"}],
            [{"name": "good", "strength": "strong"}],
            [{"name": "good", "strength": True}],
        ]
        for payload in bad:
            r, start = self._generate(payload)
            self.assertEqual(r.status_code, 400, payload)
            start.assert_not_called()

    def test_job_start_error_maps_to_400(self):
        with mock.patch.object(h3_routes.h3_jobs, "start_h3_job_ng", side_effect=ValueError("at most 1")):
            r = self.client.post("/api/ng/h3/generate",
                                 data={"prompt": "p", "loras": json.dumps([{"name": "good"}])},
                                 content_type="multipart/form-data")
        self.assertEqual((r.status_code, r.get_json()["error"]), (400, "at most 1"))


if __name__ == "__main__":
    unittest.main()
