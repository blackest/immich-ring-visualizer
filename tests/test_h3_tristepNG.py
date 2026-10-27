"""Unit tests for H3 "Fast" (the TaoMate 3-step adapter, `tristep`) and the automatic
`--lazy-adaln` flag: the h3_engineNG command line and its refusals, the adapter-attached
safety stop, h3_ref2vaNG (Fast is allowed with a RefMod, the LightX2V adapters still are not),
the h3_jobsNG enqueue and status, health, and the /generate `tristep` field. Nothing renders and
no real model files are read.

Run with: python3 -m unittest tests.test_h3_tristepNG -v
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask

import h3_engineNG as eng
import h3_jobsNG as jobs
import h3_ref2vaNG as ref2va
import routes.h3NG as h3_routes

RUNNER = ("--refmod PATH[:STRENGTH]\n--lora-qkv-permute\nRepeat the flag to stack adapters\n"
          "--sigma-subset N:I0,I1,...\n")
LAZY_RUNNER = RUNNER + "--lazy-adaln\n"
APPLIED_OK = "LoRA taomate.safetensors scale 1 mode runtime: 208 applied, 0 skipped, 0 qkv row-blocks re-ordered, 0.00s"
APPLIED_ZERO = "LoRA taomate.safetensors scale 1 mode runtime: 0 applied, 208 skipped, 0 qkv row-blocks re-ordered, 0.00s"


class _Captured(Exception):
    pass


class _Base(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.runner = self.tmp / "generate_staged.py"
        self.runner.write_text(RUNNER)
        self.adapter = self.tmp / "taomate.safetensors"
        self.adapter.write_bytes(b"x" * 64)
        self.refmod = self.tmp / "ref.safetensors"
        self.refmod.write_bytes(b"x")
        for target, name, val in (
                (eng, "_resolve_h3_python_ng", lambda c: "/py"),
                (eng, "_resolve_h3_runner_ng", lambda c: str(self.runner)),
                (eng, "_resolve_h3_dit_ng", lambda c: "/dit"),
                (eng, "_resolve_h3_compact_root_ng", lambda c: "/compact"),
                (eng, "_resolve_h3_text_config_ng", lambda c: "/text")):
            p = mock.patch.object(target, name, val)
            p.start()
            self.addCleanup(p.stop)
        p = mock.patch.object(eng, "H3_TRISTEP_CANDIDATES", ((self.adapter, 10),))
        p.start()
        self.addCleanup(p.stop)
        p = mock.patch.dict(os.environ, {}, clear=False)
        p.start()
        self.addCleanup(p.stop)
        os.environ.pop("RINGVIZ_H3_LAZY_ADALN", None)

    def _item(self, **kw):
        return {"path": str(self.refmod), "strength": 0.8, "images": 8, "videos": 0, **kw}

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

    @staticmethod
    def _value(cmd, flag):
        return cmd[cmd.index(flag) + 1]


class EngineCommandTests(_Base):
    def test_fast_adds_the_ladder_the_steps_and_the_untouched_adapter(self):
        cmd = self._cmd(tristep=True, refmods=[self._item()])
        self.assertEqual(self._value(cmd, "--steps"), "4")
        self.assertEqual(self._value(cmd, "--sigma-subset"), "50:0,16,33,49")
        self.assertEqual(self._value(cmd, "--lora"), f"{self.adapter}:1.0")  # as-is: no qkv bake
        self.assertEqual(self._value(cmd, "--refmod"), f"{self.refmod}:0.8")
        self.assertEqual(cmd.count("--lora"), 1)

    def test_fast_overrides_a_passed_step_count(self):
        self.assertEqual(self._value(self._cmd(tristep=True, steps=16), "--steps"), "4")

    def test_without_fast_there_is_no_ladder(self):
        cmd = self._cmd()
        self.assertNotIn("--sigma-subset", cmd)
        self.assertEqual(self._value(cmd, "--steps"), str(eng.H3_STEPS))

    def test_fast_works_on_every_model_unlike_turbo(self):
        for model in ("h3", "h3q8", "h3ref"):
            self.assertIn("--sigma-subset", self._cmd(model=model, tristep=True))

    def test_fast_takes_user_loras_after_it(self):
        user = self.tmp / "user.safetensors"
        user.write_bytes(b"x")
        with mock.patch.object(eng.h3_lora, "lora_for_runner_ng", return_value=(user, False)):
            cmd = self._cmd(tristep=True, loras=[(str(user), 0.7)])
        loras = [cmd[i + 1] for i, a in enumerate(cmd) if a == "--lora"]
        self.assertEqual(loras, [f"{self.adapter}:1.0", f"{user}:0.7"])

    def test_refusals(self):
        with self.assertRaisesRegex(ValueError, "its own speed adapter"):
            self._cmd(tristep=True, turbo=True)
        with self.assertRaisesRegex(ValueError, "its own speed adapter"):
            self._cmd(tristep=True, v13=True)
        self.runner.write_text("--refmod PATH\nRepeat the flag to stack adapters\n")
        with self.assertRaisesRegex(ValueError, "--sigma-subset"):
            self._cmd(tristep=True)

    def test_missing_adapter_file_is_reported(self):
        self.adapter.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Fast"):
            self._cmd(tristep=True)

    def test_a_truncated_adapter_is_not_used(self):
        with mock.patch.object(eng, "H3_TRISTEP_CANDIDATES", ((self.adapter, 10_000),)):
            self.assertIsNone(eng._resolve_h3_tristep_ng())
            with self.assertRaises(FileNotFoundError):
                self._cmd(tristep=True)

    def test_fast_with_a_refmod_is_allowed_but_the_lightx2v_adapters_still_are_not(self):
        self.assertIn("--refmod", self._cmd(tristep=True, refmods=[self._item()]))
        # turbo needs bf16 and RefMods need Ref2VA, so the pair can never be built: refused either way
        for model in ("h3", "h3ref"):
            with self.assertRaises(ValueError):
                self._cmd(model=model, turbo=True, refmods=[self._item()])
        with self.assertRaises(ValueError):
            self._cmd(model="h3q8", tristep=True, refmods=[self._item()])  # still needs h3ref


class LazyAdalnTests(_Base):
    def test_added_when_the_runner_has_it(self):
        self.runner.write_text(LAZY_RUNNER)
        self.assertIn("--lazy-adaln", self._cmd(tristep=True))
        self.assertIn("--lazy-adaln", self._cmd())  # not tied to Fast

    def test_absent_when_the_runner_lacks_it(self):
        self.assertNotIn("--lazy-adaln", self._cmd(tristep=True))

    def test_opt_out_by_environment(self):
        self.runner.write_text(LAZY_RUNNER)
        for off in ("0", "false", "OFF", "no"):
            with mock.patch.dict(os.environ, {"RINGVIZ_H3_LAZY_ADALN": off}):
                self.assertNotIn("--lazy-adaln", self._cmd())

    def test_never_with_the_legacy_adaln_turbo(self):
        self.runner.write_text(LAZY_RUNNER)
        self.assertEqual(eng._h3_lazy_adaln_argv_ng(self.runner, legacy_adaln=True), [])
        self.assertEqual(eng._h3_lazy_adaln_argv_ng(self.runner, legacy_adaln=False), ["--lazy-adaln"])


CACHE_RUNNER = LAZY_RUNNER + "--adaln-cache-dir\n"


class AdalnCacheTests(_Base):
    def setUp(self):
        super().setUp()
        self.runner.write_text(CACHE_RUNNER)
        os.environ.pop("RINGVIZ_H3_ADALN_CACHE", None)

    def test_added_with_the_default_folder_when_the_runner_has_it(self):
        folder = self._value(self._cmd(tristep=True), "--adaln-cache-dir")
        self.assertTrue(folder.endswith(os.path.join(".cache", "ringviz", "h3-adaln")), folder)

    def test_the_folder_can_be_moved(self):
        with mock.patch.dict(os.environ, {"RINGVIZ_H3_ADALN_CACHE": str(self.tmp / "tables")}):
            self.assertEqual(self._value(self._cmd(), "--adaln-cache-dir"), str(self.tmp / "tables"))

    def test_absent_when_the_runner_lacks_it(self):
        self.runner.write_text(LAZY_RUNNER)
        self.assertNotIn("--adaln-cache-dir", self._cmd(tristep=True))

    def test_opt_out_by_environment(self):
        for off in ("0", "false", "OFF", "no"):
            with mock.patch.dict(os.environ, {"RINGVIZ_H3_ADALN_CACHE": off}):
                self.assertNotIn("--adaln-cache-dir", self._cmd())

    def test_never_with_the_legacy_adaln_turbo(self):
        self.assertEqual(eng._h3_adaln_cache_argv_ng(self.runner, legacy_adaln=True), [])
        self.assertEqual(eng._h3_adaln_cache_argv_ng(self.runner, legacy_adaln=False)[0], "--adaln-cache-dir")

    def test_it_sits_beside_the_lazy_flag_and_the_log_says_so(self):
        cmd = self._cmd(tristep=True, refmods=[self._item()])
        self.assertIn("--lazy-adaln", cmd)
        self.assertIn("--adaln-cache-dir", cmd)

    def test_health_reports_it(self):
        self.assertTrue(eng.h3_health_ng("h3ref")["adaln_cache"])
        self.runner.write_text(LAZY_RUNNER)
        self.assertFalse(eng.h3_health_ng("h3ref")["adaln_cache"])


class _FakeProc:
    """Stands in for the runner subprocess: prints the given lines, then 'exits'."""
    def __init__(self, lines, rc):
        self.stdout = iter([line + "\n" for line in lines])
        self.pid = 1
        self._rc = rc
        self.killed = False

    def wait(self):
        return self._rc

    def kill(self):
        self.killed = True


class AdapterAttachedTests(_Base):
    def _run(self, lines, rc=0):
        proc = _FakeProc(lines, rc)
        out = self.tmp / "o"
        with mock.patch.object(eng.subprocess, "Popen", return_value=proc), \
             mock.patch.object(eng.os, "getpgid", side_effect=ProcessLookupError), \
             mock.patch("model_offloadNG.offload_resident_models_ng"):
            return proc, eng.generate_h3_video_ng(
                prompt="p", image_path=None, duration_s=5.0, output_dir=out, seed=1,
                config=eng.H3Config(model="h3ref"), tristep=True)

    def test_regex_reads_the_runners_own_summary_line(self):
        self.assertEqual(eng._H3_LORA_APPLIED_RE.search(APPLIED_OK).group(1), "208")
        self.assertEqual(eng._H3_LORA_APPLIED_RE.search(APPLIED_ZERO).group(1), "0")
        self.assertIsNone(eng._H3_LORA_APPLIED_RE.search("LoRA accounting: 200 wrapped, 0 unaccounted"))

    def test_zero_modules_stops_the_render_and_says_so(self):
        with self.assertRaisesRegex(RuntimeError, "attached to 0 modules"):
            self._run(["== text_encode_q8 ==", APPLIED_ZERO, "== joint_denoise =="], rc=-9)

    def test_a_healthy_adapter_does_not_trip_it(self):
        # rc 1 = the runner failing for an unrelated reason: the error must be THAT, not ours.
        with self.assertRaisesRegex(RuntimeError, "rc=1"):
            self._run([APPLIED_OK], rc=1)


class HealthTests(_Base):
    def test_fields(self):
        h = eng.h3_health_ng("h3ref")
        self.assertTrue(h["tristep_ok"] and h["tristep_supported"])
        self.assertFalse(h["lazy_adaln"])
        self.runner.write_text(LAZY_RUNNER)
        self.assertTrue(eng.h3_health_ng("h3ref")["lazy_adaln"])
        self.runner.write_text("--refmod\n")
        self.assertFalse(eng.h3_health_ng("h3ref")["tristep_supported"])
        self.adapter.unlink()
        self.assertFalse(eng.h3_health_ng("h3ref")["tristep_ok"])


class Ref2vaValidationTests(_Base):
    def _validate(self, **kw):
        args = dict(model="h3ref", image_path=None, turbo=False, v13=False, chain_windows=1,
                    runner=self.runner)
        args.update(kw)
        return ref2va.validate_refmods_ng([self._item()], **args)

    def test_fast_is_allowed_and_the_others_are_not(self):
        self.assertEqual(len(self._validate(tristep=True)), 1)
        for kw in ({"turbo": True}, {"v13": True}):
            with self.assertRaisesRegex(ValueError, "Turbo"):
                self._validate(**kw)

    def test_default_is_unchanged_for_old_callers(self):
        self.assertEqual(len(self._validate()), 1)  # no tristep argument at all


class JobTests(_Base):
    def setUp(self):
        super().setUp()
        self.queue = mock.MagicMock()
        for target, name, val in (
                (jobs, "H3GEN_DIR", str(self.tmp / "gen")),
                (jobs, "_QUEUE", self.queue),
                (jobs, "_ensure_worker_started", lambda: None),
                (jobs.h3, "_resolve_h3_runner_ng", lambda c: str(self.runner))):
            p = mock.patch.object(target, name, val)
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(lambda: jobs._JOBS.clear())

    def _start(self, **kw):
        args = dict(image_bytes=None, image_ext=".jpg", prompt="p", duration_s=5.0, model="h3ref",
                    tristep=True, refmods=[{"name": "a", **self._item()}])
        args.update(kw)
        return jobs.start_h3_job_ng(**args)

    def test_fast_job_forces_the_steps_keeps_the_refmod_and_reports_itself(self):
        job = self._start(steps=16)
        self.assertEqual((job.steps, job.tristep), (4, True))
        self.assertEqual(len(job.refmods), 1)
        status = jobs.job_status_ng(job.job_id)
        self.assertTrue(status["tristep"])
        self.assertEqual(status["steps"], 4)
        self.queue.put.assert_called_once_with(job.job_id)

    def test_a_job_without_fast_says_so(self):
        job = self._start(tristep=False, refmods=[])
        self.assertFalse(jobs.job_status_ng(job.job_id)["tristep"])

    def test_bad_combinations_fail_the_post_synchronously(self):
        for kw in ({"turbo": True, "model": "h3"}, {"v13": True, "model": "h3"}):
            with self.assertRaises(ValueError):
                self._start(**kw)
        with mock.patch.object(jobs.h3, "_resolve_h3_tristep_ng", return_value=None):
            with self.assertRaises(FileNotFoundError):
                self._start()
        self.queue.put.assert_not_called()


class RouteTests(_Base):
    def setUp(self):
        super().setUp()
        app = Flask(__name__)
        app.register_blueprint(h3_routes.h3NG_bp)
        self.client = app.test_client()

    def _post(self, **data):
        with mock.patch.object(h3_routes.h3_jobs, "start_h3_job_ng") as start:
            start.return_value = mock.Mock(job_id="j1")
            r = self.client.post("/api/ng/h3/generate", data={"prompt": "p", **data})
        return r, start

    def test_the_field_is_forwarded(self):
        r, start = self._post(tristep="true")
        self.assertEqual(r.status_code, 202)
        self.assertTrue(start.call_args.kwargs["tristep"])

    def test_absent_means_off(self):
        _r, start = self._post()
        self.assertFalse(start.call_args.kwargs["tristep"])

    def test_a_refusal_comes_back_as_400(self):
        with mock.patch.object(h3_routes.h3_jobs, "start_h3_job_ng", side_effect=ValueError("nope")):
            r = self.client.post("/api/ng/h3/generate", data={"prompt": "p", "tristep": "1", "turbo": "1"})
        self.assertEqual(r.status_code, 400)


if __name__ == "__main__":
    unittest.main()
