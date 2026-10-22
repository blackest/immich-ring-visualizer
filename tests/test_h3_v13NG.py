"""Unit tests for the lightx2v v1.3 ("faster Best") option in
h3_engineNG.generate_h3_video_ng and h3_jobsNG.start_h3_job_ng. The
subprocess is intercepted -- nothing renders.

Run with: python3 -m unittest tests.test_h3_v13NG -v
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h3_engineNG as eng
import h3_jobsNG as jobs


class _Captured(Exception):
    pass


class V13EngineTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.v13 = self.tmp / "v13.safetensors"
        self.v13.write_bytes(b"x")
        self.runner = self.tmp / "generate_staged.py"
        self.runner.write_text("--lora --video-shift")
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
                    seed=1, config=eng.H3Config(model="h3"), v13=True)
        args.update(kw)
        with mock.patch.object(eng.subprocess, "Popen", side_effect=lambda cmd, **k: (_ for _ in ()).throw(_Captured(cmd))), \
             mock.patch("model_offloadNG.offload_resident_models_ng"):
            with self.assertRaises(_Captured) as cm:
                eng.generate_h3_video_ng(**args)
        return cm.exception.args[0]

    def test_v13_forces_steps_and_adds_adapter_and_shift(self):
        cmd = self._cmd(steps=16)
        self.assertEqual(cmd[cmd.index("--steps") + 1], str(eng.H3_V13_STEPS))
        self.assertEqual(cmd[cmd.index("--lora") + 1], f"{self.v13}:1.0")
        self.assertEqual(cmd[cmd.index("--video-shift") + 1], "6")
        self.assertNotIn("--lora-adaln", cmd)

    def test_video_shift_omitted_when_runner_lacks_it(self):
        self.runner.write_text("--lora only")
        self.assertNotIn("--video-shift", self._cmd())

    def test_validation(self):
        with self.assertRaisesRegex(ValueError, "pick one"):
            self._cmd(turbo=True)
        with self.assertRaisesRegex(ValueError, "bf16"):
            self._cmd(config=eng.H3Config(model="h3q8"))
        with self.assertRaisesRegex(ValueError, "only validated"):
            self._cmd(width=768, height=448)
        self.v13.unlink()
        with self.assertRaises(FileNotFoundError):
            self._cmd()

    def test_health_reports_v13(self):
        self.assertTrue(eng.h3_health_ng("h3")["v13_ok"])
        self.v13.unlink()
        self.assertFalse(eng.h3_health_ng("h3")["v13_ok"])


class V13JobTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.v13 = Path(tmp.name) / "v13.safetensors"
        self.v13.write_bytes(b"x")
        for target, val in ((eng, ("H3_V13_LORA_PATH", self.v13)),
                            (jobs, ("H3GEN_DIR", tmp.name)),
                            (jobs, ("_JOBS", {})),
                            (jobs, ("_ensure_worker_started", mock.Mock())),
                            (jobs, ("_QUEUE", mock.Mock()))):
            p = mock.patch.object(target, *val)
            p.start()
            self.addCleanup(p.stop)

    def test_job_carries_v13_and_forced_steps(self):
        job = jobs.start_h3_job_ng(None, ".png", "a prompt", 5.0, 1, model="h3", v13=True)
        self.assertTrue(job.v13)
        self.assertEqual(job.steps, eng.H3_V13_STEPS)
        self.assertTrue(jobs.job_status_ng(job.job_id)["v13"])

    def test_job_validation_fails_synchronously(self):
        with self.assertRaisesRegex(ValueError, "pick one"):
            jobs.start_h3_job_ng(None, ".png", "p", 5.0, 1, model="h3", turbo=True, v13=True)
        with self.assertRaisesRegex(ValueError, "bf16"):
            jobs.start_h3_job_ng(None, ".png", "p", 5.0, 1, model="h3q8", v13=True)
        with self.assertRaisesRegex(ValueError, "only validated"):
            jobs.start_h3_job_ng(None, ".png", "p", 5.0, 1, model="h3",
                                 width=768, height=448, v13=True)
        self.v13.unlink()
        with self.assertRaises(FileNotFoundError):
            jobs.start_h3_job_ng(None, ".png", "p", 5.0, 1, model="h3", v13=True)
        self.assertEqual(jobs._JOBS, {})


if __name__ == "__main__":
    unittest.main()
