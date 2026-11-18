"""Klein and the macOS GPU watchdog: a killed worker is re-run once for the pictures it had not finished, on
the shortest command buffers; any other failure is not retried; the error shows the reason, not progress bars.
A fake worker stands in for the real one, so nothing touches the GPU.

Run with: python3 -m unittest tests.test_klein_watchdogNG -v
"""

import json
import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import klein_engineNG as engine

# mode comes from the request prompt: "kill" = first run dies at the watchdog, "always" = dies every time,
# "crash" = ordinary failure. Each run appends "<seeds> <ops-per-buffer>" to calls.log next to the output.
FAKE = textwrap.dedent('''
    import json, os, sys
    req = json.loads(sys.stdin.readline())
    out = req["output_dir"]
    short = os.environ.get("MLX_MAX_OPS_PER_BUFFER") == "1"
    open(os.path.join(out, "calls.log"), "a").write(json.dumps([req["seeds"], short]) + "\\n")
    mode = req["prompt"]
    def render(seed):
        p = os.path.join(out, f"k_{seed}.png")
        open(p, "wb").write(b"png")
        print("KLEIN_RESULT " + json.dumps({"png_path": p, "seed": seed, "width": 1, "height": 1}), flush=True)
    def die(msg):
        print("  6%|#6        | 1/16 [02:07<31:57, 127.83s/it]", flush=True)
        print(msg, flush=True)
        sys.exit(1)
    for i, seed in enumerate(req["seeds"]):
        if mode == "always" or (mode == "kill" and not short and i == 1):
            die("Klein: [METAL] Command buffer execution failed: Impacting Interactivity")
        if mode == "crash":
            die("ValueError: the model file is broken")
        render(seed)
''')


class WatchdogTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        worker = self.dir / "fake_worker.py"
        worker.write_text(FAKE)
        self.out = self.dir / "out"
        for p in (mock.patch.object(engine, "WORKER", worker),
                  mock.patch.object(engine, "worker_python", lambda: Path(sys.executable)),
                  mock.patch.object(engine, "snapshot_ng", lambda model: self.dir)):
            p.start()
            self.addCleanup(p.stop)

    def run_job(self, mode, n=2, **kw):
        return engine.generate_klein_ng(mode, n, 512, 512, self.out, 100, **kw)

    def calls(self):
        return [json.loads(l) for l in (self.out / "calls.log").read_text().splitlines()]

    def test_a_watchdog_kill_reruns_only_the_unfinished_seeds_on_short_buffers(self):
        logs = []
        results = self.run_job("kill", n=3, on_log=logs.append)
        self.assertEqual([r["seed"] for r in results], [100, 101, 102])
        self.assertEqual(self.calls(), [[[100, 101, 102], False], [[101, 102], True]])
        self.assertTrue(any("retrying once" in l for l in logs))

    def test_it_retries_only_once(self):
        with self.assertRaises(engine.KleinWatchdogKill):
            self.run_job("always", n=1)
        self.assertEqual(len(self.calls()), 2)

    def test_other_failures_are_not_retried_and_show_the_reason_not_the_progress_bar(self):
        with self.assertRaises(RuntimeError) as ctx:
            self.run_job("crash", n=1)
        self.assertNotIsInstance(ctx.exception, engine.KleinWatchdogKill)
        self.assertIn("the model file is broken", str(ctx.exception))
        self.assertNotIn("s/it", str(ctx.exception))
        self.assertEqual(len(self.calls()), 1)

    def test_the_watchdog_message_is_the_metal_line_not_the_progress_bar(self):
        with self.assertRaises(engine.KleinWatchdogKill) as ctx:
            self.run_job("always", n=1)
        self.assertIn("Impacting Interactivity", str(ctx.exception))
        self.assertNotIn("127.83s/it", str(ctx.exception))

    def test_a_cancel_is_not_retried(self):
        with self.assertRaises(engine.KleinCancelled):
            self.run_job("kill", n=3, should_stop=lambda: True)
        self.assertFalse((self.out / "calls.log").exists() and len(self.calls()) > 1)

    def test_short_buffers_are_only_set_for_the_retry(self):
        with mock.patch.dict(os.environ):
            os.environ.pop("MLX_MAX_OPS_PER_BUFFER", None)
            self.assertNotIn("MLX_MAX_OPS_PER_BUFFER", engine._clean_env())
        self.assertEqual(engine._clean_env(short_buffers=True)["MLX_MAX_OPS_PER_BUFFER"], "1")

    def test_reason_when_the_worker_says_nothing_but_progress(self):
        text = engine._failure_reason_ng(["  0%|          | 0/16 [00:00<?, ?it/s]"], -6)
        self.assertIn("exit code -6", text)


if __name__ == "__main__":
    unittest.main()
