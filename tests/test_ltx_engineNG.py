"""Unit tests for ltx_engineNG.py -- the LTX-2.5 render engine and the Gemma
helpers that share its subprocess lock. Every subprocess, HTTP call and
GPU-adjacent side effect (codec patching, model offload) is mocked at the
boundary; nothing here launches ltx-2-mlx or touches a model.

Run with: python3 -m unittest tests.test_ltx_engineNG -v
"""

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests

import ltx_codecNG
import ltx_engineNG as ltx
import model_offloadNG


def _run_result(stdout="", rc=0):
    return SimpleNamespace(returncode=rc, stdout=stdout)


class _Proc:
    def __init__(self, rc=0, lines=(), delay=0.0, write=None):
        self.rc, self.pid, self.killed = rc, 99999, False
        self._lines, self._delay, self._write = list(lines), delay, write
        self.stdout = self._gen()

    def _gen(self):
        if self._delay:
            time.sleep(self._delay)
        for line in self._lines:
            yield line

    def wait(self):
        if self._write:
            self._write()
        return self.rc

    def kill(self):
        self.killed = True


class PureHelperTests(unittest.TestCase):
    def test_dimension_validation(self):
        ltx._validate_ltx_dims_ng(1024, 576)
        for w, h, msg in ((1000, 576, "multiple of 64"), (1024, 100, "multiple of 64"),
                          (128, 576, "between"), (1600, 576, "between")):
            with self.assertRaisesRegex(ValueError, msg):
                ltx._validate_ltx_dims_ng(w, h)

    def test_fps_validation(self):
        ltx._validate_ltx_fps_ng(24.0)
        for bad in (7.9, 30.1):
            with self.assertRaisesRegex(ValueError, "frame_rate"):
                ltx._validate_ltx_fps_ng(bad)

    def test_duration_to_frames_is_8n_plus_1(self):
        self.assertEqual(ltx._duration_to_frames_ng(2.0), 49)      # 48/8 = 6 blocks
        self.assertEqual(ltx._duration_to_frames_ng(0.1), 9)       # at least one block
        self.assertEqual(ltx._duration_to_frames_ng(5.0, 30.0), 153)
        for s in (0.7, 1.3, 4.9):
            self.assertEqual((ltx._duration_to_frames_ng(s) - 1) % 8, 0)

    def test_clean_env_strips_malloc(self):
        with mock.patch.dict(os.environ, {"MallocStackLogging": "1", "KEEP": "y"}):
            env = ltx._clean_subprocess_env_ng()
        self.assertNotIn("MallocStackLogging", env)
        self.assertEqual(env["KEEP"], "y")

    def test_env_var_overrides_for_repo_and_weights(self):
        with mock.patch.dict(os.environ, {"LTX_MLX_DIR": "/x/ltx", "LTX_MLX_MODELS_DIR": "/x/models"}):
            self.assertEqual(ltx._resolve_ltx_repo_dir_ng(), Path("/x/ltx"))
            self.assertEqual(ltx._resolve_ltx_weights_root_ng(), Path("/x/models"))

    def test_resolvers_check_the_filesystem(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            binp = d / "ltx-2-mlx"
            binp.write_text("#!/bin/sh\n")
            cfg = ltx.LtxConfig(binary_path=str(binp), model_path=str(d / "m"),
                                gemma_path=str(d / "g"), enhance_gemma_path=str(d / "e"))
            self.assertIsNone(ltx._resolve_ltx_binary_ng(cfg))      # not executable yet
            binp.chmod(0o755)
            self.assertEqual(ltx._resolve_ltx_binary_ng(cfg), str(binp))
            for fn in (ltx._resolve_ltx_model_ng, ltx._resolve_ltx_gemma_ng,
                       ltx._resolve_ltx_enhance_gemma_ng):
                self.assertIsNone(fn(cfg))
            for name in "meg":
                (d / name).mkdir()
            self.assertEqual(ltx._resolve_ltx_model_ng(cfg), str(d / "m"))
            self.assertEqual(ltx._resolve_ltx_gemma_ng(cfg), str(d / "g"))
            self.assertEqual(ltx._resolve_ltx_enhance_gemma_ng(cfg), str(d / "e"))

    def test_health_ready_does_not_require_enhance_gemma(self):
        with mock.patch.object(ltx, "_resolve_ltx_binary_ng", return_value="/b"), \
             mock.patch.object(ltx, "_resolve_ltx_model_ng", return_value="/m"), \
             mock.patch.object(ltx, "_resolve_ltx_gemma_ng", return_value="/g"), \
             mock.patch.object(ltx, "_resolve_ltx_enhance_gemma_ng", return_value=None):
            h = ltx.ltx_health_ng()
        self.assertTrue(h["ready"])
        self.assertFalse(h["enhance_gemma_ok"])
        with mock.patch.object(ltx, "_resolve_ltx_binary_ng", return_value=None):
            self.assertFalse(ltx.ltx_health_ng()["ready"])

    def test_unload_gemma_server_is_best_effort(self):
        logs = []
        with mock.patch("requests.post") as post:
            ltx._unload_gemma_server_ng(logs.append)
        self.assertTrue(post.call_args.args[0].endswith("/unload"))
        self.assertEqual(len(logs), 1)
        with mock.patch("requests.post", side_effect=requests.exceptions.ConnectionError):
            ltx._unload_gemma_server_ng(logs.append)   # silent
        self.assertEqual(len(logs), 1)

    def test_extract_ask_rachel(self):
        self.assertIsNone(ltx._extract_ask_rachel_ng("just chatting"))
        self.assertIsNone(ltx._extract_ask_rachel_ng("ASK_RACHEL:   "))
        self.assertEqual(ltx._extract_ask_rachel_ng("ok\nASK_RACHEL: who won?"), "who won?")
        self.assertEqual(ltx._extract_ask_rachel_ng("ASK_RACHEL: q\nand more"), "q\nand more")


class SceneJsonTests(unittest.TestCase):
    def test_plain_and_fenced_json(self):
        body = {"discussion": " idea ", "options": [{"label": "A", "prompt": "p", "seconds": 4}]}
        for raw in (json.dumps(body), "```json\n" + json.dumps(body) + "\n```", "```\n" + json.dumps(body) + "\n```"):
            out = ltx._parse_scene_discuss_json_ng(raw)
            self.assertEqual(out["discussion"], "idea")
            self.assertEqual(out["options"], [{"label": "A", "prompt": "p", "seconds": 4.0}])

    def test_options_are_cleaned_and_clamped(self):
        raw = json.dumps({"options": [
            {"prompt": "one", "seconds": 999},
            {"prompt": "  ", "label": "skip"},          # empty prompt dropped
            {"prompt": "two", "seconds": "abc"},        # bad seconds -> 3.0
        ]})
        opts = ltx._parse_scene_discuss_json_ng(raw)["options"]
        self.assertEqual([o["prompt"] for o in opts], ["one", "two"])
        self.assertEqual([o["seconds"] for o in opts], [30.0, 3.0])
        self.assertEqual([o["label"] for o in opts], ["A", "C"])   # default label = position
        low = ltx._parse_scene_discuss_json_ng(json.dumps({"options": [{"prompt": "p", "label": "Z", "seconds": 0}]}))
        self.assertEqual((low["options"][0]["label"], low["options"][0]["seconds"]), ("Z", 0.5))

    def test_only_the_first_three_entries_are_considered(self):
        # The cap is applied before filtering, so a junk entry in the first
        # three uses up a slot rather than letting a fourth through.
        raw = json.dumps({"options": ["not a dict", {"prompt": ""}, {"prompt": "kept"}, {"prompt": "fourth"}]})
        opts = ltx._parse_scene_discuss_json_ng(raw)["options"]
        self.assertEqual([o["prompt"] for o in opts], ["kept"])

    def test_bad_shapes_raise(self):
        for raw, msg in (("nope", "unparseable"), ("[]", "no usable 'options'"),
                         ('{"options": []}', "no usable 'options'"),
                         ('{"options": [{"prompt": ""}]}', "no usable prompt")):
            with self.assertRaisesRegex(RuntimeError, msg):
                ltx._parse_scene_discuss_json_ng(raw)


class GenerateTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.out = Path(self._tmp.name)
        self.img = self.out / "ref.png"
        self.img.write_bytes(b"png")
        self.lora = self.out / "style.safetensors"
        self.lora.write_bytes(b"lora")
        self.logs, self.cmds, self.envs = [], [], []
        for p in (mock.patch.object(ltx, "_resolve_ltx_binary_ng", return_value="/bin/ltx"),
                  mock.patch.object(ltx, "_resolve_ltx_model_ng", return_value="/m"),
                  mock.patch.object(ltx, "_resolve_ltx_gemma_ng", return_value="/g"),
                  mock.patch.object(ltx_codecNG, "apply_codec_patch_ng", return_value=ltx_codecNG.ALREADY),
                  mock.patch.object(model_offloadNG, "offload_resident_models_ng"),
                  mock.patch("configNG.get_ltx_output_preset", return_value="web")):
            p.start()
            self.addCleanup(p.stop)
        self.offload = model_offloadNG.offload_resident_models_ng

    def _run(self, proc=None, write_mp4=True, **kw):
        args = dict(prompt="a cat", image_path=str(self.img), duration_s=2.0,
                    output_dir=self.out / "o", seed=7, config=ltx.LtxConfig())
        args.update(kw)

        def popen(cmd, **kwargs):
            self.cmds.append(cmd)
            self.envs.append(kwargs["env"])
            out = Path(cmd[cmd.index("--output") + 1])
            p = proc or _Proc()
            if write_mp4 and p._write is None:
                p._write = lambda: out.write_bytes(b"mp4data")
            return p
        with mock.patch("subprocess.Popen", popen):
            return ltx.generate_ltx_video_ng(on_log=self.logs.append, **args)

    def test_image_to_video_argv_and_result(self):
        res = self._run()
        cmd = self.cmds[0]
        flag = lambda n: cmd[cmd.index(n) + 1]
        self.assertEqual(cmd[:2], ["/bin/ltx", "generate"])
        self.assertEqual((flag("--model"), flag("--gemma"), flag("--seed")), ("/m", "/g", "7"))
        self.assertEqual((flag("--width"), flag("--height"), flag("--frames")), ("1024", "576", "49"))
        self.assertIn("--two-stages-hq", cmd)
        self.assertEqual(flag("--distilled-lora"), ltx.LTX_DISTILLED_LORA)
        self.assertEqual(flag("--image"), str(self.img))
        self.assertNotIn("--lora", cmd)
        self.assertEqual(res["seed"], 7)
        self.assertEqual((res["frames"], res["duration_s"]), (49, 2.0))
        self.assertTrue(Path(res["mp4_path"]).is_file())

    def test_text_to_video_omits_image_and_lora_is_passed_through(self):
        self._run(image_path=None, lora_path=str(self.lora), lora_strength=0.6)
        cmd = self.cmds[0]
        self.assertNotIn("--image", cmd)
        i = cmd.index("--lora")
        self.assertEqual(cmd[i + 1:i + 3], [str(self.lora), "0.6"])

    def test_custom_dims_and_fps(self):
        res = self._run(width=768, height=512, frame_rate=30.0, duration_s=1.0)
        self.assertEqual((res["width"], res["height"], res["frame_rate"]), (768, 512, 30.0))
        self.assertEqual(res["frames"], 33)

    def test_validation_and_missing_inputs_fail_before_launch(self):
        with self.assertRaises(ValueError):
            self._run(width=1000)
        with self.assertRaises(ValueError):
            self._run(frame_rate=2.0)
        with mock.patch.object(ltx, "_resolve_ltx_binary_ng", return_value=None):
            with self.assertRaisesRegex(FileNotFoundError, "binary"):
                self._run()
        with mock.patch.object(ltx, "_resolve_ltx_model_ng", return_value=None):
            with self.assertRaisesRegex(FileNotFoundError, "model"):
                self._run()
        with mock.patch.object(ltx, "_resolve_ltx_gemma_ng", return_value=None):
            with self.assertRaisesRegex(FileNotFoundError, "Gemma"):
                self._run()
        with self.assertRaisesRegex(FileNotFoundError, "reference image"):
            self._run(image_path=str(self.out / "nope.png"))
        with self.assertRaisesRegex(FileNotFoundError, "LoRA"):
            self._run(lora_path=str(self.out / "nope.safetensors"))
        self.assertEqual(self.cmds, [])

    def test_random_seed_when_none(self):
        res = self._run(seed=None)
        self.assertTrue(0 <= res["seed"] < 2**31)

    def test_codec_preset_env_offload_and_proc_hook(self):
        started = []
        proc = _Proc(lines=["loading\n", "\n", "step 1\n"])
        self._run(proc=proc, on_proc_start=started.append)
        self.assertEqual(self.envs[0]["LTX_OUTPUT_CRF"], ltx_codecNG.OUTPUT_PRESETS["web"]["crf"])
        self.offload.assert_called_once()
        self.assertEqual(started, [proc])
        self.assertIn("[ltx] loading", self.logs)
        self.assertIn("[ltx] step 1", self.logs)

    def test_codec_patch_trouble_is_logged_not_fatal(self):
        for outcome in (ltx_codecNG.MISSING, ltx_codecNG.DRIFT):
            self.logs.clear()
            with mock.patch.object(ltx_codecNG, "apply_codec_patch_ng", return_value=outcome):
                self._run()
            self.assertTrue(any(f"codec patch {outcome}" in l for l in self.logs))

    def test_failure_codes(self):
        with self.assertRaisesRegex(RuntimeError, "failed with rc=2"):
            self._run(proc=_Proc(rc=2))
        for rc in (-9, -15, 137, 143):
            with self.assertRaises(ltx.VideoJobCancelled):
                self._run(proc=_Proc(rc=rc))

    def test_watchdog_timeout(self):
        proc = _Proc(delay=0.3)
        with mock.patch.dict(os.environ, {"RINGVIZ_LTX_TIMEOUT_S": "0.05"}), \
             mock.patch.object(ltx.os, "killpg", side_effect=ProcessLookupError):
            with self.assertRaisesRegex(ltx.VideoJobCancelled, "deadline"):
                self._run(proc=proc)
        self.assertTrue(proc.killed)

    def test_missing_or_empty_mp4_raises(self):
        with self.assertRaisesRegex(RuntimeError, "no/empty mp4"):
            self._run(write_mp4=False)
        p = _Proc()
        with self.assertRaisesRegex(RuntimeError, "no/empty mp4"):
            self._run(proc=_Proc(write=None), write_mp4=False)


class EnhanceTests(unittest.TestCase):
    def setUp(self):
        for p in (mock.patch.object(ltx, "_resolve_ltx_binary_ng", return_value="/bin/ltx"),
                  mock.patch.object(ltx, "_resolve_ltx_enhance_gemma_ng", return_value="/eg"),
                  mock.patch.object(ltx, "_resolve_ltx_python_ng", return_value="/venv/python"),
                  mock.patch.object(ltx, "LTX_VISION_ENHANCE_HELPER", Path(__file__)),
                  mock.patch.object(ltx, "LTX_SCENE_TIMING_HELPER", Path(__file__)),
                  mock.patch.object(ltx, "LTX_SCENE_DISCUSS_HELPER", Path(__file__)),
                  mock.patch.object(ltx, "LTX_VISION_SCENE_DISCUSS_HELPER", Path(__file__))):
            p.start()
            self.addCleanup(p.stop)
        self.cfg = ltx.LtxConfig()

    def test_text_enhance_parses_last_marker_and_builds_argv(self):
        with mock.patch("subprocess.run", return_value=_run_result("noise\nEnhanced: old\nEnhanced:  new text \n")) as run:
            self.assertEqual(ltx.enhance_ltx_prompt_ng("cat", self.cfg, seed=3), "new text")
        cmd = run.call_args.args[0]
        self.assertEqual(cmd[:2], ["/bin/ltx", "enhance"])
        self.assertEqual(cmd[cmd.index("--mode") + 1], "i2v")
        self.assertEqual(cmd[cmd.index("--seed") + 1], "3")

    def test_text_enhance_failures(self):
        for out, rc, msg in (("boom", 1, "rc=1"), ("no marker", 0, "no 'Enhanced:'"),
                             ("x\nEnhanced:   ", 0, "empty text")):
            with mock.patch("subprocess.run", return_value=_run_result(out, rc)):
                with self.assertRaisesRegex(RuntimeError, msg):
                    ltx.enhance_ltx_prompt_ng("cat", self.cfg)
        with mock.patch.object(ltx, "_resolve_ltx_binary_ng", return_value=None):
            with self.assertRaises(FileNotFoundError):
                ltx.enhance_ltx_prompt_ng("cat", self.cfg)
        with mock.patch.object(ltx, "_resolve_ltx_enhance_gemma_ng", return_value=None):
            with self.assertRaises(FileNotFoundError):
                ltx.enhance_ltx_prompt_ng("cat", self.cfg)

    def test_image_routes_to_vision_helper_over_stdin(self):
        with mock.patch("subprocess.run", return_value=_run_result("x\nEnhanced: seen it")) as run:
            self.assertEqual(ltx.enhance_ltx_prompt_ng("cat", self.cfg, image_b64="QUJD"), "seen it")
        self.assertEqual(run.call_args.kwargs["input"], "QUJD")
        self.assertEqual(run.call_args.args[0][0], "/venv/python")
        with mock.patch("subprocess.run", return_value=_run_result("junk", 0)):
            with self.assertRaisesRegex(RuntimeError, "vision prompt enhancement produced no"):
                ltx.enhance_ltx_prompt_ng("cat", self.cfg, image_b64="QUJD")
        with mock.patch.object(ltx, "_resolve_ltx_python_ng", return_value=None):
            with self.assertRaises(FileNotFoundError):
                ltx.enhance_ltx_prompt_ng("cat", self.cfg, image_b64="QUJD")

    def test_scene_seconds_parse_and_clamp(self):
        for text, want in (("Seconds: 4.5", 4.5), ("Seconds: about 12 s", 12.0),
                           ("Seconds: 0.1", 0.5), ("Seconds: 99", 30.0)):
            with mock.patch("subprocess.run", return_value=_run_result("log\n" + text)):
                self.assertEqual(ltx.estimate_scene_seconds_ng("p", self.cfg), want)
        for out, rc, msg in (("x", 0, "no 'Seconds:'"), ("\nSeconds: lots", 0, "no parseable"),
                             ("bad", 3, "rc=3")):
            with mock.patch("subprocess.run", return_value=_run_result(out, rc)):
                with self.assertRaisesRegex(RuntimeError, msg):
                    ltx.estimate_scene_seconds_ng("p", self.cfg)

    def test_discuss_text_and_vision_paths(self):
        reply = json.dumps({"discussion": "d", "options": [{"label": "A", "prompt": "go", "seconds": 5}]})
        turns = [{"role": "user", "content": "next?"}]
        with mock.patch("subprocess.run", return_value=_run_result("log\nReply: " + reply)) as run:
            out = ltx.discuss_next_scene_ng(turns, self.cfg)
        self.assertEqual(json.loads(run.call_args.kwargs["input"]), turns)
        self.assertEqual((out["raw"], out["options"][0]["prompt"]), (reply, "go"))
        with mock.patch("subprocess.run", return_value=_run_result("log\nReply: " + reply)) as run:
            ltx.discuss_next_scene_ng(turns, self.cfg, images=["QUJD"])
        self.assertEqual(json.loads(run.call_args.kwargs["input"]), {"turns": turns, "images": ["QUJD"]})
        with mock.patch("subprocess.run", return_value=_run_result("no marker")):
            with self.assertRaisesRegex(RuntimeError, "no 'Reply:'"):
                ltx.discuss_next_scene_ng(turns, self.cfg)


class ChatTests(unittest.TestCase):
    def setUp(self):
        for p in (mock.patch.object(ltx, "_resolve_ltx_enhance_gemma_ng", return_value="/eg"),
                  mock.patch.object(ltx, "_resolve_ltx_python_ng", return_value="/venv/python"),
                  mock.patch.object(ltx, "LTX_SCENE_CHAT_HELPER", Path(__file__)),
                  mock.patch.object(ltx, "LTX_VISION_CHAT_HELPER", Path(__file__))):
            p.start()
            self.addCleanup(p.stop)
        self.cfg = ltx.LtxConfig()
        self.turns = [{"role": "user", "content": "hi"}]

    @staticmethod
    def _http(status=200, content="hello"):
        return SimpleNamespace(status_code=status, text="err body",
                               json=lambda: {"choices": [{"message": {"content": content}}]})

    def test_server_path_builds_messages_and_images_only_on_last_user_turn(self):
        turns = [{"role": "user", "content": "first"}, {"role": "assistant", "content": "a"},
                 {"role": "user", "content": "look"}]
        with mock.patch("requests.post", return_value=self._http(content=" hi there ")) as post:
            out = ltx._chat_with_gemma_server_ng(turns, self.cfg, seed=5, images=["QUJD"])
        self.assertEqual(out, {"reply": "hi there"})
        body = post.call_args.kwargs["json"]
        self.assertEqual(body["seed"], 5)
        self.assertEqual(body["messages"][0]["role"], "system")
        self.assertEqual(body["messages"][1]["content"], "first")          # earlier turn stays text
        parts = body["messages"][3]["content"]
        self.assertEqual(parts[1]["image_url"]["url"], "data:image/jpeg;base64,QUJD")
        self.assertIn("reference image", body["messages"][0]["content"])   # vision system prompt

    def test_server_down_returns_none_but_http_error_and_empty_reply_raise(self):
        for exc in (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            with mock.patch("requests.post", side_effect=exc):
                self.assertIsNone(ltx._chat_with_gemma_server_ng(self.turns, self.cfg))
        with mock.patch("requests.post", return_value=self._http(500)):
            with self.assertRaisesRegex(RuntimeError, "HTTP 500"):
                ltx._chat_with_gemma_server_ng(self.turns, self.cfg)
        with mock.patch("requests.post", return_value=self._http(content="  ")):
            with self.assertRaisesRegex(RuntimeError, "empty output"):
                ltx._chat_with_gemma_server_ng(self.turns, self.cfg)
        with mock.patch.object(ltx, "_resolve_ltx_enhance_gemma_ng", return_value=None):
            self.assertIsNone(ltx._chat_with_gemma_server_ng(self.turns, self.cfg))

    def test_falls_back_to_subprocess_text_and_vision_helpers(self):
        down = mock.patch("requests.post", side_effect=requests.exceptions.ConnectionError)
        with down, mock.patch("subprocess.run", return_value=_run_result("log\nReply: sub reply")) as run:
            self.assertEqual(ltx.chat_with_gemma_ng(self.turns, self.cfg), {"reply": "sub reply"})
        self.assertEqual(json.loads(run.call_args.kwargs["input"]), self.turns)
        with down, mock.patch("subprocess.run", return_value=_run_result("log\nReply: seen")) as run:
            ltx.chat_with_gemma_ng(self.turns, self.cfg, images=["QUJD"])
        self.assertEqual(json.loads(run.call_args.kwargs["input"])["images"], ["QUJD"])
        with down, mock.patch("subprocess.run", return_value=_run_result("junk")):
            with self.assertRaisesRegex(RuntimeError, "no 'Reply:'"):
                ltx.chat_with_gemma_ng(self.turns, self.cfg)
        with down, mock.patch("subprocess.run", return_value=_run_result("x", 1)):
            with self.assertRaisesRegex(RuntimeError, "chat failed"):
                ltx.chat_with_gemma_ng(self.turns, self.cfg)

    def test_rachel_delegation_runs_one_round_and_reports_it(self):
        replies = iter([{"reply": "ASK_RACHEL: who is PM?"}, {"reply": "It is X."}])
        calls = []

        def fake_reply(turns, config, seed=None, images=None):
            calls.append((list(turns), images))
            return next(replies)
        with mock.patch.object(ltx, "_generate_gemma_reply_ng", side_effect=fake_reply), \
             mock.patch.object(ltx, "ask_rachel_ng", return_value="X") as rachel:
            out = ltx.chat_with_gemma_ng(self.turns, self.cfg, images=["QUJD"])
        rachel.assert_called_once_with("who is PM?")
        self.assertEqual(out["reply"], "It is X.")
        self.assertEqual(out["delegated"], {"question": "who is PM?", "answer": "X"})
        followup, images = calls[1]
        self.assertIsNone(images)                                   # images only on the first pass
        self.assertEqual(followup[-1]["content"], "[Rachel says: X]")
        self.assertEqual(followup[-2], {"role": "assistant", "content": "ASK_RACHEL: who is PM?"})

    def test_rachel_failure_keeps_geminis_raw_reply(self):
        with mock.patch.object(ltx, "_generate_gemma_reply_ng", return_value={"reply": "ASK_RACHEL: q"}) as g, \
             mock.patch.object(ltx, "ask_rachel_ng", side_effect=RuntimeError("gateway down")):
            out = ltx.chat_with_gemma_ng(self.turns, self.cfg)
        self.assertEqual(out["rachel_error"], "gateway down")
        self.assertEqual(out["reply"], "ASK_RACHEL: q")
        g.assert_called_once()                                      # no second Gemma pass

    def test_plain_reply_is_not_delegated(self):
        with mock.patch.object(ltx, "_generate_gemma_reply_ng", return_value={"reply": "fine"}), \
             mock.patch.object(ltx, "ask_rachel_ng") as rachel:
            self.assertEqual(ltx.chat_with_gemma_ng(self.turns, self.cfg), {"reply": "fine"})
        rachel.assert_not_called()

    def test_ask_rachel_posts_stateless_request(self):
        resp = SimpleNamespace(raise_for_status=lambda: None,
                               json=lambda: {"choices": [{"message": {"content": " answer \n"}}]})
        with mock.patch("configNG.get_hermes_base_url", return_value="http://h:8642"), \
             mock.patch("configNG.get_hermes_api_key", return_value="k"), \
             mock.patch("configNG.get_hermes_model", return_value="m"), \
             mock.patch("requests.post", return_value=resp) as post:
            self.assertEqual(ltx.ask_rachel_ng("q?"), "answer")
        self.assertEqual(post.call_args.args[0], "http://h:8642/v1/chat/completions")
        self.assertEqual(post.call_args.kwargs["headers"], {"Authorization": "Bearer k"})
        self.assertEqual(post.call_args.kwargs["json"]["messages"], [{"role": "user", "content": "q?"}])
        self.assertFalse(post.call_args.kwargs["json"]["stream"])
        with mock.patch("configNG.get_hermes_base_url", return_value=""):
            with self.assertRaisesRegex(RuntimeError, "not configured"):
                ltx.ask_rachel_ng("q?")


if __name__ == "__main__":
    unittest.main()
