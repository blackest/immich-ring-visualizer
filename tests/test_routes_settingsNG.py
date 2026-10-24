"""Flask-test-client tests for routes/settingsNG.py.

Everything that would touch the machine (urlopen, subprocess, sockets,
sleep, the settings file) is mocked, so nothing is launched or probed.

Run with: python3 -m unittest tests.test_routes_settingsNG -v
"""

import os
import subprocess
import sys
import tempfile
import unittest
import urllib.error
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask

import routes.settingsNG as settings


def _client():
    app = Flask(__name__)
    app.register_blueprint(settings.settingsNG_bp)
    app.testing = True
    return app.test_client()


class AddressSettingsTests(unittest.TestCase):
    def test_get_returns_settings_list(self):
        with mock.patch.object(settings, "get_ng_address_settings", return_value=[{"key": "a"}]):
            r = _client().get("/api/ng/settings/addresses")
        self.assertEqual(r.get_json(), {"settings": [{"key": "a"}]})

    def test_save_passes_body_and_returns_fresh_settings(self):
        with mock.patch.object(settings, "save_ng_address_settings") as save, \
             mock.patch.object(settings, "get_ng_address_settings", return_value=[]):
            r = _client().post("/api/ng/settings/addresses", json={"k": "v"})
        save.assert_called_once_with({"k": "v"})
        self.assertTrue(r.get_json()["ok"])

    def test_save_rejects_non_object_body(self):
        with mock.patch.object(settings, "save_ng_address_settings") as save:
            r = _client().post("/api/ng/settings/addresses", json=["x"])
        self.assertEqual(r.status_code, 400)
        save.assert_not_called()

    def test_check_probes_only_probe_flagged_and_prefers_body_value(self):
        fake = [
            {"key": "ollama", "probe": True, "value": "http://saved:1"},
            {"key": "secret", "value": "nope"},
        ]
        with mock.patch.object(settings, "get_ng_address_settings", return_value=fake), \
             mock.patch.object(settings, "_probe_url", return_value=(True, None, 5)) as probe:
            r = _client().post("/api/ng/settings/addresses/check", json={"ollama": " http://edited:2 "})
        probe.assert_called_once_with("http://edited:2")
        self.assertEqual(r.get_json()["results"], {"ollama": {"ok": True, "error": None, "ms": 5}})

    def test_check_falls_back_to_saved_value(self):
        fake = [{"key": "ollama", "probe": True, "value": "http://saved:1"}]
        with mock.patch.object(settings, "get_ng_address_settings", return_value=fake), \
             mock.patch.object(settings, "_probe_url", return_value=(False, "dead", 1)) as probe:
            r = _client().post("/api/ng/settings/addresses/check", json={})
        probe.assert_called_once_with("http://saved:1")
        self.assertFalse(r.get_json()["results"]["ollama"]["ok"])


class TailscaleTests(unittest.TestCase):
    def test_status_and_up_delegate(self):
        with mock.patch.object(settings, "get_tailscale_state", return_value={"s": 1}), \
             mock.patch.object(settings, "tailscale_bring_up", return_value={"up": True}):
            c = _client()
            self.assertEqual(c.get("/api/ng/settings/tailscale").get_json(), {"s": 1})
            self.assertEqual(c.post("/api/ng/settings/tailscale/up").get_json(), {"up": True})


class ProbeUrlTests(unittest.TestCase):
    def test_success(self):
        with mock.patch.object(settings.urllib.request, "urlopen"):
            ok, err, ms = settings._probe_url("http://x")
        self.assertTrue(ok)
        self.assertIsNone(err)

    def test_http_error_counts_as_reachable(self):
        exc = urllib.error.HTTPError("http://x", 404, "nf", {}, None)
        with mock.patch.object(settings.urllib.request, "urlopen", side_effect=exc):
            self.assertTrue(settings._probe_url("http://x")[0])

    def test_connection_failure_is_unreachable(self):
        with mock.patch.object(settings.urllib.request, "urlopen", side_effect=OSError("dns")):
            ok, err, _ = settings._probe_url("http://x")
        self.assertFalse(ok)
        self.assertEqual(err, "dns")


class VenvCanImportTests(unittest.TestCase):
    def test_no_interpreter(self):
        self.assertEqual(settings._venv_can_import(None, ["a"]), (False, "interpreter not found"))

    def test_success_and_command_shape(self):
        with mock.patch.object(settings.subprocess, "run",
                               return_value=mock.Mock(returncode=0, stderr="")) as run:
            self.assertEqual(settings._venv_can_import("/py", ["a", "b"]), (True, None))
        self.assertEqual(run.call_args.args[0], ["/py", "-c", "import a, b"])

    def test_failure_reports_last_stderr_line(self):
        res = mock.Mock(returncode=1, stderr="Traceback\nModuleNotFoundError: mido\n")
        with mock.patch.object(settings.subprocess, "run", return_value=res):
            self.assertEqual(settings._venv_can_import("/py", ["mido"]),
                             (False, "ModuleNotFoundError: mido"))

    def test_failure_without_stderr(self):
        res = mock.Mock(returncode=1, stderr="")
        with mock.patch.object(settings.subprocess, "run", return_value=res):
            self.assertEqual(settings._venv_can_import("/py", ["m"]), (False, "import failed"))

    def test_exception(self):
        with mock.patch.object(settings.subprocess, "run", side_effect=OSError("boom")):
            self.assertEqual(settings._venv_can_import("/py", ["m"]), (False, "boom"))


class UpdateYtdlpTests(unittest.TestCase):
    def _post(self, **kw):
        with mock.patch.object(settings.subprocess, "run", **kw):
            return _client().post("/api/ng/settings/update-ytdlp")

    def test_success_returns_output_tail(self):
        out = "\n".join(f"line{i}" for i in range(30))
        r = self._post(return_value=mock.Mock(returncode=0, stdout=out, stderr=""))
        body = r.get_json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["output"].splitlines(), [f"line{i}" for i in range(15, 30)])

    def test_nonzero_exit(self):
        r = self._post(return_value=mock.Mock(returncode=2, stdout="", stderr="bad"))
        self.assertEqual(r.status_code, 500)
        self.assertIn("code 2", r.get_json()["error"])

    def test_timeout(self):
        r = self._post(side_effect=subprocess.TimeoutExpired("pip", 120))
        self.assertEqual(r.status_code, 504)

    def test_other_exception(self):
        r = self._post(side_effect=OSError("nope"))
        self.assertEqual(r.status_code, 500)


class PortOpenTests(unittest.TestCase):
    def test_open_and_closed(self):
        with mock.patch.object(settings.socket, "create_connection"):
            self.assertTrue(settings._port_open("h", 1))
        with mock.patch.object(settings.socket, "create_connection", side_effect=OSError):
            self.assertFalse(settings._port_open("h", 1))


class LaunchSunoTests(unittest.TestCase):
    def setUp(self):
        p = mock.patch.object(settings, "get_tailscale_hostname", return_value="ts")
        p.start()
        self.addCleanup(p.stop)
        s = mock.patch.object(settings.time, "sleep")
        s.start()
        self.addCleanup(s.stop)

    def _post(self):
        return _client().post("/api/ng/settings/launch-suno")

    def test_already_running(self):
        with mock.patch.object(settings, "_port_open", return_value=True), \
             mock.patch.object(settings.subprocess, "Popen") as popen:
            body = self._post().get_json()
        self.assertTrue(body["already_running"])
        self.assertTrue(body["url"].startswith("http://ts:"))
        popen.assert_not_called()

    def test_missing_dir(self):
        with mock.patch.object(settings, "_port_open", return_value=False), \
             mock.patch.object(settings, "SUNO_DIR", "/nonexistent/suno"), \
             mock.patch.object(settings.subprocess, "Popen") as popen:
            r = self._post()
        self.assertEqual(r.status_code, 500)
        popen.assert_not_called()

    def test_launch_then_port_comes_up(self):
        with tempfile.TemporaryDirectory() as d, \
             mock.patch.object(settings, "SUNO_DIR", d), \
             mock.patch.object(settings, "_port_open", side_effect=[False, False, True]), \
             mock.patch.object(settings.subprocess, "Popen") as popen:
            body = self._post().get_json()
        self.assertTrue(body["ok"])
        self.assertFalse(body["already_running"])
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        self.assertEqual(popen.call_args.kwargs["cwd"], d)

    def test_popen_failure(self):
        with tempfile.TemporaryDirectory() as d, \
             mock.patch.object(settings, "SUNO_DIR", d), \
             mock.patch.object(settings, "_port_open", return_value=False), \
             mock.patch.object(settings.subprocess, "Popen", side_effect=OSError("x")):
            self.assertEqual(self._post().status_code, 500)

    def test_never_comes_up(self):
        with tempfile.TemporaryDirectory() as d, \
             mock.patch.object(settings, "SUNO_DIR", d), \
             mock.patch.object(settings, "_port_open", return_value=False), \
             mock.patch.object(settings.subprocess, "Popen"):
            r = self._post()
        self.assertEqual(r.status_code, 500)
        self.assertIn("5s", r.get_json()["error"])


class LaunchComfyuiTests(unittest.TestCase):
    def setUp(self):
        s = mock.patch.object(settings.time, "sleep")
        s.start()
        self.addCleanup(s.stop)
        u = mock.patch.object(settings, "get_comfyui_base_url", return_value="http://127.0.0.1:9999")
        u.start()
        self.addCleanup(u.stop)

    def _post(self):
        return _client().post("/api/ng/settings/launch-comfyui")

    def test_already_running_includes_log_only_if_present(self):
        with tempfile.TemporaryDirectory() as d, \
             mock.patch.object(settings, "COMFYUI_DIR", d), \
             mock.patch.object(settings, "_port_open", return_value=True):
            self.assertNotIn("logPath", self._post().get_json())
            open(os.path.join(d, "comfyui_launch.log"), "w").close()
            self.assertIn("logPath", self._post().get_json())

    def test_missing_dir(self):
        with mock.patch.object(settings, "COMFYUI_DIR", "/nonexistent/comfy"), \
             mock.patch.object(settings, "_port_open", return_value=False):
            self.assertEqual(self._post().status_code, 500)

    def test_launch_uses_port_from_url_and_logs(self):
        with tempfile.TemporaryDirectory() as d, \
             mock.patch.object(settings, "COMFYUI_DIR", d), \
             mock.patch.object(settings, "_port_open", side_effect=[False, True]), \
             mock.patch.object(settings.subprocess, "Popen") as popen:
            body = self._post().get_json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["logPath"], os.path.join(d, "comfyui_launch.log"))
        argv = popen.call_args.args[0]
        self.assertEqual(argv[-2:], ["--port", "9999"])

    def test_timeout(self):
        with tempfile.TemporaryDirectory() as d, \
             mock.patch.object(settings, "COMFYUI_DIR", d), \
             mock.patch.object(settings, "_port_open", return_value=False), \
             mock.patch.object(settings.subprocess, "Popen"):
            r = self._post()
        self.assertEqual(r.status_code, 500)
        self.assertIn("30s", r.get_json()["error"])


if __name__ == "__main__":
    unittest.main()
