import unittest
from unittest import mock

import requests

import model_offloadNG as m


class _Resp:
    def __init__(self, data):
        self._d = data

    def json(self):
        return self._d


class OffloadTests(unittest.TestCase):
    def test_unloads_each_resident_ollama_model(self):
        ps = _Resp({"models": [{"name": "gemma3:12b"}, {"name": "qwen3:8b"}]})
        with mock.patch.object(m.requests, "get", return_value=ps), \
             mock.patch.object(m.requests, "post") as post:
            got = m.unload_ollama_models_ng()
        self.assertEqual(got, ["gemma3:12b", "qwen3:8b"])
        self.assertEqual(post.call_count, 2)
        self.assertEqual(post.call_args.kwargs["json"], {"model": "qwen3:8b", "keep_alive": 0})

    def test_unreachable_ollama_is_silent(self):
        with mock.patch.object(m.requests, "get", side_effect=requests.exceptions.ConnectionError):
            self.assertEqual(m.unload_ollama_models_ng(), [])

    def test_idle_ollama_posts_nothing(self):
        with mock.patch.object(m.requests, "get", return_value=_Resp({"models": []})), \
             mock.patch.object(m.requests, "post") as post:
            self.assertEqual(m.unload_ollama_models_ng(), [])
        post.assert_not_called()

    def test_offload_calls_both(self):
        with mock.patch.object(m, "unload_ollama_models_ng") as o, \
             mock.patch("ltx_engineNG._unload_gemma_server_ng") as g:
            m.offload_resident_models_ng()
        o.assert_called_once()
        g.assert_called_once()


if __name__ == "__main__":
    unittest.main()
