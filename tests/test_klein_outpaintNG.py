"""Klein LoRAs and outpaint: the listing, option checks, the padding plan, the paste-back and the route
(with the model replaced, so nothing renders).

Run with: python3 -m unittest tests.test_klein_outpaintNG -v
"""

import io
import json
import os
import struct
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image

import klein_engineNG as engine
import klein_jobsNG as jobs
import klein_lorasNG as loras
import klein_outpaintNG as op
import klein_workerNG as worker


def write_lora(folder: Path, name: str, keys, meta=None, sidecar=None) -> Path:
    header = {k: {"dtype": "F32", "shape": [1], "data_offsets": [0, 0]} for k in keys}
    if meta:
        header["__metadata__"] = meta
    raw = json.dumps(header).encode()
    path = folder / f"{name}.safetensors"
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)
    if sidecar:
        path.with_suffix(".json").write_text(json.dumps(sidecar))
    return path


FLUX_KEYS = ["diffusion_model.content_mode_blocks.image.0.q_proj.lora_A.weight",
             "diffusion_model.content_mode_blocks.image.0.q_proj.lora_B.weight"]


class LoraListing(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        patcher = mock.patch.dict(os.environ, {"KLEIN_LORAS_DIR": self.tmp.name})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)

    def test_lists_flux3_loras_with_sidecar_and_skips_others(self):
        write_lora(self.dir, "out", FLUX_KEYS, sidecar={"name": "Outpaint", "recommended_strength": 1.5, "kind": "outpaint"})
        write_lora(self.dir, "ltx", ["transformer_blocks.0.attn.to_q.lora_A.weight"])
        write_lora(self.dir, "empty", ["x.weight"])
        (self.dir / "broken.safetensors").write_bytes(b"nope")
        found = loras.list_loras_ng()
        self.assertEqual([e["name"] for e in found], ["out"])
        self.assertEqual((found[0]["title"], found[0]["recommended_strength"], found[0]["kind"]), ("Outpaint", 1.5, "outpaint"))
        self.assertFalse(found[0]["distilled_only"])

    def test_declared_flux3_counts_and_ref_time_in_marks_distilled_only(self):
        write_lora(self.dir, "d", ["a.lora_A.weight", "ref_time_in.lora_A.weight"], meta={"ss_base_model_version": "flux3"})
        (entry,) = loras.list_loras_ng()
        self.assertTrue(entry["distilled_only"])

    def test_options_resolve_names_and_refuse_the_unknown_and_the_wrong_variant(self):
        write_lora(self.dir, "a", FLUX_KEYS)
        write_lora(self.dir, "d", FLUX_KEYS + ["ref_time_in.lora_A.weight"])
        ok = engine.validate_options_ng({"loras": [{"name": "a", "scale": 1.2}]}, "base", [])
        self.assertEqual(ok["loras"], [{"name": "a", "scale": 1.2}])
        for bad, variant in (([{"name": "../a"}], "distilled"), ([{"name": "a", "scale": 9}], "distilled"),
                             ([{"name": "d"}], "base"), ([{"name": "a"}] * 5, "distilled")):
            with self.assertRaises(ValueError):
                engine.validate_options_ng({"loras": bad}, variant, [])
        self.assertEqual(engine.validate_options_ng({"loras": [{"name": "d"}]}, "distilled", [])["loras"][0]["scale"], 1.0)

    def test_worker_passes_each_lora_with_its_scale(self):
        a = worker.cli_args({"model": "/m", "variant": "distilled", "prompt": "p", "width": 256, "height": 256,
                             "refs": [], "loras": [["/x/a.safetensors", 1.5]]}, 1, Path("o.png"))
        self.assertEqual(a[a.index("--lora") + 1: a.index("--lora") + 3], ["/x/a.safetensors", "1.5"])


class Plan(unittest.TestCase):
    def test_canvas_is_a_multiple_of_16_and_keeps_the_offset(self):
        p = op.plan_ng(500, 300, 100, 0, 100, 0)
        self.assertEqual(p["offset"], (100, 0))
        self.assertEqual(p["canvas"][0] % 16, 0)
        self.assertGreaterEqual(p["canvas"][0], 700)
        self.assertEqual(p["scale"], 1.0)

    def test_big_padding_scales_everything_down_to_one_megapixel(self):
        p = op.plan_ng(2000, 1000, 1000, 500, 1000, 500)
        self.assertLessEqual(p["canvas"][0] * p["canvas"][1], op.MAX_PIXELS)
        self.assertLess(p["scale"], 1.0)

    def test_refuses_no_padding_and_negative_padding(self):
        for pads in ((0, 0, 0, 0), (-1, 0, 0, 0)):
            with self.assertRaises(ValueError):
                op.plan_ng(100, 100, *pads)

    def test_prompt_starts_with_the_trigger_instruction(self):
        self.assertTrue(op.outpaint_prompt("a quay").startswith(op.INSTRUCTION))
        self.assertTrue(op.outpaint_prompt("a quay").endswith("Scene: a quay"))
        self.assertEqual(op.outpaint_prompt(""), op.INSTRUCTION)


class PasteBack(unittest.TestCase):
    def test_prepare_pads_with_gray_and_paste_restores_the_original(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            Image.new("RGB", (320, 240), (200, 20, 20)).save(d / "in.png")
            plan = op.prepare_ng(d / "in.png", 64, 0, 64, 0, d / "canvas.png", d / "source.png")
            canvas = Image.open(d / "canvas.png")
            self.assertEqual(canvas.getpixel((5, 5)), op.GRAY)
            self.assertEqual(canvas.getpixel((200, 100)), (200, 20, 20))
            Image.new("RGB", canvas.size, (0, 0, 255)).save(d / "result.png")  # the model drew all blue
            op.paste_original_ng(d / "result.png", d / "source.png", plan["offset"])
            out = Image.open(d / "result.png")
            self.assertEqual(out.getpixel((200, 100)), (200, 20, 20))   # the original is back
            self.assertEqual(out.getpixel((5, 100)), (0, 0, 255))       # the new area is the model's
            self.assertEqual(out.size, canvas.size)


PNG_BYTES = io.BytesIO()
Image.new("RGB", (320, 240), (10, 120, 10)).save(PNG_BYTES, "PNG")


class Route(unittest.TestCase):
    def setUp(self):
        import configNG
        from flask import Flask
        from routes.kleinNG import kleinNG_bp
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        folder = Path(self.tmp.name) / "loras"
        folder.mkdir()
        write_lora(folder, "flux-3-image-klein-outpaint", FLUX_KEYS)
        for patcher in (mock.patch.dict(os.environ, {"KLEIN_LORAS_DIR": str(folder)}),
                        mock.patch.object(configNG, "KLEIN_DIR", self.tmp.name),
                        mock.patch("routes.kleinNG.KLEIN_DIR", self.tmp.name),
                        mock.patch.object(jobs, "KLEIN_DIR", self.tmp.name),
                        mock.patch.object(engine, "klein_health_ng", lambda: {"ready": True, "models": ["distilled-q8", "base-q8"]}),
                        mock.patch.object(jobs.h3_jobsNG, "busy_ng", lambda: None)):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.calls = []

        def fake(prompt, n, width, height, output_dir, seed, refs=(), model="", on_log=None, should_stop=None,
                 on_result=None, options=None):
            self.calls.append({"prompt": prompt, "refs": [Path(r).name for r in refs], "options": options,
                               "size": (width, height)})
            os.makedirs(output_dir, exist_ok=True)
            for i in range(n):
                out = Path(output_dir) / f"p{i}.png"
                Image.new("RGB", (width, height), (0, 0, 255)).save(out)
                on_result({"png_path": str(out), "seed": seed + i, "width": width, "height": height, "elapsed_seconds": 0.1})
            return []
        for patcher in (mock.patch.object(engine, "generate_klein_ng", fake),):
            patcher.start()
            self.addCleanup(patcher.stop)
        app = Flask(__name__)
        app.register_blueprint(kleinNG_bp)
        self.client = app.test_client()

    def wait(self, job_id):
        for _ in range(100):
            body = self.client.get(f"/api/ng/klein/status/{job_id}").get_json()
            if body["status"] in ("done", "error", "cancelled"):
                return body
            time.sleep(0.05)
        self.fail("job did not finish")

    def post(self, **fields):
        PNG_BYTES.seek(0)
        data = {"image": (io.BytesIO(PNG_BYTES.getvalue()), "a.png"), **fields}
        return self.client.post("/api/ng/klein/outpaint", data=data, content_type="multipart/form-data")

    def test_outpaint_job_uses_the_lora_the_padded_canvas_and_pastes_the_original_back(self):
        r = self.post(left="64", right="64", scene="a quay", seed="3")
        self.assertEqual(r.status_code, 200, r.get_json())
        body = self.wait(r.get_json()["jobId"])
        self.assertEqual(body["status"], "done", body)
        call = self.calls[-1]
        self.assertEqual(call["refs"], ["padded.png"])
        self.assertEqual(call["options"]["loras"], [{"name": "flux-3-image-klein-outpaint", "scale": 1.5}])
        self.assertTrue(call["options"]["match_ref"])
        self.assertTrue(call["prompt"].endswith("Scene: a quay"))
        out = Image.open(Path(self.tmp.name) / body["id"] / "out" / body["results"][0]["file"])
        self.assertEqual(out.getpixel((200, 100)), (10, 120, 10))  # the original was pasted back over the blue

    def test_refusals(self):
        self.assertEqual(self.post().status_code, 400)                                  # no padding
        self.assertEqual(self.post(left="10", model="base-q8").status_code, 400)        # base model
        r = self.client.post("/api/ng/klein/outpaint", data={"left": "10"}, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 400)                                            # no picture

    def test_health_lists_loras_without_paths(self):
        h = self.client.get("/api/ng/klein/health").get_json()
        self.assertEqual([l["name"] for l in h["loras"]], ["flux-3-image-klein-outpaint"])
        self.assertNotIn("path", h["loras"][0])


if __name__ == "__main__":
    unittest.main()
