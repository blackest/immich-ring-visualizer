"""Unit tests for converting kohya-format H3 LoRAs (h3_lora_convertNG.py): names, folding alpha
into lora_B, a backup, refusals, and the library listing flagging a file as convertible.

Run with: python3 -m unittest tests.test_h3_lora_convertNG -v
"""

import json
import os
import shutil
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h3_lora_convertNG as conv
import h3_loraNG
from unittest import mock


def write_safetensors(path, tensors, metadata=None):
    """tensors: {name: (dtype, ndarray-of-matching-numpy-dtype)}; BF16 passes uint16 bits."""
    header, blobs, offset = {}, [], 0
    for name, (dtype, arr) in tensors.items():
        raw = np.ascontiguousarray(arr).tobytes()
        header[name] = {"dtype": dtype, "shape": list(arr.shape), "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    if metadata:
        header["__metadata__"] = metadata
    raw_header = json.dumps(header, separators=(",", ":")).encode()
    raw_header += b" " * (-len(raw_header) % 8)
    Path(path).write_bytes(struct.pack("<Q", len(raw_header)) + raw_header + b"".join(blobs))


def read_f32(path, name):
    hlen, header = h3_loraNG._header(Path(path))
    with open(path, "rb") as fh:
        return h3_loraNG._tensor_f32(fh, hlen, header[name])


def bf16_bits(arr):
    return (np.asarray(arr, "<f4").view("<u4") >> 16).astype("<u2")


class Convert(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.path = self.tmp / "face.safetensors"
        rng = np.random.default_rng(1)
        self.down = rng.normal(size=(4, 8)).astype("<f4")
        self.up = np.round(rng.normal(size=(6, 4)), 1).astype("<f4")  # exact in bf16 after bf16_bits
        write_safetensors(self.path, {
            "lora_unet_blocks_3_attn_qkv_proj.lora_down.weight": ("F32", self.down),
            "lora_unet_blocks_3_attn_qkv_proj.lora_up.weight": ("F32", self.up),
            "lora_unet_blocks_3_attn_qkv_proj.alpha": ("F32", np.array(2.0, "<f4")),
            "lora_unet_token_refiner_blocks_0_mlp_fc1.lora_down.weight": ("F32", self.down),
            "lora_unet_token_refiner_blocks_0_mlp_fc1.lora_up.weight": ("F32", self.up),
            "lora_unet_unknown_module.lora_down.weight": ("F32", self.down),
            "lora_unet_unknown_module.lora_up.weight": ("F32", self.up),
        }, {"ss_network_dim": "4"})

    def test_names_map_onto_the_runners(self):
        self.assertEqual(conv.target_name_ng("lora_unet_blocks_12_attn_out_proj"), "blocks.12.attn.out_proj")
        self.assertEqual(conv.target_name_ng("lora_unet_token_refiner_blocks_0_mlp_fc2"),
                         "token_refiner.blocks.0.mlp.fc2")
        self.assertIsNone(conv.target_name_ng("lora_unet_blocks_x_attn_qkv_proj"))

    def test_alpha_over_rank_is_folded_into_b_and_the_rest_renamed(self):
        done = conv.convert_kohya_ng(self.path)
        self.assertEqual((done["pairs"], done["dropped"]), (2, 1))
        np.testing.assert_allclose(read_f32(self.path, "blocks.3.attn.qkv_proj.lora_B.weight"), self.up * 0.5)
        np.testing.assert_array_equal(read_f32(self.path, "blocks.3.attn.qkv_proj.lora_A.weight"), self.down)
        # no alpha in the file means alpha = rank, so B is untouched
        np.testing.assert_array_equal(read_f32(self.path, "token_refiner.blocks.0.mlp.fc1.lora_B.weight"), self.up)
        _h, header = h3_loraNG._header(self.path)
        self.assertEqual(header["__metadata__"]["converted_from"], "kohya")
        self.assertEqual(header["__metadata__"]["ss_network_dim"], "4")  # the original metadata stays
        self.assertFalse(any(k.endswith(".lora_down.weight") for k in header))

    def test_bf16_is_rounded_not_truncated(self):
        write_safetensors(self.path, {
            "lora_unet_blocks_0_mlp_fc1.lora_down.weight": ("BF16", bf16_bits(self.down)),
            "lora_unet_blocks_0_mlp_fc1.lora_up.weight": ("BF16", bf16_bits(self.up)),
            "lora_unet_blocks_0_mlp_fc1.alpha": ("F32", np.array(1.0, "<f4"))})
        conv.convert_kohya_ng(self.path)  # rank 4, alpha 1 -> 0.25 (a power of two: exact in bf16)
        np.testing.assert_allclose(read_f32(self.path, "blocks.0.mlp.fc1.lora_B.weight"),
                                   bf16_bits(self.up).astype(np.uint32).__lshift__(16).view("<f4") * 0.25)

    def test_original_is_kept_and_a_second_run_does_not_overwrite_it(self):
        before = self.path.read_bytes()
        conv.convert_kohya_ng(self.path)
        backup = self.tmp / conv.ORIGINALS_DIRNAME / "face.safetensors"
        self.assertEqual(backup.read_bytes(), before)
        self.assertFalse(conv.is_kohya_ng(self.path))
        with self.assertRaisesRegex(ValueError, "not a kohya"):
            conv.convert_kohya_ng(self.path)
        self.assertEqual(backup.read_bytes(), before)

    def test_a_file_with_nothing_convertible_is_refused_and_untouched(self):
        write_safetensors(self.path, {
            "lora_unet_odd.lora_down.weight": ("F32", self.down), "lora_unet_odd.lora_up.weight": ("F32", self.up)})
        before = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "no module"):
            conv.convert_kohya_ng(self.path)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(list(self.tmp.glob("*.kohya-tmp*")), [])


class Listing(unittest.TestCase):
    def test_a_kohya_file_is_listed_as_convertible_and_a_converted_one_as_ok(self):
        import h3_lora_libraryNG as lib
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        down, up = np.ones((2, 4), "<f4"), np.ones((4, 2), "<f4")
        write_safetensors(tmp / "k.safetensors", {
            "lora_unet_blocks_0_mlp_fc1.lora_down.weight": ("F32", down),
            "lora_unet_blocks_0_mlp_fc1.lora_up.weight": ("F32", up)})
        with mock.patch.object(lib, "H3_LORAS_DIR", tmp):
            row = lib.list_h3_loras_ng()[0]
            self.assertEqual((row["ok"], row["convertible"]), (False, True))
            conv.convert_kohya_ng(tmp / "k.safetensors")
            row = lib.list_h3_loras_ng()[0]
            self.assertEqual((row["ok"], row["convertible"]), (True, False))
            self.assertEqual(len(lib.list_h3_loras_ng()), 1)  # the backup folder is not listed


class Route(unittest.TestCase):
    def setUp(self):
        import h3_lora_libraryNG as lib
        import routes.h3_toolsNG as tools
        from flask import Flask
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        down, up = np.ones((2, 4), "<f4"), np.ones((4, 2), "<f4")
        write_safetensors(self.tmp / "k.safetensors", {
            "lora_unet_blocks_0_mlp_fc1.lora_down.weight": ("F32", down),
            "lora_unet_blocks_0_mlp_fc1.lora_up.weight": ("F32", up)})
        write_safetensors(self.tmp / "ok.safetensors", {
            "blocks.0.mlp.fc1.lora_A.weight": ("F32", down), "blocks.0.mlp.fc1.lora_B.weight": ("F32", up)})
        write_safetensors(self.tmp / "diff.safetensors", {
            "blocks.0.attn.to_q.lora_A.weight": ("F32", down)})
        patch = mock.patch.object(lib, "H3_LORAS_DIR", self.tmp)
        patch.start()
        self.addCleanup(patch.stop)
        app = Flask(__name__)
        app.register_blueprint(tools.h3ToolsNG_bp)
        self.client = app.test_client()

    def test_a_kohya_lora_is_converted_and_the_fresh_entry_returned(self):
        r = self.client.post("/api/ng/h3/loras/k/convert")
        self.assertEqual(r.status_code, 200)
        body = r.get_json()
        self.assertEqual((body["pairs"], body["lora"]["ok"], "path" in body["lora"]), (1, True, False))
        self.assertTrue((self.tmp / "_originals" / "k.safetensors").is_file())

    def test_refusals(self):
        self.assertEqual(self.client.post("/api/ng/h3/loras/nope/convert").status_code, 404)
        self.assertEqual(self.client.post("/api/ng/h3/loras/ok/convert").status_code, 400)  # already usable
        r = self.client.post("/api/ng/h3/loras/diff/convert")  # diffusers format: not convertible
        self.assertEqual(r.status_code, 400)
        self.assertEqual(self.client.post("/api/ng/h3/loras/..%2Fk/convert").status_code, 404)


if __name__ == "__main__":
    unittest.main()
