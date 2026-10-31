"""Unit tests for scripts/quantize_pruned_stream.py's planning step (which tensors are quantized, at which width).
The quantization itself needs MLX and the pruned checkpoint.

Run with: python3 -m unittest tests.test_quantize_pruned_streamNG -v
"""

import importlib.util
import unittest
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "quantize_pruned_stream", Path(__file__).resolve().parent.parent / "scripts" / "quantize_pruned_stream.py")
qps = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(qps)


class _QC:
    """The runner's QuantConfig.bits_for contract: core linears take `bits`, AdaLN takes `adaln_bits` or None."""

    def __init__(self, bits, adaln_bits):
        self.bits, self.adaln_bits = bits, adaln_bits

    def bits_for(self, path):
        if ".adaln_proj." in path:
            return self.adaln_bits
        return self.bits if path.endswith(("attn.qkv_proj", "mlp.fc1")) else None


HEADER = {
    "blocks.0.attn.qkv_proj.weight": {"dtype": "BF16", "shape": [128, 128]},
    "blocks.0.adaln_proj.linear.weight": {"dtype": "F32", "shape": [96, 64]},
    "blocks.0.adaln_proj.linear.bias": {"dtype": "F32", "shape": [96]},
    "blocks.0.mlp.fc1.weight": {"dtype": "BF16", "shape": [64, 100]},  # 100 is not a multiple of the group
    "adaln_t_table": {"dtype": "F32", "shape": [1001, 64]},
}


class Plan(unittest.TestCase):
    def kinds(self, bits, adaln_bits):
        return {k: (kind, b) for k, kind, _s, b in qps.plan_ng(HEADER, _QC(bits, adaln_bits))}

    def test_core_and_adaln_take_their_own_widths(self):
        plan = self.kinds(4, 8)
        self.assertEqual(plan["blocks.0.attn.qkv_proj.weight"], ("q", 4))
        self.assertEqual(plan["blocks.0.adaln_proj.linear.weight"], ("q", 8))

    def test_adaln_can_be_left_unquantized(self):
        plan = self.kinds(8, None)
        self.assertEqual(plan["blocks.0.adaln_proj.linear.weight"], ("F32", None))

    def test_biases_tables_and_indivisible_widths_are_copied_unchanged(self):
        plan = self.kinds(8, 8)
        self.assertEqual(plan["blocks.0.adaln_proj.linear.bias"], ("F32", None))
        self.assertEqual(plan["adaln_t_table"], ("F32", None))
        self.assertEqual(plan["blocks.0.mlp.fc1.weight"], ("BF16", None))


if __name__ == "__main__":
    unittest.main()
