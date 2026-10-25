"""Smoke tests for ring_viz.py -- the Flask entry point that wires every blueprint.
Importing it must not start a server; the checks guard against a blueprint
being dropped or two blueprints claiming the same URL.

Run with: python3 -m unittest tests.test_ring_viz -v
"""

import os
import sys
import unittest
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ring_viz


class RingVizTests(unittest.TestCase):
    def test_every_imported_blueprint_is_registered(self):
        imported = {v for k, v in vars(ring_viz).items() if k.endswith("_bp") or k.endswith("NG_bp")}
        self.assertGreater(len(imported), 10)
        registered = set(ring_viz.app.blueprints.values())
        self.assertEqual(imported - registered, set())

    def test_blueprint_names_are_unique(self):
        names = [bp.name for bp in {v for k, v in vars(ring_viz).items() if k.endswith("_bp")}]
        self.assertEqual([n for n, c in Counter(names).items() if c > 1], [])

    def test_key_routes_exist(self):
        rules = {r.rule for r in ring_viz.app.url_map.iter_rules()}
        self.assertIn("/", rules)
        self.assertIn("/immichRingNG", rules)
        self.assertTrue(any(r.startswith("/api/ng/generate/") for r in rules))
        self.assertTrue(any(r.startswith("/api/ng/h3/") for r in rules))

    def test_no_two_blueprints_serve_the_same_rule_and_method(self):
        seen = Counter()
        for r in ring_viz.app.url_map.iter_rules():
            for m in r.methods - {"HEAD", "OPTIONS"}:
                seen[(r.rule, m)] += 1
        self.assertEqual([k for k, c in seen.items() if c > 1], [])

    def test_original_index_page_renders(self):
        r = ring_viz.app.test_client().get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"<html", r.data.lower())

    def test_app_is_not_running_on_import(self):
        self.assertFalse(ring_viz.app.debug)


if __name__ == "__main__":
    unittest.main()
