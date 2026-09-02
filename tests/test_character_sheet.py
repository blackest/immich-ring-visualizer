"""Non-GPU unit tests for character_sheet.py.

Stubs hidream_engine.generate_hidream so these exercise the bookkeeping
logic (id validation, draft-character lifecycle, seed handling, view
dedup, anchor-chaining, atomic sheet write) without ever touching the
real HiDream subprocess or requiring a GPU. Run with:

    python3 -m unittest tests.test_character_sheet -v

from the repo root.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import character_sheet
import hidream_engine


def _write_tiny_png(path: Path):
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (200, 100, 50)).save(path, format="PNG")


class CharacterSheetTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        character_sheet.EXPORT_DIR = self._tmp.name

        self._src = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        _write_tiny_png(Path(self._src.name))
        self.addCleanup(lambda: os.path.exists(self._src.name) and os.remove(self._src.name))

    # ---- id validation -----------------------------------------------

    def test_safe_id_rejects_path_traversal(self):
        with self.assertRaises(ValueError):
            character_sheet._safe_id("../../etc/passwd")

    def test_safe_id_rejects_slash(self):
        with self.assertRaises(ValueError):
            character_sheet._safe_id("foo/bar")

    def test_safe_id_allows_spaces_and_hyphens(self):
        self.assertEqual(character_sheet._safe_id(" Annie Phosphene "), "Annie Phosphene")
        self.assertEqual(character_sheet._safe_id("frame-12345"), "frame-12345")

    def test_safe_id_rejects_empty(self):
        with self.assertRaises(ValueError):
            character_sheet._safe_id("   ")

    # ---- create_draft_character ---------------------------------------

    def test_create_draft_character_writes_avatar_and_bundle(self):
        bundle = character_sheet.create_draft_character("alice", self._src.name)
        self.assertEqual(bundle["id"], "alice")
        self.assertEqual(bundle["schema"], "ringviz/character_bundle@1")
        self.assertTrue(character_sheet.character_exists("alice"))
        avatar = character_sheet.character_avatar("alice")
        self.assertIsNotNone(avatar)
        self.assertTrue(avatar.is_file())
        # lives under EXPORT_DIR/<name>/character/, alongside where the
        # curated training crops for that name would already sit
        self.assertEqual(avatar.parent, Path(self._tmp.name) / "alice" / "character")

    def test_create_draft_character_defaults_pronoun_and_noun(self):
        bundle = character_sheet.create_draft_character("bob", self._src.name)
        self.assertEqual(bundle["pronoun"], "they")
        self.assertEqual(bundle["subject_noun"], "person")

    def test_create_draft_character_duplicate_raises(self):
        character_sheet.create_draft_character("carol", self._src.name)
        with self.assertRaises(character_sheet.DraftCharacterExistsError):
            character_sheet.create_draft_character("carol", self._src.name)

    def test_create_draft_character_missing_source_raises(self):
        with self.assertRaises(FileNotFoundError):
            character_sheet.create_draft_character("dave", "/no/such/file.png")

    def test_create_draft_character_bad_extension_raises(self):
        bad = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
        bad.write(b"not an image")
        bad.close()
        self.addCleanup(lambda: os.remove(bad.name))
        with self.assertRaises(ValueError):
            character_sheet.create_draft_character("erin", bad.name)

    # ---- prompt content -------------------------------------------------

    def test_view_prompt_pins_identity_attributes(self):
        prompt = character_sheet._view_prompt(
            character_sheet.CHARACTER_SHEET_VIEWS["profile_left"])
        for phrase in ("same face", "same skin tone", "same hair color",
                       "same exact hairstyle", "wearing exactly the same clothes",
                       "side view portrait from the left"):
            self.assertIn(phrase, prompt)

    def test_view_prompt_appends_wardrobe_when_given(self):
        prompt = character_sheet._view_prompt(
            character_sheet.CHARACTER_SHEET_VIEWS["front"], wardrobe="a red jacket")
        self.assertIn("They are wearing a red jacket.", prompt)

    def test_view_prompt_omits_wardrobe_clause_when_blank(self):
        prompt = character_sheet._view_prompt(character_sheet.CHARACTER_SHEET_VIEWS["front"])
        self.assertNotIn("They are wearing", prompt)

    # ---- generate_character_sheet (hidream_engine stubbed) --------------

    def _stub_generate_hidream(self, calls):
        """Returns a fake generate_hidream that records every call and
        writes a real (tiny) PNG so the real PIL compositor has something
        to open."""
        def _fake(prompt, n, width, height, output_dir, base_seed, config,
                  refs=None, on_log=None):
            calls.append({
                "prompt": prompt, "base_seed": base_seed,
                "refs": list(refs or []),
            })
            png = Path(output_dir) / f"cand_00_hidream_{len(calls)}.png"
            _write_tiny_png(png)
            return [{"png_path": str(png), "seed": base_seed,
                     "engine": "hidream-o1-dev-bf16",
                     "width": width, "height": height}]
        return _fake

    def test_generate_character_sheet_unknown_character_raises_lookup(self):
        with self.assertRaises(LookupError):
            character_sheet.generate_character_sheet("nobody")

    def test_generate_character_sheet_unknown_view_raises_value_error(self):
        character_sheet.create_draft_character("finn", self._src.name)
        with self.assertRaises(ValueError):
            character_sheet.generate_character_sheet("finn", views=["not_a_view"])

    def test_generate_character_sheet_no_avatar_raises_file_not_found(self):
        # a bundle with no avatar file next to it (hand-constructed, not
        # via create_draft_character, to isolate this edge case)
        char_dir = Path(self._tmp.name) / "ghost" / "character"
        char_dir.mkdir(parents=True)
        (char_dir / "bundle.json").write_text(json.dumps({
            "schema": "ringviz/character_bundle@1", "id": "ghost",
        }))
        with self.assertRaises(FileNotFoundError):
            character_sheet.generate_character_sheet("ghost")

    def test_generate_character_sheet_shares_one_seed_across_views(self):
        character_sheet.create_draft_character("gwen", self._src.name)
        calls = []
        with mock.patch.object(hidream_engine, "generate_hidream",
                                self._stub_generate_hidream(calls)):
            result = character_sheet.generate_character_sheet("gwen", seed=-1)
        seeds_used = {c["base_seed"] for c in calls}
        self.assertEqual(len(seeds_used), 1, "every view should share one resolved seed")
        self.assertEqual(len(calls), 3)  # default view catalogue
        # caller's original seed (-1) preserved verbatim in the sidecar,
        # separate from the resolved value actually sent to the engine
        self.assertEqual(result["result"]["seed"], -1)
        self.assertEqual(result["result"]["resolved_seed"], next(iter(seeds_used)))

    def test_generate_character_sheet_explicit_seed_passed_through(self):
        character_sheet.create_draft_character("hank", self._src.name)
        calls = []
        with mock.patch.object(hidream_engine, "generate_hidream",
                                self._stub_generate_hidream(calls)):
            result = character_sheet.generate_character_sheet("hank", seed=424242)
        self.assertTrue(all(c["base_seed"] == 424242 for c in calls))
        self.assertEqual(result["result"]["seed"], 424242)
        self.assertEqual(result["result"]["resolved_seed"], 424242)

    def test_generate_character_sheet_anchor_chain_adds_second_ref(self):
        character_sheet.create_draft_character("iris", self._src.name)
        calls = []
        with mock.patch.object(hidream_engine, "generate_hidream",
                                self._stub_generate_hidream(calls)):
            character_sheet.generate_character_sheet("iris", anchor_chain=True)
        self.assertEqual(len(calls[0]["refs"]), 1, "first view: avatar only")
        self.assertEqual(len(calls[1]["refs"]), 2, "later views: avatar + rendered front")
        self.assertEqual(len(calls[2]["refs"]), 2)

    def test_generate_character_sheet_anchor_chain_disabled(self):
        character_sheet.create_draft_character("jack", self._src.name)
        calls = []
        with mock.patch.object(hidream_engine, "generate_hidream",
                                self._stub_generate_hidream(calls)):
            character_sheet.generate_character_sheet("jack", anchor_chain=False)
        self.assertTrue(all(len(c["refs"]) == 1 for c in calls),
                        "anchor_chain=False: every view uses only the avatar ref")

    def test_generate_character_sheet_writes_sheet_png_and_sidecar_json(self):
        character_sheet.create_draft_character("kim", self._src.name)
        calls = []
        with mock.patch.object(hidream_engine, "generate_hidream",
                                self._stub_generate_hidream(calls)):
            result = character_sheet.generate_character_sheet("kim")
        sheet_png = Path(result["sheet_path"])
        self.assertTrue(sheet_png.is_file())
        sheet_json = sheet_png.with_name("sheet.json")
        self.assertTrue(sheet_json.is_file())
        meta = json.loads(sheet_json.read_text())
        self.assertEqual(meta["schema"], "ringviz/character_sheet@2")
        self.assertEqual(len(meta["views"]), 3)
        self.assertEqual(character_sheet.character_sheet_png("kim"), sheet_png)

    def test_generate_character_sheet_busy_when_lock_held(self):
        character_sheet.create_draft_character("liam", self._src.name)
        character_sheet._SHEET_LOCK.acquire()
        try:
            with self.assertRaises(character_sheet.CharacterSheetBusyError):
                character_sheet.generate_character_sheet("liam")
        finally:
            character_sheet._SHEET_LOCK.release()

    def test_generate_character_sheet_dedupes_and_validates_views(self):
        character_sheet.create_draft_character("mona", self._src.name)
        calls = []
        with mock.patch.object(hidream_engine, "generate_hidream",
                                self._stub_generate_hidream(calls)):
            character_sheet.generate_character_sheet(
                "mona", views=["front", "front", "profile_left"])
        self.assertEqual(len(calls), 2, "duplicate view name should not double-render")


    # ---- shot-list presets / style / identity-lock -----------------------

    def test_resolve_shots_extended_preset_has_fifteen_unique_shots(self):
        shots = character_sheet.resolve_shots(preset="extended")
        self.assertEqual(len(shots), 15)
        self.assertEqual(len({s.key for s in shots}), 15)

    def test_resolve_shots_unknown_preset_raises(self):
        with self.assertRaises(ValueError):
            character_sheet.resolve_shots(preset="not_a_preset")

    def test_generate_character_sheet_extended_preset_skips_composite(self):
        character_sheet.create_draft_character("olga", self._src.name)
        calls = []
        with mock.patch.object(hidream_engine, "generate_hidream",
                                self._stub_generate_hidream(calls)):
            result = character_sheet.generate_character_sheet("olga", preset="extended")
        self.assertEqual(len(calls), 15)
        self.assertIsNone(result["sheet_path"])
        self.assertIsNone(result["result"]["sheet_png"])
        self.assertEqual(len(result["result"]["views"]), 15)

    def test_generate_character_sheet_default_preset_still_composites(self):
        character_sheet.create_draft_character("pia", self._src.name)
        calls = []
        with mock.patch.object(hidream_engine, "generate_hidream",
                                self._stub_generate_hidream(calls)):
            result = character_sheet.generate_character_sheet("pia")  # preset="default"
        self.assertIsNotNone(result["sheet_path"])
        self.assertTrue(Path(result["sheet_path"]).is_file())

    def test_identity_lock_false_omits_identity_clauses(self):
        spec = character_sheet.ShotSpec("x", "a front portrait")
        locked = character_sheet._shot_prompt(spec, identity_lock=True)
        unlocked = character_sheet._shot_prompt(spec, identity_lock=False)
        self.assertIn("Keep this person exactly", locked)
        self.assertNotIn("Keep this person exactly", unlocked)
        self.assertIn("a front portrait", unlocked)

    def test_style_clause_appears_when_set(self):
        spec = character_sheet.ShotSpec("x", "a front portrait")
        prompt = character_sheet._shot_prompt(spec, style="cinematic")
        self.assertIn("Cinematic color grading", prompt)

    def test_style_none_matches_plain_default_output(self):
        spec = character_sheet.ShotSpec("x", "a front portrait")
        self.assertEqual(
            character_sheet._shot_prompt(spec, style="none"),
            character_sheet._view_prompt("a front portrait"))

    # ---- regenerate_shot -------------------------------------------------

    def test_regenerate_shot_no_sheet_yet_raises_file_not_found(self):
        character_sheet.create_draft_character("quinn", self._src.name)
        with self.assertRaises(FileNotFoundError):
            character_sheet.regenerate_shot("quinn", "front")

    def test_regenerate_shot_unknown_key_raises_lookup(self):
        character_sheet.create_draft_character("ruth", self._src.name)
        calls = []
        with mock.patch.object(hidream_engine, "generate_hidream",
                                self._stub_generate_hidream(calls)):
            character_sheet.generate_character_sheet("ruth")
        with self.assertRaises(LookupError):
            character_sheet.regenerate_shot("ruth", "not_a_shot")

    def test_regenerate_shot_reuses_refs_and_updates_sheet(self):
        character_sheet.create_draft_character("sam", self._src.name)
        calls = []
        with mock.patch.object(hidream_engine, "generate_hidream",
                                self._stub_generate_hidream(calls)):
            first = character_sheet.generate_character_sheet("sam")
        original_refs = first["result"]["views"][1]["refs"]  # profile_left: avatar + front anchor
        self.assertEqual(len(original_refs), 2)

        reroll_calls = []
        with mock.patch.object(hidream_engine, "generate_hidream",
                                self._stub_generate_hidream(reroll_calls)):
            rerolled = character_sheet.regenerate_shot("sam", "profile_left", seed=999)
        self.assertEqual(len(reroll_calls), 1)
        self.assertEqual(reroll_calls[0]["base_seed"], 999)
        self.assertEqual(reroll_calls[0]["refs"], original_refs)
        new_views = rerolled["result"]["views"]
        self.assertEqual(len(new_views), 3)  # still a 3-shot sheet
        self.assertTrue(Path(rerolled["sheet_path"]).is_file())

if __name__ == "__main__":
    unittest.main()
