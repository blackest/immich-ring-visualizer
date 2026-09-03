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
import shot_presets


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

    def test_view_prompt_default_uses_relative_hair_clause(self):
        prompt = character_sheet._view_prompt("facing the camera directly")
        self.assertIn("same hair color", prompt)

    def test_view_prompt_explicit_hair_color_names_it(self):
        prompt = character_sheet._view_prompt("facing the camera directly",
                                               hair_color="blonde")
        self.assertIn("blonde hair", prompt)
        self.assertNotIn("same hair color", prompt)

    def test_shot_prompt_hair_color_flows_through_full_clause_path(self):
        # exercises the *other* branch of _shot_prompt (background/
        # expression/style set, so it builds its own prompt rather than
        # delegating to _view_prompt) -- both branches need the same fix.
        spec = shot_presets.ShotSpec("custom_test", "turned to the side",
                                     background="A plain wall.")
        prompt = character_sheet._shot_prompt(spec, hair_color="dark red")
        self.assertIn("dark red hair", prompt)
        self.assertNotIn("same hair color", prompt)

    def test_shot_prompt_identity_lock_false_ignores_hair_color(self):
        # hair_color is an identity clause -- the free-prompt escape
        # hatch (identity_lock=False) must skip it like every other
        # identity clause, not silently append it anyway.
        spec = shot_presets.ShotSpec("custom_test", "turned to the side")
        prompt = character_sheet._shot_prompt(spec, identity_lock=False,
                                              hair_color="blonde")
        self.assertNotIn("blonde", prompt)
        self.assertNotIn("hair", prompt)

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

    def test_generate_character_sheet_each_view_gets_a_distinct_seed(self):
        # Regression test: every shot in a job used to receive the literal
        # identical base_seed (bug -- suppressed pose divergence between
        # shots on this CFG-free, clean-token-conditioned model). Each
        # shot must now get its own seed, deterministically derived from
        # the job's resolved seed plus its position in the shot list.
        character_sheet.create_draft_character("gwen", self._src.name)
        calls = []
        with mock.patch.object(hidream_engine, "generate_hidream",
                                self._stub_generate_hidream(calls)):
            result = character_sheet.generate_character_sheet("gwen", seed=-1)
        seeds_used = [c["base_seed"] for c in calls]
        self.assertEqual(len(calls), 3)  # default view catalogue
        self.assertEqual(len(set(seeds_used)), len(seeds_used),
                          "every view should get its own distinct seed")
        resolved_seed = result["result"]["resolved_seed"]
        self.assertEqual(seeds_used, [resolved_seed + i for i in range(len(seeds_used))])
        # caller's original seed (-1) preserved verbatim in the sidecar,
        # separate from the resolved value actually sent to the engine
        self.assertEqual(result["result"]["seed"], -1)

    def test_generate_character_sheet_explicit_seed_passed_through(self):
        character_sheet.create_draft_character("hank", self._src.name)
        calls = []
        with mock.patch.object(hidream_engine, "generate_hidream",
                                self._stub_generate_hidream(calls)):
            result = character_sheet.generate_character_sheet("hank", seed=424242)
        seeds_used = [c["base_seed"] for c in calls]
        self.assertEqual(seeds_used, [424242 + i for i in range(len(seeds_used))])
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

    def test_generate_character_sheet_extended_preset_skips_anchor_on_angle_shots(self):
        # The extended preset's profile/three-quarter shots have
        # use_anchor=False (shot_presets.py) precisely because a fixed,
        # near-frontal anchor reference was found to drag pose toward
        # frontal along with color -- see character_sheet.py's
        # docstring on ShotSpec.use_anchor. Confirm the job-level
        # anchor_chain=True default still skips the anchor exactly on
        # those shots, and still applies it on the frontal ones.
        character_sheet.create_draft_character("nadia", self._src.name)
        calls = []
        with mock.patch.object(hidream_engine, "generate_hidream",
                                self._stub_generate_hidream(calls)):
            character_sheet.generate_character_sheet("nadia", preset="extended",
                                                      anchor_chain=True)
        by_key = {spec.key: c for spec, c in zip(shot_presets.EXTENDED_PRESET, calls)}
        for spec in shot_presets.EXTENDED_PRESET:
            n_refs = len(by_key[spec.key]["refs"])
            if spec.key == shot_presets.EXTENDED_PRESET[0].key:
                continue  # first shot never has an anchor yet regardless
            if spec.use_anchor:
                self.assertEqual(n_refs, 2,
                    f"{spec.key!r} has use_anchor=True, expected avatar+anchor")
            else:
                self.assertEqual(n_refs, 1,
                    f"{spec.key!r} has use_anchor=False, expected avatar ref only")

    def test_generate_character_sheet_extended_preset_angle_shots_still_get_pose_text(self):
        # use_anchor=False must not affect the *prompt text* -- the pose
        # instruction and identity-lock clauses (including "same hair
        # color") still apply; only the second reference image is
        # withheld.
        character_sheet.create_draft_character("otis", self._src.name)
        calls = []
        with mock.patch.object(hidream_engine, "generate_hidream",
                                self._stub_generate_hidream(calls)):
            character_sheet.generate_character_sheet("otis", preset="extended",
                                                      anchor_chain=True)
        profile_call = next(c for spec, c in
                             zip(shot_presets.EXTENDED_PRESET, calls)
                             if spec.key == "chest_profile_left")
        self.assertIn("left cheek and left ear", profile_call["prompt"])
        self.assertIn("same hair color", profile_call["prompt"])

    def test_generate_character_sheet_left_right_pairs_have_distinct_prompts(self):
        # Regression test for the reported "left and right came out the
        # same" bug: chest_profile_left/right and
        # chest_three_quarter_left/right must produce genuinely different
        # prompt text (not just a differently-placed word), grounded in
        # which cheek/ear is visible rather than the bare word
        # "left"/"right" alone.
        character_sheet.create_draft_character("piper", self._src.name)
        calls = []
        with mock.patch.object(hidream_engine, "generate_hidream",
                                self._stub_generate_hidream(calls)):
            character_sheet.generate_character_sheet("piper", preset="extended",
                                                      anchor_chain=True)
        prompts_by_key = {
            spec.key: c["prompt"]
            for spec, c in zip(shot_presets.EXTENDED_PRESET, calls)
        }
        left_profile = prompts_by_key["chest_profile_left"]
        right_profile = prompts_by_key["chest_profile_right"]
        self.assertNotEqual(left_profile, right_profile)
        # each prompt names the *visible* cheek/ear first (before "hidden
        # from view") -- that's the side that should differ between the
        # pair, so check the "sees the ___ cheek" clause specifically
        # rather than just "cheek" appearing anywhere (both mention both
        # cheeks, since each also says which one is hidden).
        self.assertIn("sees the left cheek and left ear", left_profile)
        self.assertIn("sees the right cheek and right ear", right_profile)
        self.assertNotIn("sees the right cheek", left_profile)
        self.assertNotIn("sees the left cheek", right_profile)

        left_3q = prompts_by_key["chest_three_quarter_left"]
        right_3q = prompts_by_key["chest_three_quarter_right"]
        self.assertNotEqual(left_3q, right_3q)
        self.assertIn("left cheek angled toward camera", left_3q)
        self.assertIn("right cheek angled toward camera", right_3q)

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

    def test_sheet_shot_image_paths_returns_latest_per_shot_in_order(self):
        # Regression-adjacent coverage for the new "Add sheet to ring"
        # feature: sheet_shot_image_paths() is what feeds the ring
        # analysis pipeline, so it needs to (a) pick up shots even
        # without a finished sheet.json, (b) return the LATEST candidate
        # per shot dir (so a rerolled shot doesn't get analyzed against
        # a stale image), and (c) return them in a stable order.
        character_sheet.create_draft_character("nia", self._src.name)
        char_dir = character_sheet._character_dir("nia")

        # No shots yet.
        self.assertEqual(character_sheet.sheet_shot_image_paths("nia"), [])

        # Two shots, one of them (front) with two candidates at
        # different mtimes -- the older one first, matching how a
        # reroll actually produces files (original render, then a
        # later reroll's file with a newer timestamp).
        front_dir = char_dir / "sheet_views" / "front"
        profile_dir = char_dir / "sheet_views" / "profile_left"
        old_front = front_dir / "cand_00_hidream_1000.png"
        new_front = front_dir / "cand_00_hidream_2000.png"
        profile_png = profile_dir / "cand_00_hidream_1500.png"
        _write_tiny_png(old_front)
        _write_tiny_png(new_front)
        _write_tiny_png(profile_png)
        # Force an unambiguous mtime ordering regardless of how fast
        # these three writes actually landed on disk.
        os.utime(old_front, (1000, 1000))
        os.utime(new_front, (2000, 2000))
        os.utime(profile_png, (1500, 1500))

        paths = character_sheet.sheet_shot_image_paths("nia")
        self.assertEqual(len(paths), 2)
        # front sorts before profile_left alphabetically (shot dir order)
        self.assertEqual(Path(paths[0]), new_front)
        self.assertEqual(Path(paths[1]), profile_png)

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
