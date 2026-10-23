"""Unit tests for character_sheetNG.py -- character storage, prompt
building, shot resolution, the composite strip and the generate / reroll
flows.

The HiDream engine is mocked at generate_hidream_ng; EXPORT_DIR is pointed
at a temp dir and PNGs are real (tiny) files so the composite runs for real.

Run with: python3 -m unittest tests.test_character_sheetNG -v
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image

import character_sheetNG as cs
import shot_presetsNG as sp
from shot_presetsNG import ShotSpec


def _png(path, size=(20, 10), colour=(200, 50, 50)):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, colour).save(path, format="PNG")
    return path


class _TmpExportCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        p = mock.patch.object(cs, "EXPORT_DIR", self.root)
        p.start()
        self.addCleanup(p.stop)

    def _register(self, name="tom", avatar=True):
        src = _png(self.root / "_src.png")
        if avatar:
            cs.create_draft_character_ng(name, str(src))
        else:
            d = self.root / name / "character"
            d.mkdir(parents=True)
            (d / "bundle.json").write_text("{}")
        return name


class ValidationTests(unittest.TestCase):
    def test_render_params_accept_ints_and_numeric_strings(self):
        self.assertEqual(cs._validate_render_params_ng("512", 1024.0, "10"), (512, 1024, 10))

    def test_render_params_reject_bad_values(self):
        for args in [("x", 512, 10), (None, 512, 10), (255, 512, 10), (512, 4097, 10),
                     (512, 512, 0), (512, 512, 101)]:
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    cs._validate_render_params_ng(*args)

    def test_render_params_bounds_are_inclusive(self):
        self.assertEqual(cs._validate_render_params_ng(256, 4096, 1), (256, 4096, 1))
        self.assertEqual(cs._validate_render_params_ng(4096, 256, 100), (4096, 256, 100))

    def test_safe_id(self):
        self.assertEqual(cs._safe_id_ng("  Mary Jane_2-b "), "Mary Jane_2-b")
        for bad in ("", "   ", None, "../evil", "a/b", "x.y", "naïve"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    cs._safe_id_ng(bad)


class PromptTests(unittest.TestCase):
    def test_view_prompt_defaults_and_overrides(self):
        base = cs._view_prompt_ng("a front portrait")
        self.assertIn("same hair color", base)
        self.assertIn("a front portrait", base)
        self.assertNotIn("wearing", base.split("photorealistic.")[-1])
        coloured = cs._view_prompt_ng("a front portrait", wardrobe="red coat", hair_color="blonde")
        self.assertIn("blonde hair", coloured)
        self.assertNotIn("same hair color", coloured)
        self.assertTrue(coloured.endswith("They are wearing red coat."))

    def test_plain_shot_is_byte_identical_to_view_prompt(self):
        spec = ShotSpec("front", "a front portrait")
        self.assertEqual(cs._shot_prompt_ng(spec, "red coat", hair_color="dark red"),
                         cs._view_prompt_ng("a front portrait", "red coat", "dark red"))

    def test_shot_wardrobe_beats_job_wardrobe(self):
        spec = ShotSpec("front", "p", wardrobe="a kilt")
        self.assertIn("wearing a kilt", cs._shot_prompt_ng(spec, "red coat"))
        self.assertNotIn("red coat", cs._shot_prompt_ng(spec, "red coat"))

    def test_prompt_override_replaces_pose(self):
        spec = ShotSpec("x", "original pose", prompt_override="my own pose")
        prompt = cs._shot_prompt_ng(spec)
        self.assertIn("my own pose", prompt)
        self.assertNotIn("original pose", prompt)

    def test_extended_form_uses_expression_background_and_style(self):
        spec = ShotSpec("x", "a pose", expression="smiling", background="A beach")
        prompt = cs._shot_prompt_ng(spec, style="cinematic")
        self.assertIn("a pose, smiling,", prompt)
        self.assertIn("A beach, photorealistic.", prompt)
        self.assertIn(sp.STYLE_PRESETS["cinematic"], prompt)
        self.assertNotIn("Neutral seamless studio", prompt)

    def test_style_alone_leaves_plain_path_and_gets_default_background(self):
        prompt = cs._shot_prompt_ng(ShotSpec("x", "a pose"), style="dramatic")
        self.assertIn("Neutral seamless studio background", prompt)
        self.assertIn(sp.STYLE_PRESETS["dramatic"], prompt)

    def test_identity_lock_off_is_bare_pose_text(self):
        spec = ShotSpec("x", "just the pose", background="A beach", wardrobe="a hat")
        prompt = cs._shot_prompt_ng(spec, identity_lock=False, style="warm_film")
        self.assertTrue(prompt.startswith("just the pose A beach."))
        self.assertNotIn("Keep this person", prompt)
        self.assertIn("They are wearing a hat.", prompt)
        self.assertIn(sp.STYLE_PRESETS["warm_film"], prompt)


class ShotResolutionTests(unittest.TestCase):
    def test_preset_by_name_and_unknown_preset(self):
        self.assertEqual([s.key for s in cs.resolve_shots_ng()], ["front", "profile_left", "three_quarter"])
        self.assertEqual(len(cs.resolve_shots_ng(preset="extended")), 15)
        with self.assertRaisesRegex(ValueError, "unknown preset"):
            cs.resolve_shots_ng(preset="nope")

    def test_explicit_shots_win_and_dedupe_by_key(self):
        a, b, a2 = ShotSpec("a"), ShotSpec("b"), ShotSpec("a", "different")
        got = cs.resolve_shots_ng(shots=[a, b, a2], views=["front"], preset="extended")
        self.assertEqual(got, [a, b])

    def test_shots_must_be_nonempty_shotspecs(self):
        for bad in ([], ["front"], "front"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    cs.resolve_shots_ng(shots=bad)

    def test_views_resolve_across_presets_in_order_deduped(self):
        extended_key = sp.EXTENDED_PRESET[5].key
        got = cs.resolve_shots_ng(views=[" front ", extended_key, "front"])
        self.assertEqual([s.key for s in got], ["front", extended_key])

    def test_views_errors(self):
        with self.assertRaisesRegex(ValueError, "unknown views: bogus"):
            cs.resolve_shots_ng(views=["front", "bogus"])
        for bad in ([], ["", "  "], [1], "front"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    cs.resolve_shots_ng(views=bad)

    def test_shot_by_key(self):
        self.assertEqual(cs._shot_by_key_ng("front").key, "front")
        with self.assertRaises(ValueError):
            cs._shot_by_key_ng("nope")


class CustomShotTests(unittest.TestCase):
    def test_safe_shot_key(self):
        self.assertEqual(cs._safe_shot_key_ng("Over the shoulder!"), "custom_over_the_shoulder")
        self.assertEqual(cs._safe_shot_key_ng("custom_foo"), "custom_foo")      # idempotent
        self.assertEqual(cs._safe_shot_key_ng(cs._safe_shot_key_ng("A B")), "custom_a_b")
        self.assertEqual(cs._safe_shot_key_ng("!!!"), "custom_pose")
        self.assertEqual(cs._safe_shot_key_ng(None), "custom_pose")
        self.assertLessEqual(len(cs._safe_shot_key_ng("x" * 200)), len("custom_") + 40)
        self.assertEqual(cs._safe_shot_key_ng("front"), "custom_front")       # never collides with a preset

    def test_nothing_custom_returns_none(self):
        self.assertIsNone(cs.build_custom_shot_ng())
        self.assertIsNone(cs.build_custom_shot_ng(views=["front"]))
        self.assertIsNone(cs.build_custom_shot_ng(views=["front", "profile_left"], prompt_override="x"))

    def test_custom_prompt_beats_everything(self):
        spec = cs.build_custom_shot_ng(custom_prompt=" free text ", custom_pose="pose", views=["front"])
        self.assertEqual((spec.key, spec.prompt_override), ("custom", "free text"))

    def test_custom_pose_with_override_and_scene(self):
        spec = cs.build_custom_shot_ng(custom_pose="Looking up", prompt_override="looking way up",
                                       scene="outdoor")
        self.assertEqual(spec.key, "custom_looking_up")
        self.assertEqual(spec.pose_phrase, "looking way up")
        self.assertEqual(spec.background, sp.SCENES["outdoor"])
        plain = cs.build_custom_shot_ng(custom_pose="Looking up", custom_key="custom_k")
        self.assertEqual((plain.key, plain.pose_phrase, plain.background), ("custom_k", "Looking up", ""))

    def test_single_view_edit_keeps_base_fields(self):
        base = cs._shot_by_key_ng("profile_left")
        spec = cs.build_custom_shot_ng(views=["profile_left"], scene="interior")
        self.assertEqual((spec.key, spec.pose_phrase), ("profile_left", base.pose_phrase))
        self.assertEqual(spec.background, sp.SCENES["interior"])
        self.assertFalse(spec.use_anchor)                      # carried over from the base shot
        edited = cs.build_custom_shot_ng(views=["front"], prompt_override="a new pose")
        self.assertEqual(edited.pose_phrase, "a new pose")

    def test_unknown_scene_means_no_override(self):
        self.assertIsNone(cs.build_custom_shot_ng(views=["front"], scene="nowhere"))

    def test_preview_matches_what_would_render(self):
        got = cs.preview_shot_prompt_ng(views=["front"], wardrobe="red coat", hair_color="blonde")
        self.assertEqual(got["key"], "front")
        self.assertEqual(got["prompt"], cs._shot_prompt_ng(cs._shot_by_key_ng("front"), "red coat",
                                                           hair_color="blonde"))
        custom = cs.preview_shot_prompt_ng(custom_prompt="raw", identity_lock=False)
        self.assertEqual(custom, {"key": "custom", "prompt": "raw"})
        self.assertEqual(cs.preview_shot_prompt_ng()["key"], "front")     # first shot of the preset
        with self.assertRaises(ValueError):
            cs.preview_shot_prompt_ng(views=["bogus"])


class CompositeTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_no_images_raises(self):
        with self.assertRaises(RuntimeError):
            cs._compose_sheet_row_ng([], self.dir / "o.png")
        with self.assertRaises(RuntimeError):
            cs._compose_sheet_row_ng([None, str(self.dir / "missing.png")], self.dir / "o.png")

    def test_single_image_is_saved_unscaled(self):
        a = _png(self.dir / "a.png", (30, 20))
        out = cs._compose_sheet_row_ng([a], self.dir / "sub" / "o.png")
        with Image.open(out) as im:
            self.assertEqual(im.size, (30, 20))

    def test_strip_height_width_and_gutters(self):
        a = _png(self.dir / "a.png", (50, 100))     # -> 512 x 1024
        b = _png(self.dir / "b.png", (100, 100))    # -> 1024 x 1024
        out = cs._compose_sheet_row_ng([a, b, self.dir / "gone.png"], self.dir / "o.png")
        with Image.open(out) as im:
            w, h = im.size
        self.assertEqual(h, 1024 + 24)
        self.assertEqual(w, 512 + 1024 + 12 * 3)


class StorageTests(_TmpExportCase):
    def test_create_draft_writes_avatar_and_bundle_with_defaults(self):
        src = _png(self.root / "pic.JPG")
        bundle = cs.create_draft_character_ng("tom", str(src))
        self.assertEqual((bundle["id"], bundle["pronoun"], bundle["subject_noun"]), ("tom", "they", "person"))
        self.assertTrue(cs.character_exists_ng("tom"))
        self.assertEqual(cs.character_avatar_ng("tom").name, "avatar.jpg")   # extension lowered
        self.assertEqual(cs.character_bundle_ng("tom"), bundle)

    def test_create_draft_custom_pronoun_and_noun(self):
        src = _png(self.root / "pic.png")
        b = cs.create_draft_character_ng("tom", str(src), pronoun=" he ", subject_noun=" man ")
        self.assertEqual((b["pronoun"], b["subject_noun"]), ("he", "man"))

    def test_create_draft_rejections(self):
        src = _png(self.root / "pic.png")
        cs.create_draft_character_ng("tom", str(src))
        with self.assertRaises(cs.DraftCharacterExistsError):
            cs.create_draft_character_ng("tom", str(src))
        with self.assertRaises(FileNotFoundError):
            cs.create_draft_character_ng("ann", str(self.root / "nope.png"))
        gif = self.root / "x.gif"
        gif.write_bytes(b"x")
        with self.assertRaisesRegex(ValueError, "unsupported image type"):
            cs.create_draft_character_ng("ann", str(gif))
        empty = self.root / "e.png"
        empty.write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "must be > 0"):
            cs.create_draft_character_ng("ann", str(empty))
        with mock.patch.object(cs, "MAX_BYTES_PER_IMAGE", 10):
            with self.assertRaisesRegex(ValueError, "must be > 0"):
                cs.create_draft_character_ng("ann", str(src))
        with self.assertRaises(ValueError):
            cs.create_draft_character_ng("../bad", str(src))
        self.assertFalse(cs.character_exists_ng("ann"))

    def test_missing_things_return_empty_values(self):
        self.assertEqual(cs.character_bundle_ng("ghost"), {})
        self.assertIsNone(cs.character_avatar_ng("ghost"))
        self.assertFalse(cs.character_exists_ng("ghost"))
        self.assertEqual(cs.character_sheet_meta_ng("ghost"), {})
        self.assertIsNone(cs.character_sheet_png_ng("ghost"))
        self.assertEqual(cs.sheet_shot_image_paths_ng("ghost"), [])

    def test_corrupt_json_is_treated_as_missing(self):
        self._register()
        cdir = self.root / "tom" / "character"
        (cdir / "bundle.json").write_text("{nope")
        (cdir / "sheet.json").write_text("{nope")
        (self.root / "tom" / "character.json").write_text("{nope")
        self.assertEqual(cs.character_bundle_ng("tom"), {})
        self.assertEqual(cs.character_sheet_meta_ng("tom"), {})
        self.assertIsNone(cs.load_character_project_doc_ng("tom"))

    def test_sheet_png_follows_sheet_json_field(self):
        self._register()
        cdir = self.root / "tom" / "character"
        png = _png(cdir / "sheet.png")
        (cdir / "sheet.json").write_text(json.dumps({"sheet_png": str(png)}))
        self.assertEqual(cs.character_sheet_png_ng("tom"), png)
        (cdir / "sheet.json").write_text(json.dumps({"sheet_png": None}))   # too many shots: no composite
        self.assertIsNone(cs.character_sheet_png_ng("tom"))
        (cdir / "sheet.json").write_text(json.dumps({"sheet_png": str(cdir / "gone.png")}))
        self.assertIsNone(cs.character_sheet_png_ng("tom"))

    def test_shot_image_paths_pick_latest_candidate_per_shot(self):
        self._register()
        views = self.root / "tom" / "character" / "sheet_views"
        old, new = _png(views / "front" / "cand_1.png"), _png(views / "front" / "cand_2.png")
        os.utime(old, (200, 200))
        os.utime(new, (100, 100))                     # older mtime than "cand_1" on purpose
        _png(views / "side" / "cand_1.png")
        (views / "empty").mkdir()
        (views / "stray.txt").write_text("x")
        got = cs.sheet_shot_image_paths_ng("tom")
        self.assertEqual(got, [str(views / "front" / "cand_1.png"), str(views / "side" / "cand_1.png")])

    def test_project_doc_roundtrip_and_name_check(self):
        cs.save_character_project_doc_ng("mary", {"identity": {"name": "Mary"}, "note": "é"})
        self.assertEqual(cs.load_character_project_doc_ng("mary")["note"], "é")
        with self.assertRaises(ValueError):
            cs.save_character_project_doc_ng("../x", {})
        self.assertIsNone(cs.load_character_project_doc_ng("nobody"))

    def test_delete_character(self):
        self._register()
        self.assertTrue(cs.delete_character_ng("tom"))
        self.assertFalse((self.root / "tom").exists())
        self.assertFalse(cs.delete_character_ng("tom"))
        with self.assertRaises(ValueError):
            cs.delete_character_ng("../")


class ListCharactersTests(_TmpExportCase):
    def test_empty_or_missing_root(self):
        with mock.patch.object(cs, "EXPORT_DIR", self.root / "absent"):
            self.assertEqual(cs.list_characters_ng(), [])
        self.assertEqual(cs.list_characters_ng(), [])

    def test_merges_bundle_and_project_doc_and_sorts_newest_first(self):
        self._register("tom")
        cs.save_character_project_doc_ng("tom", {"identity": {"name": "Tommy"}, "task": "generate",
                                                 "savedAt": "2026-01-01"})
        cs.save_character_project_doc_ng("docsonly", {"savedAt": "2026-02-01"})
        self._register("bundleonly")
        _png(self.root / "tom" / "character" / "sheet_views" / "front" / "cand_1.png")
        got = {c["id"]: c for c in cs.list_characters_ng()}
        self.assertEqual([c["id"] for c in cs.list_characters_ng()], ["docsonly", "tom", "bundleonly"])
        self.assertEqual(got["tom"]["name"], "Tommy")                   # project doc name wins
        self.assertEqual((got["tom"]["shotCount"], got["tom"]["hasAvatar"], got["tom"]["task"]),
                         (1, True, "generate"))
        self.assertEqual(got["bundleonly"]["name"], "bundleonly")
        self.assertFalse(got["bundleonly"]["hasProjectDoc"])
        self.assertEqual((got["docsonly"]["shotCount"], got["docsonly"]["hasAvatar"]), (0, False))

    def test_skips_unknown_dirs_files_and_unsafe_names(self):
        (self.root / "empty").mkdir()
        (self.root / "bad.name").mkdir()
        (self.root / "stray.txt").write_text("x")
        self.assertEqual(cs.list_characters_ng(), [])


class _FakeEngine:
    """Stands in for generate_hidream_ng: writes a real PNG per call."""
    def __init__(self, fail_on=None):
        self.calls = []
        self.fail_on = fail_on

    def __call__(self, **kw):
        self.calls.append(kw)
        if self.fail_on is not None and len(self.calls) == self.fail_on:
            return []
        png = _png(Path(kw["output_dir"]) / f"cand_{kw['base_seed']}.png", (kw["width"] // 64, kw["height"] // 64))
        return [{"png_path": str(png), "seed": kw["base_seed"], "engine": "fake",
                 "width": kw["width"], "height": kw["height"]}]


class GenerateTests(_TmpExportCase):
    def setUp(self):
        super().setUp()
        self._register()
        self.engine = _FakeEngine()
        for p in (mock.patch.object(cs.hidream_engineNG, "generate_hidream_ng", self.engine),
                  mock.patch.object(cs.hidream_engineNG, "HiDreamConfig",
                                    side_effect=lambda steps: {"steps": steps})):
            p.start()
            self.addCleanup(p.stop)
        self.cdir = self.root / "tom" / "character"

    def test_default_sheet_seeds_refs_and_outputs(self):
        logs = []
        out = cs.generate_character_sheet_ng("tom", seed=100, width=512, height=256, steps=5,
                                             on_log=logs.append)
        self.assertTrue(out["ok"])
        self.assertEqual([c["base_seed"] for c in self.engine.calls], [100, 101, 102])  # seed + position
        avatar = str(self.cdir / "avatar.png")
        first, second, third = (c["refs"] for c in self.engine.calls)
        self.assertEqual(first, [avatar])
        anchor = out["result"]["views"][0]["png_path"]
        self.assertEqual(second, [avatar])         # profile_left: use_anchor False
        self.assertEqual(third, [avatar])
        self.assertEqual(self.engine.calls[0]["config"], {"steps": 5})
        self.assertTrue(self.engine.calls[0]["allow_offspec_res"])
        self.assertEqual(logs[0], "[sheet] tom: shot 1/3 (front)")
        self.assertTrue(Path(out["sheet_path"]).is_file())
        meta = json.loads((self.cdir / "sheet.json").read_text())
        self.assertEqual((meta["preset"], meta["resolved_seed"], len(meta["views"])), ("default", 100, 3))
        self.assertEqual(meta["sheet_png"], out["sheet_path"])
        self.assertTrue(anchor)

    def test_anchor_chain_adds_first_render_to_anchor_shots_only(self):
        shots = [ShotSpec("a"), ShotSpec("b"), ShotSpec("c", use_anchor=False)]
        out = cs.generate_character_sheet_ng("tom", shots=shots, seed=1)
        avatar = str(self.cdir / "avatar.png")
        anchor = out["result"]["views"][0]["png_path"]
        self.assertEqual([c["refs"] for c in self.engine.calls],
                         [[avatar], [avatar, anchor], [avatar]])
        self.assertEqual(out["result"]["preset"], "custom")

    def test_anchor_chain_off(self):
        cs.generate_character_sheet_ng("tom", shots=[ShotSpec("a"), ShotSpec("b")], anchor_chain=False, seed=1)
        self.assertEqual(len(self.engine.calls[1]["refs"]), 1)

    def test_negative_seed_picks_a_random_one(self):
        with mock.patch.object(cs.random, "randint", return_value=777):
            out = cs.generate_character_sheet_ng("tom", seed=-1, shots=[ShotSpec("a")])
        self.assertEqual(self.engine.calls[0]["base_seed"], 777)
        self.assertEqual((out["result"]["seed"], out["result"]["resolved_seed"]), (-1, 777))

    def test_prompt_uses_wardrobe_hair_and_style(self):
        cs.generate_character_sheet_ng("tom", shots=[ShotSpec("a", "a pose")], seed=1,
                                       wardrobe=" red coat ", hair_color="blonde", style="dramatic")
        prompt = self.engine.calls[0]["prompt"]
        for needle in ("red coat", "blonde hair", sp.STYLE_PRESETS["dramatic"]):
            self.assertIn(needle, prompt)

    def test_many_shots_skip_composite_and_clear_stale_one(self):
        _png(self.cdir / "sheet.png")
        shots = [ShotSpec(f"s{i}") for i in range(cs.MAX_SHOTS_FOR_COMPOSITE + 1)]
        out = cs.generate_character_sheet_ng("tom", shots=shots, seed=1)
        self.assertIsNone(out["sheet_path"])
        self.assertFalse((self.cdir / "sheet.png").exists())
        self.assertIsNone(out["result"]["sheet_png"])

    def test_validation_runs_before_any_engine_work(self):
        cases = [
            (dict(seed="abc"), ValueError), (dict(width=10), ValueError),
            (dict(preset="nope"), ValueError),
        ]
        for kw, exc in cases:
            with self.subTest(kw=kw):
                with self.assertRaises(exc):
                    cs.generate_character_sheet_ng("tom", **kw)
        with self.assertRaises(LookupError):
            cs.generate_character_sheet_ng("ghost")
        (self.cdir / "avatar.png").unlink()
        with self.assertRaises(FileNotFoundError):
            cs.generate_character_sheet_ng("tom")
        self.assertEqual(self.engine.calls, [])

    def test_engine_returning_nothing_is_an_error_and_writes_no_sheet(self):
        engine = _FakeEngine(fail_on=2)
        with mock.patch.object(cs.hidream_engineNG, "generate_hidream_ng", engine):
            with self.assertRaisesRegex(RuntimeError, "no image for shot 'profile_left'"):
                cs.generate_character_sheet_ng("tom", seed=1)
        self.assertFalse((self.cdir / "sheet.json").exists())
        self.assertFalse((self.cdir / "sheet.png").exists())

    def test_lock_is_released_after_a_failure(self):
        with mock.patch.object(cs.hidream_engineNG, "generate_hidream_ng", _FakeEngine(fail_on=1)):
            with self.assertRaises(RuntimeError):
                cs.generate_character_sheet_ng("tom", seed=1)
        self.assertTrue(cs._SHEET_LOCK_NG.acquire(blocking=False))
        cs._SHEET_LOCK_NG.release()

    def test_composite_failure_cleans_up_temp_file(self):
        with mock.patch.object(cs, "_compose_sheet_row_ng", side_effect=RuntimeError("pil died")):
            with self.assertRaisesRegex(RuntimeError, "pil died"):
                cs.generate_character_sheet_ng("tom", seed=1, shots=[ShotSpec("a")])
        self.assertEqual([p.name for p in self.cdir.glob(".sheet.*")], [])


class RegenerateTests(_TmpExportCase):
    def setUp(self):
        super().setUp()
        self._register()
        self.engine = _FakeEngine()
        for p in (mock.patch.object(cs.hidream_engineNG, "generate_hidream_ng", self.engine),
                  mock.patch.object(cs.hidream_engineNG, "HiDreamConfig",
                                    side_effect=lambda steps: {"steps": steps})):
            p.start()
            self.addCleanup(p.stop)
        self.cdir = self.root / "tom" / "character"
        cs.generate_character_sheet_ng("tom", seed=10, width=512, height=256)
        self.engine.calls.clear()

    def test_reroll_reuses_prompt_refs_and_previous_size(self):
        before = json.loads((self.cdir / "sheet.json").read_text())
        out = cs.regenerate_shot_ng("tom", "profile_left", seed=55)
        call = self.engine.calls[0]
        self.assertEqual(call["prompt"], before["views"][1]["prompt"])
        self.assertEqual(call["refs"], before["views"][1]["refs"])
        self.assertEqual((call["width"], call["height"], call["base_seed"]), (512, 256, 55))
        self.assertEqual(call["config"], {"steps": cs.DEFAULT_RENDER_STEPS})
        meta = out["result"]
        self.assertEqual(meta["views"][1]["seed"], 55)
        self.assertNotEqual(meta["views"][1]["png_path"], before["views"][1]["png_path"])
        self.assertEqual(meta["views"][0], before["views"][0])           # others untouched
        self.assertEqual(meta["last_reroll"]["shot_key"], "profile_left")
        self.assertIn("updated_at", meta)
        self.assertEqual(json.loads((self.cdir / "sheet.json").read_text())["views"][1]["seed"], 55)
        self.assertTrue(Path(out["sheet_path"]).is_file())

    def test_reroll_overrides_prompt_size_steps(self):
        cs.regenerate_shot_ng("tom", "front", seed=1, prompt="new prompt", width=1024, height=512, steps=7)
        call = self.engine.calls[0]
        self.assertEqual((call["prompt"], call["width"], call["height"]), ("new prompt", 1024, 512))
        self.assertEqual(call["config"], {"steps": 7})

    def test_reroll_seed_zero_is_respected_not_randomised(self):
        cs.regenerate_shot_ng("tom", "front", seed=0)
        self.assertEqual(self.engine.calls[0]["base_seed"], 0)

    def test_reroll_without_seed_is_random(self):
        with mock.patch.object(cs.random, "randint", return_value=4242):
            cs.regenerate_shot_ng("tom", "front")
        self.assertEqual(self.engine.calls[0]["base_seed"], 4242)

    def test_reroll_errors(self):
        with self.assertRaises(LookupError):
            cs.regenerate_shot_ng("ghost", "front")
        with self.assertRaisesRegex(LookupError, "no shot 'back'"):
            cs.regenerate_shot_ng("tom", "back")
        with self.assertRaises(ValueError):
            cs.regenerate_shot_ng("tom", "front", width=1)
        (self.cdir / "sheet.json").unlink()
        with self.assertRaises(FileNotFoundError):
            cs.regenerate_shot_ng("tom", "front")
        self.assertEqual(self.engine.calls, [])

    def test_reroll_engine_failure_leaves_sheet_json_unchanged(self):
        raw = (self.cdir / "sheet.json").read_text()
        with mock.patch.object(cs.hidream_engineNG, "generate_hidream_ng", _FakeEngine(fail_on=1)):
            with self.assertRaisesRegex(RuntimeError, "no image for shot 'front'"):
                cs.regenerate_shot_ng("tom", "front", seed=1)
        self.assertEqual((self.cdir / "sheet.json").read_text(), raw)

    def test_reroll_on_big_sheet_has_no_composite(self):
        shots = [ShotSpec(f"s{i}") for i in range(cs.MAX_SHOTS_FOR_COMPOSITE + 1)]
        cs.generate_character_sheet_ng("tom", shots=shots, seed=1, width=512, height=256)
        out = cs.regenerate_shot_ng("tom", "s2", seed=3)
        self.assertIsNone(out["sheet_path"])
        self.assertIsNone(out["result"]["sheet_png"])


if __name__ == "__main__":
    unittest.main()
