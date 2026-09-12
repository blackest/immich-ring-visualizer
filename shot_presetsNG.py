"""NG twin of shot_presets.py -- shot-list presets for character-sheet
generation.

NG-only file: identical content to shot_presets.py as it exists today,
duplicated per the NG rule in APP_ARCHITECTURE_NOTES.md so nothing under
*NG.py ever imports the original module. Pure data + one small
lookup function, no engine/route logic -- see shot_presets.py's own
docstring and ShotSpec's docstring for the full design rationale
(carried over verbatim below); update both files together if the
underlying design changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ShotSpec:
    """One entry in a shot list.

    `pose_phrase` is the camera/framing/angle clause (the old
    CHARACTER_SHEET_VIEWS value). `expression`, `background`, `wardrobe`
    are optional per-shot overrides -- empty means "use the job-level
    default" (background empty = the classic neutral studio clause,
    wardrobe empty = whatever the reference photo is already wearing).
    `prompt_override`, if set, replaces `pose_phrase` -- this is the
    "free prompt" per-shot escape hatch; it still gets wrapped in the
    identity-lock clauses unless the caller turns identity_lock off for
    the whole job.

    `use_anchor` -- whether THIS shot may receive the job-level anchor
    image (see generate_character_sheet_ng's anchor_chain) as a second
    reference, when anchor_chain is otherwise on. Default True keeps the
    original behavior. Set False for shots whose pose_phrase asks for a
    real angle change (profile/three-quarter) -- the anchor is a
    near-frontal image (the first rendered shot), and HiDream's
    multi-reference conditioning was found to carry the anchor's pose
    along with its color, overriding the text instruction to turn to
    profile. Text alone ("same hair color, same exact hairstyle" in the
    identity-lock clause) still applies to these shots.
    """
    key: str
    pose_phrase: str = ""
    expression: str = ""
    background: str = ""
    wardrobe: str = ""
    prompt_override: str = ""
    use_anchor: bool = True


# ---------------------------------------------------------------------
# "default" -- the original 3-view identity turnaround.
# ---------------------------------------------------------------------
DEFAULT_PRESET: list[ShotSpec] = [
    ShotSpec("front", "a chest-up front portrait, facing the camera directly"),
    ShotSpec("profile_left", "a side view portrait from the left, head and "
                             "shoulders, the face seen from the side",
             use_anchor=False),
    ShotSpec("three_quarter", "a three-quarter view, head and shoulders "
                              "turned halfway between front and profile",
             use_anchor=False),
]

# A handful of simple background/lighting variants, rotated across the
# extended preset instead of repeating one background 15 times. Also
# exposed to the Generate view as a per-shot "scene" dropdown (see
# character_sheetNG.build_custom_shot_ng) -- SCENES is the named,
# selectable form; _BACKGROUNDS keeps the original positional list so
# EXTENDED_PRESET's indexing below doesn't need to change.
SCENES: dict[str, str] = {
    "studio": (
        "Neutral seamless studio background, soft even lighting that "
        "matches the reference image's skin tone"
    ),
    "interior": "A plain, softly lit interior wall, natural window light",
    "outdoor": (
        "An outdoor setting with soft overcast daylight, a gently blurred "
        "natural background"
    ),
    "minimal": "A minimalist indoor space, warm ambient lighting",
}
_BACKGROUNDS = list(SCENES.values())

# ---------------------------------------------------------------------
# "extended" -- 15-shot dataset preset: framing x angle variety (close /
# chest-up / waist-up / full-body), one expression variant, background
# rotated per the note above.
# ---------------------------------------------------------------------
EXTENDED_PRESET: list[ShotSpec] = [
    ShotSpec("close_front",
             "an extreme close-up beauty shot of the face and neck only, "
             "facing the camera directly",
             background=_BACKGROUNDS[0]),
    ShotSpec("close_three_quarter",
             "an extreme close-up beauty shot of the face and neck only, "
             "turned three-quarters between front and profile",
             background=_BACKGROUNDS[0], use_anchor=False),

    ShotSpec("chest_front",
             "a chest-up portrait, head and shoulders, facing the camera "
             "directly", background=_BACKGROUNDS[1]),
    ShotSpec("chest_profile_left",
             "a chest-up portrait in full profile, head turned so the "
             "camera sees the left cheek and left ear -- the right cheek "
             "and right ear are hidden from view, facing toward the "
             "left", background=_BACKGROUNDS[1], use_anchor=False),
    ShotSpec("chest_profile_right",
             "a chest-up portrait in full profile, head turned so the "
             "camera sees the right cheek and right ear -- the left cheek "
             "and left ear are hidden from view, facing toward the "
             "right", background=_BACKGROUNDS[1], use_anchor=False),
    ShotSpec("chest_three_quarter_left",
             "a chest-up three-quarter portrait, head and shoulders, "
             "turned partway toward profile with the left cheek angled "
             "toward camera and the right side of the face turned away, "
             "halfway between front and full profile",
             background=_BACKGROUNDS[1], use_anchor=False),
    ShotSpec("chest_three_quarter_right",
             "a chest-up three-quarter portrait, head and shoulders, "
             "turned partway toward profile with the right cheek angled "
             "toward camera and the left side of the face turned away, "
             "halfway between front and full profile",
             background=_BACKGROUNDS[1], use_anchor=False),
    ShotSpec("chest_front_smiling",
             "a chest-up portrait, head and shoulders, facing the camera "
             "directly", expression="a warm natural smile",
             background=_BACKGROUNDS[2]),

    ShotSpec("waist_front",
             "a waist-up portrait, upper body visible, facing the camera "
             "directly", background=_BACKGROUNDS[2]),
    ShotSpec("waist_three_quarter",
             "a waist-up portrait, upper body visible, turned "
             "three-quarters between front and profile",
             background=_BACKGROUNDS[2], use_anchor=False),
    ShotSpec("waist_profile_left",
             "a waist-up portrait, upper body visible, seen from the left "
             "side in profile", background=_BACKGROUNDS[3], use_anchor=False),

    ShotSpec("fullbody_front",
             "a full-body shot from head to feet, standing, facing the "
             "camera directly", background=_BACKGROUNDS[3]),
    ShotSpec("fullbody_three_quarter",
             "a full-body shot from head to feet, standing, turned "
             "three-quarters between front and profile",
             background=_BACKGROUNDS[3], use_anchor=False),
    ShotSpec("fullbody_profile_left",
             "a full-body shot from head to feet, standing, seen from the "
             "left side in profile", background=_BACKGROUNDS[0], use_anchor=False),
    ShotSpec("fullbody_front_dynamic",
             "a full-body shot from head to feet, standing in a relaxed, "
             "natural candid pose, facing generally toward the camera",
             background=_BACKGROUNDS[1]),
]

assert len(EXTENDED_PRESET) == 15, "extended preset drifted from its 15-shot design"

STYLE_PRESETS: dict[str, str] = {
    "none": "",
    "cinematic": (
        "Cinematic color grading, directional key lighting with soft "
        "shadow falloff, shallow depth of field, filmic contrast"
    ),
    "editorial": (
        "Clean high-fashion editorial lighting, crisp detail, neutral "
        "color grading, magazine-quality finish"
    ),
    "documentary": (
        "Naturalistic available-light look, candid and unposed feel, "
        "true-to-life color, minimal retouching"
    ),
    "warm_film": (
        "Warm vintage film color grading, soft golden-hour-style "
        "lighting, gentle film grain, nostalgic tone"
    ),
    "dramatic": (
        "High-contrast dramatic lighting with deep, defined shadows, "
        "moody atmosphere, strong directional key light"
    ),
}


PRESETS: dict[str, list[ShotSpec]] = {
    "default": DEFAULT_PRESET,
    "extended": EXTENDED_PRESET,
}


def resolve_preset_ng(name: str) -> list[ShotSpec]:
    try:
        return PRESETS[name]
    except KeyError:
        raise ValueError(f"unknown preset {name!r} -- available: {', '.join(PRESETS)}")
