# Director style profiles (draft)

Drafts of directorial styles for the Enhanced H3 pane, written as things a camera
can show: camera, framing, light and colour, and what it points at. They are meant
to steer the OpenH3-IR compiler the way its built-in directors do.

**Status: untested.** Nothing here has been compiled or rendered yet. The source of
profiles 1-20 is a Gemini answer, pasted in by John on 2026-10-28 and tidied only
lightly. Treat it as a first draft: Gemini does not know the compiler's camera-move
vocabulary, and several items are post effects, not camera work (see Caveats).
Profile 21 (the Coens, aimed at No Country) was drafted by Claude in the same
four-part shape. 21b is Gemini's broader Coen profile, added later by John. Section 22
is a list of profiles to try on the western sentence.

## How the compiler's built-in directors work

`GET /v1/directors` on `h3ir serve` returns seven: `cameron`, `tarantino`,
`anderson`, `villeneuve`, `bigelow`, `wong`, `spielberg`. Each is a note in four
parts (camera, framing, light and colour, what it points at). The camera part names
the compiler's own moves to "reach first for" and to "leave alone". A profile only
fills in what the request leaves open: an explicit instruction in the sentence wins,
and a profile never sets shot count or cut times.

The brief also accepts a `director_profile` field. Its format has not been
confirmed (the contract only lists the field name), so check it before building on
it.

### Overlap with the built-ins

| Built-in | In the Gemini list? |
|---|---|
| anderson, villeneuve, tarantino, wong, spielberg | yes (2, 3, 5, 8, 12 below); the built-in note is probably the better source for these five, since it uses the compiler's move names |
| cameron, bigelow | no |
| Not built in | Kubrick, Fincher, Nolan, Scorsese, Mann, Scott, Cuaron, P. T. Anderson, Wright, del Toro, Miller, Park, Malick, Aronofsky, Burton, the Coens |

## Caveats before using these

- **Compiler vocabulary.** The built-ins name moves such as Static Shot, Push In,
  Pull Out, Truck Left/Right, Pan Left/Right, Tilt Up/Down, Pedestal Up/Down, Arc
  Shot, Tracking Shot, Zoom In/Out, Shake Slightly/Strongly and POV. A custom profile
  should use those names, and the "leave alone" list helps as much as the "reach for"
  list. Read `camera_moves` from `/v1/directors` for the exact list.
- **Not camera work.** Step-printing (Wong), speed ramping and variable frame rate
  (Miller), freeze-frames and split-screens (Scorsese, Wright), digital transitions
  (Park), and the Snorricam rig (Aronofsky) are post effects or rigs. The video model
  will not do them reliably. Keep them out of a profile, or expect an approximation.
- **Long takes and fast cutting.** Our windows are 5 s each and the compiler writes
  only a few shots per window. Cuaron's unbroken oners and Nolan's or Scorsese's
  crosscutting are hard to get from this setup. Expect the look, the light and the
  scale, not the structure.
- **Smoke, haze and rain** are easy for the model. **Hands, text and crowds** are not.
- Test each profile by compiling one sentence with it and reading the prompt. A
  profile only helps if it changes the shots the compiler writes.

---

## 1. Stanley Kubrick

- **Camera:** Predominantly static, or locked into mathematically precise slow push-ins and pull-outs. Rigid, linear tracking shots (Steadicam or dolly) following a character down corridors.
- **Framing:** Strict one-point perspective, lines converging on the dead centre of the frame. Wide-angle lenses (around 18 mm) to distort space and stress depth. Eye level or slightly below. Foreground left clean and open to emphasise scale.
- **Light and colour:** Practical sources (lamps, chandeliers, windows) built into the set and heavily augmented to look natural. Palette shifts between sterile cold whites and blues and saturated, threatening primary reds. High clarity, minimal haze.
- **Points at:** Symmetrical architectural spaces, isolated figures, and unblinking close-ups of faces looking slightly downward with the eyes looking up (the "Kubrick stare"). Leaves out chaotic, uncomposed background action.

## 2. Wes Anderson

- **Camera:** Highly restricted, mechanical. Almost only 90-degree pans, flat trucking shots parallel to the scene, and rapid whip-pans. No handheld or diagonal movement.
- **Framing:** Absolute planimetric symmetry, shot perpendicular to a flat background like a stage. Medium-to-long lenses flatten depth. Chest or eye level. Foreground holds perfectly arranged, stationary objects framing the characters.
- **Light and colour:** Extremely bright, high-key, minimal harsh shadow. Curated desaturated pastels (mustard yellows, soft pinks, mint greens). Crisp, hyper-clean, no organic haze, smoke or dust.
- **Points at:** Deadpan faces looking into the lens, overhead shots of hands organising objects, miniature-like detailed props. Leaves out any sense of a messy, unpredictable world.

## 3. Denis Villeneuve

- **Camera:** Slow, deliberate, majestic. Slow cranial booms and jib moves, gradual push-ins, steady tracking that mimics an observer. Never rushed.
- **Framing:** Extreme wide shots of a tiny figure against monolithic or brutalist architecture. Eye level or low angle to stress the weight of structures. Foreground often massive silhouetted geometry or architectural edges cutting the frame.
- **Light and colour:** Low-key, naturalistic light mixed with stark architectural shafts. Fiercely monochromatic or earth-toned palettes (dusty ochre, slate grey, desert beige). Thick ambient dust, sandstorm or mist for density.
- **Points at:** Scale, geometry and massive structures. Characters seen from behind looking out at vast spaces, lingering micro-expressions of dread or awe. Leaves out flashy rapid-fire cuts.

## 4. David Fincher

- **Camera:** Mechanically locked to the characters. Robotic, fluid pans, tilts and tracks that match a character's exact speed with no shake. Stationary shots are fully static on locked tripods.
- **Framing:** Medium to close-up with medium-telephoto lenses that subtly isolate characters. Level with the actor's eyes. Foreground often slightly out-of-focus shoulders or architecture (shooting through doorways).
- **Light and colour:** Low-key, top-lit or side-lit subjects with the eyes in shadow. Restricted to sickly jaundice yellows, sickly greens and deep slate blues. Clean but oily: rain, wet asphalt, dim interiors.
- **Points at:** The mechanics of a space: hands typing, computers, folders, blood splatter, and restrained intellectual reactions. Leaves out organic warmth and chaotic handheld energy.

## 5. Quentin Tarantino

- **Camera:** Energetic and kinetic. Rapid whip-pans, sudden crash-zooms, circular tracking shots around groups talking, low-angle crane tracking.
- **Framing:** Alternates wide theatrical group staging with extreme close-ups. The trunk shot (looking up from a very low, ground-level position). Dynamic foreground with people or objects passing close to the lens.
- **Light and colour:** Vibrant, high-contrast 1970s-celluloid lighting. Saturated primaries (deep yellows, blood reds). Textured air: cigarette smoke, dusty bar rooms.
- **Points at:** Characters delivering dialogue, sudden bursts of violence, feet, vintage pop-culture objects (vinyl, cigarette packs), intense reaction shots. Leaves out conventional establishing shots.

## 6. Christopher Nolan

- **Camera:** Large-scale and functional. Steady tracking shots, sweeping aerial helicopter or drone moves, subtle push-ins to build tension. Avoids handheld in favour of heavy crane and dolly rigs.
- **Framing:** Large-format IMAX framing emphasising vertical and horizontal scale. Prime lenses that mirror human sight. Grounded, human-height lens even during huge spectacle. Foreground used to frame scale (for example, looking over a cockpit dash).
- **Light and colour:** High-contrast, sharp, naturalistic. Avoids stylised digital grading. Cool blues, deep blacks and warm metallic tones. Clear, crisp air, occasionally snow, rain or debris.
- **Points at:** The physics of a scene: crashing vehicles, spinning objects, clocks, hands manipulating tools, stressed faces. Leaves out stylised gore and obvious green-screen.
- *Note:* the large, constant, pulsing score is a big part of the feel. Our briefs write a score line, so this one may carry over.

## 7. Martin Scorsese

- **Camera:** Relentlessly kinetic. Long, complex Steadicam tracking shots weaving through crowds, rapid whip-pans, fast push-ins timed to music. (Freeze-frames are a post effect.)
- **Framing:** High-energy mediums and close-ups. Shifts between eye level and sudden high-angle god's-eye views straight down. Chaotic foreground: people, smoke or glassware cutting across the lens.
- **Light and colour:** Stylised, theatrical. Neon reds, deep night blues, rich golds. Dense club smoke, cigar haze, steam from city grates, flashbulbs.
- **Points at:** Fast-talking faces, frantic hand gestures, money, weapons, sudden outbursts of violence. Leaves out sterile, quiet stillness.

## 8. Steven Spielberg

- **Camera:** The fluid master shot: a wide becomes a close-up in one unbroken dolly and pan, tracking characters through complex blocking.
- **Framing:** Wide-to-medium lenses keep characters and environment in focus. Often slightly low, looking up. Deep foreground: faces or objects very close to the lens with the background unfolding behind.
- **Light and colour:** Strong directional backlight giving silhouettes or halos. Warm cinematic golds or stylised desaturation. Thick blue volumetric haze, dust motes, shafts of light through windows.
- **Points at:** The "Spielberg face": wide-eyed awe, fear or wonder at something off camera. Leaves out abstract ambiguity; the narrative is clearly shown.

## 9. Michael Mann

- **Camera:** Visceral and immediate. Aggressive, textured handheld, jerky tracking, long-lens panning that reads as documentary or surveillance footage.
- **Framing:** Extremely tight and unconventional. Extreme telephoto compresses distance and turns backgrounds into abstract blurs. Gritty, embedded in the action. Foreground often a blurred shoulder or a weapon barrel.
- **Light and colour:** Ambient and architectural light only: fluorescent offices, amber streetlights, blue mercury-vapour lamps. Sharp, modern, often slick with night rain or coastal mist.
- **Points at:** Profiles of intensely focused men, hands gripping steering wheels or firearms, modern cityscapes, quiet micro-expressions. Leaves out theatrical, polished staging.

## 10. Ridley Scott

- **Camera:** Grand and sweeping. Continuous zooms, slow pans, multi-camera setups covering action from several angles, heavy crane use.
- **Framing:** Deeply layered. Long lenses compress foreground against massive backgrounds. Epic height: low-angle heroic stances and wide high-angle vistas. Foreground packed with environment.
- **Light and colour:** Cinematic rim light and backlight. Either icy industrial blues and silvers or hot gritty oranges and browns. Very dense smoke, low fog, swirling dust, dripping water.
- **Points at:** Textures: weathered faces, worn metal, architectural detail, flickering displays, reactions to harsh environments. Leaves out flat, unlit spaces.

## 10b. Ridley Scott on Blade Runner (1982) (Gemini, added by John 2026-10-28)

A film-specific profile, narrower than 10 above. Use it for rain-slicked neo-noir.
Gemini also notes that Blade Runner 2049 was directed by Denis Villeneuve, who built
on Scott's world with a minimalist, brutalist geometry (see 3). That makes the two a
useful pair to compile side by side on one sentence.

- **Camera:** Methodical and atmospheric. Slow, grand crane pans establishing the scale of a dystopian Los Angeles, interspersed with slow drifting push-ins during intense interrogations (the Voight-Kampff test). The movement always feels heavy, echoing the exhaustion of the characters.
- **Framing:** Heavily layered, deep-focus composition. Anamorphic lenses stretch the frame horizontally, with bokeh and flare. Grounded height, often low, looking up at the massive corporate ziggurats or crowded, rain-slicked streets. Constantly cluttered foreground: crowds passing the lens, spinning police spinners, fans cutting the frame.
- **Light and colour:** Neo-noir. High-contrast shadow with flashing, harsh neon and giant electronic billboard projections. Industrial greys, toxic ambers and cold night blues. Dense volumetric smoke, heavy artificial rain, steam rising from the streets, thick smog.
- **Points at:** The details of a synthetic world. Intense, unblinking close-ups of eyes (reflecting explosions or cityscapes), weathered architectural textures, the micro-expressions of dying androids, hands clutching photographs. Leaves out clean, traditional futuristic landscapes in favour of a messy, retrofitted city.
- *Note:* anamorphic flare and bokeh are lens effects. The video model can approximate them in the look, but you can't set them.

## 11. Alfonso Cuaron

- **Camera:** The unbroken long take. The camera acts as an invisible floating presence: large 360-degree pans, complex crane tracks and jib moves without cutting.
- **Framing:** Very wide lenses even for close-ups, so the face is tight to the glass while the surroundings stay sharp. Height moves from ground to sky. Active foreground of passing scenery or debris.
- **Light and colour:** Naturalistic soft light that shifts as the camera moves. Grounded, cool, realistic palette. Reactive atmosphere: shattering glass, floating ash, rain on the lens.
- **Points at:** Geography and chaos: bodies navigating disasters, visceral reactions, peripheral background events that add realism. Leaves out traditional close-up and cutaway editing.

## 12. Wong Kar-wai

- **Camera:** Fluid, dreamlike, fragmented. Slow handheld tracking and intimate tilts. (Step-printing, the smeared low-frame-rate motion, is a post effect.)
- **Framing:** Tight, voyeuristic, obstructed: through doorways, reflections or partitions. Tight lenses flatten space. Intimate shoulder or waist height. Out-of-focus foreground (curtains, clocks, neon) blocks the view.
- **Light and colour:** Saturated, moody neon. Toxic greens, crimson reds, smoky ambers. Cigarette smoke, steam from street stalls, heavy tropical rain.
- **Points at:** Melancholy and isolation: lonely faces, hands holding burning cigarettes, clocks, shoulders brushing in narrow corridors, glances of longing. Leaves out broad, literal exposition.

## 13. Paul Thomas Anderson

- **Camera:** Sweeping and authoritative. Long, intricate dolly tracks alongside or behind characters through vast spaces, with sudden dramatic push-ins on emotional beats.
- **Framing:** Wide, painterly compositions that slowly narrow to intense close-ups. Waist or chest height, an imposing view. Foreground carefully managed with landscape or furniture to ground the depth.
- **Light and colour:** Organic and textured. Harsh natural sunlight, stark shadow, rich amber interiors. Dusty historical palette (deep browns, burnt oranges, olive greens). Dry, dusty or hazy.
- **Points at:** Power dynamics and psychological unravelling: expressive, sweaty or unhinged faces, physical labour, landscapes, silent heavy reactions. Leaves out clean, sanitised Hollywood looks.

## 14. Edgar Wright

- **Camera:** Hyper-stylised and comedic. Rapid whip-pans, fast tracking, instant snap-zooms, movement matched to sound effects and music cues.
- **Framing:** Tight, graphic, manicured. Match-cuts. (Split-screens are a post effect.) Eye or chest height, built for physical comedy. Very dynamic foreground, with objects and actors slamming into and out of frame.
- **Light and colour:** Sharp, high-contrast, vibrant. Punchy pop-art primaries. Crisp and polished, little haze unless used for a genre parody.
- **Points at:** The mundane made epic: kettles, car keys, pint glasses in extreme close-up, frantic reactions, rapid hand movements, comic choreography. Leaves out slow lingering lulls.

## 15. Guillermo del Toro

- **Camera:** Constantly, smoothly moving. Slow drifting cranes, gentle dollies, orbital tracking that crawls around characters like a ghost.
- **Framing:** Gothic and illustrative. Medium-to-wide lenses wrap characters in their environment. Often low, giving fairy-tale or menacing scale. Heavy foreground: ornate ironwork, roots or architecture framing the action.
- **Light and colour:** Theatrical chiaroscuro. Warm amber and gold for safety, humanity or the past; cold cyan and green-amber for institutions, coldness or danger. Airborne dust motes, creeping fog, wet slimy stone.
- **Points at:** The macabre and the beautiful: elaborate creature designs, clockwork, wounds, books, expressive eyes reacting to horror or wonder. Leaves out sterile, modern or mundane places.

## 16. George Miller

- **Camera:** Relentlessly kinetic. Ultra-fast push-ins, violent whip-pans, vehicle-mounted tracking at speed. (Speed ramping and variable frame rate are post effects.)
- **Framing:** "Cross-hair" framing: the point of interest stays dead centre so the eye doesn't hunt through fast cuts. Wide lenses for massive motion. Low to the tarmac or high overhead. Foreground a blur of machine parts or dust.
- **Light and colour:** Blistering high-saturation contrast. Over-cranked desert oranges against electric cross-processed night blues. Billowing exhaust, flying sand, explosions, heat distortion.
- **Points at:** Chaos and velocity: wide adrenaline eyes, hands on gear sticks, steering wheels ripped off, tearing metal, stunts. Leaves out quiet static dialogue scenes.

## 17. Park Chan-wook

- **Camera:** Calculated, inventive, surreal. Unconventional pans, geometric tracks. (Seamless transitions that blend two spaces into one move are a post effect.)
- **Framing:** Striking graphic compositions. Extreme angles, including top-down overheads and low angles through glass floors. Aggressively stylised foreground; wallpaper or interior design splits the frame into geometric halves.
- **Light and colour:** Stylised, cold, rich. Sharp clinical light mixed with moody shadow. Bruised tones: murky greens, deep violet-blues, sharp crimson. Clean but psychologically oppressive.
- **Points at:** Macabre detail: eyes, twitching fingers, unconventional weapons, eating, subtle expressions of revenge or desire. Leaves out safe, comforting visual patterns.

## 18. Terrence Malick

- **Camera:** Weightless and exploratory. Almost all fluid, floating Steadicam, gliding forward, backward or tilting up to the sky, improvisational.
- **Framing:** Ultra-wide lenses very close to the actors. Low, looking up at the sky or sweeping along the ground at waist height. Foreground full of nature: tall grass brushing the lens, branches, skin.
- **Light and colour:** Natural light only, mostly magic hour (dawn and dusk) and strong sun backlight. Organic, unforced palette. Airy and alive: wind, lens flare, floating pollen.
- **Points at:** Tactile human connection and nature: hands in grass, faces in sunlight, water ripples, fleeting glances. Leaves out structured blocking and characters speaking directly to each other.

## 19. Darren Aronofsky

- **Camera:** Intrusive and subjective. Frantic, jittery handheld tracking. (The Snorricam, a rig strapped to the actor so the face stays still while the world moves, is hard to reproduce.)
- **Framing:** Claustrophobic and uncomfortably tight. Extreme close-ups with macro lenses. Foreground dominated by the face, skin or eyes, blurring the rest of the world.
- **Light and colour:** Harsh, gritty, high contrast. Desaturated, bleached or muddy, sometimes punctuated by blinding overexposed whites. Grainy, with sweat, dust or flickering clinical neon.
- **Points at:** Physical and mental deterioration: dilating pupils, sweat beads, scratching skin, heavy breathing, manic tics. Leaves out wide environmental context to induce claustrophobia.

## 20. Tim Burton

- **Camera:** Expressionistic and theatrical. Slow curious push-ins, sweeping crane tracks over detailed miniature landscapes, Germanic-expressionist Dutch angles to convey strangeness.
- **Framing:** High-contrast, graphic-novel framing. Lenses that stress sharp, elongated proportions. Playful height: low-angle monstrous reveals and high-angle isolation. Foreground crowded with twisted branches, wrought-iron gates or jagged silhouettes.
- **Light and colour:** Stark chiaroscuro like classic horror. Either desaturated gothic monochrome (blacks, greys, white) or hyper-saturated carnival pastels. Stage fog, graveyard mist, swirling snow.
- **Points at:** Eccentricity: pale heavily shadowed eyes, expressive hands, curly or jagged architecture, stripes, stitched clothing, quirky isolated outcasts. Leaves out mundane modern realism.

---

## Cinematographers

A cinematographer's profile is mostly **light, colour, lens and atmosphere**. It says
little about what the story points at, and only some of it about camera movement. Use
one as a look to combine with a director (for example Villeneuve plus Deakins), not as a
replacement for one. Whether the compiler can take a second style on top of a director
is untested; the simplest route is to fold the look into the `director_profile` text or
the sentence.

### Roger Deakins (Gemini, added by John 2026-10-28)

Gemini's framing: he is a chameleon, and a "Deakins film" looks different depending on
the director. His philosophy is that the photography has to serve the script, not be
"pretty pictures". Section 21 and 21b above already cover his Coen work, and 10b
covers Blade Runner's neo-noir from the director's side. The looks below are the rest:

1. **With Denis Villeneuve (Sicario, Prisoners, Blade Runner 2049): brutalist geometry and stark silhouettes.**
   Think of the Sicario border-tunnel sequence, where soldiers descend into pitch black against a deep-blue dusk sky, or the huge, hollow, orange-dust-choked ruins of Las Vegas in Blade Runner 2049. Minimalist, massive, very clean.
   *Pairs with:* Villeneuve (3).

2. **With Sam Mendes (Skyfall, 1917, Jarhead, Empire of Light): theatrical command of light and movement.**
   Skyfall's silhouetted fight in a Shanghai skyscraper, lit only by giant pulsing blue neon jellyfish billboards. 1917, built to look like one continuous, breathless, unbroken tracking shot through the WWI trenches.
   *Pairs with:* Cameron (built in) or Spielberg for scale and movement. The continuous-take idea runs into our 5 s windows (see the caveat on long takes at the top).

3. **With Frank Darabont (The Shawshank Redemption): classic, naturalistic Hollywood drama.**
   Soft, rich and emotional. Light shows time passing and state of mind: the cold, grey, oppressive shadows inside the prison walls against the blinding, saturated golden-blue freedom of the Zihuatanejo beach at the end.
   *Pairs with:* a still, patient director, such as the Coens (21) or Kubrick (1).

4. **With Andrew Dominik (The Assassination of Jesse James...): the "old photograph" look.**
   For the 1880s feel he built custom lenses (nicknamed "Deakinizers") that blur and distort the edges of the frame while the centre stays sharp, like early photography. The train-robbery scene is lit entirely by one handheld lantern cutting through thick forest steam and dark night.
   *Pairs with:* the western. *Note:* the blurred edges are a lens effect, so the video model can only approximate it. Words like "soft, blurred frame edges with a sharp centre, like an old photograph" are worth trying.

**Distilled for the compiler (a draft of mine, not from Gemini):** one hard or practical
source doing the work (a lantern, a window, neon), shaped by atmosphere (steam, dust,
haze, rain) so the light has a visible volume; silhouettes against a brighter ground;
colour held to a narrow, motivated palette per scene; nothing lit for its own sake.

---

## Additional drafts (written by Claude in the same shape, also untested)

### 21. The Coen brothers (No Country for Old Men)

- **Camera:** Locked off and patient. Holds a wide shot and lets people walk through it. Slow pushes are rare. Cuts come on action (a door, a step, a car pulling in). Nothing handheld.
- **Framing:** Wide, flat horizons with a small figure, cut against a tight close-up on a face or an object. A lot of negative space and long distances.
- **Light and colour:** Hard, bleached daylight and flat dusk, with practical lamps in motel rooms. Washed-out, desaturated browns and ochres. Plain shadows with no glamour.
- **Sound and pace:** Long silences. Wind, footsteps and creaking leather instead of a score. Sparse, calm dialogue. The menace comes from waiting.
- *Note:* our briefs always write a `non_diegetic_music` line. For this profile, say "no score, natural sound only" in the profile or in `constraints`, and check that the compiler obeys.

### 21b. Joel and Ethan Coen, across their films (Gemini, added by John 2026-10-28)

Gemini calls them "a silent, slightly mocking omniscient observer", and the profile
leans on their long collaboration with the cinematographer Roger Deakins.

- **Camera:** Methodical, sharp, and energetic when needed. Dynamic dolly tracks gliding through a space, often just above the floor, and sudden rhythmic push-ins that punctuate an absurd or shocking moment. When a character moves fast, the camera runs ahead of or behind them on a vehicle or dolly at speed, for a dizzying sense of momentum.
- **Framing:** Wide-angle lenses (Gemini says typically 27 or 32 mm) close to the actors, which exaggerates faces and keeps the environment wide and sharp behind them. Often below eye level, tilting up so characters feel imposing or cartoonish. Strong foreground: a wide-open doorway, a massive desk, or a mundane object such as a ringing telephone or a glass of water.
- **Light and colour:** Atmospheric and genre-specific. Rich, source-focused directional light with deep graphic shadows, close to film noir. The palette is curated per film: the blinding icy whites and greys of Fargo, against the dusty, tobacco-stained, sun-baked ambers and ochres of O Brother, Where Art Thou? and No Country for Old Men. Tactile atmosphere: desert heat-waves, blizzards, heavy indoor dust.
- **Points at:** The tragic, the absurd and the mundane. Expressive, sweaty or panicked faces, held after the dialogue stops so the silence is awkward. Static objects that drive the plot (a briefcase, a woodchipper, a tumbleweed, a cattle gun). Leaves out comforting "hero shots". Violence is sudden, unglamorous and often darkly comic.

**How 21 and 21b differ.** They pull in different directions, and both can be right.
21 describes No Country, which is the stillest, quietest Coen film: locked-off wides, no
score. 21b describes the broader body of work, which moves more and is funnier. For a
tense western like the cowboy scene, 21 is the closer fit. For comedy, crime or a
snowbound thriller, 21b is. If one is merged into the other, choose per film and don't
blend them, because "locked off and silent" and "dollies and push-ins" will cancel out.

**Unchecked:**
- The fast vehicle- or dolly-mounted run alongside a character is how Gemini describes
  it. I have not verified that it is a Coen signature, so treat the movement wording as
  a hint, not fact.
- The lens numbers will not matter to the video model, which takes no lens setting.
  Keep the effect (a close wide lens, exaggerated faces, a sharp background) and drop
  the millimetres.

### 22. Western variants worth trying on the cowboy-rides-into-town sentence

- **Villeneuve (3):** dust and scale, silhouettes against a bright void.
- **Bigelow (built in):** handheld, reacting, long lens, drained colour, a tense, jostled "people nervous" feel.
- **Spielberg (8 or built in):** a slow push onto a frightened face.
- **Coens (21):** stillness and silence.

## Next steps

1. Confirm the format `director_profile` expects (read the `h3ir` source or send a small test brief).
2. Add a profile box to the pane, and a small saved library of custom profiles (name plus the four parts) beside the seven built-ins.
3. Rewrite the profiles that matter in the compiler's camera-move vocabulary.
4. Compile one sentence under three profiles and read the prompts side by side, then render the best one at Draft.
