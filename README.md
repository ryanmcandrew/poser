# FPV Pose Tracker

A Blender add-on that reads a webcam feed and drives an armature in real time
from 2D/3D human pose estimation. Designed for a first-person body-cam view:
the tracked arms and hands are retargeted so they appear as the player's arms
extending forward into the camera frame.

The pose estimator is selectable -- [MediaPipe
Holistic](https://developers.google.com/mediapipe),
[RTMPose](https://github.com/open-mmlab/mmpose/tree/dev-1.x/projects/rtmpose)
or [ViTPose](https://github.com/ViTAE-Transformer/ViTPose) -- see [Pose
engines](#pose-engines).

Every arm joint (shoulder, upper arm, forearm, hand) plus every finger and
thumb joint (three phalanges per digit, both hands) is driven independently.

## Pose engines

The estimator is chosen in the panel's **Pose Engine** box and can be changed
between runs (Stop -> Start). They are not interchangeable in the same way a
quality slider is: the engines differ in what they can report, and the panel
adapts to that.

| Engine | Body | Arms / shoulders | Hands | Accuracy | Real time | 3D of its own |
| --- | --- | --- | --- | --- | --- | --- |
| MediaPipe Holistic | 33 landmarks | good | good | ***  | ***** | yes (metric world pose) |
| RTMPose | 17 / 26 / 133 | excellent | excellent (dedicated model) | ****+ | ****+ | no |
| ViTPose | 17 / 25 / 133 | excellent | excellent (dedicated model) | ***** | ** to *** | no |

**MediaPipe** is the fastest by a wide margin and the only engine that reports
a metric 3D pose itself, so it is the default and still the right choice on a
weak machine. Its weakness is exactly the thing this add-on cares most about:
shoulder, elbow and wrist placement.

**RTMPose** is the recommended upgrade. It is a top-down stack -- a person
detector proposes a box, a body model reads the joints out of that crop, and a
dedicated hand model runs on a higher-resolution crop around each wrist:

```
camera -> person detector -> body pose -> wrist/elbow ROIs -> hand pose
```

This beats raising MediaPipe's *Model Complexity* because the body and the
hands each get a model matched to their own difficulty, rather than sharing
one budget at one resolution. It is the same principle MediaPipe's Holistic
uses internally, with the models chosen per stage instead of fixed.

**ViTPose** is the same stack with a Vision Transformer backbone: the most
accurate body landmarks available here, and the heaviest. Sensible with a CUDA
GPU; expect well under real time on CPU.

### Engine settings

- **Quality** -- Lite / Balanced / Accurate picks the model size and the
  detector together (RTMPose-s/m/x, ViTPose-s/b/l, YOLOX-tiny/m/x).
- **Keypoints** -- *Body (17)* is COCO-17 and covers everything the arm
  retargeter reads. *Body + Feet (26)* adds heels and toes. *Whole Body (133)*
  adds face and both hands from one model.
- **Hand Source** -- where fingers come from. *RTMPose Hand (ROI)* is the best
  option and the architecture described above. *From Body Model* reuses the
  hand points a 133-keypoint body pass already produced: free, but limited by
  the body crop's resolution. *MediaPipe Hands* pairs an RTMPose/ViTPose body
  with MediaPipe's hand landmarker, which is fast and supplies the metric hand
  geometry the depth estimate can use. *Auto* takes the body model's hands
  when it has them and the ROI model otherwise.
- **Detect Every** -- how often the person detector runs. The detector is the
  expensive half of a top-down stack and a webcam subject does not teleport,
  so in between frames the box is tracked forward from the previous frame's
  keypoints. 5-10 costs almost nothing in accuracy.
- **Keypoint Threshold** -- keypoints below this score are reported as unseen
  rather than driving a bone. Raise it if the rig twitches toward guessed
  joints; lower it to keep a partly occluded arm alive.
- **Compute** -- CPU, CUDA or Apple CoreML. The GPU providers fall back to CPU
  if the matching `onnxruntime` build is missing, and the panel offers an
  **Install GPU Runtime** button that swaps in `onnxruntime-gpu`.

### What changes when you leave MediaPipe

RTMPose and ViTPose are **2D** estimators. Three consequences, all handled in
the panel rather than silently:

1. **Arm Depth** is forced to *2D + Limb Lengths*; there is no world pose to
   read. This is the better path anyway (see [How it works](#how-it-works)),
   but it means **Camera FOV** has to be roughly right.
2. **Your Measurements** matter. The depth lift needs your shoulder width,
   upper arm and forearm in metres. MediaPipe learns them from its own 3D
   pose; a 2D engine cannot, so unset values fall back to an average body and
   the panel says so (and the status line reads `[generic limb lengths]`).
   Measuring yourself once is the single biggest accuracy win available to
   these engines.
3. **Hand Size Drives Depth** should be on. A 2D engine reports no per-joint
   depth ordering at all, so the hand-size cue is the only thing that can tell
   an arm reaching toward the camera from one reaching away.

**Infer Width** is ignored for these engines: a top-down model crops and
rescales to its own input size, so pre-shrinking the frame saves it nothing
and only blurs the hand crops.

In multi-camera mode each camera loads its own copy of the models, so the
memory and warm-up cost scale with the number of cameras.

### Hands from RTMPose and ViTPose are flat

MediaPipe's hand model returns a depth for every landmark. RTMPose and
ViTPose do not: their hand keypoints are pixel positions, and the depth
comes back as zero. A hand whose landmarks all sit at one depth is a flat
sheet facing the lens, and the palm normal built from it can only ever
point along the view axis -- so the wrist can roll, but it cannot tilt or
turn, and the normal flips sign the moment the knuckles cross over in the
image. Switching engine for the accuracy of the body landmarks and
finding that palm direction got *worse* is this, not a regression in the
retargeter.

The addon fits the palm plane instead. The wrist and the four knuckles
are close enough to a rigid, flat shape that their image is a projection
of a known template, and fitting that template recovers the plane's tilt
-- up to the one ambiguity every planar target has, where tilting towards
or away from the camera looks identical. That ambiguity is settled by
continuity with the previous frame and by the forearm's own depth
direction, since a hand roughly continues the arm it is on.

What this does not recover is depth *within* the fingers: a curled finger
is fitted as though it lay in the palm plane. Finger bones are driven
from angles, so the cost is small, but a hand model with real depth
(MediaPipe, or a multi-camera rig) still measures what this estimates.

### Body form, not just landmarks

Whether a pose is anatomically right is a question about angles, not pixels,
and angles are also what the rig actually consumes -- a model can win on
landmark error and still produce a jittery elbow. The status block reports,
per frame:

```
elbow angle      angle(shoulder, elbow, wrist)
shoulder angle   angle(elbow, shoulder, hip)     measured against the torso,
                                                 so leaning does not change it
torso lean       angle(shoulder_mid -> hip_mid, vertical)
arm symmetry     |left angle - right angle|
```

They are computed from the depth lift when it is available and from image
coordinates otherwise, and the readout says which (`[3d]` / `[2d]`) -- a 2D
angle is a projection, so an arm swinging toward the camera barely moves it.
`pose_metrics.py` exposes the same values for scripting.

## Build

```powershell
.\scripts\runbuild.ps1
# produces dist/fpv_pose_tracker-<version>.zip
```

`runbuild.ps1` is the one to reach for: it finds a Python, runs the build from
the repository root whatever directory you are standing in, and then reads the
finished zip back to check it is actually installable. That last step matters
because the zip's top-level folder becomes the add-on's Python package name in
Blender, so a bad layout produces a zip that installs and then fails to
import -- better caught here than in Blender.

| Flag | Effect |
| --- | --- |
| `-Python <path>` | Use a specific interpreter instead of auto-detecting `py -3` / `python` / `python3`. `build.py` is standard-library only, so any Python 3 works, including Blender's own. |
| `-Clean` | Prune older zips from `dist/`, keeping the newest `-Keep`. Lists them and asks first. |
| `-Keep <n>` | How many zips `-Clean` leaves (default 5; `0` keeps only this build). |
| `-Force` | Let `-Clean` skip the confirmation, for scripted use. |
| `-WhatIf` | Show what `-Clean` would delete without deleting it. |
| `-SkipVerify` | Build without inspecting the zip. |
| `-Reveal` | Open `dist/` in Explorer on success. |

`dist/` is gitignored, so the zips in it are the only copies of past releases
that exist -- git cannot restore them and an older version cannot be rebuilt
from current source. That is why `-Clean` keeps several by default and asks
before deleting.

`python build.py` still works on its own if you would rather not go through
PowerShell; it must be run from the repository root.

## Install

1. In Blender: `Edit > Preferences > Add-ons > Install...`, pick the zip.
2. Enable **FPV Pose Tracker**.
3. On first use the panel will show an **Install Dependencies** button that
   pip-installs what the selected engine needs (`opencv-python` and `numpy`
   always; `mediapipe` for the MediaPipe engine; `rtmlib` and `onnxruntime`
   for RTMPose/ViTPose). No restart and no admin rights are needed: the
   wheels go into `scripts/fpv_pose_tracker_wheels` under your Blender
   config directory, which the add-on puts on `sys.path` at registration,
   and the installer re-imports each module before it reports success.
4. `Edit > Preferences > Add-ons > FPV Pose Tracker` lists every engine with
   its own Install button, so you can add an engine later without reinstalling
   the add-on. Model checkpoints download on first use, not at install time.

Wheels live outside the add-on folder on purpose, so reinstalling the zip does
not throw away a few hundred megabytes of downloads. If an engine reports a
module missing that you know you installed, hit **Engine Info**: it prints the
directory each module resolved from and every directory searched. The usual
cause is a `pip install --user` from outside Blender -- Blender embeds Python
with the per-user `site-packages` disabled, so wheels installed there are
invisible to it. The add-on searches that directory too, for exactly this
reason.

Blender 4.2+ also supports dragging the zip directly onto a 3D Viewport.

## Use with the test rig `fpv_arms2.blend`

1. Open `fpv_arms2.blend`.
2. Open the sidebar (`N`) in the 3D Viewport and go to the **FPV Pose** tab.
3. Pick the FPV arms armature in **Armature**.
4. Open **Bone Map** and hit **Auto-Map Bones**. Verify the guesses (arm,
   fingers, thumb). Fix any wrong assignments with the bone pickers.
5. Pick your camera in the **Webcam** dropdown (press **Scan** first to fill
   it with the device indices that exist). For a webcam in front of you pick
   **View Mode: Selfie** and keep **Mirror Webcam** on. To use more than one
   webcam, choose **Source Multiple Cameras** — see below.
6. Hit **Start Tracking**. The mapped bones snap back to rest and a
   calibration countdown starts (default 3 s). **Match the rig's rest pose
   and hold still**: for the FPS arms rig that is both arms forward toward
   the camera, palms down, fingers relaxed. The status line reads
   `calibrated` once the torso frame and both palms are sampled; after that
   the arms and hands follow you.
7. Press **Calibrate** any time to redo it (it resets the rig to rest first).
   Hit **Stop Tracking** to release the webcam.

## Sourcing multiple cameras

One webcam can only guess depth. The pose lifter recovers it from
foreshortening, but the error is always worst along the camera's own optical
axis: a hand moving straight at the lens barely moves in the image, so its
distance is the least observable thing in the frame. Everything *across* the
view, on the other hand, is measured directly in pixels.

Two cameras at an angle to each other therefore have complementary blind
spots — what is depth to one is sideways motion to the other — and combining
them removes the single biggest source of error in a selfie setup.

Place the second camera off to one side. 30-60 degrees from the first is
plenty; two cameras side by side see almost the same thing and have almost
the same blind spot.

1. Press **Scan** to find the device indices, then pick **Source Multiple
   Cameras** in the **Webcam** dropdown. Two device rows appear.
2. Set each row's **Device** index. Tick **Use** on the ones to run; two rows
   may not share an index.
3. Set each row's **FOV** if the cameras have different lenses, and drop
   **Trust** for a camera that is further away, lower resolution or worse lit.
4. Use **View Mode: Selfie**. A body cam's frame is tied to the user's own
   heading, so body-cam feeds cannot be put in a common frame.
5. Press **Start**, then **Calibrate Placement**, and move your arms through
   their whole working range for about ten seconds. See below.

Each camera runs its own full pose + hand pipeline, so two cameras cost
roughly twice the CPU. **Preview in Panel** and the preview window both show
the feeds tiled together, each with its own tracking overlay, so you can see
at a glance which camera has lost you.

### How the cameras are combined

There are two settings under **Fusion**, and the difference between them is
most of the accuracy a second camera can give.

**Fit Body to Both Views** (the default) fits a single skeleton to both
cameras' 2D landmarks at once, using your own measured limb lengths. A joint
both cameras can see is *triangulated*: its depth is measured from where the
two rays cross, not inferred. Measured against ground truth on a synthetic
two-camera rig with realistic landmark noise:

| | arm-joint error |
| --- | --- |
| averaging each camera's 3D | 75 mm rms |
| raw triangulation | 10 mm rms |
| skeleton fitted to both views | 9 mm rms |

The fit costs 0.9 ms per frame for two cameras, against roughly 17 ms for the
pose inference it waits on, so it is free in practice.

**Average Each Camera's 3D** is the older path: each camera infers 3D on its
own and the results are averaged, weighted by **Yaw** and **Depth Trust**.
It needs no calibration, which is the only reason to choose it. Its ceiling
is low and unfixable: both cameras infer depth from one image each, by the
same network and the same lift, so their errors point the same way and
averaging cancels noise but not bias.

One trap worth knowing if you use it. The Yaw weighting only does anything
when the cameras have *different* yaws. With every camera at the same yaw the
per-camera weighting matrices are identical, **Depth Trust** divides out of
the solve entirely, and what is left is a plain confidence-weighted mean —
so two cameras both left at yaw 0 are barely better than one. The panel says
so in red when that is the case.

### Calibrate Placement

Triangulation needs to know where the cameras are, and the **Yaw** slider
cannot tell it. Yaw says which way a camera faces; a ray also needs the
camera's position, and a baseline wrong by 20 cm moves a triangulated wrist
further than the monocular depth error being removed. Fitting on the Yaw
guess alone measures at 174 mm rms — *worse* than the averaging it replaces.
So the body fit does not run until the placement has been measured, and asks
for it by itself on the first frame.

Press **Calibrate Placement** and move your arms around your whole working
range for about ten seconds. Variety is what it needs, not time: it ignores
frames that repeat a pose you have already held, so standing still does not
advance it.

The rig measures itself from your own body. Each camera already estimates
your pose in metres in its own frame, so two cameras hold two copies of one
rigid point set and the transform between them is what is being solved for.
That closed-form answer is then refined against the pixels, which is the part
that was actually measured.

The result is saved with the .blend, so this is a one-off unless you move a
camera. The panel reports the fit's residual in pixels and how far apart the
cameras came out; **Clear Placement** forgets it. A fit that cannot be
trusted is rejected rather than used, and the status line says so.

**Skeleton Stiffness** controls how far the fit may stretch a limb from the
length you were measured at. Tighter rejects more landmark noise; looser
follows the cameras more closely when the measured lengths are off.

### What the rig does with the extra accuracy

Several parts of the retargeter exist purely to undo monocular depth error,
and leaving them switched on would have thrown most of the improvement away:

- **Elbow-from-angle** replaces the measured shoulder-to-wrist distance with
  the distance implied by the tracked elbow angle, because one camera reads a
  straight arm pointing at the lens as a bent one. When both cameras
  triangulated the wrist there is no compression to undo, so the correction
  is skipped for that arm — it would move a measured wrist to where an
  inference says it should be.
- **The elbow pole hint** is trusted much sooner for a triangulated elbow
  than a guessed one, and the tracked upper-arm direction is leaned on over a
  wider range.

These switch per joint, per frame, on how many cameras actually saw it, so a
camera dropping out mid-session degrades back to the monocular behaviour
rather than driving the rig on assumptions that no longer hold.

### What this does not fix

The hands are still fused the old way — weighted by apparent hand size, with
the wrist depth taken from the best single view. Finger landmarks are small,
fast and often occluded in one view, which is a different problem from arm
geometry. If your hands are the weak part, exposure and lighting will do more
than a second camera.

### When one camera is older than the other

Cameras in a pair rarely want the same settings, and an old webcam will
quietly drag the whole rig down. Each device row therefore carries its own
**Exposure** (Shared / Auto / Manual) and its own measured frame rate, shown
in the row and coloured red below 15 fps.

Exposure is the single biggest lever on an old camera's rate. Measured on a
Logitech C-series at 640x480 MJPG in normal room light:

| setting | measured |
| --- | --- |
| auto exposure | 8.9 fps |
| manual exposure -6 | 20.0 fps |

On auto, the driver picks a long shutter in anything short of bright light,
which both halves the rate and smears a moving hand into something MediaPipe
cannot read. Set that row to **Manual** and drop the exposure until a moving
hand stops smearing, then add light rather than gain. Leave the newer camera
on **Shared**.

The status box lists each camera's rate and the share of fused frames it
actually contributed. A camera sitting at a low percentage is present but not
helping, which is invisible from the fused frame rate alone.

Two settings tune the fusion:

- **Depth Trust** — how much a camera's forward axis is worth next to its
  lateral ones. The default 0.35 is what lets a camera off to the side supply
  the depth the front camera is guessing at; 1.0 weights every axis equally,
  which is a plain average and throws the benefit away.
- **Sync Window** — a camera whose newest frame is older than this sits out
  that fused frame rather than smearing a stale limb into the current pose.
  Raise it if a slower camera keeps dropping out of the fusion.

With **Record Frames** on, each camera writes its own `...-camN.jsonl` so a
session can still be replayed per camera offline.

### Troubleshooting a camera that works only once

Some webcams refuse to open roughly half the time, at random. Measured on the
same Logitech: reopening after a release succeeded 3/3 with no delay and 1/3
with a 0.4 s delay, so it is driver flakiness rather than a release race, and
waiting longer does not help. The add-on retries the same backend six times
(about 1.5 s at worst) before considering another one, which took a measured
10 of 10 Stop/Start cycles from roughly half failing to none.

Two related traps on Windows, both handled but worth knowing:

- **Never let it fall through to another backend too early.** Media Foundation
  has been measured taking over 90 seconds to open a device it then cannot
  read a frame from, and the generic fallback has returned a black 1 fps feed
  that looks like a successful open. **Scan** therefore probes DirectShow only.
- **Do not request a frame rate from a driver that will not report one.** A
  camera whose `CAP_PROP_FPS` reads -1 has been measured accepting a 60 fps
  request, running fine for that session, and then refusing the *next* open —
  which is exactly a camera that works once per Blender run. The add-on now
  skips the request on such a driver and says so in the camera line.

### A camera that shows one frame and then freezes

A webcam can stop delivering frames without ever reporting an error. Measured
on an older Logitech running alongside a second webcam: it enters a state
where every `read()` returns only after DirectShow's internal ~1000 ms
timeout, and it stays there — 0.0 fps for 15 s straight, with no recovery of
its own. From the outside that is a preview showing one frame and then never
updating, while the tracker cheerfully reports itself as running.

Reading harder cannot fix this; only reopening the device can. The capture
thread therefore watches its own delivery and, after 2.5 s with nothing
usable, releases the device and opens it again on the backend that was
working, resting a couple of seconds first because a wedged camera needs
that long before it will reopen. The preview and the status box count the
revivals (`revived 2x`), so a camera that needs constant reviving is visible
rather than silently halving your tracking.

Two supporting rules make that reliable:

- **A successful open is not the same as a working camera.** A reopened
  device has been measured coming back claiming a -1x-1 frame size with a
  garbage codec, handing back buffers that `read()` calls successful but that
  are not pictures. The add-on checks the frame and the reported mode, and
  treats a bogus mode as a failed open.
- **A refused open is retried, not fatal.** A camera that refuses roughly
  half its opens would otherwise be missing for a whole session on a coin
  flip, which in a two-camera rig quietly halves your tracking.

Cameras in a pair should also not be opened at the same instant: two
DirectShow devices opening together contend, and the loser comes up degraded.
The add-on staggers them, which is why Start takes a moment longer with two
cameras.

## Capturing fast motion

A punch lasts about 130 ms. At the 15 fps this add-on used to manage that is
two samples for the whole extension — the motion isn't attenuated, it's
absent. Two independent things decide whether fast action survives: how many
samples you get, and how much of each one the smoothing throws away.

**Frame rate** (*Capture + Performance*, needs Stop → Start):

- **Auto-Tune Exposure** is the one to press first, and usually the only
  one that matters. Exposure time, not the sensor, is what caps webcam frame
  rate: auto-exposure in an ordinary room picks something like 1/15 s, which
  pins capture at 15 fps however high a rate you request. Each stop shorter
  doubles the achievable rate until the link's bandwidth takes over, after
  which darkening buys nothing. Auto-Tune measures the rate at a range of
  exposures and picks the brightest one that still reaches the plateau. It
  takes about 5 seconds; keep the camera uncovered and pointed at your
  normal scene. Measured on the development webcam:

  | exposure | fps | brightness |
  |---|---|---|
  | -3 | 8 | 152 |
  | -4 | 16 | 118 |
  | **-5 (chosen)** | **31** | **86** |
  | -6 | 31 | 56 |

  A short exposure also keeps a moving fist sharp. At 1/15 s a hand at
  3 m/s smears ~60 px, which is enough to stop MediaPipe finding it at all —
  many of the mid-take "tracking snapped somewhere impossible" glitches are
  motion blur rather than bad prediction.
- **MJPG** and **Target FPS**. Webcams default to raw YUY2, whose bandwidth
  caps 640×480 at ~30 fps. Many cameras offer MJPG and will then do 60, but
  plenty refuse it — the development webcam does, at every resolution tried.
  The panel prints what the camera actually granted and says so explicitly
  (`Camera: DSHOW 640x480 @60 YUY2 exp=-5.0 lum=80 [MJPG refused, still
  YUY2]`). Check that line rather than assuming the request took.
- **Parallel Pose + Hands** (on by default) runs the two models on separate
  threads. They are independent and MediaPipe releases the GIL during
  inference — so this genuinely halves per-frame time. Measured on real
  models: pose 16 ms, hands 20 ms, total 21 ms rather than 36.
- **Pose Every N** runs the body model on every Nth frame only, reusing the
  last result, while hands keep running every frame. Measured 40.7 → 30.0 →
  26.3 ms for N = 1/2/3 when running sequentially. It buys nothing when
  running in parallel unless pose is the slower of the two.
- **Infer Width** downscales frames before MediaPipe sees them. Landmarks
  come back normalised, so positions and the depth maths are unaffected.
- **Compute: GPU** tries MediaPipe's GPU delegate and falls back to CPU if
  the installed wheel lacks it (common on Windows).
- **Preview FPS** limits how often the annotated preview is built. Drawing,
  resizing and converting colour is real time on the tracking thread; it now
  runs at 12 fps by default instead of once per frame.

**If the preview is black**, the panel says so at the top with the
brightness it measured, because a black feed produces no landmarks at all:
calibration never completes and the rig never moves, which looks exactly
like a broken add-on. The usual causes are a closed privacy shutter, another
app holding the camera, or an exposure set too short for the room.
**Reset Camera Exposure** (next to Scan) puts the camera back on automatic —
needed because webcam drivers keep a manual exposure even after Blender
closes, so one bad value would otherwise persist across restarts.

Frames are read by a dedicated grabber thread that keeps only the newest
one. The old loop called `cap.read()` inline, so any pass longer than a
frame interval left the driver's queue to decide what happened next — which
pinned it to an exact subharmonic of the camera rate.

The panel shows a live per-stage breakdown (`tot 21 | grab 1 hand 20 pose 20
ovl 1 | dropped 32`), also drawn on the preview HUD and written into the
`Record Frames` `.jsonl`. Use it to find which stage is actually costing you
before changing anything else.

**Smoothing.** Weights are authored as the fraction retained per frame *at
30 fps* and rescaled to the real frame interval, so the response stays
constant in milliseconds whatever the capture rate. Previously the weight
was applied per frame regardless, so the same 0.9 meant a 633 ms settling
time at 15 fps and 79 ms at 120 — every throughput improvement silently made
the rig twitchier.

The **Motion Response** presets set the arm/hand weights and the One-Euro
parameters together, and the panel shows the settling time rather than the
raw weight. How much of a 130 ms punch actually reaches the rig:

| | punch captured |
|---|---|
| old behaviour, 0.9 @ 15 fps | 19% |
| Smooth (~320 ms) | 34% |
| Balanced (~120 ms) | 68% |
| Fast Action (~50 ms) | 94% |

Fast Action shows every tracking wobble, so pair it with a high capture rate
and a short exposure. The take editor's outlier pass is there to clean up
what it lets through.

**Honest ceiling:** even at 60 fps a jab is ~8 samples. If punches are the
real target, a global-shutter camera at 90–120 fps is the actual answer —
rolling-shutter webcams also skew fast lateral motion regardless of rate.

## Recording an animation

Tracking drives the rig live; recording turns a performance into an Action.

1. Start tracking and calibrate as above, then press **Record Animation** in
   the *Animation Recording* box. (**Record on Start** begins the take with
   tracking instead, though that captures the calibration too.) The take is
   held in memory — nothing is written to an Action until you save it.
2. Press **Stop Recording**. A review window opens: the rig on top, the
   take's keyframes in an Action Editor below. The controls are in each
   sidebar, under the **FPV Pose** tab in the viewport and **FPV Take** in
   the Action Editor — the same panel also lives in the main window's
   sidebar, so the window is optional (**Open Review Window**).
3. Edit the take. With **Live Preview** on, every edit rebuilds a preview
   Action on the rig, so scrubbing the timeline shows exactly what will be
   saved:
   - **Trim Start / Trim End** cut the ends, for the frames spent getting
     into position. *Start/End at Playhead* sets them from the timeline.
   - **Find Outliers** flags samples that sit off the path their neighbours
     describe — where tracking lost you and snapped the rig somewhere you
     never went. With **Drop When Flagged** on they are removed at once.
     *Sensitivity* moves the robust threshold; *Min Jump %* is a floor
     below which nothing is flagged, so a still take doesn't flag its own
     noise. **Flag Lost Tracking** also catches frames where the tracker saw
     neither body nor hand.
   - The **Samples** list shows each flagged sample's frame, how far off the
     path it sits (as a percentage of the rig's shoulder width) and why it
     was flagged; the checkbox keeps or drops it by hand. The arrows jump the
     playhead between flagged samples, and **Drop/Keep Here** removes the
     sample at the playhead — scrub to a bad pose and press it.
   - A dropped sample is simply not keyed, so the curves interpolate
     straight across the hole. **Reset Edits** puts everything back.
4. Set **Action Name** and **Start Frame**, then **Save to Action**. With
   **Push to NLA** on, the Action also lands as a strip on an NLA track
   (default *FPV Takes*), leaving whatever action the rig already had alone.

Capture runs at the tracker's frame rate and is resampled to the scene's
frame rate on save, so the take keeps its real-time pacing. Quaternions are
put on a consistent hemisphere between keys, so no joint takes the long way
round between two frames.

## Clip library and offline pose estimation

Live tracking has one budget: whatever is left of a frame interval after
Blender has drawn. That budget is what picks the model, and it is why
ViTPose is not really a live option. Recording the performance as plain
video first and estimating pose afterwards removes the budget, which buys
three separate things: the accurate model, every frame in order with none
dropped, and the current frame's body pose placing the hand crops instead
of the previous frame's.

Everything below is in the **Clip Library + Offline Pose** panel in the
sidebar, and none of it needs tracking mode.

### Where clips live

Clips are files in a two-level folder tree:

```
<library>/<category>/<subgroup>/<name>.mp4          the footage
<library>/<category>/<subgroup>/<name>.timing.json  per-frame capture times
<library>/<category>/<subgroup>/<name>.RTMPOSE.pose.json.gz   one analysis
```

**Library** is the root; leave it blank and clips go to a `clips` folder
inside the addon. **Category** and **Subgroup** are plain folders -- pick
an existing one from the dropdown or type a new name, and it is created on
the next recording. Because they are only folders, rearranging takes in
Explorer works; press **Refresh Library** afterwards. **Whole Library**
lists everything rather than just the selected category.

### Recording a clip

Set **Clip Name**, **Countdown** (seconds of preview before capture starts,
to get out of reach of the keyboard) and **Max Length**, then press
**Record Clip**. Resolution, frame rate, MJPG and exposure come from
*Capture + Performance* in the main panel, and the advice there about short
exposures for fast motion applies with more force here: this is the only
image the accurate pass will ever see.

No pose model runs during capture, on purpose -- inference would cost frame
rate and add motion blur to the very footage the accurate pass exists to
read. A name collision appends a number rather than overwriting footage you
cannot re-shoot.

A webcam does not deliver frames at the rate it claims and a container
stores only one nominal rate, so the wall-clock time of every written frame
goes into the `.timing.json` beside the clip, and a take baked from it runs
at the speed it was performed at.

**Play in Player** hands the selected clip to whatever the desktop opens
video with, which is the quickest way to check the footage itself before
spending model time on it. The in-panel preview holds the file open while
it is showing, so this closes it first.

### Analysing a clip

Select a clip and choose an **Engine**, **Quality** and **Compute** -- these
are separate from the live settings, so live tracking can stay on MediaPipe
while clips are analysed with ViTPose-large on the GPU. **Every Nth Frame**
trades temporal detail for time on a long clip; 1 is the point of an offline
pass. **Replace Existing** guards a previous long run.

**Analyze Clip** runs in the background and reports progress; Blender stays
usable. The result is written beside the clip as one gzipped file per
engine, so the same footage can be run through MediaPipe, RTMPose and
ViTPose and the three compared directly -- the clip list shows which engines
each clip has, and the **Result** dropdown shows each one's detection rate
and cost per frame.

What is stored is raw landmarks, not bone rotations. Estimation is the
expensive fixed part; retargeting is cheap and depends on things that change
constantly (armature, limb lengths, smoothing, FOV, IK), so a clip is
analysed once per engine and replayed as often as the rig changes.

### Previewing the result

**Show Pose Preview** draws the stored landmarks over the footage at the
scrub position, in the panel. Step a frame at a time, jump to the ends, or
use **Next Gap** to jump to the next frame where the body was not detected
-- a lost stretch is what makes a take unusable and finding it by dragging a
slider over 2000 frames is hopeless. **Body** and **Hands** toggle each
overlay. What is drawn is exactly what would drive the rig, because it comes
out of the same file the replay reads.

### Calibration: what the rest pose looked like

Hand and wrist bones are not driven by the palm's absolute orientation.
They are driven by how far the palm has turned **since a reference pose**
-- the moment the performer stood in the rig's rest pose. Live tracking
gets that for free by asking the user to hold still for three seconds
before it starts driving anything.

A recorded clip cannot be asked anything. Its first frames are whatever
the take happens to open on, and calibrating off them declares mid-punch
to be the rest pose, which rotates every wrist in the resulting take by
however wrong that guess was. This is the single biggest cause of an
offline take looking worse than live tracking of the same motion.

So the **Calibration** box says where the reference comes from:

* **Saved Profile** -- a calibration recorded once and kept in the
  library. The reliable option.
* **Live Session** -- reuse the calibration from the live tracker in this
  Blender session, which is available as soon as you have tracked once.
  **Save Live Calibration** writes it to the library as a profile.
* **This Clip's Opening** -- sample from the first seconds of this clip.
  Correct only when the clip itself opens on the rest pose.

**Making a calibration clip.** Press **Record Calibration Clip**: it
shoots into a `calibration` category with a long countdown and a six
second limit, so there is time to get back from the keyboard. Hold the
rig's rest pose: facing the camera, arms and hands where the armature's
arms and hands are, palms and fingers pointing the way the rig's do.
Then analyse it with the engine you intend to use for real takes, select
it, and press **Calibrate from Clip**.
**Skip** ignores the first moment while you settle, **Hold** limits how
much of the clip is sampled, and **Save As** names the profile.

One profile covers every clip shot in the same setup: it describes the
performer and the camera, not the scene, so it lives in the library
(`_calibration/<name>.json`) rather than in the .blend, and it can prime
any rig. Record a new one when the camera moves, when someone else
performs, or when the rig's rest pose changes.

Calibrating from a clip also produces a profile you can hand to someone
else along with the footage, which is the point of keeping it as a file.

### Driving the rig from a clip

**Apply to Rig** replays the analysis through the same retargeter live
tracking uses and records it as a take. No camera is opened and nothing is
in tracking mode. The take then goes through the ordinary review editor --
trim, outlier detection, per-sample keep/drop -- and **Save to Action**
bakes it, so recorded footage gets exactly the cleanup tools a live
recording gets. **Bake Straight to an Action** skips the review step for a
clip that needs no cleanup.

**Play on Rig** sets the scene range to the rig's current action and plays
it, which is the whole point of the offline path: a recorded performance
plays back on the model with nothing running.

Applying needs an armature, a bone map (**Auto-Map**), no live tracker
holding the rig, and a calibration source that actually exists; the panel
says so when one is missing, and **Apply to Rig** refuses rather than
calibrating off the performance behind your back.

## How it works

- A worker thread reads frames from OpenCV (`cv2.VideoCapture`) and runs the
  selected engine's body + hand models on each frame.
- **Every engine reports into one schema** (`pose_schema.py`): MediaPipe's
  33-point pose order. COCO-17, Halpe-26, COCO-25 and COCO-WholeBody-133 are
  lookup-table conversions into those slots, with unmapped joints reported as
  visibility 0 -- which is already how the pipeline spells "not measured".
  Hand landmarks need no conversion: MMPose numbers its 21 hand points in the
  same order MediaPipe does. So the retargeter, depth lifter and multi-camera
  solver never learn which engine produced a frame (`pose_engines.py` is the
  only seam).
- **Multi-camera fusion** (`multicam.py`) runs one of those pipelines per
  device and combines their pose frames with a weighted least squares. Each
  camera contributes its estimate with an information matrix
  `c_i * R_i @ diag(1, depth_trust, 1) @ R_i.T`, where `R_i` is its Yaw
  rotation into the reference frame and `c_i` is the landmark's confidence in
  that camera (visibility for pose joints, apparent hand size for fingers).
  The fused point is `(sum W_i)^-1 @ sum W_i @ R_i @ p_i`, which collapses to
  "just use that camera" when only one sees a landmark and to "take each axis
  from whichever camera measures it across its view" when two cameras sit at
  a right angle. Wrist distance is not averaged — it is measured from one
  lens — so it comes from whichever camera has the best view of that hand.
- Both landmarkers run in MediaPipe VIDEO mode (temporal tracking, far
  steadier than per-frame detection). **Model Complexity** 0/1/2 picks the
  lite/full/heavy pose model.
- **Sides are decided by geometry, not labels.** MediaPipe's pose labels are
  anatomical as-seen-in-the-image and the hand landmarker's handedness
  assumes a mirrored feed, and real builds don't always match their docs.
  The tracker instead takes the shoulder on the image's left as the user's
  left (for a mirrored selfie feed or an unmirrored body cam), attaches each
  hand to the pose wrist it sits on in the image, and learns the label
  conventions from that for frames where the pose drops out. The preview
  HUD shows `pose*` when the pose labels are being swapped.
- **Arm depth** (default *2D + Limb Lengths*): MediaPipe's 3D depth is a
  learned guess that compresses when a limb points at the camera. Instead the
  tracker lifts the 2D landmarks along their camera rays using the camera's
  **horizontal FOV** and your limb lengths (estimated from the metric world
  landmarks): the shoulder depth comes from the shoulder line's pixel width,
  then each elbow/wrist is placed on its ray at exactly one limb length from
  its parent. Foreshortening in the image thereby becomes real depth. Set
  **Camera FOV** to your webcam's horizontal FOV (60–78° for most webcams).
  *MediaPipe World* is available as the alternative source.
- **Hand size drives depth** (default on): each ray placement above has two
  answers, joint toward the camera or away, and when an arm points at the
  lens the image cannot tell them apart. The tracker measures each hand's
  distance from the camera from its size in the image (MediaPipe's metric
  hand model fitted to the image landmarks with a perspective-n-point solve)
  and picks the elbow/wrist roots whose wrist depth matches it. MediaPipe's
  image depth ordering is only the fallback when no hand is tracked. The
  metric hand model is an average hand, so **Hand Size** scales it; with
  **Learn** on, the scale is also refined from frames where the arm lies in
  the image plane (hanging down or out to the side). The preview HUD shows
  which cue chose the roots (`roots[L:han R:ima]`), each hand's distance in
  metres (`handz[L=0.52]`) and the current hand-size factor (`x1.03`).
- **Record Frames** (Camera section) writes every frame's raw MediaPipe
  output to a `.jsonl` file. `python tools/replay_diag.py <file>` re-runs the
  depth pipeline on it outside Blender and prints each wrist's forward
  position, hand distance, elbow angle and which cue chose the roots;
  `--fov`, `--no-hand-depth`, `--hand-size` and `--csv` let you compare
  settings on the same frames. Use it to hand a session over for analysis.
- Hand landmarks give the 21 joints per hand, converted (with the frame's
  aspect ratio, so vertical directions aren't squashed) into a body frame:
  X = your right, Y = forward, Z = up.
- **Calibration** measures rather than assumes conventions. From the frames
  you hold the rest pose it derives
  - the map from MediaPipe's pose axes into the rig's armature space (torso
    frame: shoulder line, shoulders→hips, forward from the arms/nose). This
    absorbs the Y-up/Y-down question and the mirror reflection outright;
  - each palm's rest basis (fingers, across-palm, palm normal). The hand
    bone is then driven by the *rotation since calibration* applied to the
    bone's rest orientation, so which local axis of your rig's hand bone is
    the palm normal never has to be guessed.
- **Arms** use analytic 2-bone IK anchored at the rig's rest shoulder. In
  Selfie mode the wrist target is placed relative to the torso centre and
  scaled by rig/user shoulder width and reach, so touching your hands
  together brings the rig's hands together even though this rig's
  shoulders are 0.83 m apart. Arm straightness comes from the tracked elbow
  angle (blended in as the arm straightens) instead of the raw wrist depth,
  which MediaPipe compresses when the arm points at the camera. For
  strongly bent arms (fist up) the upper arm follows the tracked
  shoulder→elbow direction directly.
- Fingers are driven per phalanx from landmark-pair directions, mapped
  through the bone's parent-local space with a shortest-arc rotation.
- Child bones use the parent's matrix computed *this frame* including its
  pose scale/location, so a stray -1 scale on an arm bone no longer inverts
  the whole chain (and **Reset Pose on Start** clears such leftovers anyway).
- **Recording** keeps each frame's mapped-bone rotations plus the
  armature-space elbow/wrist/palm positions. The positions are what makes
  cleanup tractable: a bad frame is one whose joints sit far from the
  midpoint of their neighbours, which is one number per frame instead of a
  quaternion per bone. Constant-velocity motion has a deviation of zero no
  matter how fast it is, so honest movement doesn't get flagged; a
  snap-and-return does. The threshold is median + k×MAD over the take, so a
  few violent snaps can't raise the bar until they look normal, and it is
  divided by the rig's shoulder width so the same settings work at any rig
  scale. Actions are built through Blender 4.4+'s layered/slotted Action API
  (`Action.fcurves` no longer exists), with a fallback for 3.6–4.3.

## Tips

- Selfie mode: keep **Mirror Webcam** on. Body Cam mode: turn it off.
- Accuracy checklist: correct **Camera FOV**; **Model Complexity** 2 if the
  machine keeps up; even, front-facing lighting; sit so your shoulders and
  hips are in frame during calibration; keep the camera at chest height so
  the shoulder line stays roughly parallel to the sensor.
- If hands end up pointing the wrong way, press **Calibrate** and make sure
  you are really holding the rig's rest pose during the countdown.
- Hands don't quite meet on the FPS arms rig: its shoulders are 0.83 m apart
  with 0.5 m of reach, so its own centre line is at the edge of its reach.
- Raise **Smoothing** to reduce jitter (at the cost of some latency).
- Lower **Model Complexity** to 0 for faster tracking on slow machines.
- If dependency install fails, run Blender's Python manually:
  ```
  "C:\Program Files\Blender Foundation\Blender X.Y\X.Y\python\bin\python.exe" -m pip install opencv-python mediapipe numpy
  ```
