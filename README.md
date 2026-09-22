# FPV Pose Tracker

A Blender add-on that reads a webcam feed and drives an armature in real time
using [MediaPipe Holistic](https://developers.google.com/mediapipe) pose and
hand-landmark estimation. Designed for a first-person body-cam view: the
tracked arms and hands are retargeted so they appear as the player's arms
extending forward into the camera frame.

Every arm joint (shoulder, upper arm, forearm, hand) plus every finger and
thumb joint (three phalanges per digit, both hands) is driven independently.

## Build

```bash
python build.py
# produces dist/fpv_pose_tracker-<version>.zip
```

## Install

1. In Blender: `Edit > Preferences > Add-ons > Install...`, pick the zip.
2. Enable **FPV Pose Tracker**.
3. On first use the panel will show an **Install Dependencies** button that
   pip-installs `opencv-python`, `mediapipe`, and `numpy` into Blender's
   bundled Python. Blender may need to be run as admin the first time on
   Windows. Restart Blender afterwards if imports still fail.

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

## How it works

- A worker thread reads frames from OpenCV (`cv2.VideoCapture`) and runs
  MediaPipe pose + hand landmarkers on each frame.
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
