"""Person segmentation with SAM 3.1: one entry point, `track`, and two prompting modes.

`prompt` - SAM 3 finds the person itself. Every frame is scored against one text prompt and
nothing else: no detector box, no keypoints, no points of any kind. The first detection that
beats `birth_threshold` starts the track and the tracker's memory carries the mask from
there. This is Meta's own SAM 3 video policy, run over core's weight-carrying primitives:

- a detection scores sigmoid(query) * sigmoid(presence), not the query alone: the presence
  head is the model saying whether the concept is in this frame at all, and on this prompt
  it is the entire signal (see SAM3Config)
- detections are thinned by NMS on plain IoU, because several queries fire on one body
- a track is born from the first detection that beats the birth threshold
- for its first `hotstart_frames` frames a track is on probation: if the detector fails to
  find it on `hotstart_unmatched` of them it was never a person - a poster, a reflection, a
  figure in the background that scored once - so it is killed, everything it produced is
  thrown away and the search for a birth starts again. That is why the output is buffered
  until the track is out of its window.
- every `recondition_every` frames a detection that still agrees with the track re-anchors
  it as a conditioning frame, and the non-conditioning memories around it are cleared so the
  fresh anchor is not outvoted by the drift it is there to correct
- the tracker only moves forwards, so once the clip is done the frames before the birth are
  filled by propagating over them in a mirrored index space (see `propagate_backwards`)

With `max_objects` above 1 the same policy runs once per track and Meta's multi-object rules
decide between the tracks (`segment_by_prompt_multi`); at 1 none of that code runs.

`box_keypoint` - there is no text prompt. The person is described to SAM by what the pose
pipeline already knows about each frame (`pose_data`): the detector's box and the body
keypoints, as a box and positive points on the joints, along the limbs and down the torso,
plus negative points on the previous frame's background and on whatever the mask is known
to have annexed. That mask then propagates through the tracker's memory, which is what
carries the mask through motion blur, and the tracker is re-seeded from a fresh prompt every
`reseed_interval` frames and whenever its mask stops covering the keypoints - so the memory
cannot drift away from the person the way a tracker prompted only once does.

Both modes drive core's primitives (`_compute_backbone_frame`, `track_step`,
`_condition_with_masks`, `_deferred_memory_encode`, `_forward_sam_heads`) directly. Core's
`SAM3Model.forward_video` / `track_video_with_detection` are not used: their detection policy
is a much cruder one - it thresholds the raw query score, its NMS measures overlap as
max(IoU, IoM), it has no false-positive guard and no reconditioning.

The model is ComfyUI's own SAM3 implementation loaded from `models/checkpoints`
(`sam3.1_multiplex_fp16.safetensors`, fetched on first use when missing), so it is loaded
and offloaded by ComfyUI like every other model. The same checkpoint carries the CLIP the
prompt is encoded with.
"""
import json
import os
from dataclasses import dataclass, field, fields

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import comfy.sd
import folder_paths
from comfy import model_management as mm
from comfy.ldm.sam3.tracker import MultiplexState, _prep_frame, fill_holes_in_mask_scores
from comfy.ops import cast_to_input
from comfy.utils import ProgressBar, common_upscale

from . import log
from .bbox import parse_bboxes as parse_boxes
from .models.download import download

DEFAULT_SAM3 = "sam3.1_multiplex_fp16.safetensors"
DEFAULT_SAM3_URL = "https://huggingface.co/Comfy-Org/sam3.1/resolve/main/checkpoints/sam3.1_multiplex_fp16.safetensors"

MODE_PROMPT = "prompt"
MODE_BOX_KEYPOINT = "box_keypoint"
# prompt-only produces the better mask; box+keypoint is the v1 behaviour and the fallback
MODES = (MODE_PROMPT, MODE_BOX_KEYPOINT)

# What the person is called to SAM. A bare noun scores higher and more evenly across clips,
# but it also tracks whoever the detector likes best, which on these clips is as often a
# figure in the background; the phrase names the one the generation is about. The
# thresholds in SAM3Config were measured with this prompt.
PROMPT = "main person in the foreground"


def _field(default, lo, hi, step, doc):
    return field(default=default, metadata={"min": lo, "max": hi, "step": step, "tooltip": doc})


# The two sides of each A/B switch below: "ours" is the policy this module has always run,
# "meta" is Meta's (easy-sam3 @ 88fe578) ported onto core's primitives.
OURS, META = "ours", "meta"


def _choice(default, choices, doc):
    return field(default=default, metadata={"choices": tuple(choices), "tooltip": doc})


@dataclass
class SAM3Config:
    """The tunables of both modes. Each tooltip starts with the mode it affects, and a field is
    read in that mode only: a field changed from its default that the run does not read (the
    other mode's, the multi-object ones at max_objects 1, the tracker ones with `temporal` off
    in box_keypoint mode) is named in one console line, never raised on.

    `[prompt]` defaults are measured on SAM 3.1 with PROMPT (1116 frames across eight clips):
    the person's presence-corrected score never fell below 0.62, and the best detection that
    was not the person never passed 0.29 after NMS. (The query score alone carries no
    information here - it never left 0.96-0.995.) Meta thresholds the same presence-corrected
    score (sam3_image.py joins them into pred_logits), but its values were set on SAM 3.0, and
    SAM 3.1's scores sit on a different scale, so they do not transfer. The multi-object fields
    are Meta's own values (the IoU and frame-count thresholds of its video policy, as vendored
    by easy-sam3): they compare masks
    with masks, not scores, so they carry over. `[box_keypoint]` defaults are the swept values
    of the box+keypoint branch."""

    # --- [prompt] ---
    # Where the track is born. In the middle of the empty band: roughly twice the junk and
    # well under the weakest frame a clip has opened on (Meta's 0.70 sits 0.002 under one).
    birth_threshold: float = _field(0.50, 0.0, 1.0, 0.01, "[prompt] a detection this confident starts a track")
    # At the band's floor, so a frame where presence collapses still produces a detection
    # for the probation window to match the track against.
    detection_threshold: float = _field(0.30, 0.0, 1.0, 0.01, "[prompt] SAM 3.1 detections below this presence-corrected score are dropped (Pose Config's detection_threshold is the person detector's)")
    # Plain IoU, not core's max(IoU, IoM): IoM calls a hand inside a body the same detection.
    nms_iou: float = _field(0.10, 0.0, 1.0, 0.01, "[prompt] two detections that overlap this much are the same body")
    match_iou: float = _field(0.50, 0.0, 1.0, 0.01, "[prompt] a detection and a track are the same person above this IoU")
    hotstart_frames: int = _field(15, 1, 120, 1, "[prompt] how many frames a new track stays on probation")
    hotstart_unmatched: int = _field(8, 1, 120, 1, "[prompt] a new track unmatched on this many probation frames is dropped")
    # A conditioning frame every this many frames stops the mask drifting off the body.
    recondition_every: int = _field(16, 1, 512, 1, "[prompt] every this many frames an agreeing detection re-anchors a track")
    # The IoU is what makes an anchor safe - it has to be the same body the track already
    # has - so the score only has to keep the frames where presence collapsed from becoming
    # anchors. It sits at the lower quartile of the hardest clip measured, which is where
    # Meta's 0.80 already is; lower rejects nothing, the person never scored under 0.62.
    recondition_score: float = _field(0.80, 0.0, 1.0, 0.01, "[prompt] a detection must score this much to re-anchor a track")
    recondition_iou: float = _field(0.80, 0.0, 1.0, 0.01, "[prompt] and overlap the track this much")
    # Cleaning at the decoder's own 288x288 output rather than after the upsample is both
    # cheaper and sharper: bilinear upsampling smears a speck into something with area.
    fill_hole_area: int = _field(16, 0, 4096, 1, "[prompt] islands and holes smaller than this many pixels of the 288x288 decoder output are removed")
    # The tracker reads the last num_maskmem - 1 frames of non-conditioning memory; clearing
    # that many frames either side of a fresh anchor is what makes it an anchor.
    memory_gap: int = _field(7, 0, 64, 1, "[prompt] frames of non-conditioning memory cleared either side of a fresh anchor")
    # --- [prompt] A/B against Meta's policy (specs/mask-process.md, A1-A9) ---
    # Each switch defaults to ours; "meta" is Meta's behaviour ported onto core's primitives.
    # Which side wins is decided by the A/B on the test clips, not here. A6 and A7 are read with
    # any max_objects, the others with max_objects 1 only.
    anchor_memory: str = _choice(OURS, (OURS, "clear_past", META), "[prompt] A1, memory around a re-anchor. ours: the stored non-conditioning memory is cleared and none is encoded for memory_gap frames after; clear_past: cleared, then encoded as usual; meta: neither (Meta clears only its per-object copies, which propagation does not read)")
    anchor_output: str = _choice(OURS, (OURS, META), "[prompt] A2, a re-anchor frame. ours: outputs the detection and stores it as conditioning memory; meta: outputs the propagated mask and stores that mask's memory in the conditioning entry")
    conditioning_frames: str = _choice(OURS, (OURS, META), "[prompt] A3, conditioning frames attended. ours: the birth frame and the newest anchor; meta: the 4 newest, the birth frame not kept")
    memory_selection: str = _choice(OURS, (OURS, META), "[prompt] A4, spatial memory read. ours: the last 6 frames; meta: the last 6 frames whose object score says the person was there (Meta's memory selection, by object score alone: core does not return the decoder's IoU)")
    anchor_score_gate: str = _choice(OURS, (OURS, META), "[prompt] A5, re-anchoring. ours: any agreeing detection; meta: only while the tracker's own object-score logit is above 0.8")
    anchor_matching: str = _choice(OURS, (OURS, META), "[prompt] A6, which detection re-anchors which track. ours: each track takes the best-scoring detection agreeing with it; meta: each detection goes to the one track it overlaps most (the last such detection wins)")
    unmatched_counting: str = _choice(OURS, (OURS, META), "[prompt] A7, probation. ours: a frame the detector misses counts as unmatched even when the track's mask is empty; meta: only when the mask is not empty")
    seed_cleaning: str = _choice(OURS, (OURS, META), "[prompt] A9, a detection that starts or re-anchors a track. ours: its specks and pinholes are removed before it conditions the tracker; meta: the tracker gets it raw and only the birth frame's output is cleaned")
    # --- [prompt, max_objects > 1]: only read when several objects are tracked ---
    # Every track runs the single-object policy above unchanged; these decide between tracks.
    # A further person is born from a detection this confident while another track is live
    # (the first track is still born at birth_threshold). Measured on two single-person clips
    # side by side (513 frames, two pairs): with PROMPT the second person scored 0.31-0.65 at
    # worst and 0.42-0.75 at the 5th percentile, the best non-person detection 0.12; with the
    # prompt "person" the people never fell below 0.95 and the best non-person reached 0.29.
    # 0.50 is above every non-person detection under either prompt, and every run bore both
    # people on frame 0. The phrase in PROMPT favours one person, so for several people a
    # plain noun gives the wider band.
    new_object_threshold: float = _field(0.50, 0.0, 1.0, 0.01, "[prompt, max_objects > 1] a detection this confident that touches no track starts a further track")
    # Meta's assoc_iou_thresh: loose on purpose, a detection that touches a track at all is
    # that track's, not a new person.
    assoc_iou: float = _field(0.10, 0.0, 1.0, 0.01, "[prompt, max_objects > 1] a detection overlapping any track this much is not a new object")
    # Meta's hotstart_dup_thresh: two tracks that keep claiming the same detection are one
    # person; the younger one goes if it is still on probation.
    duplicate_frames: int = _field(8, 1, 120, 1, "[prompt, max_objects > 1] a track on probation sharing a detection with an older track on this many frames is a duplicate and dropped")
    # Meta's suppress_overlapping_based_on_recent_occlusion_threshold: of two tracks this
    # overlapped, the one that came out of occlusion more recently is the one that jumped
    # onto the other person, and it is blanked on that frame.
    occlusion_iou: float = _field(0.70, 0.0, 1.0, 0.01, "[prompt, max_objects > 1] of two tracks overlapping this much, the more recently occluded one is suppressed")
    # Meta's _suppress_shrinked_masks: a track that keeps less than this share of its area
    # once every pixel goes to the highest-scoring track is noise under another one, and is
    # kept out of the memory.
    shrink_keep: float = _field(0.30, 0.0, 1.0, 0.01, "[prompt, max_objects > 1] a track keeping less than this share of its area against the others is kept out of the memory")

    # --- [box_keypoint] ---
    # How long the tracker may propagate before it is re-seeded from a fresh prompt. The
    # tracker's own object score decays without reconditioning (the mask dies after a few
    # hundred frames), so it is re-anchored well before that; the memory is what keeps the
    # mask through frames the pose model loses to motion blur.
    reseed_interval: int = _field(24, 1, 512, 1, "[box_keypoint] frames propagated before the tracker is re-seeded from an anchor frame; ignored with temporal off")
    # Propagating for longer than this without an anchor is not worth the drift risk: re-seed
    # from whatever the frame offers. Three re-seed intervals.
    max_propagate: int = _field(72, 1, 2048, 1, "[box_keypoint] frames propagated at most before re-seeding from whatever the frame offers; ignored with temporal off")
    # Re-seeding only helps from a frame the pose model is sure about: re-seeding on schedule
    # landed on a motion-blurred frame, which replaced a good propagated mask with a bad
    # prompt. A frame is an anchor when the detector found the person, enough keypoints are
    # confident, they are confident on average, and the pose is no less complete than the one
    # the current segment was seeded from - the blurred frames lose the knees while the head
    # and the torso stay confident, so the count is what tells them apart.
    min_anchor_keypoints: int = _field(8, 1, 20, 1, "[box_keypoint] an anchor frame has at least this many confident body keypoints; ignored with temporal off")
    min_anchor_conf: float = _field(0.5, 0.0, 1.0, 0.01, "[box_keypoint] and this mean confidence over them; ignored with temporal off")
    min_anchor_completeness: float = _field(0.9, 0.0, 1.0, 0.01, "[box_keypoint] and at least this share of the keypoints the running segment was seeded with; ignored with temporal off")
    # A propagated mask that stops covering this fraction of the frame's confident keypoints
    # has come off the person: re-seed immediately.
    min_tracked_recall: float = _field(0.9, 0.0, 1.0, 0.01, "[box_keypoint] a propagated mask covering less of the confident keypoints than this is re-seeded; ignored with temporal off")
    # The decoder is told what the person is, never what it is not, and on a close-up in a
    # cluttered room it annexes whatever is adjacent - a stuffed toy, a patch of wall. These
    # many points of the previous frame's background, inside the box and this far (as a
    # fraction of the box diagonal) from the mask it had, are given as negative prompts.
    negative_points: int = _field(8, 0, 64, 1, "[box_keypoint] negative points on the previous frame's background inside the box")
    negative_margin: float = _field(0.04, 0.0, 0.5, 0.01, "[box_keypoint] how far, in box diagonals, those points stay from the previous mask")
    # The clearance above is exactly what the annexed object falls into: it is adjacent to the
    # body, so no point from the previous frame's background ever lands on it. What the mask
    # gains from one frame to the next with no confident keypoint anywhere in it is that
    # annexation caught happening, and it is the only evidence that names the object. Regions
    # this big against the mask are remembered and given as negative points at every later
    # prompt, until the person moves onto them. The size is where a shadow on a wall stops being
    # caught (5% left one growing until it read as a second object) and where the negative points
    # start eating the person instead (2% cost a clip a point of peak mask coverage).
    min_annexed_fraction: float = _field(0.03, 0.0, 1.0, 0.005, "[box_keypoint] a keypoint-free region the mask gains, this large against the mask, is remembered as background")
    # 4 left 2 fragmented / 24 specks on the hardest clip, 8 left 1 / 20, 12 left 3 / 22.
    annexed_points: int = _field(8, 0, 64, 1, "[box_keypoint] negative points spread over the remembered annexed background")
    # Where the mask logits are cut. Slightly below zero because the boundary is soft exactly
    # where the thin parts are - loose hair, fingers, the edge of a foot - and they were left
    # a few pixels outside the mask, which the block mask downstream then makes obvious.
    mask_threshold: float = _field(-1.0, -10.0, 10.0, 0.1, "[box_keypoint] the prompted mask's logits are cut here; with uniform_mask_threshold on, every frame of both modes")
    # Islands smaller than this fraction of the largest region are decoder noise (specks in
    # shadows and edges), not the person; downstream block masks would blow them up.
    min_island_fraction: float = _field(0.01, 0.0, 1.0, 0.005, "[box_keypoint] regions smaller than this share of the largest one are removed")
    # A hole is only evidence of dropped clothing or hair while it is small against the body.
    # The background between a raised arm and the torso is enclosed just as completely, and
    # filling that swallows the whole triangle. Measured on eight clips, every enclosed hole
    # above this fraction of the mask's own area was such a gap (the smallest confirmed one was
    # 1.2%) and every one below it was clothing or hair the decoder had dropped.
    max_hole_fraction: float = _field(0.01, 0.0, 1.0, 0.005, "[box_keypoint] enclosed holes up to this share of the mask's area are filled")
    refine: bool = field(default=True, metadata={"tooltip": "[box_keypoint] feed the prompted mask back to the decoder once to refine it"})
    temporal: bool = field(default=True, metadata={"tooltip": "[box_keypoint] propagate with the tracker between prompts; off prompts every frame on its own (not Pose Config's temporal)"})

    # --- [all modes] (specs/mask-process.md M4) ---
    # Off keeps each mode's own cuts: prompt mode at 0 everywhere, box_keypoint at
    # mask_threshold on the frames it prompts and at 0 on the frames the tracker propagates -
    # which is why its mask gains area at every re-seed. On, one threshold cuts everything.
    uniform_mask_threshold: bool = field(default=False, metadata={"tooltip": "[all modes] M4, on: mask_threshold cuts every frame's mask logits in both modes (prompted, propagated, tracked backwards, every object); off: prompt mode cuts at 0, box_keypoint cuts its prompted frames at mask_threshold and its propagated frames at 0"})
    # Prompt mode's birth frame and every re-anchor frame show the mask the tracker is
    # conditioned with, binarised at 0 to +/-10, so with M4 on mask_threshold does not move
    # their boundary and the area steps every recondition_every frames. Last, after the M4
    # switch it depends on, so saved workflows keep their widget positions.
    m4_anchor_frames: str = _choice("mask", ("mask", "output"), "[prompt] M4, with uniform_mask_threshold on and max_objects 1: what the birth and re-anchor frames show. mask: the binarised mask the tracker is conditioned with, which mask_threshold does not move; output: the detection's logits cut at mask_threshold like every other frame (a birth with seed_cleaning meta and an anchor with anchor_output meta already show that). The tracker is conditioned the same either way")

    def __post_init__(self):
        for f in fields(self):
            choices = f.metadata.get("choices")
            if choices and getattr(self, f.name) not in choices:
                raise ValueError(f"SAM3Config.{f.name} is {getattr(self, f.name)!r}; it must be one of {choices}")


# The [box_keypoint] fields only the tracker reads: with `temporal` off nothing is propagated.
TRACKER_FIELDS = ("reseed_interval", "max_propagate", "min_anchor_keypoints", "min_anchor_conf",
                  "min_anchor_completeness", "min_tracked_recall")
# The [prompt] A/B switches the single-object policy reads and segment_by_prompt_multi does not.
SINGLE_OBJECT_FIELDS = ("anchor_memory", "anchor_output", "conditioning_frames", "memory_selection",
                        "anchor_score_gate", "seed_cleaning", "m4_anchor_frames")


def changed_fields(config, tag=None, names=()):
    """The fields of `config` whose tooltip starts with `tag`, or that are named in `names`, and
    that differ from their default: what a run that does not read them has to say it ignored."""
    return [f.name for f in fields(config)
            if ((tag is not None and f.metadata["tooltip"].startswith(tag + " ")) or f.name in names)
            and getattr(config, f.name) != f.default]


# Conditioning frames are attended in full and forever, so a clip's worth of them would
# grow the memory attention without bound. Two is what the policy needs: the birth frame,
# which says which person this is, and the newest anchor, which says where they are now.
MAX_CONDITIONING_FRAMES = 2
# Meta's side of the switches: max_cond_frames_in_attn (A3), the tracker object-score logit a
# re-anchored track must have (A5, HIGH_CONF_THRESH on the raw logit), and the memory
# selection threshold on the rescaled object score (A4, mf_threshold).
META_CONDITIONING_FRAMES = 4
META_ANCHOR_OBJECT_LOGIT = 0.8
META_MEMORY_SCORE = 0.01

# --- box_keypoint mode layout constants ----------------------------------------------------

SAM3_SIZE = 1008
# Body keypoints used as positive points: nose, neck, both shoulders, both hips, both ankles
# and both feet in the body layout (see guard.BODY_NAMES). The feet matter because the ankle
# is at the top of the foot: prompted from the ankle alone the decoder cut the shoe off.
PROMPT_KEYPOINTS = (0, 1, 2, 5, 8, 11, 10, 13, 18, 19)
R_SHOULDER, L_SHOULDER, R_HIP, L_HIP = 2, 5, 8, 11
# The body layout carries one foot point per side, the midpoint of that side's two toe
# keypoints; pose2d_utils.split_kp2ds_for_aa averages wholebody 17/18 into the left one and
# 20/21 into the right one, and drops the heels, so these are the only feet there are.
R_ANKLE, L_ANKLE, R_FOOT, L_FOOT = 10, 13, 19, 18
# Keypoints sit on joints, so the body between them carries no positive evidence and the
# decoder drops parts of it: a close-up lost the lower half of a jacket, and a dancer's
# legs dropped out of the mask on the frames they moved fastest, while the knees the pose
# model was sure of sat outside it. These fractions along a limb or the torso put points
# on the body itself, halfway between the joints.
TORSO_FRACTIONS = (0.35, 0.65)
LIMB_FRACTIONS = (0.5,)
# Limbs to put a point on, as (from, to) keypoints: thighs, shins, upper arms and feet.
LIMBS = ((R_HIP, 9), (L_HIP, 12), (9, R_ANKLE), (12, L_ANKLE), (R_SHOULDER, 3), (L_SHOULDER, 6),
         (R_ANKLE, R_FOOT), (L_ANKLE, L_FOOT))
# A binary mask handed to the tracker as logits: +/- this value.
MASK_LOGIT_SCALE = 10.0
# A hand-placed point is the user's word on that spot and wins over the points derived from
# the pose and the previous mask: a derived point of the opposite label closer to it than
# this share of the box diagonal would contradict it in the same prompt, so it is dropped.
# Hand-placed points themselves are always passed as given.
HAND_POINT_CLEARANCE = 0.04


# --- model ---------------------------------------------------------------------------------

_loaded = {"name": None, "model": None, "clip": None}


def load_sam3(name=DEFAULT_SAM3):
    """(model, clip): the SAM3 checkpoint `name` as a ComfyUI model and the CLIP that encodes
    the prompt (None when the checkpoint carries no text encoder), loaded once and kept; the
    default checkpoint is downloaded into models/checkpoints when it is missing."""
    if _loaded["name"] == name:
        return _loaded["model"], _loaded["clip"]
    path = folder_paths.get_full_path("checkpoints", name)
    if path is None and name == DEFAULT_SAM3:
        path = os.path.join(folder_paths.get_folder_paths("checkpoints")[0], name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        download(DEFAULT_SAM3_URL, path)
    if path is None:
        raise FileNotFoundError(f"SAM3 checkpoint {name} is not in models/checkpoints")
    _loaded["name"], _loaded["model"], _loaded["clip"] = None, None, None
    with log.step(f"loading {name}"):
        model, clip = comfy.sd.load_checkpoint_guess_config(path, output_vae=False, output_clip=True)[:2]
    _loaded["name"], _loaded["model"], _loaded["clip"] = name, model, clip
    return model, clip


def _multiplex_parts(model):
    """(sam3, detector, tracker, vision backbone) of a loaded multiplex SAM 3.1 checkpoint,
    raising when the checkpoint or this ComfyUI does not have what both modes drive."""
    sam3 = model.model.diffusion_model
    detector, tracker = getattr(sam3, "detector", None), getattr(sam3, "tracker", None)
    backbones = getattr(detector, "backbone", None)
    if not isinstance(backbones, (dict, torch.nn.ModuleDict)) or "vision_backbone" not in backbones:
        raise ValueError(f"expected a SAM3 checkpoint with a detector vision backbone, found {type(sam3).__name__}")
    backbone = backbones["vision_backbone"]
    if not getattr(backbone, "multiplex", False):
        raise ValueError("this node needs a multiplex SAM 3.1 checkpoint; a plain SAM 3 has "
                         f"no detector-driven tracker. Use {DEFAULT_SAM3}")
    missing = [n for n in ("_compute_backbone_frame", "track_step", "_condition_with_masks",
                           "_deferred_memory_encode", "_forward_sam_heads") if not hasattr(tracker, n)]
    if missing:
        raise ValueError(f"this ComfyUI's SAM3 tracker has no {', '.join(missing)}; the tracking "
                         "loop is built on those primitives. Update ComfyUI")
    return sam3, detector, tracker, backbone


def _propagation_backbone(backbone):
    """The backbone function the tracker's `_compute_backbone_frame` calls: the trunk once,
    then the propagation neck on the cached trunk."""
    def backbone_fn(frame, frame_idx=None):
        trunk_out = backbone.trunk(frame)
        _, _, feats, positions = backbone(frame, tracker_mode="propagation",
                                          cached_trunk=trunk_out, tracker_only=True)
        return feats, positions, trunk_out
    return backbone_fn


def iou(masks_a, masks_b):
    """[A, B] intersection over union of two sets of mask logits, thresholded at 0."""
    a = (masks_a > 0).float().flatten(1)
    b = (masks_b > 0).float().flatten(1)
    intersection = a @ b.T
    return intersection / (a.sum(1, keepdim=True) + b.sum(1, keepdim=True).T - intersection).clamp(min=1)


# --- prompt mode ---------------------------------------------------------------------------

def encode_prompt(clip, detector, prompt, device, dtype):
    """`prompt` as the (embedding, attention mask) pair the detector wants, encoded the way
    CLIPTextEncode encodes it and put through the language backbone's resizer."""
    conditioning = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
    embedding = conditioning[0][0].to(device=device, dtype=dtype)
    mask = conditioning[0][1].get("attention_mask")
    if mask is None:
        mask = torch.ones(embedding.shape[0], embedding.shape[1], dtype=torch.int64)
    resizer = detector.backbone["language_backbone"]["resizer"]
    return resizer(embedding), mask.to(device).bool()


def detect_person(detector, backbone, trunk_out, embedding, text_mask, config):
    """The detections on one frame, sorted by score and thinned by NMS: (masks, scores).

    This is core's `forward_from_trunk` with one line added - it throws away `dec_out`, and
    with it the presence logit that the score is meant to be multiplied by."""
    features = [conv(trunk_out) for conv in backbone.convs]
    positions = [cast_to_input(backbone.position_encoding(f), f) for f in features]
    _, scores, masks, dec_out = detector._detect(features, positions, embedding, text_mask)
    scores = (scores.sigmoid() * dec_out["presence"].sigmoid())[0].float()
    masks = masks[0]

    order = scores.argsort(descending=True)
    order = order[scores[order] >= config.detection_threshold]
    masks, scores = masks[order], scores[order]
    keep = []
    for i in range(masks.shape[0]):
        if keep and iou(masks[i:i + 1], masks[keep]).max() >= config.nms_iou:
            continue
        keep.append(i)
    return masks[keep], scores[keep]


def clean_logits(masks, fill_hole_area):
    """Mask logits with the decoder's specks and pinholes taken out, at the decoder's own
    resolution. Shape is [n, H, W] in, [n, H, W] out; fill_holes_in_mask_scores wants the
    channel."""
    return fill_holes_in_mask_scores(masks.unsqueeze(1).float(), max_area=fill_hole_area)[:, 0]


def to_frame_size(low_res, H, W, threshold=0.0):
    """One frame's [1, 1, h, w] mask logits as an [H, W] float mask of the frame, cut at
    `threshold`."""
    upsampled = F.interpolate(low_res.float(), size=(H, W), mode="bilinear", align_corners=False)
    return (upsampled[0, 0] > threshold).float().cpu()


def output_cut(config):
    """The logit threshold prompt mode's masks and box_keypoint's propagated masks are cut at:
    0, or mask_threshold with uniform_mask_threshold on (M4)."""
    return config.mask_threshold if config.uniform_mask_threshold else 0.0


def shown_logits(raw, cut, fill_hole_area, cleaned=None):
    """The [n, 1, h, w] logits a prompt-mode frame shows, from the tracker's or detector's `raw`
    ones: `cleaned` (clean_logits of `raw`, the memory's copy; computed when not given) when
    the output is cut at 0. At any other cut (M4) the specks and pinholes are cleaned relative
    to the cut: cleaning at 0 sets a removed speck to -0.1 and a filled hole to 0.1, which a cut
    of -1 would put straight back in (and one of +1 back out). Only the pixels the cleaning
    changes are moved, to just past the cut; every other pixel keeps its raw value."""
    if cut == 0:
        return cleaned if cleaned is not None else clean_logits(raw[:, 0], fill_hole_area).unsqueeze(1)
    raw = raw[:, 0].float()
    relative = clean_logits(raw - cut, fill_hole_area)
    return torch.where(relative != raw - cut, relative + cut, raw).unsqueeze(1)


def low_res_logits(masks):
    """The [h, w] logits of a [1, 1, h, w] (or [1, h, w]) tensor as the fp16 CPU copy the logits
    dump keeps."""
    return masks.reshape(masks.shape[-2:]).detach().to("cpu", torch.float16).clone()


def seed_from(det_masks, index, config):
    """The detection that starts or re-anchors a track, as it is handed to the tracker: cleaned
    (ours) or raw (Meta, A9)."""
    if config.seed_cleaning == OURS:
        return clean_logits(det_masks[index:index + 1], config.fill_hole_area)
    return det_masks[index:index + 1].float()


def choose_anchor(det_scores, overlap, config):
    """The detection that re-anchors the single track, or None: of the detections scoring
    recondition_score and overlapping the track by recondition_iou, the first - the best
    scoring, they are sorted - (ours) or the last (Meta's per-detection loop, where a later
    detection overwrites an earlier one for the same track, A6). `overlap` is [D]."""
    fresh = ((det_scores >= config.recondition_score) & (overlap >= config.recondition_iou)).nonzero()
    if not fresh.shape[0]:
        return None
    return int(fresh[0]) if config.anchor_matching == OURS else int(fresh[-1])


def anchor_detections(det_scores, overlap, tracks, config):
    """{track: detection} re-anchoring several tracks; `overlap` is [D, K], `tracks` the columns
    still live. Ours: every track takes the first detection agreeing with it, so two tracks can
    take one detection. Meta (A6): every agreeing detection goes to the one track it overlaps
    most, the last detection winning a track two agree with."""
    anchors = {}
    if config.anchor_matching == OURS:
        for k in tracks:
            fresh = ((det_scores >= config.recondition_score) & (overlap[:, k] >= config.recondition_iou)).nonzero()
            if fresh.shape[0]:
                anchors[k] = int(fresh[0])
        return anchors
    if not tracks:
        return anchors
    columns = torch.tensor(tracks, device=overlap.device)
    for d in range(overlap.shape[0]):
        best, j = overlap[d, columns].max(dim=0)
        if float(det_scores[d]) >= config.recondition_score and float(best) >= config.recondition_iou:
            anchors[tracks[int(j)]] = d
    return anchors


def counts_unmatched(matched, track_masks, config):
    """Whether a probation frame counts against the track: unmatched, and for Meta (A7) only
    when the track's mask is not empty."""
    return not matched and (config.unmatched_counting == OURS or bool((track_masks > 0).any()))


def prune_conditioning(conditioned, how):
    """Drop conditioning frames past the limit (A3). Ours keeps the birth frame, which says
    which person this is, and the newest anchor, which says where they are now; Meta keeps the
    META_CONDITIONING_FRAMES temporally closest, which for a forward pass are the newest."""
    limit, keep_first = (MAX_CONDITIONING_FRAMES, True) if how == OURS else (META_CONDITIONING_FRAMES, False)
    while len(conditioned) > limit:
        del conditioned[sorted(conditioned)[1 if keep_first else 0]]


def object_present(out):
    """Meta's memory score for a stored frame, by the object score alone (A4): the logit,
    rescaled to 0..1 above 0, has to pass META_MEMORY_SCORE. Meta multiplies in the decoder's
    predicted IoU, which core's track_step does not return."""
    logit = float(out["object_score_logits"].float().max())
    return logit > 0 and (torch.sigmoid(torch.tensor(logit)).item() * 2 - 1) > META_MEMORY_SCORE


def memory_view(output_dict, frame_idx, tracker):
    """The output_dict the tracker should read on `frame_idx` under Meta's memory selection
    (A4): the frame before, plus the newest stored frames the person was present on, placed at
    frame_idx - 1, frame_idx - 2, ... in that order. Core's lookup reads memories and object
    pointers by index, so re-indexing them is how the selected frames reach it - the temporal
    positions become ranks, as in Meta's frame_filter."""
    stored = output_dict["non_cond_frame_outputs"]
    selected = [t for t in sorted(stored, reverse=True) if t < frame_idx and object_present(stored[t])]
    selected = selected[:tracker.max_obj_ptrs_in_encoder - 1]
    if frame_idx - 1 in stored and frame_idx - 1 not in selected:
        selected.insert(0, frame_idx - 1)   # the frame before is always read, present or not
    return {"cond_frame_outputs": output_dict["cond_frame_outputs"],
            "non_cond_frame_outputs": {frame_idx - rank: stored[t] for rank, t in enumerate(selected, start=1)}}


def forget_old_memory(stored, frame_idx, lookback, how):
    """Drop the non-conditioning outputs the lookup cannot reach any more. Ours: older than
    `lookback` frames. Meta (A4): all but the newest frame and the newest lookback - 1 frames the
    person was present on, however old."""
    if how == OURS:
        for old in list(stored):
            if old < frame_idx - lookback:
                del stored[old]
        return
    present = [t for t in sorted(stored, reverse=True) if t != frame_idx and object_present(stored[t])]
    keep = {frame_idx, *present[:lookback - 1]}
    for old in list(stored):
        if old not in keep:
            del stored[old]


def propagate_backwards(tracker, backbone, backbone_fn, frames, seed, birth, emit, device, dtype, config):
    """Fill in the frames before `birth` by tracking backwards from the detection that
    started the track: `emit(frame, output, raw)` is called with the tracker's output for every
    frame from birth - 1 down to 0, and the logits it had before they were cleaned.

    Core's memory lookups only run forwards - `collect_memory_tokens` asks for frame_idx - 1,
    frame_idx - 2 and so on - so the frames go in mirrored: frame t is tracked under the
    index MIRROR - t, which turns "the frame before" into the real frame after it, the one
    already tracked. Nothing else in the tracker is direction-aware, because every position
    encoding it builds from those indices is relative. This pass propagates and nothing
    else: no detection, no reconditioning, no probation. It is filling a gap at the head of
    the clip from a track that has already proved itself, not deciding anything."""
    N = frames.shape[0]
    mirror = 2 * N
    size = tracker.image_size
    output_dict = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
    mux = MultiplexState(1, tracker.num_multiplex, device, dtype)

    frame = _prep_frame(frames, slice(birth, birth + 1), device, dtype, size)
    vision_feats, vision_pos, feat_sizes, high_res, trunk_out = tracker._compute_backbone_frame(
        backbone_fn, frame, frame_idx=birth)
    tracker._condition_with_masks(seed, mirror - birth, vision_feats, vision_pos, feat_sizes,
                                  high_res, output_dict, N, mux, backbone, frame, trunk_out, threshold=0.0)

    for f in range(birth - 1, -1, -1):
        frame = _prep_frame(frames, slice(f, f + 1), device, dtype, size)
        vision_feats, vision_pos, feat_sizes, high_res, _ = tracker._compute_backbone_frame(
            backbone_fn, frame, frame_idx=f)
        current = tracker.track_step(
            frame_idx=mirror - f, is_init_cond_frame=False, current_vision_feats=vision_feats,
            current_vision_pos_embeds=vision_pos, feat_sizes=feat_sizes, mask_inputs=None,
            output_dict=output_dict, num_frames=N, propagation_high_res=high_res,
            multiplex_state=mux, run_mem_encoder=False)
        raw = current["pred_masks"]
        current["pred_masks"] = clean_logits(raw[:, 0], config.fill_hole_area).unsqueeze(1)
        tracker._deferred_memory_encode(current, 1, vision_feats, feat_sizes, mux, device)
        output_dict["non_cond_frame_outputs"][mirror - f] = current
        for old in list(output_dict["non_cond_frame_outputs"]):
            if old < mirror - f - max(tracker.num_maskmem, tracker.max_obj_ptrs_in_encoder):
                del output_dict["non_cond_frame_outputs"][old]
        emit(f, current, raw)


def _logits_record(logits, N):
    """The per-frame slots a segment function fills for the logits dump, or None when nobody
    asked: `logits` is the caller's dict, given "logits" (the [h, w] fp16 low-res logits each
    output frame was cut from, None for a frame without output) and "cut" (how: "prompt",
    "prompted" or "propagated"; segment_by_prompt marks its birth frame "birth" and its
    re-anchor frames "anchor"). segment_by_prompt adds "raw": whether a frame's logits are the
    ones before shown_logits cleaned them."""
    if logits is None:
        return None
    logits.update({"logits": [None] * N, "cut": [None] * N})
    return logits


def segment_by_prompt(model, clip, images, prompt, config, result=None, logits=None):
    """[N, H, W] float masks of the person in `images` [N, H, W, 3], from the text prompt
    alone. `result`, if given, is filled with what happened for the log; `logits`, if given, a
    dict, receives each output frame's low-res logits (see `_logits_record`).

    The [prompt] A/B switches of `config` pick ours or Meta's side of each policy step; at
    their defaults this is the policy described in the module docstring."""
    if clip is None:
        raise ValueError("the SAM3 checkpoint carries no text encoder, so the prompt cannot be "
                         f"encoded; use a full SAM3 checkpoint such as {DEFAULT_SAM3}")
    c = config
    N, H, W, _ = images.shape
    device, dtype = mm.get_torch_device(), model.model.get_dtype()
    frames = images[..., :3].movedim(-1, 1)

    mm.load_model_gpu(model)
    _, detector, tracker, backbone = _multiplex_parts(model)
    embedding, text_mask = encode_prompt(clip, detector, prompt, device, dtype)
    size = tracker.image_size
    backbone_fn = _propagation_backbone(backbone)
    lookback = max(tracker.num_maskmem, tracker.max_obj_ptrs_in_encoder)
    cut = output_cut(c)
    # M4: birth and anchor frames show the detection cut at `cut`, not the conditioning mask
    detection_shown = c.uniform_mask_threshold and c.m4_anchor_frames == "output"
    record = _logits_record(logits, N)
    if record is not None:
        record["raw"] = [False] * N

    masks = torch.zeros(N, H, W)
    output_dict = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
    mux = seed = None
    birth = -1            # the frame the live track was born on, -1 while there is none
    unmatched = 0         # frames of the probation window the track went unmatched
    quiet_until = -1      # no non-conditioning memory is kept up to here, after an anchor
    pending = []          # (frame, mask, logits, raw, how) produced by a track still on probation
    counts = {"false starts": 0, "reconditioned": 0}
    pbar = ProgressBar(N)

    def put(f, mask, low, raw, how="prompt"):
        masks[f] = mask
        if record is not None:
            record["logits"][f], record["cut"][f], record["raw"][f] = low, how, raw

    with torch.inference_mode():
        for f in range(N):
            frame = _prep_frame(frames, slice(f, f + 1), device, dtype, size)
            vision_feats, vision_pos, feat_sizes, high_res, trunk_out = tracker._compute_backbone_frame(
                backbone_fn, frame, frame_idx=f)
            det_masks, det_scores = detect_person(detector, backbone, trunk_out, embedding, text_mask, c)
            how = "prompt"

            if birth < 0:
                if det_scores.numel() == 0 or det_scores[0] < c.birth_threshold:
                    pbar.update(1)
                    continue
                mux = MultiplexState(1, tracker.num_multiplex, device, dtype)
                seed = seed_from(det_masks, 0, c)
                current = tracker._condition_with_masks(
                    seed, f, vision_feats, vision_pos, feat_sizes, high_res,
                    output_dict, N, mux, backbone, frame, trunk_out, threshold=0.0)
                birth, unmatched, quiet_until, how = f, 0, -1, "birth"
                # Meta tracks the raw detection but outputs it cleaned (A9). `dumped`: the
                # logits before the output's cleaning, None when the frame shows the
                # conditioning mask itself
                if c.seed_cleaning == OURS and not detection_shown:
                    shown, dumped = current["pred_masks"], None
                else:
                    dumped = det_masks[:1].unsqueeze(1)
                    shown = shown_logits(dumped, cut, c.fill_hole_area)
            else:
                lookup = output_dict if c.memory_selection == OURS else memory_view(output_dict, f, tracker)
                current = tracker.track_step(
                    frame_idx=f, is_init_cond_frame=False, current_vision_feats=vision_feats,
                    current_vision_pos_embeds=vision_pos, feat_sizes=feat_sizes, mask_inputs=None,
                    output_dict=lookup, num_frames=N, propagation_high_res=high_res,
                    multiplex_state=mux, run_mem_encoder=False)
                raw = current["pred_masks"]
                current["pred_masks"] = clean_logits(raw[:, 0], c.fill_hole_area).unsqueeze(1)
                shown, dumped = shown_logits(raw, cut, c.fill_hole_area, current["pred_masks"]), raw

                overlap = iou(det_masks, current["pred_masks"][:, 0])[:, 0] if det_masks.shape[0] else None
                matched = overlap is not None and float(overlap.max()) >= c.match_iou

                if f - birth < c.hotstart_frames and counts_unmatched(matched, current["pred_masks"], c):
                    unmatched += 1
                    if unmatched >= c.hotstart_unmatched:
                        # The detector cannot find this track on half of its first frames, so
                        # it was never the person; drop it and everything it drew.
                        log.info(f"the track born on frame {birth} went unmatched on {unmatched} of its "
                                 f"first {f - birth + 1} frames, so it was not the person; dropped")
                        counts["false starts"] += 1
                        output_dict = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
                        mux, birth, pending = None, -1, []
                        pbar.update(1)
                        continue

                anchor = None
                if f % c.recondition_every == 0 and overlap is not None and (
                        c.anchor_score_gate == OURS
                        or float(current["object_score_logits"].float().max()) > META_ANCHOR_OBJECT_LOGIT):
                    anchor = choose_anchor(det_scores, overlap, c)
                if anchor is None:
                    if f > quiet_until:
                        tracker._deferred_memory_encode(current, 1, vision_feats, feat_sizes, mux, device)
                    output_dict["non_cond_frame_outputs"][f] = current
                    forget_old_memory(output_dict["non_cond_frame_outputs"], f, lookback, c.memory_selection)
                else:
                    # An anchor only anchors if it is alone: drop the spatial memory of the
                    # frames around it, the ones carrying the drift. Everything still in the
                    # dict is within reach of the lookup, and the frames after it are held off
                    # by quiet_until. The object pointers stay - it is the memory that is
                    # cleared, not the frame. (A1: clear_past only clears, Meta does neither.)
                    if c.anchor_memory != META:
                        for old in output_dict["non_cond_frame_outputs"].values():
                            old["maskmem_features"] = old["maskmem_pos_enc"] = None
                    if c.anchor_memory == OURS:
                        quiet_until = f + c.memory_gap
                    propagated = current
                    current = tracker._condition_with_masks(
                        seed_from(det_masks, anchor, c), f, vision_feats,
                        vision_pos, feat_sizes, high_res, output_dict, N, mux, backbone, frame, trunk_out,
                        threshold=0.0)
                    shown, dumped, how = current["pred_masks"], None, "anchor"
                    if c.anchor_output == META:
                        # A2: the frame shows the propagated mask, and the conditioning entry
                        # remembers that mask - encoded as a propagated one, the way Meta
                        # re-encodes every frame's memory from the propagated masks
                        tracker._deferred_memory_encode(propagated, 1, vision_feats, feat_sizes, mux, device)
                        current["maskmem_features"] = propagated["maskmem_features"]
                        current["maskmem_pos_enc"] = propagated["maskmem_pos_enc"]
                        shown = shown_logits(raw, cut, c.fill_hole_area, propagated["pred_masks"])
                    elif detection_shown:
                        dumped = det_masks[anchor:anchor + 1].unsqueeze(1)
                        shown = shown_logits(dumped, cut, c.fill_hole_area)
                    counts["reconditioned"] += 1
                    prune_conditioning(output_dict["cond_frame_outputs"], c.conditioning_frames)

            mask = to_frame_size(shown, H, W, cut)
            low = low_res_logits(shown if dumped is None else dumped) if record is not None else None
            if f - birth >= c.hotstart_frames:
                for held in pending:
                    put(*held)
                pending = []
                put(f, mask, low, dumped is not None, how)
            else:
                pending.append((f, mask, low, dumped is not None, how))
            pbar.update(1)

        for held in pending:
            put(*held)
        if birth > 0:
            log.info(f"the track starts on frame {birth}; tracking backwards to fill frames 0-{birth - 1}")
            def emit(f, current, raw):
                put(f, to_frame_size(shown_logits(raw, cut, c.fill_hole_area, current["pred_masks"]), H, W, cut),
                    low_res_logits(raw) if record is not None else None, True)
            propagate_backwards(tracker, backbone, backbone_fn, frames, seed, birth, emit, device, dtype, c)
            counts["tracked backwards"] = birth

    segmented = int((masks.flatten(1).any(dim=1)).sum())
    if segmented == 0:
        log.warning(f"no frame of this clip scored above {c.birth_threshold} for '{prompt}'; there is no mask")
    else:
        log.info(f"segmented {segmented} of {N} frame(s) from '{prompt}', tracked from frame {birth}"
                 + (f", reconditioned {counts['reconditioned']} time(s)" if counts["reconditioned"] else ""))
    counts["frames segmented"] = segmented
    counts["tracked from frame"] = birth if segmented else 0
    if result is not None:
        result.update({k: v for k, v in counts.items() if v})
    return masks


# --- prompt mode, several objects ----------------------------------------------------------

def suppress_recently_occluded(masks, last_occluded, threshold):
    """[K] bool: which of the K tracks' mask logits [K, h, w] to blank on this frame. Meta's
    `_get_objects_to_suppress_based_on_most_recently_occluded`: of two tracks that overlap by
    `threshold` IoU or more, the one whose mask was empty (or suppressed) more recently gives
    way. `last_occluded` [K] is each track's last such frame, -1 for never."""
    K = masks.shape[0]
    to_suppress = torch.zeros(K, dtype=torch.bool, device=masks.device)
    if K <= 1:
        return to_suppress
    pairs = torch.triu(iou(masks, masks) >= threshold, diagonal=1)
    li, lj = last_occluded[:, None], last_occluded[None, :]
    suppress_i = pairs & (li > lj) & (lj > -1)
    suppress_j = pairs & (lj > li) & (li > -1)
    return suppress_i.any(dim=1) | suppress_j.any(dim=0)


def suppress_shrunk(masks, keep_share):
    """The K tracks' mask logits [K, h, w] as the memory encoder should see them. Meta's
    `_suppress_object_pw_area_shrinkage`, as core writes it for its own multi-object memory
    encode: give every pixel to the track scoring highest there, and a track left with less
    than `keep_share` of its area is noise under another one and is blanked. The masks
    themselves are not made disjoint here - that happens at output."""
    K = masks.shape[0]
    if K <= 1:
        return masks
    winner = masks.argmax(dim=0, keepdim=True)
    own = torch.arange(K, device=masks.device)[:, None, None]
    exclusive = torch.where(winner == own, masks, masks.clamp(max=-10.0))
    before = (masks > 0).sum(dim=(-1, -2)).float().clamp(min=1)
    after = (exclusive > 0).sum(dim=(-1, -2)).float()
    keep = (after / before) >= keep_share
    return torch.where(keep[:, None, None], masks, masks.clamp(max=-10.0))


def non_overlapping(binary, probs):
    """[K, H, W] bool masks made disjoint, Meta's `_apply_object_wise_non_overlapping_constraints`
    at output: a pixel two objects claim goes to the one with the higher object score `probs`
    [K] (the earlier-born one on a tie)."""
    K = binary.shape[0]
    if K <= 1:
        return binary
    scores = torch.where(binary, probs[:, None, None], torch.zeros((), dtype=probs.dtype))
    winner = scores.argmax(dim=0)
    return binary & (winner[None] == torch.arange(K)[:, None, None])


def segment_by_prompt_multi(model, clip, images, prompt, config, max_objects, object_index, result=None):
    """[N, H, W] float masks of up to `max_objects` people in `images` [N, H, W, 3], from the
    text prompt alone: the union of all of them when `object_index` is -1, else only object
    `object_index`, objects numbered in birth order (on the same frame, the higher score
    first). Raises when fewer objects than that were tracked.

    Every track runs the single-object policy of `segment_by_prompt` on its own tracker state -
    birth, probation, reconditioning, memory clearing, backwards fill - and Meta's
    multi-object parts decide between them: association (a detection touching any track by
    `assoc_iou` is not a new object; new objects past `max_objects` are dropped, lowest score
    first; the first track is born at `birth_threshold`, further ones at
    `new_object_threshold`), duplicate removal on probation, suppression of the more recently occluded of two
    overlapping tracks, the shrinkage check before memory encoding, and disjoint masks at
    output. Every output is held until the clip is done, so a track dropped on probation
    leaves nothing behind. Meta's keep-alive counters and masklet confirmation are not here:
    both are switched off in the configuration easy-sam3 ships (keep-alive only suppresses
    with suppress_unmatched_only_within_hotstart=False, confirmation with
    masklet_confirmation_enable=True), so they would be dead code. Of the [prompt] A/B switches
    only anchor_matching (A6) and unmatched_counting (A7) are read here, and the
    uniform_mask_threshold cut (M4)."""
    if clip is None:
        raise ValueError("the SAM3 checkpoint carries no text encoder, so the prompt cannot be "
                         f"encoded; use a full SAM3 checkpoint such as {DEFAULT_SAM3}")
    c = config
    N, H, W, _ = images.shape
    device, dtype = mm.get_torch_device(), model.model.get_dtype()
    frames = images[..., :3].movedim(-1, 1)

    mm.load_model_gpu(model)
    _, detector, tracker, backbone = _multiplex_parts(model)
    embedding, text_mask = encode_prompt(clip, detector, prompt, device, dtype)
    size = tracker.image_size
    backbone_fn = _propagation_backbone(backbone)
    lookback = max(tracker.num_maskmem, tracker.max_obj_ptrs_in_encoder)
    cut = output_cut(c)

    born = []             # every track ever started, in birth order
    live = []             # the tracks still running
    duplicates = {}       # (older id, younger id) -> probation frames they shared a detection
    outputs = [dict() for _ in range(N)]   # frame -> {track id: (low-res logits on the CPU, score)}
    counts = {"false starts": 0, "duplicates": 0, "reconditioned": 0, "suppressed": 0}
    pbar = ProgressBar(N)

    def drop(track, why):
        log.info(f"the track born on frame {track['birth']} {why}; dropped")
        track["removed"] = True
        for f in range(track["birth"], N):
            outputs[f].pop(track["id"], None)

    with torch.inference_mode():
        for f in range(N):
            frame = _prep_frame(frames, slice(f, f + 1), device, dtype, size)
            vision_feats, vision_pos, feat_sizes, high_res, trunk_out = tracker._compute_backbone_frame(
                backbone_fn, frame, frame_idx=f)
            det_masks, det_scores = detect_person(detector, backbone, trunk_out, embedding, text_mask, c)
            D = det_masks.shape[0]

            for t in live:
                current = tracker.track_step(
                    frame_idx=f, is_init_cond_frame=False, current_vision_feats=vision_feats,
                    current_vision_pos_embeds=vision_pos, feat_sizes=feat_sizes, mask_inputs=None,
                    output_dict=t["output"], num_frames=N, propagation_high_res=high_res,
                    multiplex_state=t["mux"], run_mem_encoder=False)
                t["raw"] = current["pred_masks"]
                current["pred_masks"] = clean_logits(t["raw"][:, 0], c.fill_hole_area).unsqueeze(1)
                t["current"] = current
                t["score"] = float(current["object_score_logits"].float().sigmoid().flatten()[0])
            K = len(live)
            overlap = (iou(det_masks, torch.cat([t["current"]["pred_masks"][:, 0] for t in live]))
                       if K and D else torch.zeros(D, K, device=device))

            # association: a detection that touches any track is that track's
            threshold = c.new_object_threshold if K else c.birth_threshold
            new = [d for d in range(D) if float(det_scores[d]) >= threshold
                   and not bool((overlap[d] >= c.assoc_iou).any())]
            new = new[:max(0, max_objects - K)]      # detections are sorted, best first

            # probation: unmatched tracks, and the younger of two tracks sharing a detection
            for k, t in enumerate(live):
                matched = D > 0 and float(overlap[:, k].max()) >= c.match_iou
                if f - t["birth"] < c.hotstart_frames and counts_unmatched(matched, t["current"]["pred_masks"], c):
                    t["unmatched"] += 1
                    if t["unmatched"] >= c.hotstart_unmatched:
                        drop(t, f"went unmatched on {t['unmatched']} of its first {f - t['birth'] + 1} frames")
                        counts["false starts"] += 1
            for d in range(D):
                sharing = [k for k in range(K) if float(overlap[d, k]) >= c.assoc_iou]
                if len(sharing) < 2:
                    continue
                oldest = min(sharing, key=lambda k: live[k]["id"])
                for k in sharing:
                    if k != oldest:
                        key = (live[oldest]["id"], live[k]["id"])
                        duplicates[key] = duplicates.get(key, 0) + 1
            for (older, younger), shared in duplicates.items():
                t = born[younger]
                if (not t["removed"] and f - t["birth"] < c.hotstart_frames
                        and shared >= c.duplicate_frames):
                    drop(t, f"shared a detection with the track born on frame {born[older]['birth']} "
                            f"on {shared} frames")
                    counts["duplicates"] += 1
            kept = [k for k, t in enumerate(live) if not t["removed"]]

            # reconditioning, per track
            anchors = {}
            if f % c.recondition_every == 0 and D:
                anchors = anchor_detections(det_scores, overlap, kept, c)

            # the tracks against each other, then memory
            if kept:
                stacked = torch.cat([live[k]["current"]["pred_masks"][:, 0] for k in kept])
                last = torch.tensor([live[k]["last_occluded"] for k in kept], device=device)
                suppressed = suppress_recently_occluded(stacked, last, c.occlusion_iou)
                empty = ~(stacked > 0).any(dim=(-1, -2))
                for j, k in enumerate(kept):
                    if k in anchors:
                        suppressed[j] = False
                    if bool(suppressed[j]) or bool(empty[j]):
                        live[k]["last_occluded"] = f
                    if bool(suppressed[j]):
                        live[k]["current"]["pred_masks"][:] = -10.0
                        stacked[j] = -10.0
                        counts["suppressed"] += 1
                memory = suppress_shrunk(stacked, c.shrink_keep)
            for j, k in enumerate(kept):
                t = live[k]
                output = t["output"]
                if k not in anchors:
                    current = t["current"]
                    if f > t["quiet_until"]:
                        encoded = {"pred_masks": memory[j:j + 1, None]}
                        tracker._deferred_memory_encode(encoded, 1, vision_feats, feat_sizes, t["mux"], device)
                        current["maskmem_features"] = encoded["maskmem_features"]
                        current["maskmem_pos_enc"] = encoded["maskmem_pos_enc"]
                    output["non_cond_frame_outputs"][f] = current
                    for old in list(output["non_cond_frame_outputs"]):
                        if old < f - lookback:
                            del output["non_cond_frame_outputs"][old]
                else:
                    # as in segment_by_prompt: an anchor only anchors if it is alone
                    for old in output["non_cond_frame_outputs"].values():
                        old["maskmem_features"] = old["maskmem_pos_enc"] = None
                    t["quiet_until"] = f + c.memory_gap
                    a = anchors[k]
                    t["current"] = tracker._condition_with_masks(
                        clean_logits(det_masks[a:a + 1], c.fill_hole_area), f, vision_feats, vision_pos,
                        feat_sizes, high_res, output, N, t["mux"], backbone, frame, trunk_out, threshold=0.0)
                    counts["reconditioned"] += 1
                    conditioned = output["cond_frame_outputs"]
                    while len(conditioned) > MAX_CONDITIONING_FRAMES:
                        del conditioned[sorted(conditioned)[1]]
                shown = t["current"]["pred_masks"]
                if k not in anchors and not bool(suppressed[j]):
                    shown = shown_logits(t["raw"], cut, c.fill_hole_area, shown)
                outputs[f][t["id"]] = (shown.float().cpu(), t["score"])
            live = [live[k] for k in kept]

            # births, best score first
            for d in new:
                seed = clean_logits(det_masks[d:d + 1], c.fill_hole_area)
                t = {"id": len(born), "birth": f, "seed": seed, "unmatched": 0, "quiet_until": -1,
                     "last_occluded": -1, "removed": False, "score": float(det_scores[d]),
                     "mux": MultiplexState(1, tracker.num_multiplex, device, dtype),
                     "output": {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}}
                t["current"] = tracker._condition_with_masks(
                    seed, f, vision_feats, vision_pos, feat_sizes, high_res,
                    t["output"], N, t["mux"], backbone, frame, trunk_out, threshold=0.0)
                outputs[f][t["id"]] = (t["current"]["pred_masks"].float().cpu(), t["score"])
                born.append(t)
                live.append(t)
            pbar.update(1)

        objects = [t for t in born if not t["removed"]]
        for t in objects:
            if t["birth"] > 0:
                log.info(f"object {objects.index(t)} starts on frame {t['birth']}; tracking backwards "
                         f"to fill frames 0-{t['birth'] - 1}")

                def emit(f, current, raw, track=t):
                    score = float(current["object_score_logits"].float().sigmoid().flatten()[0])
                    shown = shown_logits(raw, cut, c.fill_hole_area, current["pred_masks"])
                    outputs[f][track["id"]] = (shown.float().cpu(), score)
                propagate_backwards(tracker, backbone, backbone_fn, frames, t["seed"], t["birth"], emit,
                                    device, dtype, c)

    if object_index >= len(objects):
        raise ValueError(f"object_index {object_index}: {len(objects)} object(s) were tracked "
                         f"for '{prompt}' (numbered from 0)")
    masks = torch.zeros(N, H, W)
    ids = [t["id"] for t in objects]
    for f in range(N):
        present = [i for i in ids if i in outputs[f]]
        if not present:
            continue
        binary = torch.stack([to_frame_size(outputs[f][i][0], H, W, cut) > 0 for i in present])
        binary = non_overlapping(binary, torch.tensor([outputs[f][i][1] for i in present]))
        if object_index < 0:
            masks[f] = binary.any(dim=0).float()
        elif ids[object_index] in present:
            masks[f] = binary[present.index(ids[object_index])].float()

    segmented = int((masks.flatten(1).any(dim=1)).sum())
    log.info(f"tracked {len(objects)} object(s) from '{prompt}', born on frame(s) "
             f"{[t['birth'] for t in objects]}; {segmented} of {N} frame(s) segmented")
    counts["objects tracked"] = len(objects)
    counts["frames segmented"] = segmented
    if result is not None:
        result.update({k: v for k, v in counts.items() if v})
    return masks


# --- box_keypoint mode: mask cleaning at video resolution ----------------------------------

def drop_islands(mask, min_fraction):
    """Remove connected regions of a binary uint8 mask smaller than `min_fraction` of its
    largest one."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 2:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.flatnonzero(areas >= areas.max() * min_fraction) + 1
    return np.isin(labels, keep).astype(np.uint8)


def fill_holes(mask, max_fraction):
    """Fill the small background regions the mask fully encloses: a person has no holes, so a
    small hole is clothing or hair the decoder dropped. A large one is not - it is the
    background showing between a limb and the body, and it is left alone. Small is up to
    `max_fraction` of the mask's own area."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats((mask == 0).astype(np.uint8), connectivity=4)
    if count <= 2:
        return mask
    H, W = mask.shape
    limit = max_fraction * int(mask.sum())
    out = mask.copy()
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if x > 0 and y > 0 and x + w < W and y + h < H and area <= limit:   # encloses, and small
            out[labels == i] = 1
    return out


def clean_mask(mask, config):
    """A person's mask: no small enclosed holes, no islands far smaller than the body."""
    return drop_islands(fill_holes(mask.astype(np.uint8), config.max_hole_fraction), config.min_island_fraction)


# --- box_keypoint mode: prompt points ------------------------------------------------------

def body_points(kps, threshold, aspect=1.0):
    """Points on the body between the joints, in normalised coordinates: down the torso and
    along every limb whose two ends the pose model is sure of. `aspect` is W / H, needed
    where an x distance has to be stepped along y."""
    points = []
    if kps[R_SHOULDER][2] >= threshold and kps[L_SHOULDER][2] >= threshold:
        shoulder = ((kps[R_SHOULDER][0] + kps[L_SHOULDER][0]) / 2, (kps[R_SHOULDER][1] + kps[L_SHOULDER][1]) / 2)
        hips = [kps[i] for i in (R_HIP, L_HIP) if kps[i][2] >= threshold]
        if hips:
            hip = (sum(h[0] for h in hips) / len(hips), sum(h[1] for h in hips) / len(hips))
            points += [(shoulder[0] + t * (hip[0] - shoulder[0]), shoulder[1] + t * (hip[1] - shoulder[1])) for t in TORSO_FRACTIONS]
        else:
            # the hips are out of frame: step down from the shoulders by their own width, in
            # frame units - the coordinates are normalised per axis, so the x span has to be
            # scaled by the aspect ratio before it can be added to y
            span = (abs(kps[R_SHOULDER][0] - kps[L_SHOULDER][0]) or 0.15) * aspect
            points += [(shoulder[0], min(shoulder[1] + t * span, 0.99)) for t in (1.0, 2.0)]
    for a, b in LIMBS:
        if kps[a][2] >= threshold and kps[b][2] >= threshold:
            points += [(kps[a][0] + t * (kps[b][0] - kps[a][0]), kps[a][1] + t * (kps[b][1] - kps[a][1])) for t in LIMB_FRACTIONS]
    return points


def box_bounds(bbox, W, H):
    """The box clipped to the frame as integers, or None when nothing usable is left."""
    x1, y1, x2, y2 = (int(max(0, min(v, limit))) for v, limit in zip(bbox[:4], (W, H, W, H)))
    return (x1, y1, x2, y2) if x2 - x1 >= 8 and y2 - y1 >= 8 else None


def spread_points(free, x1, y1, count):
    """Up to `count` points taken from a boolean map, one per cell of a grid over it so they
    are spread rather than clustered, offset to (x1, y1) and returned in pixels."""
    if not free.any():
        return []
    rows = cols = int(np.ceil(np.sqrt(count)))
    h, w = free.shape
    points = []
    for r in range(rows):
        for c in range(cols):
            cell = free[r * h // rows:(r + 1) * h // rows, c * w // cols:(c + 1) * w // cols]
            ys, xs = np.nonzero(cell)
            if len(ys):
                middle = len(ys) // 2
                points.append((x1 + c * w // cols + int(xs[middle]), y1 + r * h // rows + int(ys[middle])))
    return points[:count]


def background_points(previous_mask, bbox, W, H, count, margin):
    """Up to `count` points inside the box that the previous frame's mask says are background,
    kept clear of the mask by `margin` of the box diagonal and spread over the box, in pixels."""
    bounds = box_bounds(bbox, W, H)
    if previous_mask is None or not previous_mask.any() or bounds is None:
        return []
    x1, y1, x2, y2 = bounds
    inside = previous_mask[y1:y2, x1:x2]
    if not inside.any():
        return []
    clearance = int(margin * float(np.hypot(x2 - x1, y2 - y1)))
    free = cv2.distanceTransform((~inside).astype(np.uint8), cv2.DIST_L2, 3) > clearance
    return spread_points(free, x1, y1, count)


def annexed_points(annexed, bbox, W, H, count):
    """Up to `count` points spread over the background the decoder is already known to have annexed, inside
    the box, in pixels."""
    bounds = box_bounds(bbox, W, H)
    if annexed is None or bounds is None:
        return []
    x1, y1, x2, y2 = bounds
    return spread_points(annexed[y1:y2, x1:x2], x1, y1, count)


def confident_pixels(kps, W, H, threshold):
    """The frame's body keypoints at or above `threshold` as integer pixel coordinates."""
    return [(int(min(max(x * W, 0), W - 1)), int(min(max(y * H, 0), H - 1)))
            for x, y, c in kps if c >= threshold]


def remember_annexed(annexed, mask, previous, kps, W, H, min_fraction, threshold):
    """The running map of background the decoder has annexed.

    A region the mask gained since the previous frame that is large against the mask and holds
    no confident keypoint is not the person appearing, it is an adjacent object being taken
    in; it joins the map. A region already on the map that a keypoint has since landed in is
    the person having moved there, and the whole of it leaves - otherwise a negative point
    would be put on the body itself. Clearing only a radius around the keypoints instead was
    tried and is worse: the stale remainder ate into the person on a clip where they fill the
    frame. `min_fraction` is the gained-region size against the mask, `threshold` the keypoint
    confidence."""
    area = int(mask.sum())
    gained = mask & ~previous
    points = confident_pixels(kps, W, H, threshold)
    if area and gained.any():
        count, labels, stats, _ = cv2.connectedComponentsWithStats(gained.astype(np.uint8), connectivity=8)
        occupied = {labels[y, x] for x, y in points}
        keep = [i for i in range(1, count)
                if stats[i, cv2.CC_STAT_AREA] >= min_fraction * area and i not in occupied]
        if keep:
            fresh = np.isin(labels, keep)
            annexed = fresh if annexed is None else (annexed | fresh)
    if annexed is None:
        return None
    count, labels, _, _ = cv2.connectedComponentsWithStats(annexed.astype(np.uint8), connectivity=8)
    occupied = {labels[y, x] for x, y in points}
    keep = [i for i in range(1, count) if i not in occupied]
    return np.isin(labels, keep) if keep else None


def prompt_for(frame_index, bboxes, pose_metas, W, H, device, dtype, config, min_keypoint_conf, previous_mask=None,
               annexed=None, extra_positive=(), extra_negative=()):
    """(box_inputs, point_inputs) for the SAM decoder in its 1008x1008 space, or (None, None)
    when the frame has neither a detection nor a confident keypoint. The points are the
    keypoints and the body points as positives, plus background points from the previous
    frame's mask and points from whatever the decoder has already annexed as negatives.
    `extra_positive` / `extra_negative` are hand-placed points in pixels, added as given; a
    derived point of the opposite label within HAND_POINT_CLEARANCE of one is dropped."""
    sx, sy = SAM3_SIZE / W, SAM3_SIZE / H
    bbox = bboxes[frame_index]
    box_inputs = None
    if bbox is not None and bbox[-1] > 0:
        box_inputs = torch.tensor([[[bbox[0] * sx, bbox[1] * sy], [bbox[2] * sx, bbox[3] * sy]]], device=device, dtype=dtype)
    kps = pose_metas[frame_index]["keypoints_body"]
    threshold = min_keypoint_conf
    positive = [(kps[k][0] * W, kps[k][1] * H) for k in PROMPT_KEYPOINTS if kps[k][2] >= threshold]
    positive += [(x * W, y * H) for x, y in body_points(kps, threshold, W / H)]
    negative = []
    if bbox is not None and bbox[-1] > 0:
        negative = (background_points(previous_mask, bbox, W, H, config.negative_points, config.negative_margin)
                    + annexed_points(annexed, bbox, W, H, config.annexed_points))
    if extra_positive or extra_negative:
        positive, negative = _clear_of_hand_points(positive, negative, extra_positive, extra_negative,
                                                   bbox if bbox is not None and bbox[-1] > 0 else (0, 0, W, H))
    positive += list(extra_positive)
    negative += list(extra_negative)
    point_inputs = None
    if positive:
        coords = [(x * sx, y * sy) for x, y in positive + negative]
        labels = [1] * len(positive) + [0] * len(negative)
        point_inputs = {"point_coords": torch.tensor([coords], device=device, dtype=dtype),
                        "point_labels": torch.tensor([labels], dtype=torch.int32, device=device)}
    return box_inputs, point_inputs


def _clear_of_hand_points(positive, negative, extra_positive, extra_negative, bbox):
    """The derived `positive` / `negative` points without those that land within
    HAND_POINT_CLEARANCE of the box diagonal of a hand-placed point of the opposite label, in
    their order; how many were dropped is logged."""
    reach = HAND_POINT_CLEARANCE * float(np.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1]))

    def clear(points, against):
        if not against:
            return points
        hand = np.asarray(against, dtype=np.float64)
        return [p for p in points if np.hypot(hand[:, 0] - p[0], hand[:, 1] - p[1]).min() > reach]

    kept_positive, kept_negative = clear(positive, extra_negative), clear(negative, extra_positive)
    dropped = len(positive) - len(kept_positive), len(negative) - len(kept_negative)
    if any(dropped):
        log.info(f"hand-placed points win: {dropped[0]} derived positive and {dropped[1]} derived negative point(s) "
                 f"within {reach:.0f} px of a hand-placed point of the other label dropped")
    return kept_positive, kept_negative


def keypoint_recall(mask, kps, W, H, threshold):
    """Fraction of the frame's body keypoints at or above `threshold` that fall inside `mask`, 1.0
    when the pose model is sure of none of them."""
    confident = confident_pixels(kps, W, H, threshold)
    if not confident:
        return 1.0
    return sum(1 for x, y in confident if mask[y, x]) / len(confident)


def confident_count(pose_meta, threshold):
    return sum(1 for _, _, c in pose_meta["keypoints_body"] if c >= threshold)


def is_anchor(frame_index, bboxes, pose_metas, config, min_keypoint_conf, reference_count=0):
    """Whether this frame's detection and pose are good enough to re-seed the tracker from.
    `reference_count` is how many keypoints the running segment was seeded with: a frame
    that sees fewer of them is a worse view of the person, not a better anchor."""
    bbox = bboxes[frame_index]
    if bbox is None or bbox[-1] <= 0:
        return False
    conf = [c for _, _, c in pose_metas[frame_index]["keypoints_body"] if c >= min_keypoint_conf]
    return (len(conf) >= max(config.min_anchor_keypoints, config.min_anchor_completeness * reference_count)
            and sum(conf) / len(conf) >= config.min_anchor_conf)


# --- box_keypoint mode: decoder and tracker ------------------------------------------------

def decode(sam3, frame, point_inputs, box_inputs, refine):
    """Mask logits for one 1008x1008 frame from box and point prompts, with an optional
    refinement pass that feeds the first mask back to the decoder. This is
    SAM3Model.forward_segment with the image encoder run once: its features do not depend on
    the prompt, so the refinement pass only runs the SAM heads."""
    _, _, feats, _ = sam3.detector.backbone["vision_backbone"](frame, tracker_mode="interactive")
    high_res, backbone_feat = list(feats[:-1]), feats[-1]
    tracker = sam3.tracker
    no_mem = getattr(tracker, "interactivity_no_mem_embed", None)
    if no_mem is None:
        no_mem = getattr(tracker, "no_mem_embed", None)
    if no_mem is not None:
        B, C, H, W = backbone_feat.shape
        flat = backbone_feat.flatten(2).permute(0, 2, 1)
        backbone_feat = (flat + cast_to_input(no_mem, flat)).view(B, H, W, C).permute(0, 3, 1, 2)
    num_pts = 0 if point_inputs is None else point_inputs["point_labels"].size(1)
    _, logits, _, _ = tracker._forward_sam_heads(
        backbone_features=backbone_feat, point_inputs=point_inputs, mask_inputs=None, box_inputs=box_inputs,
        high_res_features=high_res, multimask_output=(0 < num_pts <= 1))
    if refine:
        _, logits, _, _ = tracker._forward_sam_heads(
            backbone_features=backbone_feat, point_inputs=None, mask_inputs=logits, box_inputs=None,
            high_res_features=high_res, multimask_output=False)
    return logits


def propagate(sam3, frames_chw, first_mask, device, dtype, H, W, threshold=0.0, logits_out=None):
    """Masks for frames_chw[1:], propagated by the tracker's memory from `first_mask` on
    frames_chw[0], yielded one [H, W] bool array per frame: a frame is computed only when the
    caller asks for it, so a caller that stops accepting frames stops the tracker there.

    This is exactly what core's `track_video_with_detection` computes when it is given an
    initial mask and no detector: the mask conditions frame 0, every later frame is a plain
    track_step with the pinholes filled, and the output is the tracker's high-res mask cut at
    `threshold` (0 unless uniform_mask_threshold, M4) and resized to the frame. Written out here
    so no part of core's detection policy runs. `logits_out`, a list, receives each returned
    frame's low-res logits before the pinholes are filled - the ones the high-res mask was
    upsampled from - before that frame is yielded."""
    tracker, backbone = sam3.tracker, sam3.detector.backbone["vision_backbone"]
    backbone_fn = _propagation_backbone(backbone)
    N = frames_chw.shape[0]
    size = tracker.image_size
    idev = mm.intermediate_device()
    initial = (first_mask[None, None].to(device, dtype) * 2 - 1) * MASK_LOGIT_SCALE
    output_dict = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
    mux = MultiplexState(1, tracker.num_multiplex, device, dtype)
    lookback = max(tracker.num_maskmem, tracker.max_obj_ptrs_in_encoder)
    for f in range(N):
        # inference mode per frame, not around the loop: a generator suspended inside it
        # would leave the caller in inference mode
        with torch.inference_mode():
            frame = _prep_frame(frames_chw, slice(f, f + 1), device, dtype, size)
            vision_feats, vision_pos, feat_sizes, high_res, trunk_out = tracker._compute_backbone_frame(
                backbone_fn, frame, frame_idx=f)
            if f == 0:
                tracker._condition_with_masks(
                    initial, 0, vision_feats, vision_pos, feat_sizes, high_res, output_dict, N, mux,
                    backbone, frame, trunk_out)
                continue  # frame 0 is the given mask, not an output
            current = tracker.track_step(
                frame_idx=f, is_init_cond_frame=False, current_vision_feats=vision_feats,
                current_vision_pos_embeds=vision_pos, feat_sizes=feat_sizes, mask_inputs=None,
                output_dict=output_dict, num_frames=N, propagation_high_res=high_res,
                multiplex_state=mux, run_mem_encoder=False)
            if logits_out is not None:
                logits_out.append(low_res_logits(current["pred_masks"]))
            current["pred_masks"] = fill_holes_in_mask_scores(current["pred_masks"], max_area=16)
            if tracker.num_maskmem > 0:
                tracker._deferred_memory_encode(current, 1, vision_feats, feat_sizes, mux, device)
            output_dict["non_cond_frame_outputs"][f] = current
            for old in list(output_dict["non_cond_frame_outputs"]):
                if old < f - lookback:
                    del output_dict["non_cond_frame_outputs"][old]
            # one frame at a time: bilinear resizes every frame of a batch independently
            mask = (current["pred_masks_high_res"][0, 0] > threshold).to(idev).float()[None, None]
            mask = F.interpolate(mask, size=(H, W), mode="bilinear", align_corners=False)[0, 0] > 0.5
        yield mask.cpu().numpy()


def segment_by_pose(model, images, bboxes, pose_metas, config, min_keypoint_conf, extra_positive=(), extra_negative=(),
                    result=None, logits=None):
    """[N, H, W] float masks of the detected person in `images` [N, H, W, 3].

    `bboxes[i]` is frame i's detector box as (x1, y1, x2, y2, score); a score of -1 means
    nothing was detected. `pose_metas[i]["keypoints_body"]` are its body keypoints,
    normalised to the frame, with confidence; `min_keypoint_conf` is the confidence the pose was
    made with, at or above which a keypoint is prompted, counted and checked. `extra_positive` / `extra_negative` are
    hand-placed points in pixels of frame 0, added to frame 0's prompt.

    A frame the detector and the pose model agree on is prompted with that box and those
    keypoints; the tracker's memory then propagates the mask, which is what carries it
    through frames the pose model loses to motion blur. The tracker is re-seeded from a
    fresh prompt every `config.reseed_interval` frames - but only on a frame that is an anchor,
    and immediately when the propagated mask stops covering the keypoints. Whatever the mask
    annexes along the way is remembered and prompted against from then on. With
    `config.temporal` off, every frame is prompted on its own, without the tracker. `result`, if given, is
    filled with what happened for the log; `logits`, if given, a dict, receives each output
    frame's low-res logits (see `_logits_record`)."""
    c = config
    refine, temporal = c.refine, c.temporal
    N, H, W, _ = images.shape
    mm.load_model_gpu(model)
    device, dtype = mm.get_torch_device(), model.model.get_dtype()
    sam3 = _multiplex_parts(model)[0]
    frames_chw = images[..., :3].movedim(-1, 1)
    masks = torch.zeros(N, H, W)
    pbar = ProgressBar(N)
    counts = {"prompted": 0, "propagated": 0, "re-seeded early": 0, "kept at low recall": 0, "no prompt": 0,
              "empty prompt": 0}
    i = 0
    seed_frame, seed_count = -1, 0
    annexed = None
    cut = output_cut(c)
    record = _logits_record(logits, N)

    def keep(index, mask, low=None, how=None):
        """Store frame `index`'s mask and update what the decoder is known to have annexed.
        The annexation shows itself on the frame it happens, which is almost never a frame
        that is prompted, so it has to be carried to the next prompt."""
        nonlocal annexed
        was = masks[index - 1].numpy() > 0.5 if index > 0 and masks[index - 1].any() else None
        masks[index] = torch.from_numpy(mask).float()
        if record is not None:
            record["logits"][index], record["cut"][index] = low, how
        if was is not None:
            annexed = remember_annexed(annexed, mask.astype(bool), was,
                                       pose_metas[index]["keypoints_body"], W, H,
                                       c.min_annexed_fraction, min_keypoint_conf)

    while i < N:
        carried = False  # this frame's prompt failed and the track is carried on from i - 1
        previous = masks[i - 1].numpy() > 0.5 if i > 0 and masks[i - 1].any() else None
        box_inputs, point_inputs = prompt_for(i, bboxes, pose_metas, W, H, device, dtype, c, min_keypoint_conf,
                                              previous, annexed,
                                              *((extra_positive, extra_negative) if i == 0 else ()))
        if i == 0 and (extra_positive or extra_negative) and (box_inputs is None or point_inputs is None):
            log.warning("the hand-placed points are for frame 0, which has no box or no positive point to prompt "
                        "with; they are not used")
        if box_inputs is None or point_inputs is None:
            # nothing to describe the person with: carry the track on if there is one, else
            # leave this frame without a mask (the guard reports it as pose, not mask)
            counts["no prompt"] += 1
            if not (temporal and i > 0 and masks[i - 1].any()):
                pbar.update(1)
                i += 1
                continue
            carried = True
        else:
            frame = common_upscale(frames_chw[i:i + 1], SAM3_SIZE, SAM3_SIZE, "bilinear", crop="disabled").to(device, dtype)
            with torch.inference_mode():
                decoded = decode(sam3, frame, point_inputs, box_inputs, refine)
                seed = (F.interpolate(decoded.float(), size=(H, W), mode="bilinear",
                                      align_corners=False)[0, 0] > c.mask_threshold).cpu().numpy()
            if seed.any():
                keep(i, clean_mask(seed, c), low_res_logits(decoded) if record is not None else None, "prompted")
                counts["prompted"] += 1
                seed_frame, seed_count = i, confident_count(pose_metas[i], min_keypoint_conf)
                pbar.update(1)
                i += 1
            else:
                # the decoder found nothing here; propagating from an empty mask is wasted work
                counts["empty prompt"] += 1
                if not (temporal and i > 0 and masks[i - 1].any()):
                    pbar.update(1)
                    i += 1
                    continue
                carried = True
        if not temporal or i >= N:
            continue
        # propagate from the last frame that has a mask; accept frames while the mask still
        # covers the keypoints, and stop at the first anchor once the interval is up
        while i < N:
            end = min(i + c.reseed_interval, N)
            lows = [] if record is not None else None
            tracked = propagate(sam3, frames_chw[i - 1:end], masks[i - 1], device, dtype, H, W, cut, lows)
            stop = False
            start = i
            for k, mask in enumerate(tracked, start=start):
                if keypoint_recall(mask, pose_metas[k]["keypoints_body"], W, H, min_keypoint_conf) < c.min_tracked_recall:
                    if carried and k == start:
                        # the prompt for this frame just failed, so re-seeding it would only
                        # repeat the same failure forever: keep the tracked mask and move on
                        counts["kept at low recall"] += 1
                    else:
                        counts["re-seeded early"] += 1
                        stop = True
                        break
                keep(k, clean_mask(mask, c), lows[k - start] if lows is not None else None, "propagated")
                counts["propagated"] += 1
                pbar.update(1)
                i = k + 1
                since_seed = i - seed_frame if seed_frame >= 0 else c.reseed_interval
                if since_seed >= c.reseed_interval and i < N and is_anchor(i, bboxes, pose_metas, c, min_keypoint_conf, seed_count):
                    stop = True
                    break
                if since_seed >= c.max_propagate:
                    stop = True
                    break
            carried = False  # only the frame whose prompt just failed is exempt, not later windows
            if stop:
                break
    if result is not None:
        result.update({k: v for k, v in counts.items() if v})
    log.info("segmented " + ", ".join(f"{v} frame(s) {k}" for k, v in counts.items() if v))
    return masks


# --- inputs --------------------------------------------------------------------------------

def parse_coords(coords, W, H, name="coords"):
    """Hand-placed points as a list of (x, y) in pixels, from the JSON string KJNodes'
    PointsEditor and easy-sam3 pass around: either `[{"x": 50, "y": 120}, ...]` in pixels, or
    easy-sam3's `{"points": [[x, y], ...]}` normalised to the frame. None or an empty string
    is no points. A point outside the frame raises."""
    if coords is None or not str(coords).strip():
        return []
    try:
        data = json.loads(coords)
    except json.JSONDecodeError as e:
        raise ValueError(f"{name} must be a JSON string, found {coords!r} ({e})") from e
    if isinstance(data, dict) and "points" in data:
        raw = data["points"] or []
        if not all(isinstance(p, (list, tuple)) and len(p) == 2 for p in raw):
            raise ValueError(f'{name}: expected {{"points": [[x, y], ...]}} normalised, found {coords!r}')
        points = [(float(x) * W, float(y) * H) for x, y in raw]
    elif isinstance(data, list):
        if not all(isinstance(p, dict) and "x" in p and "y" in p for p in data):
            raise ValueError(f'{name}: expected [{{"x": x, "y": y}}, ...] in pixels, found {coords!r}')
        points = [(float(p["x"]), float(p["y"])) for p in data]
    else:
        raise ValueError(f'{name}: expected a JSON list of {{"x", "y"}} or {{"points": [...]}}, found {coords!r}')
    for x, y in points:
        if not (0 <= x < W and 0 <= y < H):
            raise ValueError(f"{name}: point ({x}, {y}) is outside the {W}x{H} frame")
    return points


def parse_bboxes(bboxes, N):
    """Boxes as the per-frame list `segment_by_pose` reads: N arrays (x1, y1, x2, y2, 1.0) in
    pixels, from anything the shared `bbox.parse_bboxes` accepts. A supplied box counts as a
    detection (score 1.0)."""
    return [np.array([*b, 1.0]) for b in parse_boxes(bboxes, N)]


def pose_keypoint_conf(pose_data):
    """The keypoint confidence the pose was made with (PoseConfig.min_keypoint_conf, carried in
    pose_data["pose_config"]): one threshold for every node that reads pose_data, so the mask is
    prompted from the same keypoints the pose images draw and the guards check."""
    pose_config = pose_data.get("pose_config") if isinstance(pose_data, dict) else None
    if not isinstance(pose_config, dict) or "min_keypoint_conf" not in pose_config:
        raise ValueError("pose_data has no pose_config.min_keypoint_conf; it must come from Pose Detection "
                         "or WanAnimate Preprocess")
    return pose_config["min_keypoint_conf"]


def pose_inputs(pose_data, N):
    """(bboxes, pose_metas, min_keypoint_conf) from pose_data the way segment_by_pose reads them:
    each frame's detections box as (x1, y1, x2, y2, score), the `pose_metas_original` dicts and
    the pose's keypoint confidence threshold."""
    if not isinstance(pose_data, dict) or "pose_metas_original" not in pose_data or "detections" not in pose_data:
        found = sorted(pose_data) if isinstance(pose_data, dict) else type(pose_data).__name__
        raise ValueError(f"box_keypoint mode needs pose_data with 'pose_metas_original' and 'detections', found {found}")
    metas, detections = pose_data["pose_metas_original"], pose_data["detections"]
    if len(metas) != N or len(detections) != N:
        raise ValueError(f"pose_data covers {len(metas)} pose frames and {len(detections)} detections, "
                         f"the images are {N} frames")
    bboxes = [np.array([*d["bbox"], d["score"]], dtype=np.float64) for d in detections]
    return bboxes, metas, pose_keypoint_conf(pose_data)


# --- entry point ---------------------------------------------------------------------------

# Set to a callable to receive every run's low-res logits (see `track`); the test dump sets it,
# since the node's outputs are masks only. None collects nothing.
LOGITS_SINK = None


def track(sam3_model, images, pose_data=None, bboxes=None, positive_coords=None, negative_coords=None,
          mode=MODE_PROMPT, prompt=PROMPT, max_objects=1, object_index=-1, config=None, logits_sink=None):
    """[N, H, W] float mask of the person (or people) in `images` [N, H, W, 3], the one function
    the SAM3 node calls. `sam3_model` is the (model, clip) pair from `load_sam3`.

    `mode` picks how the person is described to SAM:
    - "prompt": the text `prompt` alone, under the tracking policy in `config` (SAM3Config).
      pose_data, bboxes and the coords are not used.
    - "box_keypoint": pose_data (required) gives every frame's box and body keypoints; the
      positive points are computed from them frame by frame and the negatives from the running
      mask. `bboxes`, when given, replace pose_data's detection boxes. `positive_coords` /
      `negative_coords` are hand-placed points on frame 0 (KJNodes / easy-sam3 JSON), added to
      frame 0's prompt; a derived point of the opposite label near one is dropped (see
      HAND_POINT_CLEARANCE). `prompt`, `max_objects`, `object_index` and the [prompt] config
      fields are not used: one person, the one the pose describes.

    `max_objects` is how many tracks may be born and kept. 1 is the single-person policy
    (`segment_by_prompt`); above 1, prompt mode only, `segment_by_prompt_multi` runs instead and
    nothing of it runs at 1. `object_index` -1 is the union of every tracked object, k is object
    k alone, objects numbered from 0 in birth order; asking for an object that was not tracked
    raises with the count found.

    Whatever the mode does not read is ignored, never raised on, so the mode can be switched
    without rewiring: a connected input, a widget or a config field changed from its default
    that the run ignores is named in one console line.

    `logits_sink` (or, when it is None, the module's LOGITS_SINK) is called once when the run is
    done, with `(logits, info)`: `logits` is a list of N [h, w] float16 CPU tensors - the low-res
    mask logits each output frame was cut from, None for a frame with no output - and `info` a
    dict {"mode", "cut", "size", "threshold", "mask_threshold"}, in prompt mode also "raw" and
    "fill_hole_area". "cut" says per frame how its mask came from its logits, so a threshold can
    be swept offline:
    - "prompt": bilinear to `size` (H, W), then > threshold; where "raw" is true for the frame,
      the logits are the ones before the output's speck and pinhole cleaning, cleaned relative
      to the threshold by `shown_logits(logits, threshold, fill_hole_area)` first
    - "birth" / "anchor": the same as "prompt", on the frame the track was born on and on the
      frames it was re-anchored on (where "raw" is false, the logits are the conditioning mask)
    - "prompted" (box_keypoint): bilinear to `size`, > threshold, then clean_mask
    - "propagated" (box_keypoint): bilinear to the tracker's 1008 x 1008, > threshold, bilinear
      to `size` as 0/1, > 0.5, then clean_mask
    "threshold" is the cut this run used on "prompt" / "propagated" frames and "mask_threshold"
    the one on "prompted" frames. Not collected with max_objects above 1."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, found {mode!r}")
    if not isinstance(max_objects, int) or max_objects < 1:
        raise ValueError(f"max_objects must be an integer of at least 1, found {max_objects!r}")
    if not isinstance(object_index, int) or object_index < -1:
        raise ValueError(f"object_index must be -1 (every object) or an object number, found {object_index!r}")
    if mode == MODE_PROMPT and object_index >= max_objects:
        raise ValueError(f"object_index {object_index}: at most {max_objects} object(s) are tracked "
                         f"(max_objects), numbered from 0")
    if config is None:
        config = SAM3Config()
    if not isinstance(config, SAM3Config):
        raise TypeError(f"config must be a SAM3Config, found {type(config).__name__}")
    if images.dim() != 4 or images.shape[-1] < 3:
        raise ValueError(f"images must be [N, H, W, 3], found {list(images.shape)}")
    model, clip = sam3_model
    N, H, W = images.shape[:3]
    sink = logits_sink if logits_sink is not None else LOGITS_SINK
    collected = {} if sink is not None and not (mode == MODE_PROMPT and max_objects > 1) else None
    dump = {"logits": collected} if collected is not None else {}

    result = {}
    with log.step(f"segmenting the person with SAM3 ({mode}) on {N} frames", result):
        if mode == MODE_PROMPT:
            unused = [n for n, v in (("pose_data", pose_data), ("bboxes", bboxes), ("positive_coords", positive_coords),
                                     ("negative_coords", negative_coords)) if v is not None]
            unused += [f"sam3_config.{n}" for n in changed_fields(config, "[box_keypoint]")
                       if not (n == "mask_threshold" and config.uniform_mask_threshold)]
            if max_objects == 1 and not config.uniform_mask_threshold:
                unused += [f"sam3_config.{n} (uniform_mask_threshold off)" for n in changed_fields(config, names=("m4_anchor_frames",))]
            if max_objects == 1:
                unused += [f"sam3_config.{n} (max_objects 1)" for n in changed_fields(config, "[prompt, max_objects > 1]")]
            else:
                unused += [f"sam3_config.{n} (max_objects > 1)" for n in changed_fields(config, names=SINGLE_OBJECT_FIELDS)]
                if sink is not None:
                    unused.append("the logits sink (max_objects > 1)")
            if unused:
                log.info(f"prompt mode segments from the text alone; {', '.join(unused)} not used")
            if not prompt or not prompt.strip():
                raise ValueError("prompt mode needs a text prompt, found an empty one")
            if max_objects == 1:
                mask = segment_by_prompt(model, clip, images, prompt, config, result=result, **dump)
            else:
                mask = segment_by_prompt_multi(model, clip, images, prompt, config, max_objects,
                                               object_index, result=result)
        else:
            if pose_data is None:
                raise ValueError("box_keypoint mode prompts from the pose; connect pose_data")
            frame_boxes, pose_metas, keypoint_conf = pose_inputs(pose_data, N)
            unused = [f"{n} {v!r}" for n, v, default in (("prompt", prompt, PROMPT), ("max_objects", max_objects, 1),
                                                         ("object_index", object_index, -1)) if v != default]
            if bboxes is not None:
                unused.append("pose_data's person boxes (bboxes replace them)")
            unused += [f"sam3_config.{n}" for n in changed_fields(config, "[prompt]") + changed_fields(config, "[prompt, max_objects > 1]")]
            if not config.temporal:
                unused += [f"sam3_config.{n} (temporal off)" for n in changed_fields(config, names=TRACKER_FIELDS)]
            if unused:
                log.info(f"box_keypoint mode prompts the one person the pose describes; {', '.join(unused)} not used")
            if bboxes is not None:
                frame_boxes = parse_bboxes(bboxes, N)
            mask = segment_by_pose(model, images, frame_boxes, pose_metas, config, keypoint_conf,
                                   extra_positive=parse_coords(positive_coords, W, H, "positive_coords"),
                                   extra_negative=parse_coords(negative_coords, W, H, "negative_coords"),
                                   result=result, **dump)
        if mode == MODE_PROMPT and object_index == 0 and max_objects == 1 and not bool(mask.any()):
            raise ValueError("object_index 0: 0 objects were tracked (numbered from 0)")
        coverage = mask.mean(dim=(1, 2))
        result["frames without a mask"] = int((coverage == 0).sum())
        result["mask coverage"] = f"{coverage.min() * 100:.1f}-{coverage.max() * 100:.1f}%"
    if collected is not None:
        info = {"mode": mode, "cut": collected["cut"], "size": (H, W),
                "threshold": output_cut(config), "mask_threshold": config.mask_threshold}
        if "raw" in collected:
            info.update(raw=collected["raw"], fill_hole_area=config.fill_hole_area)
        sink(collected["logits"], info)
    return mask
