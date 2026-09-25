"""The SAM 3.1 Multiplex tunables, SAM3_1MultiplexConfig, and what both modes share: the fields a
run does not read and the record of the logits dump."""
from dataclasses import dataclass, field, fields


def _field(default, lo, hi, step, doc):
    return field(default=default, metadata={"min": lo, "max": hi, "step": step, "tooltip": doc})


# The two sides of each A/B switch below: "ours" is the policy the SAM 3.1 Multiplex pipeline has
# always run, "meta" is Meta's (easy-sam3 @ 88fe578) ported onto core's primitives.
OURS, META = "ours", "meta"

# What a prompt-mode re-anchor frame shows (`anchor_output`).
DETECTION, PROPAGATED = "detection", "propagated"


def _choice(default, choices, doc):
    return field(default=default, metadata={"choices": tuple(choices), "tooltip": doc})


@dataclass
class SAM3_1MultiplexConfig:
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
    detection_threshold: float = _field(0.30, 0.0, 1.0, 0.01, "[prompt] SAM 3.1 Multiplex detections below this presence-corrected score are dropped (Pose Config's detection_threshold is the person detector's)")
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
    # --- [prompt] A/B against Meta's policy (specs/mask-process.md, A6 and A7) ---
    # Each switch defaults to ours; "meta" is Meta's behaviour ported onto core's primitives.
    # Read with any max_objects.
    anchor_matching: str = _choice(OURS, (OURS, META), "[prompt] A6, which detection re-anchors which track. ours: each track takes the best-scoring detection agreeing with it; meta: each detection goes to the one track it overlaps most (the last such detection wins)")
    unmatched_counting: str = _choice(OURS, (OURS, META), "[prompt] A7, probation. ours: a frame the detector misses counts as unmatched even when the track's mask is empty; meta: only when the mask is not empty")
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
    mask_threshold: float = _field(-1.0, -10.0, 10.0, 0.1, "[box_keypoint] the prompted mask's logits are cut here")
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
    temporal: bool = field(default=True, metadata={"tooltip": "[box_keypoint] propagate with the tracker between prompts; off prompts every frame on its own"})

    # --- [prompt, max_objects 1] ---
    # Last, so the widgets of saved workflows keep their positions.
    # The detector mask a re-anchor frame is conditioned with drops a motion-blurred limb the
    # tracker still carries on that frame; easy-sam3 shows the tracker's mask there. The track
    # is re-anchored with the detection either way: only what the frame shows changes. The default
    # shows the propagated mask, as easy-sam3 does.
    anchor_output: str = _choice(PROPAGATED, (DETECTION, PROPAGATED), "[prompt, max_objects 1] what a re-anchor frame shows. propagated (default): the mask the tracker propagated onto that frame; detection: the detector mask the track is re-anchored with. The track is re-anchored with the detection either way")

    def __post_init__(self):
        for f in fields(self):
            choices = f.metadata.get("choices")
            if choices and getattr(self, f.name) not in choices:
                raise ValueError(f"SAM3_1MultiplexConfig.{f.name} is {getattr(self, f.name)!r}; it must be one of {choices}")


# The [box_keypoint] fields only the tracker reads: with `temporal` off nothing is propagated.
TRACKER_FIELDS = ("reseed_interval", "max_propagate", "min_anchor_keypoints", "min_anchor_conf",
                  "min_anchor_completeness", "min_tracked_recall")


def changed_fields(config, tag=None, names=()):
    """The fields of `config` whose tooltip starts with `tag`, or that are named in `names`, and
    that differ from their default: what a run that does not read them has to say it ignored."""
    return [f.name for f in fields(config)
            if ((tag is not None and f.metadata["tooltip"].startswith(tag + " ")) or f.name in names)
            and getattr(config, f.name) != f.default]


def logits_record(logits, N):
    """The per-frame slots a segment function fills for the logits dump, or None when nobody
    asked: `logits` is the caller's dict, given "logits" (the [h, w] fp16 low-res logits each
    output frame was cut from, None for a frame without output) and "cut" (how: "prompt",
    "prompted" or "propagated"; segment_by_prompt marks its birth frame "birth" and its
    re-anchor frames "anchor"). segment_by_prompt adds "raw": whether a frame's logits are the
    ones before the output's speck and pinhole cleaning (clean_logits)."""
    if logits is None:
        return None
    logits.update({"logits": [None] * N, "cut": [None] * N})
    return logits


def report_counts(result, counts):
    """Fill `result`, the caller's dict for the log, if it gave one, with the `counts` that are
    not zero."""
    if result is not None:
        result.update({k: v for k, v in counts.items() if v})
