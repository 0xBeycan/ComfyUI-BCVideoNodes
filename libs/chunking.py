"""Chunk length math for chaining Wan Animate / Wan Animate 2 / SCAIL-2 generations.

Pure Python on purpose: no torch, no ComfyUI. The long-video loop
(pipelines/long_video.py) and the Wan Animate adapters import this; the
tests import only this.

Wan latents are 4k+1 pixel frames. Every chunk length lives on that grid and
is at least MIN_CHUNK frames. After the first chunk, each chunk re-seeds from
the previous output and the node trims the seeded frames back off, so a later
chunk only contributes ``length - overlap`` new frames.
"""

MIN_CHUNK = 5


def snap_down(frames):
    """Largest 4k+1 <= frames, never below MIN_CHUNK."""
    frames = int(frames)
    if frames < MIN_CHUNK:
        return MIN_CHUNK
    return ((frames - 1) // 4) * 4 + 1


def snap_up(frames):
    """Smallest 4k+1 >= frames, never below MIN_CHUNK."""
    frames = int(frames)
    if frames < MIN_CHUNK:
        return MIN_CHUNK
    return -(-(frames - 1) // 4) * 4 + 1


def overlap_for_motion_frames(motion_frames):
    """Pixel frames the core node reports as ``trim_image`` when it is seeded
    with ``motion_frames`` frames of continue_motion.

    Mirrors WanAnimateToVideo and WanAnimate2ToVideo alike: the seed frames
    occupy ``((n - 1) // 4) + 1`` latent frames, and trim_image is
    ``max(0, latents * 4 - 3)``. Equals n when n is on the 4k+1 grid, i.e.
    this is also "largest 4k+1 <= n": WanAnimateToVideo's
    continue_motion_max_frames widget (default 5) gives 5, WanAnimate2ToVideo's
    CONTINUE_MOTION_FRAMES = 1 gives 1.
    """
    motion_frames = int(motion_frames)
    if motion_frames <= 0:
        return 0
    latents = ((motion_frames - 1) // 4) + 1
    return max(0, latents * 4 - 3)


def next_chunk_length(produced, total_frames, frames_per_chunk, overlap):
    """Length of the chunk to run next, given frames kept so far.

    The last chunk shrinks: the remaining need (plus the overlap that will be
    trimmed) is rounded UP to the grid and capped at frames_per_chunk. Calling
    this with the real produced count each iteration is what keeps the loop
    exact even if the overlap assumption turns out wrong.
    """
    chunk = _grid_chunk(frames_per_chunk, overlap)
    need = int(total_frames) - int(produced)
    if need <= 0:
        return 0
    if produced == 0:
        return min(chunk, snap_up(need))
    return min(chunk, snap_up(need + overlap))


def full_chunk_length(produced, total_frames, frames_per_chunk, overlap):
    """Like next_chunk_length, but every chunk, the last one included, is frames_per_chunk on the
    4k+1 grid: for a model trained on full-length segments only. The last chunk then runs past
    total_frames and the caller cuts the output back to it."""
    chunk = _grid_chunk(frames_per_chunk, overlap)
    return chunk if int(total_frames) > int(produced) else 0


def _grid_chunk(frames_per_chunk, overlap):
    """frames_per_chunk on the 4k+1 grid; raises when it does not exceed the overlap."""
    chunk = snap_down(frames_per_chunk)
    if overlap >= chunk:
        raise ValueError(
            "frames_per_chunk ({} -> {} on the 4k+1 grid) must exceed the overlap ({}).".format(
                frames_per_chunk, chunk, overlap
            )
        )
    return chunk


FIT, FULL = "fit", "full"
# the last_chunk widget: its values, each with the chunk length policy the plan and the loop share
# and why the plan then samples past total_frames (for the hold log line)
LAST_CHUNK = {
    FIT: (next_chunk_length, "the last chunk is snapped up to 4k+1"),
    FULL: (full_chunk_length, "the last chunk runs the full frames_per_chunk"),
}


def plan_chunks(total_frames, frames_per_chunk, overlap, chunk_length=next_chunk_length):
    """Chunk lengths that cover total_frames, as the loop would run them with the length policy
    `chunk_length` (next_chunk_length or full_chunk_length)."""
    total_frames = int(total_frames)
    if total_frames < 1:
        raise ValueError("total_frames must be at least 1.")
    plan = []
    produced = 0
    while produced < total_frames:
        length = chunk_length(produced, total_frames, frames_per_chunk, overlap)
        plan.append(length)
        produced += length if len(plan) == 1 else length - overlap
    return plan


def produced_frames(plan, overlap):
    """Frames a plan yields after per-chunk trimming."""
    if not plan:
        return 0
    return plan[0] + sum(length - overlap for length in plan[1:])


def format_plan(plan, produced, total_frames, pose_frames, overlap):
    text = " + ".join(str(length) for length in plan)
    text += " -> {} produced -> {} frames (pose {}, overlap {})".format(produced, total_frames, pose_frames, overlap)
    return text
