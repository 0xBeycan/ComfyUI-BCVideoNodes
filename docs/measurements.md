# Measured against the earlier workflow

One Wan 2.2 Animate replacement workflow (background video and character mask connected), run
once with the earlier packs and once with this pack, on a 1080 x 1920, 30 fps clip of 612 frames
loaded at 720p: 609 frames (4n+1), sampled as `81 x 7 + 49` with overlap 1 and the same sampler
settings in both runs. RTX PRO 6000 Blackwell (96 GB), a 126.5 GiB container RAM limit,
ComfyUI 79be670e; each run in a freshly started ComfyUI with no model loaded.

- Old: the packs as of 2026-09-22, VideoHelperSuite (4d907be) and KJNodes (d3cfe21) for loading,
  resizing, the mask steps and saving, this pack's sampler as of 2026-09-19
  (`WanAnimateLongVideoSampler`) and ComfyUI-BCNodes' Video Comparer of then.
- New: this pack (263d5f9) and ComfyUI-BCNodes (075ad7a). These runs predate WanAnimate
  Preprocess's `final_mask` / `bg_images` and the samplers' `gc.collect()` (below).
- The preprocess is the same third-party one in both runs (Kijai's WanAnimatePreprocess pose,
  easy-sam3's SAM 3 track), so it is not part of the comparison.

Per node: the ComfyUI-BCNodes Process Monitor's time, RAM rise (the node's peak minus its start,
the container's working set sampled every 100 ms), output size (what ComfyUI keeps in its cache)
and VRAM peak (torch's allocator).

| Whole workflow | Old | This pack |
|----------------|-----|-----------|
| Wall time, from queue | 1034.6 s | 1028.4 s |
| Peak RAM, process (VmHWM) | 101,614,804 kB (96.91 GiB) | 79,366,792 kB (75.69 GiB) |
| Peak RAM, container working set | 98.77 GiB | 77.58 GiB |
| Peak VRAM (the sampler) | 47.85 GiB | 53.93 GiB |

| Stage | Old nodes: time, RAM rise, output | This pack's nodes: time, RAM rise, output |
|-------|-----------------------------------|-------------------------------------------|
| Load at 720 x 1280 | VHS Load Video (1080 x 1920 frames): 5.4 s, 15.35 GiB, 14.18 GiB; then KJ Image Resize v2 (lanczos, crop): 13.2 s, 12.57 GiB, 6.27 GiB | Load Video: 12.2 s, 6.74 GiB, 6.28 GiB (frames and audio) |
| Final mask and background | KJ GrowMaskWithBlur (expand 10): 1.2 s, 6.33 GiB, 4.18 GiB; BlockifyMask (32): 1.0 s, 4.19 GiB, 2.09 GiB; DrawMaskOnImage (black): 1.3 s, 19.73 GiB, 6.27 GiB | ComfyUI-BCNodes MaskGrow (grow 10, blur 0): 0.8 s, 2.09 GiB, 2.09 GiB; Blockify Mask (32): 0.5 s, 2.10 GiB, 2.09 GiB; Draw Mask On Image (black): 0.5 s, 6.29 GiB, 6.27 GiB |
| Sampler | This pack's of 2026-09-19: 860.9 s, 31.05 GiB, 6.27 GiB; VRAM 47.85 GiB | Wan Animate Long Video Sampler: 853.6 s, 27.85 GiB, 6.27 GiB; VRAM 53.93 GiB |
| Save and compare | VHS Video Combine (h264, crf 19): 2.1 s, 0.72 GiB; Video Comparer: 7.1 s, 0.01 GiB | Save Video (h264-mp4, crf 19, medium): 3.5 s, 0.58 GiB; Video Comparer: 6.5 s, 1.11 GiB |

The RAM peak is the sampler's, and most of the difference is what the old graph keeps in
ComfyUI's cache until the prompt ends: the full-size frames next to the resized ones (14.18 GiB)
and GrowMaskWithBlur's second output (2.09 GiB). The cache held 50.52 GiB after the old sampler
and 34.23 GiB after this pack's; the sampler's own RAM rise is 3.20 GiB smaller (computed
from the table). Load Video takes 6.4 s less than the loader and the resize together (computed).

VRAM: the sampler's peak varies from run to run with either pack. Core's `WanAnimateToVideo`
leaves the Wan VAE encoder's full-resolution features (`[1, 96, 2, 1280, 720]` and
`[1, 3, 2, 1280, 720]`, bf16) in a reference cycle, so they stay on the GPU into the chunk's
sampling until Python's own collector happens to run. Probed after one call: 4.06 GiB of them
still allocated with the old packs, 7.30 GiB with this pack, none in a third probe where the
collector had already run; one `gc.collect()` freed them all (0.28 GiB allocated after, both
packs). On a 161-frame run of each pack, 7.47-12.49 GiB
was allocated when a chunk's sampling started and its peak was 43.77-49.69 GiB. The samplers now
call `gc.collect()` right after each core conditioning call: 74.9 and 78.1 ms per call, measured
locally with about 590k tracked objects, against about 100 s of sampling per chunk here.

The last chunk decodes only the latent frames `total_frames` needs ([Length math](long-video-samplers.md#length-math)); this run
needed all of them. Measured on this GPU with the Wan 2.1 VAE in bf16, on 81-frame windows of
the clip (21 latent frames): decoding the first 1 or 11 latent frames gives the frames of
decoding all 21, bit for bit; the first 2 or 20 give the last latent frame's 4 frames up to
1.87/255 apart (73 of 77 frames identical as 8-bit). The odd counts measured exact, the even ones
not.

This pack's own preprocess on the same 609 frames (WanAnimate Preprocess, prompt mode, ViTPose-H,
263d5f9), between Load Video and the WanAnimate Preprocess Guard: 64.4 s (person detection 5.5 s,
keypoints 5.9 s, pose images 0.5 s, the SAM 3.1 Multiplex track 51.4 s, face crops 0.1 s), a VRAM
peak of 4.51 GiB, a RAM rise of 15.20 GiB of which 10.15 GiB are its outputs; the guard 3.0 s;
80.3 s from queue, VmHWM 23,520,984 kB (22.43 GiB). For reference, in the runs above the
third-party preprocess took 11.7 / 14.3 s for its pose, 1.6 s for drawing it and 105.8 / 111.9 s
for easy-sam3's SAM 3 track (fp32, VRAM peak 9.45 / 9.75 GiB): another model, so the times only.

A replacement workflow that animates a single image into 81 frames (the image repeated) showed
no difference: 148.0 s and 147.2 s from queue, VmHWM 41.37 and 40.79 GiB.
