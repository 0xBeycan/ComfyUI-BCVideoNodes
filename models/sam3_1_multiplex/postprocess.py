"""SAM 3.1 Multiplex mask logits after the decoder: specks and pinholes cleaned out, and the low-res
copy the logits dump keeps."""
import torch


def clean_logits(masks, fill_hole_area):
    """Mask logits with the decoder's specks and pinholes taken out, at the decoder's own
    resolution. Shape is [n, H, W] in, [n, H, W] out; fill_holes_in_mask_scores wants the
    channel."""
    from comfy.ldm.sam3.tracker import fill_holes_in_mask_scores
    return fill_holes_in_mask_scores(masks.unsqueeze(1).float(), max_area=fill_hole_area)[:, 0]


def clean_channel_logits(raw, fill_hole_area):
    """clean_logits of [n, 1, h, w] logits, the tracker's and the detector's shape, keeping the
    channel."""
    return clean_logits(raw[:, 0], fill_hole_area).unsqueeze(1)


def low_res_logits(masks):
    """The [h, w] logits of a [1, 1, h, w] (or [1, h, w]) tensor as the fp16 CPU copy the logits
    dump keeps."""
    return masks.reshape(masks.shape[-2:]).detach().to("cpu", torch.float16).clone()
