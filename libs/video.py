"""Helpers on video tensors: IMAGE batches, frames first."""


def hold_last(video, frames):
    """``video`` with its last frame repeated up to ``frames`` frames."""
    import torch

    return torch.cat((video, video[-1:].expand(frames - video.shape[0], *video.shape[1:])), dim=0)


def as_numpy(images):
    """An IMAGE batch as a numpy array: a tensor's own memory, anything else through np.asarray."""
    import numpy as np
    import torch

    return images.numpy() if isinstance(images, torch.Tensor) else np.asarray(images)
