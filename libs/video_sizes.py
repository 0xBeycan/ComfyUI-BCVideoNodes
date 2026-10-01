"""The sizes the video nodes work to: the model table Load Video loads to, and the ladder Conform
Video fits to. Sizes are [width, height], portrait; a landscape frame gets them swapped."""
import math

PORTRAIT, LANDSCAPE, AUTO = "portrait", "landscape", "auto"
ORIENTATIONS = [AUTO, LANDSCAPE, PORTRAIT]

# model -> its frame rule (counts of the form frames * n + 1) and its resolution labels (the short
# edge) with their sizes. Wan: a /16 grid (VAE /8 x patch 2x2). SCAIL: a /32 grid (the pose runs at
# half resolution through the /16 grid); 512p and 704p are the authors' sizes. A new model adds a row.
MODELS = {
    "Wan": {"frames": 4, "sizes": {"480p": [480, 832], "720p": [720, 1280]}},
    "SCAIL": {"frames": 4, "sizes": {"512p": [512, 896], "704p": [704, 1280]}},
}
# every model's labels, in table order: the resolution widget's values
RESOLUTIONS = [label for model in MODELS.values() for label in model["sizes"]]

# the platform ladder Conform Video fits to (9:16, landscape 16:9)
CONFORM_SIZES = {"480p": [480, 854], "720p": [720, 1280], "1080p": [1080, 1920]}


def model_size(model, resolution):
    """The portrait [width, height] of `resolution` for `model`. Raises ValueError, saying what to
    pick, when the model is unknown or the label is not one of its resolutions."""
    if model not in MODELS:
        raise ValueError(f"model {model!r} is not one of {', '.join(MODELS)}; pick one of them.")
    sizes = MODELS[model]["sizes"]
    if resolution not in sizes:
        raise ValueError(f"resolution {resolution!r} does not belong to model {model}; "
                         f"pick one of {', '.join(sizes)}.")
    return list(sizes[resolution])


def orientation_of(width, height):
    """portrait when the frame is taller than wide, otherwise landscape (a square is landscape)."""
    return PORTRAIT if height > width else LANDSCAPE


def oriented(size, orientation):
    """The portrait `size` [width, height] turned to `orientation`: landscape swaps it."""
    width, height = size
    return [width, height] if orientation == PORTRAIT else [height, width]


def conform_size(width, height):
    """The CONFORM_SIZES entry a `width` x `height` frame fits to, oriented as the frame: the one
    whose scale on the short edge is closest to 1, measured as |log(target / short edge)|. The
    midpoints (about 588 and 882) are not integers, so no short edge ties; above 1080 it is 1080p."""
    short = min(width, height)
    label = min(CONFORM_SIZES, key=lambda k: abs(math.log(CONFORM_SIZES[k][0] / short)))
    return oriented(CONFORM_SIZES[label], orientation_of(width, height))
