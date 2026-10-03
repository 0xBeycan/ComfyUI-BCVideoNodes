"""The sizes the video nodes work to: the model table Load Video loads to, and the ladder Conform
Video fits to. Sizes are [width, height], portrait; a landscape frame gets them swapped."""
import math

PORTRAIT, LANDSCAPE, AUTO = "portrait", "landscape", "auto"
ORIENTATIONS = [AUTO, LANDSCAPE, PORTRAIT]

# the platform ladder Conform Video fits to (9:16, landscape 16:9)
CONFORM_SIZES = {"480p": [480, 854], "720p": [720, 1280], "1080p": [1080, 1920]}

# the model entry that is no model: no frame rule, no grid
NO_MODEL = "None"

# the resolution of every model that keeps the video's own pixels: no resize, only cuts (the other
# orientation's crop, then the model's grid); its size in the table is None
SOURCE = "source"

# model -> its frame rule (counts of the form frames * n + 1), its grid (the side lengths its core
# node takes are multiples of it) and its resolution labels (the short edge) with their sizes.
# Wan: a /16 grid (VAE /8 x patch 2x2; core's WanAnimateToVideo and WanAnimate2ToVideo take width
# and height in steps of 16). SCAIL: a /32 grid (the pose runs at half resolution through the /16
# grid; core's WanSCAILToVideo takes steps of 32); 512p and 704p are the authors' sizes. None: no
# model, so no frame rule (frames * n + 1 with frames 1 is any count) and no grid, Conform Video's
# ladder. A new model adds a row.
MODELS = {
    "Wan": {"frames": 4, "grid": 16, "sizes": {"480p": [480, 832], "720p": [720, 1280], SOURCE: None}},
    "SCAIL": {"frames": 4, "grid": 32, "sizes": {"512p": [512, 896], "704p": [704, 1280], SOURCE: None}},
    NO_MODEL: {"frames": 1, "grid": 1, "sizes": {**CONFORM_SIZES, SOURCE: None}},
}
# every model's labels, each once, the sized ones in table order, then source: the resolution
# widget's values
RESOLUTIONS = [*dict.fromkeys(label for model in MODELS.values() for label, size in model["sizes"].items() if size),
               SOURCE]


def model_size(model, resolution):
    """The portrait [width, height] of `resolution` for `model`, None for SOURCE. Raises
    ValueError, saying what to pick, when the model is unknown or the label is not one of its
    resolutions."""
    if model not in MODELS:
        raise ValueError(f"model {model!r} is not one of {', '.join(MODELS)}; pick one of them.")
    sizes = MODELS[model]["sizes"]
    if resolution not in sizes:
        raise ValueError(f"resolution {resolution!r} does not belong to model {model}; "
                         f"pick one of {', '.join(sizes)}.")
    return None if sizes[resolution] is None else list(sizes[resolution])


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
