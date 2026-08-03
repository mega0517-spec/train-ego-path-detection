import base64
import json
import os
import subprocess
import urllib.request

import numpy as np
from PIL import Image, ImageDraw

# The whole point of control-conditioned transfer here is that the rails do not move,
# so that the existing annotations stay valid. Every prompt states it explicitly.
GEOMETRY_CLAUSE = (
    "The railway track layout, the position of the rails in the frame and the camera"
    " viewpoint remain exactly unchanged."
)

DEFAULT_DOMAIN_PROMPTS = {
    "night": "The same forward-facing railway scene at night, lit by the train headlight"
    " and distant coloured signal lights, dark sky, artificial light reflecting on the"
    " rail heads.",
    "rain": "The same forward-facing railway scene during heavy rain, soaked reflective"
    " rails, puddles on the track bed, water spray and a dark overcast sky.",
    "fog": "The same forward-facing railway scene in dense fog, low visibility, muted"
    " desaturated colours, the track fading into the mist in the distance.",
    "snow": "The same forward-facing railway scene in winter, snow covering the ground"
    " and the track bed, snow flurries in the air, flat overcast light.",
    "dawn": "The same forward-facing railway scene at dawn, low sun near the horizon"
    " facing the camera, long shadows across the track, strong backlight and lens flare.",
    "overcast": "The same forward-facing railway scene under a heavy overcast sky, flat"
    " diffuse light, dull wet surfaces, no visible shadows.",
}

NEGATIVE_PROMPT = (
    "changed track layout, moved or missing rails, additional tracks, different camera"
    " angle, warped perspective, distorted geometry, blurry"
)

# NVIDIA's guidance is that structurally consistent results need multi-control tuning
# rather than a single modality: edge preserves the layout the annotations depend on,
# depth keeps the perspective consistent.
DEFAULT_CONTROL_WEIGHTS = {"edge": 0.5, "depth": 0.3}

# Modelled on the Cosmos Transfer 2.5 spec conventions. The schema differs between
# Cosmos versions and runtimes, so treat this as a starting point and override it with
# --spec-template when it does not match the runtime you use.
DEFAULT_SPEC_TEMPLATE = {
    "prompt": "{prompt}",
    "negative_prompt": "{negative_prompt}",
    "input_video_path": "{source_path}",
    "output_video_path": "{output_path}",
    "control": "{control}",
}


class _SafeDict(dict):
    """Substitution mapping that leaves unknown placeholders untouched."""

    def __missing__(self, key):
        return "{" + key + "}"


def build_jobs(names, domains, prompts, limit=None, seed=42):
    """Builds the list of generation jobs, one per source frame and target domain.

    Args:
        names (list): Names of the source images.
        domains (list): Names of the target domains.
        prompts (dict): Prompt of each domain.
        limit (int, optional): Maximum number of source frames to use. Defaults to None.
        seed (int, optional): Random seed of the source frames sampling. Defaults to 42.

    Returns:
        list: Generation jobs.
    """
    names = sorted(names)
    if limit is not None and limit < len(names):
        indices = np.random.default_rng(seed).choice(len(names), limit, replace=False)
        names = [names[i] for i in sorted(indices)]
    jobs = []
    for name in names:
        stem, extension = os.path.splitext(os.path.basename(name))
        for domain in domains:
            jobs.append(
                {
                    "source": name,
                    "domain": domain,
                    "output": f"{stem}__{domain}{extension}",
                    "prompt": f"{prompts[domain]} {GEOMETRY_CLAUSE}",
                    "negative_prompt": NEGATIVE_PROMPT,
                }
            )
    return jobs


def scale_annotation(annotation, source_shape, target_shape):
    """Scales an annotation to a different image shape.

    Generation may not return the source resolution, in which case the absolute rail
    coordinates have to follow. This assumes the generated frame is a plain resize of
    the source; a crop or a letterbox would invalidate it, which the edge retention
    check is there to catch.

    Args:
        annotation (dict): Annotation with its "left_rail" and "right_rail" points.
        source_shape (tuple): Shape (W, H) of the source image.
        target_shape (tuple): Shape (W, H) of the target image.

    Returns:
        dict: Scaled annotation.
    """
    if source_shape == target_shape:
        return annotation
    xfactor = (target_shape[0] - 1) / max(source_shape[0] - 1, 1)
    yfactor = (target_shape[1] - 1) / max(source_shape[1] - 1, 1)
    return {
        rail: [
            [int(round(x * xfactor)), int(round(y * yfactor))]
            for x, y in annotation[rail]
        ]
        for rail in ("left_rail", "right_rail")
    }


def rail_edge_ratio(img, annotation, radius=2):
    """Measures how much image gradient concentrates along the annotated rails.

    The ratio to the mean gradient of the whole image makes the measure invariant to
    the global contrast, which a domain transfer changes on purpose (a foggy frame is
    flatter everywhere). What it captures is whether strong edges are still located
    where the annotation says the rails are.

    Args:
        img (PIL.Image.Image): Image to measure.
        annotation (dict): Annotation with its "left_rail" and "right_rail" points.
        radius (int, optional): Half-width (in pixels) of the band sampled along the rails. Defaults to 2.

    Returns:
        float: Ratio of the mean gradient along the rails to the mean gradient of the image.
    """
    gray = np.array(img.convert("L"), dtype=np.float32)
    yderivative, xderivative = np.gradient(gray)
    gradient = np.hypot(xderivative, yderivative)
    band = Image.new("L", img.size, 0)
    draw = ImageDraw.Draw(band)
    for rail in ("left_rail", "right_rail"):
        points = [tuple(point) for point in annotation[rail]]
        if len(points) >= 2:
            draw.line(points, fill=255, width=2 * radius + 1)
    band = np.array(band) > 0
    background = gradient.mean()
    if not band.any() or background <= 0:
        return 0.0
    return float(gradient[band].mean() / background)


def edge_retention(source_img, generated_img, annotation, radius=2):
    """Measures whether the generated frame kept the rails where the annotation expects them.

    This is deliberately independent of any detection model: a low value means the
    geometry moved and the annotation is no longer valid, whereas a model-based check
    would confuse that with the target domain simply being harder to detect in.

    A generated frame of a different resolution is resized back to the source one before
    measuring, so that both ratios are computed on the same grid.

    Read the result as a floor test, not as a fraction: a frame that is blurrier than the
    source (because it was generated at a lower resolution, or simply smoothed) loses more
    background texture than rail contrast, which pushes the ratio above 1. That direction
    is harmless for a minimum threshold, and a displaced path still scores far below it.

    Args:
        source_img (PIL.Image.Image): Source image.
        generated_img (PIL.Image.Image): Generated image.
        annotation (dict): Annotation of the source image.
        radius (int, optional): Half-width (in pixels) of the band sampled along the rails. Defaults to 2.

    Returns:
        float: Ratio of the generated rail edge concentration to the source one (1.0 means fully retained).
    """
    if generated_img.size != source_img.size:
        generated_img = generated_img.resize(source_img.size, Image.BILINEAR)
    source_ratio = rail_edge_ratio(source_img, annotation, radius)
    if source_ratio <= 0:
        return 0.0
    return float(rail_edge_ratio(generated_img, annotation, radius) / source_ratio)


def load_generated(path, frame=0):
    """Loads a generated frame, from an image or from a video output.

    Args:
        path (str): Path to the generated file.
        frame (int, optional): Index of the frame to extract from a video output. Defaults to 0.

    Returns:
        PIL.Image.Image or None: Generated frame, or None if it could not be read.
    """
    if os.path.splitext(path)[1].lower() in (".mp4", ".avi", ".webm", ".mov"):
        import cv2  # lazy import, only needed for video outputs

        capture = cv2.VideoCapture(path)
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
        read, extracted = capture.read()
        capture.release()
        if not read:
            return None
        return Image.fromarray(cv2.cvtColor(extracted, cv2.COLOR_BGR2RGB))
    try:
        return Image.open(path).convert("RGB")
    except (OSError, ValueError):
        return None


def render_spec(template, values):
    """Renders a generation spec by substituting the placeholders of a template.

    Args:
        template (dict): Spec template, whose strings may contain "{placeholder}" fields.
        values (dict): Values to substitute.

    Returns:
        dict: Rendered spec.
    """

    def render(value):
        if isinstance(value, str):
            # a string that is nothing but one placeholder of a non-string value (such
            # as "{control}") is replaced by that value itself, not by its repr
            name = value.strip()[1:-1]
            if value.strip().startswith("{") and value.strip().endswith("}"):
                if name in values and not isinstance(values[name], str):
                    return values[name]
            return value.format_map(_SafeDict(values))
        if isinstance(value, dict):
            return {key: render(item) for key, item in value.items()}
        if isinstance(value, list):
            return [render(item) for item in value]
        return value

    return render(template)


def generate_with_command(command, values):
    """Generates a frame by running an external command.

    Args:
        command (str): Command template, whose "{placeholder}" fields are substituted.
        values (dict): Values to substitute (including "spec_path", "source_path" and "output_path").

    Returns:
        tuple: Whether the command succeeded, and its error output if it did not.
    """
    rendered = command.format_map(_SafeDict(values))
    result = subprocess.run(rendered, shell=True, capture_output=True, text=True)
    return result.returncode == 0, result.stderr.strip()[:500]


def _find_encoded(payload, field):
    """Recursively looks up a field holding base64 content in a response body."""
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key == field and isinstance(value, str):
                return value
            found = _find_encoded(value, field)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = _find_encoded(item, field)
            if found is not None:
                return found
    return None


def generate_with_nim(
    endpoint, spec, output_path, api_key=None, field="image", timeout=600
):
    """Generates a frame by calling a NIM (or API catalog) endpoint.

    Args:
        endpoint (str): URL of the inference endpoint.
        spec (dict): Request body.
        output_path (str): Path to write the generated file to.
        api_key (str, optional): Bearer token. Defaults to None.
        field (str, optional): Name of the response field holding the base64 content. Defaults to "image".
        timeout (int, optional): Request timeout in seconds. Defaults to 600.

    Returns:
        tuple: Whether the generation succeeded, and the error message if it did not.
    """
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(spec).encode(),
        headers={"Content-Type": "application/json"},
    )
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content_type = response.headers.get("Content-Type", "")
            payload = response.read()
    except Exception as error:  # noqa: BLE001 - any transport error is a failed job
        return False, str(error)[:500]

    if content_type.startswith(("image/", "video/")):
        with open(output_path, "wb") as f:
            f.write(payload)
        return True, ""
    try:
        encoded = _find_encoded(json.loads(payload), field)
    except json.JSONDecodeError:
        return False, f"unexpected response content type: {content_type}"
    if encoded is None:
        return False, f"no '{field}' field in the response"
    with open(output_path, "wb") as f:
        f.write(base64.b64decode(encoded))
    return True, ""
