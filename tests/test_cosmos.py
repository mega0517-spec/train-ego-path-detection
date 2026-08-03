import json
import os
import sys

import numpy as np
import pytest
from PIL import Image, ImageDraw

from src.utils.cosmos import (
    DEFAULT_DOMAIN_PROMPTS,
    DEFAULT_SPEC_TEMPLATE,
    GEOMETRY_CLAUSE,
    build_jobs,
    edge_retention,
    load_generated,
    rail_edge_ratio,
    render_spec,
    scale_annotation,
)

IMG_WIDTH = 320
IMG_HEIGHT = 180
RAIL_TOP = 60


def make_annotation(shift=0):
    """Two straight rails converging with distance, reaching the bottom row."""
    ys = np.linspace(IMG_HEIGHT - 1, RAIL_TOP, 20).astype(int)
    ratios = (IMG_HEIGHT - 1 - ys) / (IMG_HEIGHT - 1 - RAIL_TOP)
    return {
        "left_rail": [[int(100 + 44 * r) + shift, int(y)] for y, r in zip(ys, ratios)],
        "right_rail": [[int(220 - 44 * r) + shift, int(y)] for y, r in zip(ys, ratios)],
    }


def make_scene(shift=0, contrast=1.0, size=None, seed=0):
    """Builds a rail-like scene: textured background, track bed, two bright rails."""
    rng = np.random.default_rng(seed)
    img = Image.fromarray(  # always drawn at full resolution, resized at the end
        rng.normal(90, 10, (IMG_HEIGHT, IMG_WIDTH, 3)).clip(0, 255).astype(np.uint8)
    )
    draw = ImageDraw.Draw(img)
    annotation = make_annotation(shift)
    bed = annotation["left_rail"] + annotation["right_rail"][::-1]
    draw.polygon([tuple(p) for p in bed], fill=(70, 70, 70))
    for rail in ("left_rail", "right_rail"):
        draw.line([tuple(p) for p in annotation[rail]], fill=(235, 235, 235), width=3)
    if size is not None:
        img = img.resize(size, Image.BILINEAR)
    if contrast != 1.0:  # blend toward gray, as a fog or night transfer would
        array = np.array(img, dtype=np.float32) * contrast + 128 * (1 - contrast)
        img = Image.fromarray(array.clip(0, 255).astype(np.uint8))
    return img


SOURCE = make_scene()
ANNOTATION = make_annotation()


class TestBuildJobs:
    def test_one_job_per_frame_and_domain(self):
        jobs = build_jobs(["b.jpg", "a.jpg"], ["night", "fog"], DEFAULT_DOMAIN_PROMPTS)
        assert len(jobs) == 4
        assert [job["source"] for job in jobs] == ["a.jpg", "a.jpg", "b.jpg", "b.jpg"]

    def test_output_names_are_traceable_and_unique(self):
        jobs = build_jobs(["a.jpg"], ["night", "fog"], DEFAULT_DOMAIN_PROMPTS)
        assert jobs[0]["output"] == "a__night.jpg"
        assert len({job["output"] for job in jobs}) == len(jobs)

    def test_prompts_state_the_geometry_constraint(self):
        # the annotations only stay valid if the generation keeps the rails in place
        jobs = build_jobs(["a.jpg"], list(DEFAULT_DOMAIN_PROMPTS), DEFAULT_DOMAIN_PROMPTS)
        assert all(GEOMETRY_CLAUSE in job["prompt"] for job in jobs)
        assert len({job["prompt"] for job in jobs}) == len(jobs)

    def test_limit_subsamples_deterministically(self):
        names = [f"{i}.jpg" for i in range(20)]
        first = build_jobs(names, ["night"], DEFAULT_DOMAIN_PROMPTS, limit=5)
        second = build_jobs(names, ["night"], DEFAULT_DOMAIN_PROMPTS, limit=5)
        assert len(first) == 5
        assert [job["source"] for job in first] == [job["source"] for job in second]


class TestScaleAnnotation:
    def test_same_shape_is_untouched(self):
        assert scale_annotation(ANNOTATION, (320, 180), (320, 180)) is ANNOTATION

    def test_coordinates_follow_the_new_shape(self):
        halved = scale_annotation(ANNOTATION, (320, 180), (160, 90))
        assert abs(halved["left_rail"][0][0] - ANNOTATION["left_rail"][0][0] / 2) <= 1
        assert halved["left_rail"][0][1] == 89  # the bottom row stays the bottom row


class TestRailEdgeRatio:
    def test_rails_concentrate_the_gradient(self):
        assert rail_edge_ratio(SOURCE, ANNOTATION) > 3

    def test_a_wrong_annotation_does_not(self):
        assert rail_edge_ratio(SOURCE, make_annotation(shift=40)) < 1.5


class TestEdgeRetention:
    """The check that decides whether a generated frame keeps its annotation.

    It has to tolerate the global contrast change a transfer applies on purpose, while
    catching rails that moved away from where the annotation says they are.
    """

    def test_identical_frame_retains_fully(self):
        assert edge_retention(SOURCE, SOURCE.copy(), ANNOTATION) == pytest.approx(1)

    @pytest.mark.parametrize("contrast", [0.35, 0.15])
    def test_contrast_loss_is_tolerated(self, contrast):
        # a night or foggy frame is flatter everywhere, which must not read as a move
        assert edge_retention(SOURCE, make_scene(contrast=contrast), ANNOTATION) > 0.85

    @pytest.mark.parametrize("shift", [10, 40])
    def test_moved_rails_are_caught(self, shift):
        assert edge_retention(SOURCE, make_scene(shift=shift), ANNOTATION) < 0.7

    def test_cropped_output_is_caught(self):
        # a crop shifts the content, so the annotation no longer lines up
        cropped = SOURCE.crop((30, 0, IMG_WIDTH, IMG_HEIGHT)).resize(
            (IMG_WIDTH, IMG_HEIGHT), Image.BILINEAR
        )
        assert edge_retention(SOURCE, cropped, ANNOTATION) < 0.7

    def test_resized_output_is_not_falsely_rejected(self):
        # a lower-resolution frame is blurrier, which inflates the ratio rather than
        # lowering it, so the minimum threshold stays safe
        resized = make_scene(size=(IMG_WIDTH // 2, IMG_HEIGHT // 2))
        assert edge_retention(SOURCE, resized, ANNOTATION) > 0.85

    def test_moved_rails_are_caught_even_when_resized(self):
        moved = make_scene(shift=20, size=(IMG_WIDTH // 2, IMG_HEIGHT // 2))
        assert edge_retention(SOURCE, moved, ANNOTATION) < 0.7


class TestRenderSpec:
    def test_placeholders_are_substituted(self):
        spec = render_spec(
            DEFAULT_SPEC_TEMPLATE,
            {"prompt": "p", "negative_prompt": "n", "source_path": "/s.jpg", "output_path": "/o.jpg"},
        )
        assert spec["prompt"] == "p"
        assert spec["input_video_path"] == "/s.jpg"

    def test_a_lone_placeholder_keeps_the_value_type(self):
        # "{control}" must become the weights mapping, not its text representation
        spec = render_spec(DEFAULT_SPEC_TEMPLATE, {"control": {"edge": 0.5}})
        assert spec["control"] == {"edge": 0.5}

    def test_nested_structures_are_rendered(self):
        assert render_spec({"a": {"b": ["{p}", 3]}, "c": True}, {"p": "x"}) == {
            "a": {"b": ["x", 3]},
            "c": True,
        }

    def test_unknown_placeholders_survive(self):
        assert render_spec({"x": "{unknown}"}, {})["x"] == "{unknown}"


class TestLoadGenerated:
    def test_image_output_is_loaded(self, tmp_path):
        path = tmp_path / "frame.png"
        SOURCE.save(path)
        assert load_generated(str(path)).size == (IMG_WIDTH, IMG_HEIGHT)

    def test_unreadable_output_returns_none(self, tmp_path):
        path = tmp_path / "broken.png"
        path.write_bytes(b"not an image")
        assert load_generated(str(path)) is None


SHIFT_COMMAND = (
    'python3 -c "import sys;from PIL import Image;import numpy as np;'
    "a=np.array(Image.open(sys.argv[1]));"
    'Image.fromarray(np.roll(a,40,axis=1)).save(sys.argv[2])" {source_path} {output_path}'
)


@pytest.fixture
def source_dataset(tmp_path):
    """Writes a couple of annotated source frames on disk."""
    images_dir = tmp_path / "sources"
    images_dir.mkdir()
    annotations = {}
    for i in range(2):
        make_scene(seed=i).save(images_dir / f"frame{i}.png")
        annotations[f"frame{i}.png"] = ANNOTATION
    annotations_path = tmp_path / "annotations.json"
    with open(annotations_path, "w") as f:
        json.dump(annotations, f)
    return str(images_dir), str(annotations_path)


def run_cosmos_transfer(argv):
    """Runs the script entry point with the given arguments."""
    import cosmos_transfer

    previous = sys.argv
    sys.argv = ["cosmos_transfer.py"] + argv
    try:
        cosmos_transfer.main(cosmos_transfer.parse_arguments())
    finally:
        sys.argv = previous


class TestCosmosTransferScript:
    def test_preserved_geometry_is_imported(self, tmp_path, source_dataset):
        images_dir, annotations_path = source_dataset
        output = tmp_path / "out"
        run_cosmos_transfer(
            ["--annotations", annotations_path, "--images", images_dir,
             "--output", str(output), "--domains", "night,fog",
             "--backend", "command", "--command", "cp {source_path} {output_path}",
             "--visualize", str(tmp_path / "vis"), "--device", "cpu"]
        )
        with open(output / "cosmos_egopath.json") as f:
            produced = json.load(f)
        assert sorted(produced) == [
            "frame0__fog.png", "frame0__night.png", "frame1__fog.png", "frame1__night.png"
        ]
        assert all({"left_rail", "right_rail"} == set(v) for v in produced.values())
        assert len(os.listdir(output / "images")) == 4
        assert len(os.listdir(output / "specs")) == 4
        assert len(os.listdir(tmp_path / "vis")) == 4

    def test_broken_geometry_is_rejected(self, tmp_path, source_dataset):
        images_dir, annotations_path = source_dataset
        output = tmp_path / "out"
        run_cosmos_transfer(
            ["--annotations", annotations_path, "--images", images_dir,
             "--output", str(output), "--domains", "night",
             "--backend", "command", "--command", SHIFT_COMMAND, "--device", "cpu"]
        )
        with open(output / "cosmos_egopath.json") as f:
            assert json.load(f) == {}
        with open(output / "report.json") as f:
            report = json.load(f)
        assert report["accepted"] == 0
        assert all(record["reasons"] == ["edge_retention"] for record in report["images"])

    def test_specs_only_run_then_imports_external_results(self, tmp_path, source_dataset):
        images_dir, annotations_path = source_dataset
        output = tmp_path / "out"
        argv = ["--annotations", annotations_path, "--images", images_dir,
                "--output", str(output), "--domains", "night", "--device", "cpu"]

        run_cosmos_transfer(argv)  # default backend only writes the specs
        assert (output / "specs" / "frame0__night.json").exists()
        assert not (output / "cosmos_egopath.json").exists()

        # generation happens elsewhere, then the same command imports what appeared
        make_scene().save(output / "images" / "frame0__night.png")
        run_cosmos_transfer(argv)
        with open(output / "cosmos_egopath.json") as f:
            assert list(json.load(f)) == ["frame0__night.png"]
