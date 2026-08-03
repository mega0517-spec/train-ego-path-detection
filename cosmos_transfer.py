import argparse
import json
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
from PIL import Image

from src.utils.common import simple_logger
from src.utils.cosmos import (
    DEFAULT_CONTROL_WEIGHTS,
    DEFAULT_DOMAIN_PROMPTS,
    DEFAULT_SPEC_TEMPLATE,
    build_jobs,
    edge_retention,
    generate_with_command,
    generate_with_nim,
    load_generated,
    render_spec,
    scale_annotation,
)
from src.utils.evaluate import compute_iou
from src.utils.postprocessing import rails_to_mask
from src.utils.visualization import draw_egopath


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Ego-Path Detection Cosmos Transfer Data Generation Script"
    )
    parser.add_argument(
        "--annotations",
        type=str,
        required=True,
        help="Path to the annotations file of the source images (e.g. 'rs19_egopath.json').",
    )
    parser.add_argument(
        "--images",
        type=str,
        required=True,
        help="Path to the directory containing the source images.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the destination directory, which receives the generation specs, the generated images and the resulting annotations.",
    )
    parser.add_argument(
        "--domains",
        type=str,
        default="night,rain,fog",
        help=f"Comma-separated target domains to generate ('all' for every known domain). Known domains: {', '.join(DEFAULT_DOMAIN_PROMPTS)}.",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default=None,
        help="Path to a JSON file mapping domain names to prompts, to override or extend the built-in ones.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of source frames to use. If not specified, uses all of them.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed of the source frames sampling.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="none",
        choices=["none", "command", "nim"],
        help="How to run the generation: 'none' only writes the specs, so that generation can be run separately (rerun this script afterwards to import the results), 'command' runs an external command per job, 'nim' calls an inference endpoint.",
    )
    parser.add_argument(
        "--command",
        type=str,
        default=None,
        help="Command template run for each job with the 'command' backend, whose '{spec_path}', '{source_path}' and '{output_path}' fields are substituted.",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default=None,
        help="URL of the inference endpoint used by the 'nim' backend.",
    )
    parser.add_argument(
        "--api-key-env",
        type=str,
        default="NVIDIA_API_KEY",
        help="Name of the environment variable holding the bearer token of the endpoint.",
    )
    parser.add_argument(
        "--response-field",
        type=str,
        default="image",
        help="Name of the response field holding the base64 content, for endpoints that do not return raw bytes.",
    )
    parser.add_argument(
        "--spec-template",
        type=str,
        default=None,
        help="Path to a JSON spec template to render for each job. If not specified, uses a built-in template modeled on the Cosmos Transfer 2.5 conventions, which may need to be adapted to the runtime in use.",
    )
    parser.add_argument(
        "--control-weights",
        type=str,
        default=",".join(f"{k}={v}" for k, v in DEFAULT_CONTROL_WEIGHTS.items()),
        help="Comma-separated control modality weights (e.g. 'edge=0.5,depth=0.3'). Edge preserves the layout the annotations depend on; segmentation can move the geometry and should be used with care.",
    )
    parser.add_argument(
        "--video-frame",
        type=int,
        default=0,
        help="Index of the frame to extract when the generation returns a video instead of an image.",
    )
    parser.add_argument(
        "--min-edge-retention",
        type=float,
        default=0.7,
        help="Minimum ratio of rail edge concentration between the generated and the source frame. This is the criterion that decides whether the annotation is still valid.",
    )
    parser.add_argument(
        "--verify-with",
        type=str,
        default=None,
        help="Name of a trained model used to additionally report the IoU it reaches on the generated frames (e.g. 'twinkling-rocket-21'). Optional: the geometric check does not need a model.",
    )
    parser.add_argument(
        "--min-teacher-iou",
        type=float,
        default=float("nan"),
        help="Minimum IoU of the verification model on the generated frame. Disabled by default on purpose: a low IoU also happens when the geometry is intact but the target domain is simply hard, which is exactly the data worth keeping.",
    )
    parser.add_argument(
        "--visualize",
        type=str,
        default=None,
        help="Path to the destination directory for the visual outputs (source annotation drawn on the generated frame, prefixed with its acceptance status).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda", "mps"]
        + [f"cuda:{x}" for x in range(torch.cuda.device_count())],
        help="Device to use for the verification model ('cpu', 'cuda', 'cuda:x' or 'mps').",
    )
    return parser.parse_args()


def teacher_mask(detector, img):
    """Runs a detector on an image and returns its prediction as a binary mask."""
    prediction = detector.detect(img)
    if isinstance(prediction, list):
        return rails_to_mask(prediction, img.size)
    return prediction


def main(args):
    logger = simple_logger(__name__, "info")
    base_path = os.path.dirname(__file__)

    prompts = dict(DEFAULT_DOMAIN_PROMPTS)
    if args.prompts is not None:
        with open(args.prompts) as f:
            prompts.update(json.load(f))
    domains = sorted(prompts) if args.domains == "all" else args.domains.split(",")
    unknown = [domain for domain in domains if domain not in prompts]
    if unknown:
        raise ValueError(f"No prompt for domain(s): {', '.join(unknown)}")

    control = {}
    for item in filter(None, args.control_weights.split(",")):
        modality, _, weight = item.partition("=")
        control[modality.strip()] = float(weight)

    spec_template = DEFAULT_SPEC_TEMPLATE
    if args.spec_template is not None:
        with open(args.spec_template) as f:
            spec_template = json.load(f)

    with open(args.annotations) as f:
        annotations = json.load(f)
    names = [
        name for name in annotations if os.path.exists(os.path.join(args.images, name))
    ]
    if not names:
        raise ValueError(f"No annotated image found in {args.images}")

    images_dir = os.path.join(args.output, "images")
    specs_dir = os.path.join(args.output, "specs")
    for directory in (images_dir, specs_dir):
        os.makedirs(directory, exist_ok=True)
    if args.visualize is not None:
        os.makedirs(args.visualize, exist_ok=True)

    jobs = build_jobs(names, domains, prompts, args.limit, args.seed)
    logger.info(
        f"\n{len(jobs)} jobs: {len(jobs) // len(domains)} source frames"
        + f" x {len(domains)} domains ({', '.join(domains)})"
    )

    # 1. write one spec per job, so that generation can be run by this script or separately
    for job in jobs:
        job["source_path"] = os.path.abspath(os.path.join(args.images, job["source"]))
        job["output_path"] = os.path.abspath(os.path.join(images_dir, job["output"]))
        job["spec_path"] = os.path.join(
            specs_dir, f"{os.path.splitext(job['output'])[0]}.json"
        )
        with open(job["spec_path"], "w") as f:
            json.dump(
                render_spec(spec_template, {**job, "control": control}), f, indent=2
            )
    with open(os.path.join(args.output, "jobs.json"), "w") as f:
        json.dump(jobs, f, indent=2)
    logger.info(f"Specs written to {specs_dir}")

    # 2. generate, unless the generation is run separately
    if args.backend != "none":
        pending = [job for job in jobs if not os.path.exists(job["output_path"])]
        logger.info(
            f"\nGenerating {len(pending)} frames with the '{args.backend}' backend..."
        )
        progress = simple_logger(f"{__name__}_generation", "info", terminator="\r")
        failures = []
        for i, job in enumerate(pending):
            if args.backend == "command":
                if args.command is None:
                    raise ValueError("--command is required with the 'command' backend")
                done, error = generate_with_command(
                    args.command, {**job, "control": control}
                )
            else:
                if args.endpoint is None:
                    raise ValueError("--endpoint is required with the 'nim' backend")
                with open(job["spec_path"]) as f:
                    spec = json.load(f)
                done, error = generate_with_nim(
                    args.endpoint,
                    spec,
                    job["output_path"],
                    os.environ.get(args.api_key_env),
                    args.response_field,
                )
            if not done:
                failures.append((job["output"], error))
            progress.info(f"Generated {i + 1}/{len(pending)} frames")
        logger.info("")
        if failures:
            logger.info(f"\n{len(failures)} generation failures, first ones:")
            for name, error in failures[:5]:
                logger.info(f"  {name}: {error}")

    # 3. verify the generated frames and import the ones whose geometry was preserved
    detector = None
    if args.verify_with is not None:
        from src.utils.interface import Detector

        detector = Detector(
            model_path=os.path.join(base_path, "weights", args.verify_with),
            crop_coords=None,
            runtime="pytorch",
            device=args.device,
        )

    generated = [job for job in jobs if os.path.exists(job["output_path"])]
    if not generated:
        logger.info(
            "\nNo generated frame found yet."
            + f"\nRun the generation on the specs in {specs_dir},"
            + " then rerun this command to verify and import the results."
        )
        return

    logger.info(f"\nVerifying {len(generated)} generated frames...")
    progress = simple_logger(f"{__name__}_verification", "info", terminator="\r")
    source_ious = {}
    accepted = {}
    records = []
    for i, job in enumerate(generated):
        source_img = Image.open(job["source_path"]).convert("RGB")
        generated_img = load_generated(job["output_path"], args.video_frame)
        record = {
            "output": job["output"],
            "source": job["source"],
            "domain": job["domain"],
        }
        if generated_img is None:
            record.update({"accepted": False, "reasons": ["unreadable"]})
            records.append(record)
            continue

        annotation = annotations[job["source"]]
        scaled = scale_annotation(annotation, source_img.size, generated_img.size)
        metrics = {
            "edge_retention": edge_retention(source_img, generated_img, annotation),
            "rescaled": source_img.size != generated_img.size,
        }
        if detector is not None:
            if job["source"] not in source_ious:
                source_ious[job["source"]] = compute_iou(
                    teacher_mask(detector, source_img),
                    rails_to_mask(
                        [annotation["left_rail"], annotation["right_rail"]],
                        source_img.size,
                    ),
                )
            metrics["source_iou"] = source_ious[job["source"]]
            metrics["teacher_iou"] = compute_iou(
                teacher_mask(detector, generated_img),
                rails_to_mask(
                    [scaled["left_rail"], scaled["right_rail"]], generated_img.size
                ),
            )
            metrics["iou_drop"] = metrics["source_iou"] - metrics["teacher_iou"]

        reasons = []
        if metrics["edge_retention"] < args.min_edge_retention:
            reasons.append("edge_retention")
        if detector is not None and metrics["teacher_iou"] < args.min_teacher_iou:
            reasons.append("teacher_iou")
        record.update({"accepted": not reasons, "reasons": reasons, "metrics": metrics})
        records.append(record)

        if not reasons:
            accepted[job["output"]] = scaled
        if args.visualize is not None:
            status = "accepted" if not reasons else "rejected"
            vis = draw_egopath(
                generated_img, [scaled["left_rail"], scaled["right_rail"]]
            )
            vis.save(os.path.join(args.visualize, f"{status}_{job['output']}"))
        progress.info(f"Verified {i + 1}/{len(generated)} frames")
    logger.info("")

    annotations_path = os.path.join(args.output, "cosmos_egopath.json")
    with open(annotations_path, "w") as f:
        json.dump(accepted, f)
    report_path = os.path.join(args.output, "report.json")
    with open(report_path, "w") as f:
        json.dump(
            {
                "domains": domains,
                "control_weights": control,
                "min_edge_retention": args.min_edge_retention,
                "verify_with": args.verify_with,
                "accepted": len(accepted),
                "total": len(records),
                "images": records,
            },
            f,
            indent=2,
        )

    logger.info(
        f"\nAccepted {len(accepted)}/{len(records)} generated frames"
        + f" ({len(accepted) / len(records) * 100:.2f}%)"
    )
    for domain in domains:
        matching = [r for r in records if r["domain"] == domain]
        if matching:
            kept = sum(r["accepted"] for r in matching)
            logger.info(f"  {domain}: {kept}/{len(matching)}")
    if detector is not None:
        drops = [
            r["metrics"]["iou_drop"]
            for r in records
            if r["accepted"] and "iou_drop" in r.get("metrics", {})
        ]
        if drops:
            logger.info(
                f"\nAmong accepted frames, the verification model loses {sum(drops) / len(drops):.4f} IoU on average."
                + "\nA large drop with an intact geometry is expected on hard domains,"
                + " and is precisely the coverage this data is meant to add."
            )
    logger.info(
        f"\nAnnotations saved to {annotations_path}\nReport saved to {report_path}"
    )
    logger.info(
        "\nTo train on them, set in configs/global.yaml:"
        + f'\n  pseudo_annotations_path: "{os.path.abspath(annotations_path)}"'
        + f'\n  pseudo_images_path: "{os.path.abspath(images_dir)}"'
    )


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
