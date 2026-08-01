# Using NVIDIA Cosmos for Synthetic Training Data

This document describes how [NVIDIA Cosmos](https://www.nvidia.com/en-us/ai/cosmos/), a family of world foundation models (WFMs) for physical AI, can be used to expand and diversify the training data of TEP-Net, the ego-path detection model implemented in this repository.

## Why synthetic data for ego-path detection

TEP-Net is trained on ego-path annotations for the [RailSem19](https://www.wilddash.cc/railsem19) dataset (~8,500 images). While RailSem19 is diverse, it under-represents several conditions that matter for a deployed autonomous train perception system:

- Adverse weather (heavy rain, fog, snow-covered tracks)
- Night-time and low-sun / backlit scenes
- Rare infrastructure layouts (complex switch zones, unusual catenary or platform configurations)
- Degraded visual conditions (dirty lens, glare, motion blur)

Collecting and annotating real footage for these long-tail cases is expensive. World foundation models can synthesize photorealistic variants of such scenes, which is exactly the gap NVIDIA Cosmos targets for robotics and autonomous-vehicle workloads.

## The Cosmos model family

Cosmos was reorganized in 2026. **Cosmos 3** (released 2026-05-31) replaces the previous split into separate model families with a single unified omni-model that combines world generation, physical reasoning and action generation. It ships in three tiers, all under the NVIDIA Open Model License:

| Tier | Size | Target hardware |
| ---- | :--: | --------------- |
| **Cosmos3-Super** | 64B | Data center: H200 / B200 / GB200 |
| **Cosmos3-Nano** | 16B | Data center and workstation: RTX Pro 6000 / H100 / B200 |
| **Cosmos3-Edge** | 4B | Edge and on-device: Jetson AGX Orin / Thor / RTX Pro 6000 |

The previous generation — **Cosmos Predict 2.5** (video generation) and **Cosmos Transfer 2.5** (structure-conditioned domain transfer) — moved to limited maintenance on 2026-06-01, but its weights and NIM deployment documentation remain available. It is still the practical choice on pre-Hopper hardware (see [Hardware requirements](#hardware-requirements)), and the 2B Transfer model is the smallest entry point into this workflow.

The capability this repository depends on is **structure-preserving generation**: re-rendering a scene while keeping its geometry fixed. It exists in both generations — as the dedicated Transfer model in 2.5, and as `transfer` hints (`edge`, `blur`, `depth`, `seg`, `wsm`) in the unified Cosmos 3 generator. Scene reasoning, previously the separate Cosmos Reason model, is likewise folded into Cosmos 3.

## Getting access

Three routes, in increasing order of setup effort:

1. **NVIDIA API Catalog ([build.nvidia.com](https://build.nvidia.com))** — hosted NIM endpoints running on DGX Cloud, free for NVIDIA Developer Program members for prototyping, with no local GPU required. Rate limits make it unsuitable for bulk generation, but it is the right way to check that rail geometry survives the transfer before committing any hardware.
2. **Hugging Face checkpoints** — for bulk generation on your own GPU:
   - Create a read-scoped Hugging Face token and run `hf auth login`.
   - Accept the NVIDIA Open Model License. Note that the pipeline also pulls **Cosmos-Guardrail1**, whose license must be accepted separately on its own model page. The guardrail filters generated content, so a fraction of any batch is dropped silently — budget for it when estimating yield.
   - Checkpoints download automatically on first inference; `HF_HOME` relocates the cache.
3. **Self-hosted NIM containers** — require an `NGC_API_KEY`. Developer Program members may run NIM microservices on up to 16 GPUs free of charge for research and development. The container exposes an OpenAI-compatible REST endpoint (port 8000) and supports Helm deployment. A 90-day NVIDIA AI Enterprise trial covers enterprise support and API stability terms.

## Hardware requirements

This is the binding constraint, and it differs sharply between the two generations:

| | Cosmos 3 | Cosmos Transfer 2.5 (2B) |
| --- | --- | --- |
| Architecture floor | NIM `Cosmos3-Generator`: Hopper or newer (CC ≥ 9.0); `nvfp4` additionally requires Blackwell (CC ≥ 10.0) | Ampere or newer (RTX 30 series, A100) |
| Single-GPU VRAM | tier-dependent, see the tier table above | 65.4 GB |
| Tested GPUs | per tier | B200, H100 NVL, H100 PCIe, H20, H200 NVL, B300, RTX PRO 6000 Blackwell SE |

Two consequences worth stating plainly:

- **Neither RTX 4090 (Ada, CC 8.9) nor A100 (Ampere, CC 8.0) appears in the target hardware of any Cosmos 3 tier**, and the NIM `Cosmos3-Generator` deployment requires Hopper (CC ≥ 9.0), which both fall below. That floor is deployment-specific rather than uniform across the family — the Edge tier targets Jetson AGX Orin, which is Ampere — so check the tier and runtime you actually intend to use before ruling hardware in or out.
- For Transfer 2.5 both clear the architecture floor, but the 65.4 GB requirement leaves **A100 80GB** as the only one of the two with headroom; A100 40GB and RTX 4090 24GB fall short. Multi-GPU inference is supported. Community reports describe running at 480p within roughly 24 GB, but that is not an officially supported configuration and neither GPU appears in the tested list.

Driver and OS for Transfer 2.5: driver ≥ 570.124.06 (CUDA 12.8.1), Linux x86-64 with glibc ≥ 2.35.

**Reduced resolution is unusually cheap for this project.** `input_shape` is `[3, 512, 512]`, so generated frames are resized to 512×512 for training, and the teacher used for pseudo-labeling infers at that same resolution. Generating at 480p therefore costs much less here than it would in a video-production workflow. Generation is also entirely offline: it does not affect the training or TensorRT inference requirements of this repository, so a GPU that cannot run Cosmos can still handle all the training and pseudo-labeling work.

## Recommended pipeline for this repository

### 1. Annotation-preserving augmentation by control-conditioned transfer (primary approach)

The key property of structure-preserving generation is that it keeps the scene geometry fixed. If a RailSem19 frame is re-rendered as "same scene at night, in heavy rain", the rails do not move in image space, so the existing polyline annotations in `rs19_egopath.json` remain valid for the generated image — the annotation comes for free, and none of the teacher-error concerns of pseudo-labeling apply.

Workflow:

1. Select annotated frames from the RailSem19 image directory (`images_path` in `configs/global.yaml`).
2. Generate with the source frame as control input — Cosmos 3 with a `transfer` hint, or Cosmos Transfer 2.5.
3. Prompt for the target domain: night, rain, fog, snow, low sun, etc.
4. Save each output under a new filename and duplicate the corresponding annotation entry in the JSON file under that filename.

**Choosing control modalities matters here.** The modalities do different things, and the wrong choice silently destroys the property this whole approach relies on:

- **`edge` is the essential one** — it preserves the structure, shape and layout of the source, which is exactly what keeps the annotations valid.
- **`depth`** maintains 3D realism and perspective consistency, and is a good companion to `edge` for rail scenes, where convergence with distance carries most of the geometric information.
- **`seg` transforms objects and backgrounds completely.** RailSem19 ships dense semantic labels that are tempting to use as a control signal, but applying `seg` to the rail and track-bed classes can move the geometry and invalidate the annotations. Use it conservatively, and verify (step 3 below).

NVIDIA's own guidance is that high-fidelity, structurally consistent results need **multi-control tuning** (a combination such as `edge` + `depth`) rather than any single modality.

Because `src/utils/dataset.py` (`PathsDataset`) simply reads a JSON dict keyed by image filename and loads images from a flat directory, no code changes are needed: place the generated images alongside the originals (or point `images_path` at a merged directory) and extend the annotations file with entries for the new filenames.

### 2. Novel scene generation + pseudo-labeling (secondary approach)

Cosmos can also generate entirely new rail scenes from a text prompt, or plausible continuations of existing cab-ride footage from a start frame or clip. These outputs have no ground-truth ego-path, so they must be labeled. `pseudo_label.py` implements this: it runs a trained teacher model over a directory of images, converts its predictions to the annotation format of this repository, and keeps only the predictions it can justify. See [Pseudo-labeling](#pseudo-labeling) below.

For domains where the teacher is unreliable even after filtering (e.g. dense fog, which is far outside RailSem19), annotate a small curated set by hand in the same JSON polyline format instead.

### 3. Quality filtering

Generated data should be filtered before training:

- `pseudo_label.py` rejects predictions the teacher is not confident about (see the criteria below).
- Use the reasoning capability of Cosmos 3 (or any VLM) to reject frames with implausible physics or missing/hallucinated tracks, before pseudo-labeling.
- For Transfer outputs, the source annotation is the reference: run the teacher on the generated frame and compare its prediction to the *original* annotation. A large IoU drop means the generation distorted the geometry, so the frame should be dropped even though its annotation is nominally still valid. `pseudo_label.py --calibrate` performs exactly this comparison.

## Pseudo-labeling

`pseudo_label.py` labels a directory of unannotated images with a trained teacher model and filters the results by confidence.

```bash
python pseudo_label.py  twinkling-rocket-21  # teacher model (most accurate released model, IoU 0.9769)
                        /path/to/cosmos_frames  # directory containing the images to label
                        --output annotations/cosmos_pseudo.json  # destination annotations file
                        --report annotations/cosmos_report.json  # per-image metrics, useful to tune the thresholds
                        --visualize output/cosmos_vis  # overlays, prefixed with their acceptance status
                        --device cuda
```

The output is a standard annotations file (`{"image.png": {"left_rail": [[x, y], ...], "right_rail": [[x, y], ...]}}`), directly usable as `pseudo_annotations_path` (see [Training with pseudo-labels](#training-with-pseudo-labels)). Only accepted images are written, unless `--keep-rejected` is passed.

### Confidence criteria

No ground truth is available on generated images, so confidence is estimated from four independent signals. A pseudo-label is kept only if it passes every applicable criterion; the report records which ones failed.

| Metric | Signal | Default | Applies to |
| ------ | ------ | :-----: | ---------- |
| `flip_iou` | IoU between the prediction and the prediction on the mirrored image, mirrored back. The training pipeline flips images at random, so the model is equivariant to it — disagreement means the input is out of distribution. This is the most informative single signal. | ≥ 0.85 | all methods |
| `prob_confidence` | Mean `max(p, 1-p)` over the predicted probabilities: how far the model is from its decision boundary. | ≥ 0.90 | classification, segmentation |
| `uncertain_ratio` | Ratio of pixels whose probability falls in the ambiguous `]0.1, 0.9[` band. | ≤ 0.10 | segmentation |
| `ensemble_iou` | Mean IoU between the teacher and the models passed to `--ensemble`. The most reliable signal, at the cost of one extra inference per model. | ≥ 0.85 | all methods |
| `bottom_gap`, `height_ratio`, `min_width`, `width_monotonicity`, `jitter` | Geometric plausibility: the path must start at the bottom of the image, be long enough, narrow with distance (perspective) and not wander (RMS deviation from a quadratic fit). | see `--help` | all methods |

Each threshold is a CLI flag (`--min-flip-iou`, `--max-jitter`, …); passing `nan` disables that criterion.

Regression models expose no probabilistic output, so `prob_confidence` and `uncertain_ratio` are skipped for them — a segmentation or classification teacher gives a stricter filter.

### Calibrating the thresholds

The defaults are a starting point, not a calibration. Before a large generation run, run the script on *annotated* images to see how each metric relates to the IoU actually achieved:

```bash
python pseudo_label.py  twinkling-rocket-21  /path/to/rs19_val/jpgs/rs19_val
                        --calibrate /path/to/rs19_egopath.json  # ground truth for these images
                        --limit 500  --device cuda
```

This reports, per metric, its correlation with the true IoU and a sweep over candidate thresholds showing the resulting kept ratio and the mean IoU of the kept and dropped images. Pick thresholds that drop the low-IoU tail without discarding most of the data, then tighten them for generated images, which are further out of distribution than the real ones used for calibration. Nothing is written to the annotations file in this mode.

## Training with pseudo-labels

`train.py` accepts pseudo-labels through two optional keys of `configs/global.yaml`:

```yaml
pseudo_annotations_path: "/path/to/annotations/cosmos_pseudo.json"
pseudo_images_path: "/path/to/cosmos_frames"  # "null" to reuse "images_path"
```

These images are appended to the **training set only**. The validation and test splits keep being drawn from `annotations_path` alone, so the reported IoU stays comparable to the reference results and is never measured against the teacher's own output. Leaving both keys at `null` (the default) reproduces the original behavior exactly.

## Practical notes

- **Pseudo-label collapse:** a student trained on its teacher's output inherits the teacher's errors and cannot exceed it on the domains where both fail. Pseudo-labels are worth adding for *coverage* of new conditions, not for accuracy on the existing ones — always check the test IoU against a baseline trained without them.
- **Cropping:** `--crop` defaults to `none`, which is the right choice for independent generated frames. `auto` is only meaningful for consecutive frames of a same video, since the autocropper averages over a sequence. Note that a crop excluding the bottom of the image produces paths that do not reach the last row, which are rejected (the training augmentation requires them).
- **Augmentation interaction:** the existing ColorJitter augmentation (`brightness`/`contrast`/`saturation`/`hue` in `configs/global.yaml`) already covers mild photometric variation. Cosmos-generated data is complementary: it changes scene content (precipitation, illumination sources, shadows), which photometric jitter cannot.
- **Compute:** see [Hardware requirements](#hardware-requirements). The short version is that the generation step, not this repository's training pipeline, is what constrains the choice of GPU.
- **Model generation to target:** prefer Cosmos 3 if the hardware allows it, since 2.5 is only maintained, not developed. On pre-Hopper GPUs, Transfer 2.5 (2B) is the only option of the two and remains a reasonable starting point — its multi-control documentation is mature and its weights stay published.
- **Licensing:** RailSem19-derived annotations are CC BY-NC-SA 4.0. Images generated *from* RailSem19 frames are derivatives — the non-commercial share-alike terms carry over. Cosmos model outputs themselves are governed by the NVIDIA Open Model License, which permits use of outputs for training. Review both before distributing a synthetic dataset.

## References

- [NVIDIA Cosmos product page](https://www.nvidia.com/en-us/ai/cosmos/)
- [NVIDIA/cosmos — the Cosmos 3 platform repository](https://github.com/NVIDIA/cosmos)
- [Cosmos 3: Omnimodal World Models for Physical AI (technical report)](https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf)
- [Develop Physical AI Reasoning, World, and Action Models with NVIDIA Cosmos 3](https://developer.nvidia.com/blog/develop-physical-ai-reasoning-world-and-action-models-with-nvidia-cosmos-3/)
- [NVIDIA NIM for Cosmos WFM — supported models and hardware](https://docs.nvidia.com/nim/cosmos/3.0.0/support-matrix.html)
- [Cosmos Cookbook — control modalities](https://nvidia-cosmos.github.io/cosmos-cookbook/core_concepts/control_modalities/overview.html)
- [cosmos-transfer2.5 — setup and inference requirements](https://github.com/nvidia-cosmos/cosmos-transfer2.5)
- [Free NIM access for NVIDIA Developer Program members](https://developer.nvidia.com/blog/access-to-nvidia-nim-now-available-free-to-developer-program-members)
- [World Simulation with Video Foundation Models for Physical AI (Cosmos-Predict 2.5 / Transfer 2.5 report)](https://arxiv.org/abs/2511.00062)
- [Cosmos World Foundation Model Platform for Physical AI (original 1.0 paper)](https://arxiv.org/abs/2501.03575)
