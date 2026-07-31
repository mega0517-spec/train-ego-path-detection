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

Cosmos consists of three complementary model families, all released under the NVIDIA Open Model License with checkpoints on Hugging Face and NGC:

| Model | What it does | Relevance here |
| ----- | ------------ | -------------- |
| **Cosmos Predict** | Generates physically plausible video/images from text, image, or video conditioning (Text2World, Image2World, Video2World; Predict 2.5 unifies these in one flow-based model) | Generate novel rail scenes from prompts or extend short clips |
| **Cosmos Transfer** | Structure-conditioned domain/style transfer: re-renders a scene while preserving its geometry, guided by control inputs (segmentation, depth, edge maps, or source video) | **Best fit** — re-style already-annotated frames into new weather/lighting while keeping rail geometry (and therefore annotations) valid |
| **Cosmos Reason** | Physical-AI vision-language model for reasoning about scenes | Automated curation / quality filtering of generated data |

## Recommended pipeline for this repository

### 1. Annotation-preserving augmentation with Cosmos Transfer (primary approach)

The key property of Cosmos Transfer is that it preserves scene structure. If a RailSem19 frame is re-rendered as "same scene at night, in heavy rain", the rail geometry in image space does not move, so the existing polyline annotations in `rs19_egopath.json` remain valid for the generated image.

Workflow:

1. Select annotated frames from the RailSem19 image directory (`images_path` in `configs/global.yaml`).
2. Run Cosmos Transfer conditioned on the source frame (optionally with the RailSem19 semantic segmentation maps as an additional control signal — the dataset ships with dense labels, and rail/track-bed classes give a strong structural prior).
3. Prompt for the target domain: night, rain, fog, snow, low sun, etc.
4. Save each output under a new filename and duplicate the corresponding annotation entry in the JSON file under that filename.

Because `src/utils/dataset.py` (`PathsDataset`) simply reads a JSON dict keyed by image filename and loads images from a flat directory, no code changes are needed: place the generated images alongside the originals (or point `images_path` at a merged directory) and extend the annotations file with entries for the new filenames.

### 2. Novel scene generation with Cosmos Predict + pseudo-labeling (secondary approach)

Cosmos Predict can generate entirely new rail scenes (Text2World) or plausible continuations of existing cab-ride footage (Video2World / Image2World). These outputs have no ground-truth ego-path, so they must be labeled. `pseudo_label.py` implements this: it runs a trained teacher model over a directory of images, converts its predictions to the annotation format of this repository, and keeps only the predictions it can justify. See [Pseudo-labeling](#pseudo-labeling) below.

For domains where the teacher is unreliable even after filtering (e.g. dense fog, which is far outside RailSem19), annotate a small curated set by hand in the same JSON polyline format instead.

### 3. Quality filtering

Generated data should be filtered before training:

- `pseudo_label.py` rejects predictions the teacher is not confident about (see the criteria below).
- Use Cosmos Reason (or any VLM) to reject frames with implausible physics or missing/hallucinated tracks, before pseudo-labeling.
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
- **Compute:** Cosmos checkpoints range from ~2B to ~14B parameters. Generation is offline (dataset preparation, not training-time), so a single modern NVIDIA GPU with sufficient VRAM (or NVIDIA API Catalog / NIM endpoints) is enough; it does not affect the training or TensorRT inference requirements of this repo.
- **Licensing:** RailSem19-derived annotations are CC BY-NC-SA 4.0. Images generated *from* RailSem19 frames are derivatives — the non-commercial share-alike terms carry over. Cosmos model outputs themselves are governed by the NVIDIA Open Model License, which permits use of outputs for training. Review both before distributing a synthetic dataset.

## References

- [NVIDIA Cosmos product page](https://www.nvidia.com/en-us/ai/cosmos/)
- [Cosmos World Foundation Model Platform for Physical AI (paper)](https://arxiv.org/abs/2501.03575)
- [NVIDIA Cosmos GitHub organization](https://github.com/nvidia-cosmos)
- [Cosmos world foundation models and physical AI data tools announcement](https://nvidianews.nvidia.com/news/nvidia-announces-major-release-of-cosmos-world-foundation-models-and-physical-ai-data-tools)
- [World Simulation with Video Foundation Models for Physical AI (Cosmos-Predict 2.5 / Transfer 2.5 report)](https://arxiv.org/abs/2511.00062)
