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

### 2. Novel scene generation with Cosmos Predict (secondary approach)

Cosmos Predict can generate entirely new rail scenes (Text2World) or plausible continuations of existing cab-ride footage (Video2World / Image2World). These outputs have no ground-truth ego-path, so they require labeling:

- **Pseudo-labeling:** run one of the strong trained models from this repo (e.g. `twinkling-rocket-21`, IoU 0.9769) on generated frames, keep high-confidence predictions as labels, and optionally verify with Cosmos Reason or manual spot checks. This is most useful when the generated domain is close enough to the training domain for the teacher to be reliable.
- **Manual annotation:** for domains where pseudo-labels are unreliable (e.g. dense fog), annotate a small curated set by hand in the same JSON polyline format.

### 3. Quality filtering

Generated data should be filtered before training:

- Use Cosmos Reason (or any VLM) to reject frames with implausible physics or missing/hallucinated tracks.
- Reject frames where the teacher model's prediction disagrees strongly with the source annotation (for Transfer outputs, the source annotation is the reference — large IoU drop between prediction and annotation signals that the generation distorted the geometry).

## Practical notes

- **Training integration:** the dataset split in `train.py` is proportional (`train_prop`/`val_prop`/`test_prop` over sorted keys). Keep the validation and test sets composed of *real* images only — either by holding synthetic filenames out of the annotation file used for evaluation, or by adapting the index selection — so reported IoU stays comparable to the paper's numbers.
- **Augmentation interaction:** the existing ColorJitter augmentation (`brightness`/`contrast`/`saturation`/`hue` in `configs/global.yaml`) already covers mild photometric variation. Cosmos-generated data is complementary: it changes scene content (precipitation, illumination sources, shadows), which photometric jitter cannot.
- **Compute:** Cosmos checkpoints range from ~2B to ~14B parameters. Generation is offline (dataset preparation, not training-time), so a single modern NVIDIA GPU with sufficient VRAM (or NVIDIA API Catalog / NIM endpoints) is enough; it does not affect the training or TensorRT inference requirements of this repo.
- **Licensing:** RailSem19-derived annotations are CC BY-NC-SA 4.0. Images generated *from* RailSem19 frames are derivatives — the non-commercial share-alike terms carry over. Cosmos model outputs themselves are governed by the NVIDIA Open Model License, which permits use of outputs for training. Review both before distributing a synthetic dataset.

## References

- [NVIDIA Cosmos product page](https://www.nvidia.com/en-us/ai/cosmos/)
- [Cosmos World Foundation Model Platform for Physical AI (paper)](https://arxiv.org/abs/2501.03575)
- [NVIDIA Cosmos GitHub organization](https://github.com/nvidia-cosmos)
- [Cosmos world foundation models and physical AI data tools announcement](https://nvidianews.nvidia.com/news/nvidia-announces-major-release-of-cosmos-world-foundation-models-and-physical-ai-data-tools)
- [World Simulation with Video Foundation Models for Physical AI (Cosmos-Predict 2.5 / Transfer 2.5 report)](https://arxiv.org/abs/2511.00062)
