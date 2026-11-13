# Data and material licensing

The MIT license in `LICENSE` applies to this project's analysis software. It does not relicense third-party datasets.

## Perception Test derived material

The Perception Test derived question units, split assignments, project predictions, scores, aggregate results and project figures are distributed under Creative Commons Attribution 4.0 International:
https://creativecommons.org/licenses/by/4.0/
Full legal text: https://creativecommons.org/licenses/by/4.0/legalcode

Source: Google DeepMind, *Perception Test: A Diagnostic Benchmark for Multimodal Video Models*, Patraucean et al., NeurIPS 2023.
Official source and license: https://github.com/google-deepmind/perception_test#license-and-disclaimer
Upstream copyright notice: Copyright 2022 DeepMind Technologies Limited.
Project additions and transformations: Copyright 2026 Quan Ji.

Changes include grouping source events into video/action counting units, defining video-level splits and eligibility, constructing prompts, generating model predictions, and producing analyses and figures. Figure 6 contains selected source-video frames arranged and annotated as a qualitative illustration. It retains the Perception Test attribution and license; it is not a wholly original image. No source video archive or extracted-frame directory is included.

When redistributing this material, retain appropriate attribution, link to CC BY 4.0, and state modifications. The upstream project's software license is Apache-2.0; that software is not relicensed by this repository's MIT notice.

## Countix

Countix is a separate source. It is not part of Perception Test.
Official Countix project: https://sites.google.com/view/repnet
Question-answering distribution used in this project: https://huggingface.co/datasets/kekeYeah/Countix-QA

At the preparation of this release, the Countix-QA page provided no dataset card or explicit license file. This release therefore includes source identifiers, this project's numerical model predictions and aggregate analyses, but does not redistribute Countix-QA videos, source question text or ground-truth annotations. The two `countix*_units.jsonl` files are ID/split indexes only. Countix prediction JSONL files contain ID fields, numerical predictions and, where available, the frame budget. Obtain source labels from the cited upstream distribution under its applicable terms.

CC BY 4.0 is offered for this project's own prediction outputs and aggregate analyses. This does not grant rights to upstream Countix videos, labels or question text. `06_documents/COUNTIX_REPRODUCTION.md` explains how externally obtained labels can be joined to the released predictions.

## Attribution and dependencies

Third-party packages, models and datasets remain subject to their own licenses. They are referenced rather than bundled here. No model weights, adapter weights or hidden-state caches are included. Historical analysis files are marked in `08_history/README.md`; inclusion is not an endorsement of a withdrawn method.
