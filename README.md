# simple_kd

[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Accelerate](https://img.shields.io/badge/accelerate-distributed-orange.svg)](https://huggingface.co/docs/accelerate/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

simple_KD is a library that combines non data free methods implemented by repositories such as mdistiller and the Knowledge Distillation Zoo as well as popular data free distillation methods and implements them with pytorch-style structure and calling. The library also uses the Hugging Face
accelerate library in order to support multiple GPUs.

---

## Table of Contents

- [Overview](#overview)
- [Conveniences](#conveniences)
- [Supported Distillation Methods](#supported-distillation-methods)
  - [Logit-Based Distillation](#logit-based-distillation)
  - [Feature & Attention-Based Distillation](#feature--attention-based-distillation)
  - [Relational Distillation](#relational-distillation)
  - [Dynamic & Skeptical Arbitration](#dynamic--skeptical-arbitration)
  - [Data-Free Knowledge Distillation (DFKD)](#data-free-knowledge-distillation-dfkd)
  - [Synthetic Noise & Inductive Bias Generators](#synthetic-noise--inductive-bias-generators)
- [Installation](#installation)
- [Data & Hook Utilities](#data--hook-utilities-simple_kd.utils)
- [Included Examples](#included-examples)
- [Performance Results](#performance-results)
- [Testing & Quality Verification](#testing--quality-verification)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Overview

**`simple_kd`** is designed to collect all the relevant knowledge distillation methods when implementing, testing, and scaling in a single place, avoiding digging and understanding multiple and usually deprecated repositories. Whether you are performing standard logit matching, intermediate feature alignment, relational metric transfer, or zero-data distillation with adversarial generators, `simple_kd` provides composable, highly tested primitives. Currently the methods are only concerning Classification tasks, with plans in the future to also include segmentation and other kinds of problems.

### Conveniences:
- **Unified APIs**: All loss functions provide both standard PyTorch `nn.Module` classes and zero-overhead functional counterparts (`simple_kd.losses.functional`).
- **Distributed by Default**: Built from the ground up on Hugging Face [Accelerate](https://huggingface.co/docs/accelerate/) — effortlessly runs on CPU, single GPU, Multi-GPU Distributed Data Parallel (DDP), and mixed precision (`fp16`, `bf16`).
- **Data-Free Distillation Ready**: Full end-to-end training loops for **DAFL**, **DFAD**, and **ZSKT**, along with other methods, including unified generator statistics tracking (entropy, confidence, diversity spread) and gradient clipping.
- **Synthetic Noise Datasets**: Generates structured noise datasets (Dead Leaves, Fourier Feature Inversion, Wavelet Steerable Pyramids, Untrained StyleGAN2) allowing networks to learn visual inductive biases without natural data.
- **Simplicity**: Designed from scratch to be easy to pick up and familiar to the pytorch syntax.
- **Zero-Leak Hook Management**: `FeatureSelector` automatically manages PyTorch forward hooks and garbage collection for multi-layer representation distillation.

---

## Conveniences

- **A vast amount of available techniques**: Logit matching (KD, DKD), spatial attention (AT), representation similarity (SP, SSP), geometric relationships (RKD-Distance, RKD-Angle, PKT), and skeptical student gating.
- **Modular Data-Free Engines**: Plug-and-play adversarial and generative distillation algorithms.
- **Telemetry & Artifacts**: Automatic CSV training history logging, metric summaries, and Parquet prediction matrices for evaluation.
- **Plethora of examples**: Multiple of examples covering all the possible usecases, from utilizing the existing pre-defined training pipelines to integrating complex distillation processes in a custom pipeline.

---

## Supported Distillation Methods

### Logit-Based Distillation
| Method | Class | Functional | Key Idea |
| :--- | :--- | :--- | :--- |
| **KD** | `KDLoss` | `kd_loss` | The original implementation, matching the teacher and student probabilities through the KL divergence |
| **DKD** | `DKDLoss` | `dkd_loss` | The incorrect classes carry possible more knowledge than the correct one (dark knowledge) |

### Feature & Attention-Based Distillation
| Method | Class | Functional | Key Idea |
| :--- | :--- | :--- | :--- |
| **AT** | `ATLoss` | `at_loss` | Attention map matching between the teacher and the student models |
| **CC** | `CCLoss` | `cc_loss` | Attempts to improve correlation between the teacher and the student |
| **SP** | `SPLoss` | `sp_loss` | Attempts to match the teacher and student representation space by keeping similar pairwise similarities |
| **SSP** | `SSPLoss` | `ssp_loss` | Based on the [fitnets](https://arxiv.org/pdf/1412.6550) approach. Passes the stages through spatial pooling and then directly matches the teacher and student features | Custom |

### Relational Distillation
| Method | Class | Functional | Key Idea |
| :--- | :--- | :--- | :--- |
| **NST**| `NSTLoss`| `nst_loss`| Student mimics neuron co-activations from the teacher. |
| **RKD-Dist** | `RKDDistLoss` | `rkd_distance_loss` | Attempts to match the representation space of the teacher through the distance of couples |
| **RKD-Angle** | `RKDAngleLoss` | `rkd_angle_loss` | Attempts to match the representation space of the teacher via the angle formed by a triplet of samples |
| **PKT** | `PKTLoss` | `pkt_loss` | Distills through the representation space rather than the final predictions |

### Dynamic & Skeptical Arbitration
| Method | Class | Functional | Key Idea |
| :--- | :--- | :--- | :--- |
| **Skeptical** | `SkepticalLoss` | `skeptical_loss` | Attempts to distill from a nasty teacher by distilling from earlier layers, which remain mostly unaffected |

### Data-Free Knowledge Distillation (DFKD)
| Method | Function | Key Idea |
| :--- | :--- | :--- |
| **DAFL** | `dafl_train` | Multi-objective Generator based method for data free training of the student network|
| **DFAD** | `dfad_train` | Minimize the discrepancy between the teacher and the student. Directly applicable to segmentation tasks |
| **ZSKT** | `zskt_train` | A generator creates examples that the student poorly matches the teacher, and then trains on them |
| **Gaussian distillation** | `gaussian_train`| Utilizes gaussian noise (or even synthetic noise) in order to train through the saved batch statistics |
| **ODSD** | `odsd_train` | Choose low noise open world data to avoid domain shift issues and the generator module altogether |
| **FastDFKD(TODO)** | `fdfkd_train` | Use cached examples with common features to speed up the training process  immensely|
| **ZSKD(TODO)** | `zskt_train` | Adversarial generator training with attention matching and student divergence |

### Synthetic Noise & Inductive Bias Generators

The library has also integrated several options first mentioned in the Paper [Learning to See by Looking at Noise](https://arxiv.org/pdf/2106.05963) in order to create synthetic noise to use in place of gaussian noise and even the generators. Specifically the noise types are as follows:

| Generator Model | Module | Characteristics |
| :--- | :--- | :--- |
| **Dead Leaves** | `simple_kd.generators.synthetic_noise.dead_leaves` | Occlusion modeling with power-law scale distribution ($r^{-\sigma}$) producing natural $1/f^2$ statistics. |
| **Feature Visualization** | `simple_kd.generators.synthetic_noise.feature_vis` | Optimizes the value of a single neuron or a combination of neurons to create an image. |
| **Statistical Noise** | `simple_kd.generators.synthetic_noise.statistical` | Combines different combination of properties such as Spectrum, Wavelet-marginal models as well as color histograms to generate more complex noise  |
| **Untrained StyleGAN2** | `simple_kd.generators.synthetic_noise.stylegan` | Utilizes an untrained StyleGAN2 in order to create effective training data from it. |

<!-- """image: Visual gallery showing samples generated by Dead Leaves, Untrained StyleGAN2, Statistical Noise, and Feature Inversion""" -->

---

## Installation

### Prerequisites
- Python >= 3.10
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- 4.0 <= opencv-python <= 5.0
- pyarrow>=22.0
- scikit-learn>=1.7
- accelerate>=0.20.0

### Install from Source
```bash
git clone https://github.com/evanlour/simple_KD.git
cd simple_KD
pip install -e .
```

### Optional Dependencies
For visualization (Plotly, Matplotlib, Seaborn) or developer tooling (pytest, ruff, mypy):
```bash
pip install -e ".[viz,dev]"
```

---

## Data & Hook Utilities

- **`Dataset`, `LabelFreeDataset` & `InMemoryDataset`** ([simple_kd/utils/dataset.py](./simple_kd/utils/dataset.py)):
  Simple ready-to-use Dataset classes to load .npy files, either using a dataloader or directly in memory.
- **`FeatureSelector` & `apply_hooks`** ([simple_kd/utils/feature_selector.py](./simple_kd/utils/feature_selector.py)):
  Registers non-intrusive forward hooks on intermediate layers to capture activation representations for AT, SP, or feature distillation.
- **`SimpleGradClip`** ([simple_kd/utils/simple_grad_clip.py](./simple_kd/utils/simple_grad_clip.py)):
  Clean wrapper supporting gradient clipping by **norm**, **value**, or **adaptive** gradient clipping [(AGC)](https://arxiv.org/pdf/2102.06171).
- **`inspect_dataset`** ([simple_kd/utils/inspect_dataset.py](./inspect_dataset.py)):
  Quick statistical diagnostics (sample counts, shape, mean, std, class balance) for arbitrary PyTorch datasets.

---

## Included Examples

The [examples](./examples) directory provides turnkey scripts ready to run:

| Script | Description | Run Command |
| :--- | :--- | :--- |
| [train_example.py](./examples/train_example.py) | Train a baseline teacher model with `accelerate`. | `python ./examples/train_example.py` |
| [distill_example.py](./examples/distill_example.py) | Classical distillation pipeline using the DKD method | `python ./examples/distill_example.py` |
| [dafl_example.py](./examples/dafl_example.py) | Data-Free Learning (DAFL) with simultaneous generator updates. | `python ./examples/dafl_example.py` |
| [dfad_example.py](./examples/dfad_example.py) | Data-Free Adversarial Distillation (DFAD) with min-max game. | `python ./examples/dfad_example.py` |
| [zskt_example.py](./examples/zskt_example.py) | Zero-Shot Knowledge Transfer (ZSKT) adversarial training. | `python ./examples/zskt_example.py` |
| [gaussian_example.py](./examples/gaussian_example.py) | Distillation with pure synthetic Gaussian noise. | `python ./examples/gaussian_example.py` |
| [generate_synthetic_noise.py](./examples/generate_synthetic_noise.py) | Generates Dead Leaves and StyleGAN synthetic datasets. | `python ./examples/generate_synthetic_noise.py` |
| [select_layers.py](./examples/select_layers.py) | Inspect model layers and verify hook extraction. | `python ./examples/select_layers.py` |
| [prepare_data.py](./examples/prepare_data.py) | Prepares(and downloads) the dataset that will be tested. Currently only CIFAR100 | `python ./examples/prepare_data.py` |
| [use_train_profiler.py](./examples/use_train_profiler.py) | Runs PyTorch Profiler on a sample training loop. | `python ./examples/use_train_profiler.py` |

---

## Performance Results

The following table summarizes the performance on CIFAR100 of various distillation methods implemented in `simple_kd`. Some notes about the training:

- For the Non Data Free methods the stock hyperparameters for each method were used. Train times were similar and around the 8 minute mark
- For the Data Free methods, as there are a lot of variables, the hyperparameters used are hardcoded inside the examples. As Data free methods are usually time-intensive, i tried to keep a total train time of around an hour for each. I will attach more info for the methods below. For reference, my hardware consists of an Nvidia RTX3060 and 16GB or RAM
-  I set the DAFL epochs to 100  instead of 2000 used in the original paper and used a more aggresive optimizer setup as well as a much higher info entropy loss (80 instead of 20) to prevent generator collapse
- For the gaussian distillation 50K images of StyleGAN seems to not be enough as training goes almost immediately to 40+% and then very slowly increases. A test with more images and/or combined synthetic noise must be tested (TODO)
- The same goes for the ODSD method, more images with more variance might be needed(TODO).
- ZSKT method currently does not work, it needs further testing on why that is (TODO).

| **Method**                       | **Accuracy** | **Loss** |  **F1 Score(macro)** |
| :----------------                | :----------: | :------: | :------------------: |
| **Teacher(Resnet34) baseline**   |    73.16%    |   1.052  |         0.7294       |
| **Student(Resnet18) baseline**   |    72.84%    |   1.066  |         0.7271       |
| **Student with low data**        |    52.26%    |   1.926  |         0.5172       |
| **Non Data free methods**        | 
| **AT**                           |    57.01%    |   1.694  |         0.5639       |
| **CC**                           |    52.83%    |   1.862  |         0.5227       |
| **DKD**                          |    67.05%    |   1.321  |         0.6669       |
| **KD**                           |    60.18%    |   1.611  |         0.5968       |
| **NST**                          |    58.97%    |   1.524  |         0.5812       |
| **PKT**                          |    53.28%    |   1.863  |         0.5270       |
| **RKD Angle**                    |    56.31%    |   1.738  |         0.5575       |
| **RKD Dist**                     |    54.15%    |   1.813  |         0.5348       |
| **SP**                           |    54.47%    |   1.819  |         0.5394       |
| **SSP**                          |    54.57%    |   1.785  |         0.5390       |
| **Data free methods**            |
| **DAFL**                         |    58.93%    |   1.522  |         0.5822       |
| **DFAD**                         |    13.49%    |   4.629  |         0.1012       |
| **Gaussian (Noise)**             |    62.64%    |   1.851  |         0.6192       |
| **Gaussian (StyleGAN)**          |    49.15%    |   2.257  |         0.4804       |
| **ODSD (50% StyleGAN)**          |    26.96%    |   3.048  |         0.2320       |
| **ZSKT**                         |     -        |    -     |         -            |

All the common hyperparameters between the methods such as the optimizer, the scheduler, the batch size etc for the non data free methods are the same. The data free methods vary a lot, so each method needs its own tuning.

For distillation training only 20% of the train data were kept.

These benchmarks are not representative of performance in real world problems but they give us an idea of how they work. Some methods are better suited for specific problems.

---

## Testing & Quality Verification

Run unit tests and verification across all loss functions and synthetic noise generators:

```bash
# Verify all loss methods against PyTorch accelerator
python tests/verify_loss_methods.py

# Verify synthetic noise generation routines
python tests/check_synthetic_noise.py
```

Run code formatting and static type checking:
```bash
ruff check .
ruff format .
```

---

## Acknowledgements

This project builds upon the following repositories and methods. Thanks to their authors and maintainers.

| **Name / Method** | **Paper** | **GitHub Repo** |
| :--- | :--- | :--- |
| **AGC (Adaptive Gradient Clipping)** | [High-Performance Large-Scale Image Recognition Without Normalization](https://arxiv.org/pdf/2102.06171) | [https://github.com/huggingface/pytorch-image-models/blob/main/timm/utils/agc.py](https://github.com/huggingface/pytorch-image-models/blob/main/timm/utils/agc.py) |
| **AT loss** | [PAYING MORE ATTENTION TO ATTENTION:IMPROVING THE PERFORMANCE OF CONVOLUTIONAL NEURAL NETWORKS VIA ATTENTION TRANSFER](https://arxiv.org/pdf/1612.03928) | [https://github.com/szagoruyko/attention-transfer](https://github.com/szagoruyko/attention-transfer) |
| **CC loss** | [Correlation Congruence for Knowledge Distillation](https://openaccess.thecvf.com/content_ICCV_2019/papers/Peng_Correlation_Congruence_for_Knowledge_Distillation_ICCV_2019_paper.pdf) | - |
| **DAFL** | [Data-Free Learning of Student Networks](https://arxiv.org/pdf/1904.01186) | [https://github.com/autogyro/DAFL](https://github.com/autogyro/DAFL) |
| **DFAD** | [Data-Free Adversarial Distillation](https://arxiv.org/pdf/1912.11006) | [https://github.com/VainF/Data-Free-Adversarial-Distillation](https://github.com/VainF/Data-Free-Adversarial-Distillation) |
| **DKD loss** | [Decoupled Knowledge Distillation](https://arxiv.org/pdf/2203.08679) | [https://github.com/megvii-research/mdistiller](https://github.com/megvii-research/mdistiller) |
| **FastDFKD** | [Up to 100× Faster Data-free Knowledge Distillation](https://arxiv.org/pdf/2112.06253) | [https://github.com/zju-vipa/Fast-Datafree](https://github.com/zju-vipa/Fast-Datafree) |
| **Feature Vis tools** | - | [Lucent](https://github.com/greentfrapp/lucent), [Lucid](https://github.com/tensorflow/lucid) |
| **Gaussian** | [Discovering and Overcoming Limitations of Noise-engineered Data-free Knowledge Distillation](https://proceedings.neurips.cc/paper_files/paper/2022/file/1f96b24df4b06f5d68389845a9a13ed9-Paper-Conference.pdf) | [https://github.com/Piyush-555/GaussianDistillation](https://github.com/Piyush-555/GaussianDistillation) |
| **KD loss** | [Distilling the Knowledge in a Neural Network](https://arxiv.org/pdf/1503.02531) | - |
| **Knowledge Distillation Zoo** | - | [https://github.com/AberHu/Knowledge-Distillation-Zoo](https://github.com/AberHu/Knowledge-Distillation-Zoo) |
| **NST loss** | [Like What You Like: Knowledge Distill via Neuron Selectivity Transfer](https://arxiv.org/pdf/1707.01219) | - |
| **StyleGAN2-ADA** | [Training Generative Adversarial Networks with Limited Data](https://arxiv.org/pdf/2006.06676) | [https://github.com/nvlabs/stylegan2-ada-pytorch](https://github.com/nvlabs/stylegan2-ada-pytorch) |
| **Synthetic Noise Datasets** | [Learning to See by Looking at Noise](https://arxiv.org/pdf/2106.05963) | [https://github.com/mbaradad/learning_with_noise](https://github.com/mbaradad/learning_with_noise) |
| **mdistiller** | [DOT: A Distillation-Oriented Trainer](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhao_DOT_A_Distillation-Oriented_Trainer_ICCV_2023_paper.pdf) | [https://github.com/megvii-research/mdistiller](https://github.com/megvii-research/mdistiller) |
| **ODSD** | [Sampling to Distill: Knowledge Transfer from Open-World Data](https://arxiv.org/pdf/2307.16601v2) | - |
| **PKT loss** | [Learning Deep Representations with Probabilistic Knowledge Transfer](https://arxiv.org/pdf/1803.10837) | [https://github.com/passalis/probabilistic_kt](https://github.com/passalis/probabilistic_kt) |
| **RKD (Angle, Dist) loss** | [Relational Knowledge Distillation](https://arxiv.org/pdf/1904.05068) | [https://github.com/lenscloth/RKD](https://github.com/lenscloth/RKD) |
| **Skeptical loss** | [Analyzing the Confidentiality of Undistillable Teachers in Knowledge Distillation](https://openreview.net/pdf?id=Laz0L5tjml) | [https://github.com/ksouvik52/Skeptical2021](https://github.com/ksouvik52/Skeptical2021) |
| **SP loss** | [Similarity-Preserving Knowledge Distillation](https://arxiv.org/pdf/1907.09682) | [https://github.com/DongGeun-Yoon/SPKD](https://github.com/DongGeun-Yoon/SPKD) |
| **SSP Loss** | (*Original method, based on*) [FITNETS: HINTS FOR THIN DEEP NETS](https://arxiv.org/pdf/1412.6550) | [https://github.com/adri-romsor/FitNets](https://github.com/adri-romsor/FitNets) |
| **ZSKD** | [Zero-Shot Knowledge Distillation in Deep Networks](https://arxiv.org/pdf/1905.08114) | [https://github.com/vcl-iisc/ZSKD](https://github.com/vcl-iisc/ZSKD) |
| **ZSKT** | [Zero-shot Knowledge Transfer via Adversarial Belief Matching](https://arxiv.org/pdf/1905.09768) | [https://github.com/polo5/ZeroShotKnowledgeTransfer](https://github.com/polo5/ZeroShotKnowledgeTransfer) |

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
