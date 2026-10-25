# Deep DIC: Displacement Field Estimation from Speckle Images

This project implements **Deep Digital Image Correlation (DIC)** using deep learning. Given a reference (initial) grayscale speckle image and a deformed grayscale speckle image, a model predicts a dense displacement field \((u_x, u_y)\) at each pixel.

The project is implemented in PyTorch and includes three model architectures, a training pipeline, inference and evaluation utilities, and saved experimental results.

## Features

- Three neural network architectures for dense displacement-field estimation.
- Charbonnier endpoint loss for training.
- Training pipeline with checkpoint saving and CSV/JSON result export.
- Single-sample inference and endpoint-error evaluation.
- Displacement-field visualization using quiver plots.
- Training and evaluation loss-curve plotting.

## Model Architectures

All three models accept two single-channel images and predict a displacement field with shape `[B, 2, H, W]`, where the two output channels represent \(u_x\) and \(u_y\).

| Model | Architecture | Key components |
|---|---|---|
| `crossflow` | `TransformerDIC` | ResNet-34 backbone, Transformer encoder, cross-attention, global matching, and displacement decoder |
| `efficientnet_encoder_decoder` | `EffNetEncDec` | EfficientNet-B2 backbone, transposed-convolution decoder, and skip connections |
| `efficientnet_transformer` | `EffnetTranformer` | EfficientNet-B2 backbone, Transformer encoder/decoder, and transposed-convolution upsampling |

The CrossFlow model uses intermediate ResNet-34 features and global matching to estimate correspondences. The EfficientNet variants explore alternative feature-extraction and decoding strategies.

The models are implemented in:

- `models/crossflow.py`
- `models/efficientnet_encoder_decoder.py`
- `models/efficientnet_transformer.py`

Use the corresponding model class and checkpoint when training or performing inference. Refer to the current training script and model registry for the supported model-selection arguments.

## Repository Structure

```text
.
├── models/
│   ├── __init__.py
│   ├── crossflow.py
│   ├── efficientnet_encoder_decoder.py
│   ├── efficientnet_transformer.py
│   └── checkpoints/
├── scripts/
│   ├── train.py
│   ├── dataloader.py
│   ├── data_prep.py
│   ├── engine.py
│   ├── helpers.py
│   ├── predictions.py
│   └── plotting.py
├── results/
├── AI_mfg_data/          # Downloaded separately; not tracked in this repository
├── AI_mfg.ipynb
├── train.ipynb
├── predict.ipynb
├── requirements.txt
├── .gitattributes
└── .gitignore
```

**Main directories and files**

- `models/`: Model architecture definitions.
- `models/checkpoints/`: Saved model checkpoints tracked using Git LFS.
- `scripts/train.py`: Training entry point.
- `scripts/dataloader.py`: Dataset and data-loader utilities.
- `scripts/data_prep.py`: Dataset filtering and preparation.
- `scripts/engine.py`: Training and evaluation loops and checkpoint saving.
- `scripts/helpers.py`: Model registry, model construction, and helper functions.
- `scripts/predictions.py`: Single-sample inference and displacement-error evaluation.
- `scripts/plotting.py`: Loss curves and displacement visualization.
- `results/`: Saved experiment metrics and plots.
- `AI_mfg_data/`: Local dataset directory populated by downloading the external dataset.
- The Jupyter notebooks provide interactive workflows for training, prediction, and exploration.

The tree describes the intended layout; the exact files present may vary with the current repository version.

## Requirements and Installation

Python dependencies are listed in `requirements.txt`.

Clone this project repository:

```bash
git clone https://github.com/keerthi806/AI_in_manf_project.git
cd AI_in_manf_project
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

For PyTorch installation instructions tailored to your operating system and hardware, see the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).

### Git LFS

This project uses [Git Large File Storage (Git LFS)](https://git-lfs.com/) for `.pth` model checkpoints.

Install Git LFS if necessary, then initialize it and download the checkpoint files:

```bash
git lfs install
git lfs pull
```

Checkpoint availability depends on which model files have been committed and uploaded to the repository.

## Dataset and Preprocessing

This project uses the publicly available dataset from the following external repository:

**Dataset source:** [Deep-DIC: Deep Learning-Based Digital Image Correlation](https://github.com/RuYangNU/Deep-Dic-deep-learning-based-digital-image-correlation)

The dataset is maintained separately by its original authors and is **not included in this project repository**. Please refer to the original repository for dataset download instructions, usage terms, and attribution requirements.

Download or clone the dataset separately:

```bash
git clone https://github.com/RuYangNU/Deep-Dic-deep-learning-based-digital-image-correlation.git
```

Configure the `--data-path` argument to point to the dataset directory expected by this project. If necessary, organize or configure the downloaded files to match the layout below.

### Expected Dataset Layout

The current data-loading pipeline expects a structure similar to:

```text
AI_mfg_data/
├── train_images/
│   └── *.png
└── gt_labels/
    └── *.mat
```

The expected files include grayscale reference/deformed image pairs and corresponding ground-truth displacement fields stored in MATLAB `.mat` files.

Based on the project's data-loading implementation:

- Images are loaded using Pillow and converted to tensors using `torchvision.transforms.ToTensor()`.
- Grayscale image tensors have values in the range `[0, 1]`.
- Ground-truth displacement fields have shape `(H, W, 2)` and are transposed to `(2, H, W)` for training.
- The data-preparation code pairs images by sample identifier and filters for samples with matching ground-truth labels.
- The loader expects displacement data under MATLAB keys such as `Disp_field_1` or `Disp_field_2`, depending on the implementation.

**Important:** The external dataset's actual folder structure, filenames, and MATLAB label keys may differ from the expected layout above. Check `scripts/data_prep.py` and `scripts/dataloader.py` and adjust the data path or preprocessing configuration as necessary.

If the dataset is stored elsewhere, provide its location through `--data-path`.

## Training

Run the training script from the project root.

Example command:

```bash
python scripts/train.py \
    --model model \
    --data-path AI_mfg_data \
    --data-frac 0.3 \
    --test-size 0.2 \
    --batch-size 32 \
    --seed 42 \
    --lr 1e-3 \
    --epochs 10 \
    --device mps \
    --save-dir models/checkpoints \
    --results-dir results \
    --plot
```

This is an example configuration. Adjust the arguments to match your dataset, hardware, and experiment.

The `mps` device option is intended for compatible Apple Silicon systems. Select another supported device if necessary.

Check the available arguments in the current training script:

```bash
python scripts/train.py --help
```

The training pipeline saves checkpoints and experiment results according to its configured output directories. The best checkpoint is selected using the loss criterion implemented in the training code.

The exact model-selection names and available command-line arguments should match the current implementation in `scripts/train.py` and `scripts/helpers.py`.

## Inference and Evaluation

The project provides a `predict()` function for inference and utilities for comparing predicted displacement fields against ground truth.

The following is an example for the CrossFlow architecture. It assumes the checkpoint contains a compatible state dictionary and that `img1` and `img2` are already prepared in the format expected by `predict()`.

```python
import torch
from models.crossflow import TransformerDIC
from scripts.predictions import predict

model = TransformerDIC(pretrained=False)

checkpoint = torch.load(
    "models/checkpoints/crossflow.pth",
    map_location="cpu"
)
model.load_state_dict(checkpoint)

pred = predict(
    model=model,
    img1=img1,
    img2=img2,
    accelerator=False
)
```

The prediction is a displacement tensor with shape `[2, H, W]`.

For another architecture, instantiate the corresponding model class and load its matching checkpoint.

### Evaluation Metrics

The displacement-comparison utilities report:

- **Mean EPE:** Mean endpoint error across pixels.
- **Median EPE:** Median endpoint error across pixels.
- **EPE RMSE:** Root mean square endpoint error.
- **Mean absolute \(u_x\) error:** Mean absolute error in the horizontal displacement component.
- **Mean absolute \(u_y\) error:** Mean absolute error in the vertical displacement component.
- **Maximum EPE:** Largest endpoint error among evaluated pixels.

Endpoint error is calculated from the difference between the predicted and ground-truth displacement vectors.

The project also provides quiver plots for visual comparison of predicted and ground-truth displacement fields.

Refer to `scripts/predictions.py` for the current inference and evaluation workflow. If it relies on configured relative paths, run it from the directory expected by the script.

## Results

The `results/` directory contains saved experiment artifacts, including CSV/JSON metrics and loss-curve plots.

Results should be interpreted in the context of their corresponding training configurations and evaluation procedures. Single-sample comparisons are useful for inspecting predictions, but they do not establish generalization across the full dataset.

For a reliable comparison between architectures, evaluate all models on the same held-out test samples using a consistent evaluation protocol.

## Git LFS and Model Checkpoints

Model checkpoints use the `.pth` extension and are stored using [Git Large File Storage (Git LFS)](https://git-lfs.com/).

To configure LFS tracking for `.pth` files:

```bash
git lfs install
git lfs track "*.pth"
git add .gitattributes
```

The `.gitattributes` file must be committed to the repository so Git knows which files should be stored through LFS.

After cloning the project repository, retrieve the checkpoint files using:

```bash
git lfs install
git lfs pull
```

Checkpoint files should be tracked through Git LFS rather than committed as ordinary Git blobs. GitHub's applicable LFS file-size, storage, and bandwidth limits still apply.

## Limitations and Future Work

- Broader evaluation on held-out data is needed to establish model generalization.
- Reproducibility depends on compatible dependency versions and consistent training configurations.
- Future work could include automated tests, configuration files, and systematic architecture ablation studies.
- Additional evaluation across different displacement magnitudes and deformation patterns could help characterize model performance.

## Acknowledgements

This project uses the publicly available dataset from [RuYangNU/Deep-Dic-deep-learning-based-digital-image-correlation](https://github.com/RuYangNU/Deep-Dic-deep-learning-based-digital-image-correlation).

Please consult the original repository for its recommended citation, licensing information, and dataset attribution requirements.
