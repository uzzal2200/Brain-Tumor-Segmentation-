# EffiDec3D — Brain Tumor Segmentation

Dear reader,

Thank you for visiting this repository. This project demonstrates a 3D brain tumor segmentation workflow implemented in a Jupyter Notebook. The notebook `EffiDec3D_Brain_tumor_segmentation.ipynb` contains data loading, model definition, training, evaluation, and visualization steps.

## Overview

- Project: 3D brain tumor segmentation
- Model: EffiDec3D-style encoder-decoder 3D segmentation network
- Data: Multi-modal MRI volumes (T1, T1c, T2, FLAIR) in NIfTI format

## Repository contents

- `EffiDec3D_Brain_tumor_segmentation.ipynb` — Main notebook (data, training, evaluation, visualization)
- `read.md` — A concise guide (English)
- `README.md` — This file (conventional repository entry)

## Requirements

Recommended: Python 3.8 or newer.

Core packages:

- numpy
- scipy
- matplotlib
- scikit-learn
- nibabel (for NIfTI input/output)
- torch (PyTorch) and torchvision
- jupyter

## Quick setup (Windows — PowerShell)

Open PowerShell in the repository root and run:

```powershell
# Create and activate a virtual environment
python -m venv .venv; .\.venv\Scripts\Activate.ps1

# Upgrade pip and install required packages
python -m pip install --upgrade pip
pip install numpy scipy matplotlib scikit-learn jupyter nibabel torch torchvision

# Start the Jupyter Notebook (from the repository root)
jupyter notebook EffiDec3D_Brain_tumor_segmentation.ipynb
```

If you have a CUDA-capable GPU, please install the matching PyTorch+CUDA build from https://pytorch.org for best performance.

## Dataset layout (example)

This project typically expects NIfTI volumes organized per-subject. Example layout:

```
data/
  patient_001/
    T1.nii.gz
    T1c.nii.gz
    T2.nii.gz
    FLAIR.nii.gz
    segmentation.nii.gz  # ground truth mask
```

Adjust paths in the notebook's data-loading cells if your dataset is organized differently.

## Running & training notes

- Run notebook cells top-to-bottom.
- Full 3D training is resource-intensive — prefer GPU. On CPU, reduce batch size and consider downsampling.
- Model checkpoints are saved/loaded with PyTorch (`torch.save()` / `torch.load()`) — check the notebook for checkpoint paths.

## Troubleshooting

- PyTorch install / CUDA issues: follow official instructions at https://pytorch.org
- NIfTI read errors: ensure `nibabel` is installed and the file path is correct
- If kernel dies when training: lower batch size or use mixed precision / smaller crops

## License & usage

No explicit license file is included in this repository. Please contact the repository owner before reusing or redistributing the code for commercial purposes.

## Next steps — I can help with any of the following

- Create a `requirements.txt` with pinned versions
- Rename `read.md` to `README.md` (if you prefer to keep a single README) or remove `read.md`
- Add a small sanity-check script that imports key packages and prints versions
- Configure a `conda` environment file or provide CUDA-specific PyTorch install commands

## Contact / Support

If you want me to perform any of the next steps above, tell me which one and I will create the file or run the check for you.

With best regards,
Your collaborator

---

Note: The notebook in this repository is actually `EffiDec3D_Brain_tumor_segmentation (1).ipynb`. The following section below is an expanded README that mirrors that notebook's exact workflow, configuration, and commands. Use it if you want a reproducible, notebook-aligned README.

## Notebook-aligned README (details from the notebook)

This section summarizes the actual workflow implemented in the notebook titled `EffiDec3D_Brain_tumor_segmentation (1).ipynb`.

### Purpose
- Demonstrates a 2D encoder-decoder segmentation model (named `EffiDec3D_Medium` in the notebook) trained on the LGG MRI segmentation dataset (Kaggle: mateuszbuda/lgg-mri-segmentation).

### Notebook highlights
- Downloads the Kaggle dataset using `opendatasets`.
- Builds a pandas DataFrame of image/mask pairs and labels images by whether the mask contains positive pixels (used for oversampling).
- Uses Keras `ImageDataGenerator` for images and masks, with an `adjust_data` function that normalizes images and thresholds masks.
- Custom metrics and losses: `dice_coef`, `dice_loss`, `bce_dice_loss`, and `iou`.
- Architecture details:
  - `dynamic_selector`: a channel-wise SE-style selector
  - `nas_decoder_block`: a NAS-inspired decoder block mixing two branch outputs by learned weights
  - `enc_block`: simple conv-conv-pool encoder blocks
  - `EffiDec3D_Medium(input_shape=(256,256,3))`: assembled encoder-decoder model
- Training config in the notebook: optimizer Adam(3e-4), loss `bce_dice_loss`, metrics `binary_accuracy`, `dice_coef`, `iou`.
- Checkpoint: `ModelCheckpoint("EffiDec3D_Medium_best.keras", save_best_only=True)`

### Important hyperparameters (as used in the notebook)
- Input size: 256 x 256 (3 channels)
- Batch size: 16
- Epochs: 50

### Dependencies (minimum list)
- Python 3.8+
- opendatasets
- tensorflow (2.x)
- numpy
- pandas
- matplotlib
- scikit-learn
- opencv-python

### Install (Windows PowerShell)
```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install opendatasets tensorflow numpy pandas matplotlib scikit-learn opencv-python
```

### Dataset
- Kaggle dataset used in the notebook: https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation/data
- The notebook downloads it with `opendatasets` and expects a structure similar to:

  `./lgg-mri-segmentation/kaggle_3m/*/*_mask*`

### Run steps (summary)
1. Create and activate virtual environment and install dependencies (see above).
2. Open Jupyter and run the notebook:

```powershell
jupyter notebook "EffiDec3D_Brain_tumor_segmentation (1).ipynb"
```

3. Run cells sequentially: dataset download, generator creation, model definition, and `model.fit(...)`.

### Notes
- `opendatasets` may require Kaggle credentials (`kaggle.json`) placed under `%USERPROFILE%\.kaggle\` or follow prompts from the package.
- If you hit OOM during training, reduce the batch size or input resolution.
- GFLOPs profiling in the notebook uses TensorFlow profiler internals and may require additional TF packages; it is optional.

### Outputs
- The main artifact produced by the notebook is the checkpoint file `EffiDec3D_Medium_best.keras`.

### Optional extras I can add
- `requirements.txt` with pinned versions
- `tools/check_env.py` to validate imports and versions
- Merge `read.md` and `README.md` into a single canonical `README.md`

If you'd like any of these extras created, tell me which and I will add them.
