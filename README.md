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
