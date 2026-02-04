**From Voxels to Diagnosis: Automated Brain Tumor Segmentation**
*A Comparative Analysis of Volumetric (3D) and Pseudo-3D (2.5D) Deep Learning Architectures.*

##  Overview

Manual segmentation of brain tumors is a bottleneck in clinical workflows, taking hours of expert radiologist time. This project automates the segmentation of **Edema, Necrotic Core, and Enhancing Tumor** regions using the **BraTS 2021 Dataset**.

We implemented and compared two architectures:

1. **3D U-Net:** captures full volumetric context ().
2. **2.5D U-Net:** optimizes computational efficiency by stacking adjacent slices.

**The Impact:** Reducing segmentation time from **hours to seconds** while maintaining high fidelity.

---

## Key Features (Our "Secret Sauce")

### 1. ROI-Based Z-Score Normalization

Standard normalization fails on MRI scans because 90% of the volume is black background (air). This skews the mean and suppresses tissue contrast.

* **Our Solution:** We implemented a custom normalization strategy that calculates statistics (mean/std) **only on non-zero voxels (the brain)**.
* **Result:** Significantly higher contrast for soft tissue and tumor boundaries.

### 2. Handling Class Imbalance

Tumors represent a tiny fraction of the brain volume. To prevent the model from bias towards the background:

* **Tumor-Centered Sampling:** 50% of training patches are forced to center on tumor regions.
* **Hybrid Loss Function:** We combine **Dice Loss** (Shape overlap) + **Categorical Cross-Entropy** (Pixel accuracy).

### 3. Multi-Modal Fusion

We stack 4 MRI modalities into a single 4D input tensor `(H, W, D, 4)`:

* **FLAIR:** Highlights Edema.
* **T1ce:** Highlights Active Tumor.
* **T1/T2:** Anatomical structure.

---

## Installation & Usage

### Prerequisites

* Python 3.8+
* Git LFS (Required for downloading the model weights)

### 1. Clone the Repo

```bash
git clone https://github.com/YOUR_USERNAME/BraTS-Tumor-Segmentation-UNet-Analysis.git
cd BraTS-Tumor-Segmentation-UNet-Analysis

```


### 3. Run the Streamlit App

To launch the interactive demo:

```bash
streamlit run app/app.py



*Built for the [Hackathon Name] 2024.*
