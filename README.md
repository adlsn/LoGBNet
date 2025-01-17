# Hierarchical LoG Bayesian Neural Network for Enhanced Aorta Segmentation

[Delin An](https://github.com/adlsn)<sup>1</sup>, Pan Du<sup>1</sup>, [Pengfei Gu](https://pgu-nd.github.io/)<sup>2</sup>, [Jian-Xun Wang](https://www.engineering.cornell.edu/faculty-directory/jian-xun-wang)<sup>3</sup>, and [Chaoli Wang](https://sites.nd.edu/chaoli-wang/)<sup>1</sup>

University of Notre Dame<sup>1</sup>, The University of Texas Rio Grande Valley<sup>2</sup>, Cornell University<sup>3</sup>

<div>
  <img src='method.png'>
</div>

## Introduction

This repository contains the implementation of **Bayesian Hierarchical Laplacian of Gaussian Neural Network (sLoGNN)** for enhanced 3D aorta segmentation. The framework integrates Bayesian principles with a hierarchical Laplacian of Gaussian (LoG) module to achieve high geometric fidelity and multiscale blood vessel recognition, particularly for small-radius vessels. 

### Key Features:
- A **Bayesian LoG module** for uncertainty quantification and robust feature extraction.
- A **UNet-inspired 3D architecture** with multiscale encoder-decoder pathways.
- **ASPP refinement** for capturing multiscale contextual information.

This method is particularly suited for medical imaging tasks requiring accurate segmentation with geometric preservation, such as aortic dissection analysis and computational fluid dynamics (CFD) simulation preparation.

---

## Method

The proposed framework integrates several innovative components:

1. **Bayesian LoG Module**:
   - Utilizes precomputed 3D Laplacian of Gaussian kernels to enhance multiscale feature extraction.
   - Computes KL divergence for uncertainty quantification during training, aiding robust segmentation.

2. **Hierarchical Encoder-Decoder Architecture**:
   - Inspired by UNet, the architecture includes multiple down-sampling and up-sampling pathways.
   - Each stage refines spatial and contextual features while preserving critical geometric details.

3. **ASPP Refinement**:
   - Adds Atrous Spatial Pyramid Pooling (ASPP) to refine combined features from the decoder and LoG module.
   - Enables capturing long-range dependencies and multiscale contextual information.

4. **Bayesian Loss**:
   - Combines the Dice loss for segmentation accuracy with KL divergence from the Bayesian LoG module, ensuring both performance and reliability.

---

## Experiments

We evaluate our method on a 3D aorta segmentation dataset, with comparisons against baseline approaches.

### Dataset
- Aorta segmentation dataset with both **large branches** and **small-radius supra-aortic branches**.
- Data is preprocessed using resampling and intensity normalization.

### Training
- **Input Size**: 3D volumes of size 64×64×64.
- **Loss Function**: Bayesian Dice Loss (combination of Dice and KL divergence).
- **Optimizer**: Adam optimizer with a learning rate of 1e-4 and cosine annealing scheduler.

### Evaluation Metrics
- **Dice Similarity Coefficient (DSC)** for segmentation quality.
- **Uncertainty Analysis** using KL divergence values from the Bayesian LoG module.

### Results
1. **Quantitative Evaluation**:
   - Achieved superior Dice scores compared to state-of-the-art methods.
   - Demonstrated robustness in detecting small vessels.
2. **Qualitative Analysis**:
   - Produced smoother boundaries and reduced false positives in complex regions.
3. **Uncertainty Visualization**:
   - Visualized uncertainty maps highlighting areas of low confidence, useful for clinical decision-making.

---

<div align="center">
  <img src='qualitative_results.png'>
</div>

## Installation, Train, and Test
The code is developed by Python. After cloning the repository, follow the steps below for installation:
1. Create and activate the conda environment
```python
conda create --name logb python=3.11
conda activate logb
```
2. Install dependencies
```python
pip install -r requirements.txt
```
3. Train
```python
python train.py
```

4. Test
```python
python test.py
```

## Dependencies
* Python (3.11), other versions should also work
* PyTorch (2.3.0), other versions should also work

## Contact
Should you have any questions, please send emails to dan3@nd.edu.




