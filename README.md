# Endoscopic Adaptive Transformer (EAT) for Enhanced Polyp Segmentation

This repository provides training and evaluation code for **EAT**.

[![huggingface weights](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-deepang/eat-yellow)](https://huggingface.co/deepang/eat)&nbsp;

CVCliniDB weights & logs: [![huggingface weights](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-deepang/CVCliniDB-yellow)](https://huggingface.co/deepang/eat/tree/main/CVCliniDB)&nbsp;

Kvasir-SEG weights & logs: [![huggingface weights](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-deepang/Kvasir--SEG-yellow)](https://huggingface.co/deepang/eat/tree/main/Kvasir)

EDD2020 weights & logs: [![huggingface weights](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-deepang/EDD2020-yellow)](https://huggingface.co/deepang/eat/tree/main/EDD2020)

POLYPGEN weights & logs: [![huggingface weights](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-deepang/POLYPGEN-yellow)](https://huggingface.co/deepang/eat/tree/main/POLYPGEN)

SUN-SEG weights & logs: [![huggingface weights](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-deepang/SUN--SEG-yellow)](https://huggingface.co/deepang/eat/tree/main/SUN-SEG)


## 🧩 Requirements
- A Python environment with dependencies from `requirements.txt`
- A CUDA-capable GPU and a CUDA-enabled PyTorch build
- CUDA toolkit available to compile `DCNv4_op/` (required)
- On Windows: run `train.sh/verify.sh` via WSL or translate the commands to `torchrun` in your shell


---

## 🚀 Quick Start

### 1) Install dependencies
```bash
pip install -r requirements.txt
```

### 2) Build DCNv4 (Required)
EAT depends on **DCNv4**. The source code is provided in `DCNv4_op/` and must be compiled/installed with CUDA.

Build & install:
```bash
cd DCNv4_op
bash make.sh
```

Editable install:
```bash
cd DCNv4_op
pip install -v -e .
```

Note: `DCNv4_op/setup.py` requires CUDA and will raise an error if CUDA is not available.

### 3) (Optional) Pretrained backbone weights
`config.yml` points to an optional backbone weight file:
```text
./pretrained/pvt_v2_b2.pth
```
If the file is missing, the backbone will be initialized without loading pretrained weights.

### 4) Configure dataset paths
Set `trainer.dataset_choose` and the corresponding `dataset.<name>.data_root` in `config.yml`.

Supported datasets and their folder conventions:

- **CVC-ClinicDB** (`trainer.dataset_choose: CVC_ClinicDB`)
	- Loader: `src/CVCLoader.py`
	- `data_root` should contain:
		- `Original/` (images)
		- `GroundTruth/` (masks)
	- Note: please rename `Ground Truth` -> `GroundTruth` (remove spaces) if needed.

- **Kvasir-SEG** (`trainer.dataset_choose: Kvasir_SEG`)
	- Loader: `src/CVCLoader.py`
	- `data_root` should contain:
		- `images/`
		- `masks/`

- **EDD2020 (Seg)** (`trainer.dataset_choose: EDD_seg`)
	- Loader: `src/EDDLoader.py`
	- `data_root` should contain:
		- `originalImages/` (images)
		- `masks/` (multi-class masks)
	- Mask naming: for an image `XXX.jpg/png`, masks are looked up as `masks/XXX_<class>.tif`, where `<class>` is one of `BE`, `cancer`, `HGD`, `polyp`, `suspicious`.

- **PolypGen** (`trainer.dataset_choose: PolypGen`)
	- Loader: `src/PolypGenLoder.py`
	- `data_root` is expected to contain multiple subfolders (e.g., different centers). For each subfolder:
		- `images/`
		- `masks/`
	- Mask naming: for `images/NAME.<ext>`, the loader expects `masks/NAME_mask.jpg`.

- **SUN-SEG** (`trainer.dataset_choose: Sun_seg`)
	- Loader: `src/SunsegLoader.py`
	- `data_root` should contain:
		- `TrainDataset/Frame/<video_or_folder>/*` and `TrainDataset/GT/<video_or_folder>/*.png`
		- `TestHardDataset/Unseen/Frame/<video_or_folder>/*` and `TestHardDataset/Unseen/GT/<video_or_folder>/*.png`

### 5) Train / Evaluate
Single GPU:
```bash
python train.py
python verify.py
```

Multi-GPU (torchrun):
```bash
torchrun --nproc_per_node 2 --master_port 29400 train.py
torchrun --nproc_per_node 4 --master_port 29400 verify.py
```

Example bash scripts (set env vars like `CUDA_VISIBLE_DEVICES`):
```bash
bash train.sh
bash verify.sh
```


---

## 🔍 Main Contributions  
- **Endoscopic Adaptive Transformer (EAT)**: A hierarchical framework leveraging adaptive perceptive-field mechanisms to handle polyp morphology variability.  
- **Adaptive Perception Module (APM)**: Uses parallel Adaptive Perception Units (APUs) to capture edge features and structural context via deformable convolution.  
- **Comprehensive Feature Extractor (CFE)**: Integrates global context and local details through self-attention mechanisms.  
- **State-of-the-Art Performance**: Validated on 5 datasets with superior accuracy and robustness.  

---

## 🧠 Network Architecture  
### Overview
EAT employs a Pyramid Vision Transformer (PVT) backbone for multi-scale feature extraction. The architecture processes shallow features with deformable convolution and deeper features via APM, followed by CFE-based fusion:  

![Overview](./figures/Fig1.png)  

**Key Innovations**:  
- **APM** adjusts observational scope dynamically to match polyp morphology.  
- **Deformable convolution** enhances boundary precision in shallow layers.  
- **Dual-stream fusion** balances high-resolution details and semantic context.  

### APM(Adaptive Perception Module)
![Fig3.png](figures/Fig3.png)

### APU(Adaptive Perception Unit)
![Fig4.png](figures/Fig4.png)
---

## 📊 Performance  
### Single-Target Segmentation
####  CVC-ClinicDB & Kvasir-SEG

![ClinicDB & Kvasir-SEG](./figures/table1.png)

####  PolypGen & SUN-SEG-Hard(Unseen)
![PolypGen & SUN-SEG-Hard](figures/table3.png)


### Multi-Target Segmentation
####  EDD 2020
![table2.png](figures/table2.png)

---

### 📂 Data Description

We evaluate our model on five benchmark datasets with rigorous data partitioning protocols to ensure reproducibility and fair comparison:

### Dataset Name: CVC-ClinicDB

Size: 612 colonoscopy images (384×288)

Challenge: Polyp segmentation in colonoscopy images

- **Split**: 80% training / 10% validation / 10% testing
- High-quality annotated polyp images for medical diagnosis

### Dataset Name: Kvasir-SEG

Size: 1,000 annotated endoscopic images

Challenge: Gastrointestinal polyp segmentation

- **Split**: 80% training / 10% validation / 10% testing
- Comprehensive polyp segmentation benchmark dataset

### Dataset Name: EDD 2020

Size: 386 GI tract frames with 5 lesion types (160 NDBE, 88 Susp., 74 HGD, 53 Cancer, 127 Polyp masks)

Challenge: Endoscopy Disease Detection and Segmentation

- **Split**: 80% training / 10% validation / 10% testing
- Multi-target segmentation with various gastrointestinal lesions

### Dataset Name: PolypGen

Size: Multi-center dataset (3,762 polyp-positive frames)

Challenge: Cross-center polyp segmentation generalization

- **Split**: Cross-center evaluation design
- **Training**: Centers data1-data5 (comprehensive training set)
- **Testing**: Center data6 (completely unseen test set)

### Dataset Name: SUN-SEG

Size: 158,690 colonoscopy frames (49,136 polyp-positive + 109,554 negative from 285 polyp-positive and 728 normal video clips)

Challenge: Large-scale colonoscopy polyp segmentation

- **Split**: Official dataset partition
- **Test Set**: SUN-SEG-Hard (Unseen) subset
- **Challenge**: Includes artifact-rich frames for robust evaluation

---

### 🎨 Visualization  
#### Single-Target Segmentation  
![Fig9.png](figures/Fig9.png)

#### Multi-Target Segmentation  
![Fig11.png](figures/Fig11.png)
---
