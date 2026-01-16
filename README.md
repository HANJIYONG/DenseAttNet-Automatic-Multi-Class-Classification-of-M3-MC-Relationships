# DenseAttNet: Automatic Multi-Class Classification of M3-MC Relationships

This repository contains the official implementation of the paper:  
**"Automatic Multi-Class Classification of 3D Relationships between the Mandibular Third Molar and Canal on CBCT Using a Geometry-Aware Network"**.

## Directory Structure
```
├── data/
│   ├── samples/                # 25 subset samples from Tooth Fairy dataset
│   └── meta_df.csv             # Metadata containing paths and labels
├── Segmentation/
│   ├── dataset.py              # Data loader for CBCT segmentation
│   └── train.py                # Training script for segmentation
├── Classification/
│   ├── dataset.py              # Data loader for DenseAttNet (SDM included)
│   ├── model.py                # DenseAttNet architecture (Ours)
│   └── train.py                # Training script for classification
├── utils/
│   └── roi_crop.py             # Automated ROI extraction based on centroids
├── inference_sample.py            # End-to-end inference script (Seg -> ROI -> Class)
└── README.md
```

## Pre-trained Weights
Due to institutional data privacy policies and intellectual property protection, pre-trained weights are hosted on Google Drive.

-  [Request Access to Weights](https://drive.google.com/drive/u/1/folders/1eQG2ZPlmWfjDmHe5XEFjEC4UMFKP9j4_)

- Please request access using your institutional email address. Access will be granted for academic and research purposes upon review.

## Inference Example
- To run the full pipeline (Segmentation → ROI Cropping → Classification) on a single sample:
```
- python inference_sample.py --input_path ./data/samples/case_01
```
---

## Dataset

- We provide 25 sample cases derived from the public Tooth Fairy dataset to demonstrate the code's functionality.
---

## License & Usage
- Academic Use Only: This code and the associated weights are provided for research and educational purposes.

- Non-Commercial: Commercial use, redistribution, or modification for commercial purposes is strictly prohibited without prior written consent from the authors.

- Citation: If you find this work useful, please cite our paper.

