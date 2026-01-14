# DenseAttNet: Automatic Multi-Class Classification of M3-MC Relationships

This repository contains the official implementation of the paper:  
**"Automatic Multi-Class Classification of 3D Relationships between the Mandibular Third Molar and Canal on CBCT Using a Geometry-Aware Network"**.

## Overview
This study proposes **DenseAttNet**, a geometry-aware 3D deep learning framework designed to classify the spatial relationship between the Mandibular Third Molar (M3) and the Mandibular Canal (MC) into five categories.

### Key Features:
- **Geometry-Awareness**: Integration of Signed Distance Maps (SDM) to enhance spatial relationship learning.
- **Attention Mechanism**: 3D self-attention blocks to focus on critical anatomical interfaces.
- **Clinical Transparency**: Developed and reported following **CLAIM** and **TRIPOD-AI** guidelines.

---

## Model Architecture
Our model utilizes a modified 3D DenseNet backbone with integrated attention gates. For more details on the architecture (growth rate, layers, etc.), please refer to Table S1 in the Supplementary Materials of the paper.
