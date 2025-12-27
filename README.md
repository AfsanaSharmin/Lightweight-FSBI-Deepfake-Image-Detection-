# Lightweight-FSBI-Deepfake-Image-Detection-
Mobile-FSBI: Lightweight Frequency-Enhanced Deepfake Detection for Edge Devices
##Overview

Mobile-FSBI is a lightweight deepfake image detection framework designed for resource-constrained environments such as mobile phones, embedded systems, and edge AI devices.
The project combines Self-Blended Images (SBI) with frequency-domain analysis using Discrete Wavelet Transform (DWT) and a MobileNetV3-Small backbone to achieve strong detection performance with minimal computational cost.

Unlike heavy deepfake detectors that rely on large CNNs (e.g., EfficientNet, Xception), Mobile-FSBI focuses on efficiency, portability, and practical deployment, while preserving competitive accuracy.

## Key Features

Self-Blended Image (SBI) Generation
Generates realistic pseudo-fake images without relying on external deepfake generators, improving generalization.

Frequency-Domain Enhancement (DWT)
Uses one-level Discrete Wavelet Transform to extract stable low-frequency artifacts indicative of manipulation.

Lightweight CNN Backbone
Employs MobileNetV3-Small (~2.5M parameters), suitable for mobile and edge deployment.

End-to-End PyTorch Pipeline
Includes dataset preparation, SBI generation, DWT processing, model training, and evaluation.

Cross-Dataset Evaluation
Evaluated on unseen datasets to analyze real-world generalization performance.

## Methodology Pipeline

Input Image (Real Face)

Self-Blended Image (SBI) Generation

Face landmarks → convex hull mask

Controlled blending of augmented variants

Frequency Feature Extraction

Apply DWT on RGB channels

Retain LL (low-frequency) sub-bands

Fuse frequency information with RGB

Classification

MobileNetV3-Small classifier

Binary prediction: Real vs Fake

## Experimental Results
In-Domain Performance (DF40-derived dataset)

Accuracy: ~88%

F1-score: ~0.88

ROC-AUC: ~0.954

Cross-Dataset Performance (Celeb-DF subset)

ROC-AUC: ~0.67
(Highlights the well-known domain shift challenge in deepfake detection)

Despite its small size, Mobile-FSBI achieves strong in-domain performance and serves as a solid baseline for lightweight forensic models.

## Implementation Details

Framework: PyTorch

Backbone: MobileNetV3-Small (ImageNet pretrained)

Optimizer: AdamW

Loss: Weighted Cross-Entropy

Input Size: 224 × 224

Training Platform: Google Colab (NVIDIA T4 GPU)

Precision: Mixed Precision (AMP)

## Applications

Mobile deepfake detection

Browser-based content verification

Edge AI and IoT security

Lightweight forensic analysis pipelines

## Limitations

Limited training data compared to large-scale benchmarks

Single-level DWT with fixed fusion strategy

No full mobile deployment (TFLite / quantization) yet

##  Future Work

Multi-dataset and multi-domain training

Learnable frequency fusion mechanisms

Domain generalization and meta-learning

Video-level deepfake detection

Mobile deployment with quantization (TFLite / ONNX)

## Author
Afsana SHarmin|
PhD Researcher|AL, ML learning, Deep Learning
