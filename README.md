# 🕵️‍♂️ Fusion Gate Deepfake Detector

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Live%20Demo-Hugging%20Face-blue)](https://huggingface.co/spaces/minus2/Fusion-Gate-Deepfake-Detector)
[![GitHub Actions CI/CD](https://img.shields.io/github/actions/workflow/status/ArkaRouth/FusionGate_DeepFake_Detection/sync_to_hf.yml?label=HF%20Sync)](https://github.com/ArkaRouth/FusionGate_DeepFake_Detection/actions)

A multimodal Machine Learning pipeline that detects deepfake videos by fusing Spatial (pixel-level) and Temporal (lip-sync) anomaly detection algorithms.

## 🚀 Live Demonstration
You do not need to install anything to test this AI. 
**[Click here to try the live web application on Hugging Face](https://huggingface.co/spaces/minus2/Fusion-Gate-Deepfake-Detector)**.

## 🧠 System Architecture
This system does not rely on a single point of failure. It utilizes a custom "Fusion Gate" architecture that dynamically weighs two separate neural networks based on the input video's quality.

1. **Spatial Stream (ResNet-50):** Analyzes individual frames for visual artifacts, unnatural edges, and pixel blending anomalies.
2. **Temporal Stream (SyncNet):** Analyzes the audio-visual offset, calculating the Euclidean distance between lip landmarks and audio frequencies to detect AI voice dubbing.
3. **The Dynamic Fusion Gate:** An algorithmic decision boundary. If a video is highly compressed or blurry (low Alpha score), the system dynamically shifts trust to the Temporal stream and raises the classification threshold to prevent false positives.

## ⚙️ Key Features
* **Lossless Pre-processing:** Utilizes FFmpeg (`-crf 18`) to standardize video formats without degrading crucial high-definition pixel data before analysis.
* **Dynamic Step-Cutoff Logic:** Mathematically adjusts strictness (5.8 for low-quality vs. 6.5 for HD) based on Laplacian variance edge-detection.
* **Automated CI/CD Pipeline:** Deployed via GitHub Actions, automatically syncing repository pushes directly to the Hugging Face cloud server.
