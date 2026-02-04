
# LCM-Style Audio Segmentation Pipeline

This repository contains a robust pipeline for segmenting long audio files into semantically meaningful clips using ASR and machine learning-based text segmentation. 

This approach replicates the data processing logic used in the "Large Concept Model" (LCM) architecture. It ensures that audio segments are neither too short (fragmented) nor too long (diluted), making them suitable for embedding generation (e.g., SONAR, WavLM).

## Features

- **ASR-Guided Segmentation:** Uses `faster-whisper` for high-accuracy word-level timestamps.
- **Semantic Splitting:** Uses `wtpsplit` (SaT) to split text based on semantic probabilities rather than simple punctuation.
- **Dynamic Length Constraints:** - Merges segments shorter than 10 characters.
  - Forces splits on segments longer than 256 characters using a "safety valve" logic.
- **Robustness:** Handles device placement (CPU/GPU) automatically and patches temporary directory issues common in WSL/Colab environments.

## Installation

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
