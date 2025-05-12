# ESC-50 Audio Classification

This repository contains a PyTorch implementation for classifying audio events using the ESC-50 dataset.

## Features
- Uses torchaudio's ESC-50 dataset loader
- CNN-based audio classifier
- Mel spectrogram transformation
- Easy training pipeline

## Installation
```
pip install -r requirements.txt
```

## Training
Run the training script (downloads ESC-50 dataset automatically):
```
python train_esc50.py
```

## Output
The trained model will be saved as `audio_classifier_esc50.pth`.
