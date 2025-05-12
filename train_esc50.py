import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchaudio.datasets import ESC50
import torchaudio.transforms as T
from audio_classifier import AudioClassifier

transform = T.MelSpectrogram(sample_rate=44100, n_mels=128)

def collate_fn(batch):
    spectrograms = []
    labels = []
    for waveform, _, label, _, _ in batch:
        spec = transform(waveform).squeeze(0).unsqueeze(0)
        spec = torch.nn.functional.interpolate(spec, size=(128, 128))
        spectrograms.append(spec)
        labels.append(label)
    return torch.stack(spectrograms), torch.tensor(labels)

def train():
    dataset = ESC50("./data", download=True)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    model = AudioClassifier(num_classes=50)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(5):
        total_loss = 0
        for x, y in dataloader:
            optimizer.zero_grad()
            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")
    torch.save(model.state_dict(), "audio_classifier_esc50.pth")

if __name__ == "__main__":
    train()
