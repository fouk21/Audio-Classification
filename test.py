import torch
import torchaudio
import torchaudio.transforms as T
from audio_classifier import AudioClassifier

def load_model(path='audio_classifier_esc50.pth'):
    model = AudioClassifier(num_classes=50)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model

def preprocess_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    transform = T.MelSpectrogram(sample_rate=44100, n_mels=128)
    spec = transform(waveform).squeeze(0).unsqueeze(0)
    spec = torch.nn.functional.interpolate(spec, size=(128, 128))
    return spec.unsqueeze(0)  # Add batch dimension

def predict(model, file_path):
    input_tensor = preprocess_audio(file_path)
    with torch.no_grad():
        output = model(input_tensor)
    predicted_class = torch.argmax(output, dim=1).item()
    return predicted_class

if __name__ == "__main__":
    audio_path = "your_audio_file.wav"  # Replace with the path to your test audio file
    model = load_model()
    prediction = predict(model, audio_path)
    print(f"Predicted class index: {prediction}")
