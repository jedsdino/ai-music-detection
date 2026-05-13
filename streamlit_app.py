import streamlit as st
import torch
import torch.nn as nn
import torchaudio
import librosa
import numpy as np

# run app command: streamlit run streamlit_app.py --server.enableCORS false --server.enableXsrfProtection false

st.title("AI Music Detection Application")
st.write("Generative AI music has been infecting music platforms. As a way to differentiate between AI music and human-created music, I trained a convolutional neural network to " \
"detect the difference between the two. However, the model is admittedly not the greatest."
)

# add a button to upload the image file from user
uploaded_file = st.file_uploader("Choose an audio file you want to analyze...", type = ["wav", "mp3"])

st.write("Based on the file you uploaded, the AI music detector has labeled this as.")

# ====== AUDIOCNN CLASS DEFINTIION ======
class AudioCNN(nn.Module):
    def __init__(self):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size = 3, padding = 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size = 3, padding = 1)
        self.bn3 = nn.BatchNorm2d(128)

        # Increased pooling size to capture more detail
        self.avg_pool = nn.AdaptiveAvgPool2d((4, 4))

        # Updated input size: 128 filters * 4 * 4 pooling = 2048
        # Increased hidden units to 512 to widen the bottleneck
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ====== PREPROCESS THE UPLOADED FILE ======
def preprocess_audio(file, duration = 5, sr = 22050): 
    waveform, original_sr = torchaudio.load(file)

    if original_sr != sr:
        resampler = torchaudio.transforms.Resample(original_sr, sr)
        waveform = resampler(waveform)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    target_samples = sr * duration
    if waveform.shape[1] < target_samples:
        padding = target_samples - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    else:
        waveform = waveform[:, target_samples]

    spectrogram_transform = torchaudio.transforms.Spectrogram(n_fft = 400, power = 2)
    spectrogram = spectrogram_transform(waveform)

    spectrogram = spectrogram.unsqueeze(0)

    return spectrogram

# ====== LOAD THE MODEL ======
loaded_model = AudioCNN()
loaded_model.load_state_dict(torch.load("/workspaces/ai-music-detection/CONVERGED_conv2d3_88.pth", map_location=torch.device('cpu')))

loaded_model.eval()

# ====== RUN FILE THROUGH MODEL ======
if uploaded_file is not None:
    st.audio(uploaded_file, format = "audio/wav")

    with st.spinner("Analyzing audio..."):
        input_tensor = preprocess_audio(uploaded_file)

        with torch.no_grad():
            output = loaded_model(input_tensor)
            predicton = torch.argmax(output, dim = 1).item()

        label_map = {0: "Human-Created", 1: "AI-Generated"}
        result = label_map.get(predicton, "Unknown")

        st.subheader(f"Result: {result}")