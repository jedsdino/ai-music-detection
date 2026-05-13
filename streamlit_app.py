import streamlit as st
import torch
import torch.nn as nn
import librosa
import torchaudio
import numpy as np
import os
import io
import torch.nn.functional as F

# run app command: streamlit run streamlit_app.py --server.enableCORS false --server.enableXsrfProtection false

st.title("AI Music Detection Application")
st.write("Generative AI music has been infecting music platforms. As a way to differentiate between AI music and human-created music, I trained a convolutional neural network to " \
"detect the difference between the two. However, the model is admittedly not the greatest."
)

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
    y, _ = librosa.load(uploaded_file, sr = sr, duration = duration, mono = True)

    target_samples = sr * duration
    if len(y) < target_samples:
        y = np.pad(y, (0, target_samples - len(y)))
    else:
        y = y[:target_samples]

    waveform = torch.from_numpy(y).unsqueeze(0)

    spectrogram_transform = torchaudio.transforms.Spectrogram(power = 2)
    spectrogram = spectrogram_transform(waveform)

    spectrogram = spectrogram.unsqueeze(0)

    return spectrogram

# ====== LOAD THE MODEL ======
@st.cache_resource
def load_my_model():
    current_dir = os.path.dirname(__file__)
    model_path = os.path.join(current_dir, "CONVERGED_conv2d3_88.pth")
    model = AudioCNN()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

loaded_model = load_my_model()

# ====== RUN FILE THROUGH MODEL ======
# 1. Update the uploader to accept multiple files
uploaded_files = st.file_uploader(
    "Choose audio files you want to analyze...", 
    type=["wav", "mp3"], 
    accept_multiple_files=True
)

if uploaded_files: 
    st.write(f"Analyzing {len(uploaded_files)} files...")
    
    results_data = []

    for uploaded_file in uploaded_files:
        uploaded_file.seek(0)
        
        with st.spinner(f"Analyzing {uploaded_file.name}..."):
            input_tensor = preprocess_audio(uploaded_file)
            
            with torch.no_grad():
                logits = loaded_model(input_tensor)
                probabilities = torch.nn.functional.softmax(logits, dim=1)
                confidence, prediction = torch.max(probabilities, dim=1)
            
            label_map = {0: "Human-Created", 1: "AI-Generated"}
            result = label_map.get(prediction.item(), "Unknown")
            score = f"{confidence.item() * 100:.2f}%"
            
            results_data.append({
                "File Name": uploaded_file.name,
                "Prediction": result,
                "Confidence": score
            })

    st.subheader("Batch Results")
    st.table(results_data)

    with st.expander("Listen to uploaded files"):
        for uploaded_file in uploaded_files:
            st.write(f"File: {uploaded_file.name}")
            st.audio(uploaded_file)