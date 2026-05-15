import streamlit as st
import torch
import torch.nn as nn
import torchaudio
import librosa
import numpy as np
import random
import os
import io
import matplotlib.pyplot as plt
import torch.nn.functional as F

# ====== WEBSITE TITLE/HEADING ======

st.title("AI Music Detection Application")
st.write("Generative AI music has been infecting music platforms. As a way to differentiate between AI music and human-created music, I trained a convolutional neural network to " \
"detect the difference between the two. However, the model is admittedly not the greatest."
)

# ====== AUDIOCNN CLASS DEFINTIION ======
class AudioCNN(nn.Module):
    def __init__(self):
        super(AudioCNN, self).__init__()

        # BLOCK 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size = 3, padding = 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2)

        # BLOCK 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(64)

        # BLOCK 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size = 3, padding = 1)
        self.bn3 = nn.BatchNorm2d(128)

        self.avg_pool = nn.AdaptiveAvgPool2d((4, 4))
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
@st.cache_data
def preprocess_audio(file_content, duration=5, sr=22050):
    temp_file = io.BytesIO(file_content)
    total_duration = librosa.get_duration(path=temp_file)
    
    # 2. random start
    if total_duration > duration:
        max_start = total_duration - duration
        start_time = random.uniform(0, max_start)
    else:
        start_time = 0  # start at 0 if 5 seconds

    # load five seconds
    temp_file.seek(0)
    y, _ = librosa.load(temp_file, sr=sr, offset=start_time, duration=duration, mono=True)

    # pad
    target_samples = sr * duration
    if len(y) < target_samples:
        y = np.pad(y, (0, target_samples - len(y)))
    else:
        y = y[:target_samples]

    waveform = torch.from_numpy(y).unsqueeze(0)
    spectrogram_transform = torchaudio.transforms.Spectrogram(power=2)
    spectrogram = spectrogram_transform(waveform)
    spectrogram = spectrogram.unsqueeze(0)

    # return spect and start time
    return spectrogram, start_time

# ====== LOAD THE MODEL ======
@st.cache_resource
def load_my_model():
    current_dir = os.path.dirname(__file__)
    model_path = os.path.join(current_dir, "FINAL_AI_MUSIC_DETECTOR.pth")
    model = AudioCNN()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

loaded_model = load_my_model()

# ====== RUN FILE THROUGH MODEL ======
# update the uploader to accept multiple files
uploaded_files = st.file_uploader(
    "Choose audio files you want to analyze...", 
    type=["wav", "mp3"], 
    accept_multiple_files=True
)

if uploaded_files: 
    # list of files
    file_names = [f.name for f in uploaded_files]
    
    # select
    selected_filename = st.selectbox("Select a file to analyze and play:", file_names)

    # find file + selection
    selected_file = next(f for f in uploaded_files if f.name == selected_filename)

    st.write(f"Analyzing: **{selected_file.name}**")
    
    # reset
    selected_file.seek(0)

    with st.spinner(f"Processing {selected_file.name}..."):
        input_tensor, start_timestamp = preprocess_audio(selected_file.getvalue())        
        
        with torch.no_grad():
            logits = loaded_model(input_tensor)
            probabilities = F.softmax(logits, dim=1)
            confidence, prediction = torch.max(probabilities, dim=1)
        
        label_map = {0: "Human-Created", 1: "AI-Generated"}
        result = label_map.get(prediction.item(), "Unknown")

    st.info(f"Analyzed a 5-second segment starting at **{start_timestamp:.2f}s**")

    # display results
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Prediction", result)
    with col2:
        st.metric("Confidence", f"{confidence.item() * 100:.2f}%")

    # 5. audio player
    st.audio(selected_file)

    st.subheader(f"Spectrogram for {selected_filename}")

    spec_np = input_tensor.squeeze().numpy()
        
        # transform
    log_spec = np.log10(spec_np + 1e-10)

        # plot
    fig, ax = plt.subplots(figsize=(10, 4))
    img = ax.imshow(log_spec, aspect='auto', origin='lower', cmap='viridis')
        
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title('Power Spectrogram (Log Scale)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Frequency Bin')

        # display
    st.pyplot(fig)

st.title("How Does This Work?")
st.write("Some convolutional neural networks")

# ====== DISPLAY SPECTS ======
current_dir = os.path.dirname(__file__)

# FMA
fma_audio_path = os.path.join(current_dir, "02 - Frenic - Ulan Bator.wav")
st.audio(fma_audio_path)
fma_spect_path = os.path.join(current_dir, "fma_spect.png")
st.image(fma_spect_path)

# SONICS
sonics_audio_path = os.path.join(current_dir, "00 - sonics_subset - fake 15114 udio 1.wav")
st.audio(sonics_audio_path)
sonics_spect_path = os.path.join(current_dir, "sonics_spect.png")
st.image(sonics_spect_path)