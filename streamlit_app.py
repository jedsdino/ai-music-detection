import streamlit as st
import torch
import torch.nn as nn
import librosa
import librosa.display
import torchaudio
import numpy as np
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
@st.cache_data
def preprocess_audio(file_content, duration = 5, sr = 22050): 
    y, _ = librosa.load(io.BytesIO(file_content), sr = sr, duration = duration, mono = True)

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
    # 1. Create a list of file names for the dropdown
    file_names = [f.name for f in uploaded_files]
    
    # 2. Add the selection widget
    selected_filename = st.selectbox("Select a file to analyze and play:", file_names)

    # 3. Find the file object that matches the selection
    # (Next, we find the specific file object in our list that matches the selected name)
    selected_file = next(f for f in uploaded_files if f.name == selected_filename)

    st.write(f"Analyzing: **{selected_file.name}**")
    
    # Reset file pointer to ensure it reads from the start
    selected_file.seek(0)

    with st.spinner(f"Processing {selected_file.name}..."):
        # We pass the bytes to our (ideally) cached function
        input_tensor = preprocess_audio(selected_file.getvalue())
        
        with torch.no_grad():
            logits = loaded_model(input_tensor)
            probabilities = F.softmax(logits, dim=1)
            confidence, prediction = torch.max(probabilities, dim=1)
        
        label_map = {0: "Human-Created", 1: "AI-Generated"}
        result = label_map.get(prediction.item(), "Unknown")

    # 4. Display results for just the selected file
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Prediction", result)
    with col2:
        st.metric("Confidence", f"{confidence.item() * 100:.2f}%")

    # 5. Audio Player for the selection
    st.audio(selected_file)

    st.subheader(f"Spectrogram for {selected_filename}")

    # 1. Convert the tensor back to a numpy array
    # We need to remove the batch and channel dimensions for plotting
    # input_tensor shape is [1, 1, freq, time] -> we want [freq, time]
    spec_np = input_tensor.squeeze().numpy()

    # 2. Convert power to decibels (log scale) for better visualization
    spec_db = librosa.power_to_db(spec_np, ref=np.max)

    # 3. Create the plot
    fig, ax = plt.subplots()
    img = librosa.display.specshow(spec_db, sr=22050, x_axis='time', y_axis='hz', ax=ax)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set(title='Mel-frequency spectrogram')

    # 4. Display in Streamlit
    st.pyplot(fig)

st.title("How Does This Work?")
st.write("Some convolutional neural networks")

# ====== DISPLAY SPECTS ======
current_dir = os.path.dirname(__file__)

# FMA
fma_spect_path = os.path.join(current_dir, "fma_spect.png")
st.image(fma_spect_path)

# SONICS
sonics_spect_path = os.path.join(current_dir, "sonics_spect.png")
st.image(sonics_spect_path)