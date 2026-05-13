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
if uploaded_file is not None:
    # We display the audio player
    file_type = "audio/wav" if uploaded_file.name.endswith("wav") else "audio/mpeg"
    st.audio(uploaded_file, format=file_type)
    
    if st.button("Analyze Audio"):
        with st.spinner("Analyzing audio..."):
            uploaded_file.seek(0) 
            input_tensor = preprocess_audio(uploaded_file)

            with torch.no_grad():
                logits = loaded_model(input_tensor)
                
                probabilities = F.softmax(logits, dim=1)
                
                confidence, prediction = torch.max(probabilities, dim=1)
                
                conf_score = confidence.item() * 100  
                pred_idx = prediction.item()

            label_map = {0: "Human-Created", 1: "AI-Generated"}
            result = label_map.get(pred_idx, "Unknown")

            color = "green" if pred_idx == 0 else "red"
            
            st.markdown(f"### Result: :{color}[{result}]")
            st.progress(conf_score / 100)
            st.write(f"**Confidence Score:** {conf_score:.2f}%")