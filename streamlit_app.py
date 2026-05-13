import streamlit as st
import torch
import torch.nn as nn
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

# ====== AUDIO CNN CLASS ======

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

# ====== LOAD THE MODEL ======
loaded_model = AudioCNN()
loaded_model.load_state_dict(torch.load("/workspaces/ai-music-detection/CONVERGED_BASELINE_NONOISE_SPECT_10SEC_CONV1D_94.4.pth", map_location=torch.device('cpu')))

loaded_model.eval()

# 