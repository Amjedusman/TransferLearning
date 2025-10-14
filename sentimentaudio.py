# ======================================
# 🎯 Audio Emotion Detection using Wav2Vec2 + Streamlit
# ======================================

import streamlit as st
import torch
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
from io import BytesIO
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="🎧 Audio Emotion Detection", layout="centered")

st.title("🎧 Audio Emotion Detection using Wav2Vec2")
st.write("Upload a voice clip to detect the **emotion** expressed in the speech.")

# ===========================
# 1️⃣ Upload Audio File
# ===========================
uploaded_file = st.file_uploader("📁 Upload an audio file (.wav, .mp3, .flac)", type=["wav", "mp3", "flac"])

if uploaded_file:
    # Load audio
    speech_array, sampling_rate = librosa.load(uploaded_file, sr=16000)
    duration = len(speech_array) / sampling_rate

    st.audio(uploaded_file, format="audio/wav")
    st.write(f"✅ Loaded audio: {uploaded_file.name}")
    st.write(f"🕒 Duration: {duration:.2f} seconds | Sampling rate: {sampling_rate} Hz")

    # ===========================
    # 2️⃣ Audio Preprocessing
    # ===========================
    st.subheader("🎵 Audio Waveform")
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(np.linspace(0, len(speech_array)/sampling_rate, len(speech_array)), speech_array)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Audio Waveform")
    st.pyplot(fig)

   # Spectrogram
    st.subheader("🎶 Spectrogram")
    fig, ax = plt.subplots(figsize=(10, 4))

    # Compute STFT (Short-Time Fourier Transform)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(speech_array)), ref=np.max)

    # Plot using librosa.display
    import librosa.display  # make sure it's imported
    img = librosa.display.specshow(D, sr=sampling_rate, x_axis='time', y_axis='hz', cmap='magma', ax=ax)

    # Add colorbar safely
    fig.colorbar(img, ax=ax, format='%+2.0f dB')

    ax.set_title("Spectrogram (dB)")
    st.pyplot(fig)

    # ===========================
    # 3️⃣ Feature Extraction & Model
    # ===========================
    st.subheader("🧠 Emotion Classification")

    model_name = "superb/wav2vec2-base-superb-er"
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)

    inputs = extractor(speech_array, sampling_rate=16000, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.nn.functional.softmax(logits, dim=-1)[0].detach().numpy()
    predicted_id = torch.argmax(logits, dim=-1).item()
    label = model.config.id2label[predicted_id]

    # ===========================
    # 4️⃣ Output Results
    # ===========================
    st.success(f"✅ Predicted Emotion: **{label.upper()}**")

    # ===========================
    # 5️⃣ Visualization
    # ===========================
    st.subheader("📊 Emotion Probability Distribution")

    emotions = list(model.config.id2label.values())
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(emotions, probs, color='skyblue')
    ax.set_xlabel("Emotion")
    ax.set_ylabel("Probability")
    ax.set_title("Emotion Classification Probabilities")
    plt.xticks(rotation=45)
    st.pyplot(fig)
else:
    st.info("⬆️ Upload a .wav or .mp3 file to start the analysis.")
