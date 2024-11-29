import streamlit as st
import numpy as np
import librosa
import tensorflow as tf

# Define the list of emotions corresponding to your model's output classes
emotion_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# Function to extract features from audio
def extract_features(data, sample_rate):
    result = np.array([])

    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))

    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))

    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc))

    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))

    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))

    return result

# Function to predict emotion from audio
def predict_emotion(audio_path, model_path):
    data, sample_rate = librosa.load(audio_path, sr=None)
    features = extract_features(data, sample_rate)
    features = np.expand_dims(features, axis=0)

    model = tf.keras.models.load_model(model_path)
    predictions = model.predict(features)
    predicted_index = np.argmax(predictions, axis=1)[0]
    predicted_emotion = emotion_labels[predicted_index]

    return predicted_emotion

# Predefined model path
MODEL_PATH = 'SER_model.h5'

# Streamlit UI
st.title("Emotion Recognition from Audio")
st.write("Upload an audio file to predict the emotion")

audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "ogg"])

if audio_file is not None:
    with st.spinner('Processing...'):
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_file.getbuffer())

        predicted_emotion = predict_emotion("temp_audio.wav", MODEL_PATH)
        st.audio(audio_file, format='audio/wav')
        st.success(f"The predicted emotion is: {predicted_emotion}")
else:
    st.write("Please upload an audio file.")
