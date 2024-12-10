import os
import numpy as np
import librosa
import noisereduce as nr
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder

def load_label_encoder(encoder_path):
    return LabelEncoder().fit(np.load(encoder_path,allow_pickle=True))

def load_model(model_path):
    return keras.models.load_model(model_path)

def reduce_noise(audio_path):
    y, sr = librosa.load(audio_path)
    y_denoised = nr.reduce_noise(y, sr)
    return y_denoised

def extract_features(segment, sample_rate):
    mfcc_features = librosa.feature.mfcc(y=segment, sr=sample_rate, n_mfcc=13)
    return mfcc_features

def preprocess_input_features(features, max_length):
    if features.shape[1] < max_length:
        features = np.pad(features, ((0, 0), (0, max_length - features.shape[1])), 'constant')
    elif features.shape[1] > max_length:
        features = features[:, :max_length]
    return features.reshape(-1, features.shape[0], features.shape[1])

def predict_rnn(model, label_encoder, input_features):
    max_length = 100
    input_features = preprocess_input_features(input_features, max_length)
    input_features = input_features.reshape(1, input_features.shape[1], input_features.shape[2])
    prediction = model.predict(input_features)
    
    # Determine the correct axis for argmax based on array dimensions
    axis = 1 if prediction.ndim == 3 else -1
    
    predicted_label_index = np.argmax(prediction, axis=axis)
    predicted_label = label_encoder.classes_[predicted_label_index][0]
    return predicted_label


if __name__ == "__main__":
    # Paths
    model_path = 'Bird_Classification_Model.keras'
    encoder_path = 'label_encoder_classes.npy'

    # Load label encoder and model
    label_encoder = load_label_encoder(encoder_path)
    model = load_model(model_path)

    # Example: Replace 'your_audio_file.ogg' with the path to your audio file
    # audio_path = 'Training Dataset/afgfly1/XC134487.ogg'
    audio_path = input('Enter the path to audio data: ')

    # Reduce noise
    denoised_signal = reduce_noise(audio_path)

    # Extract the specified time segment
    sample_rate = librosa.get_samplerate(audio_path)
    segment_start_time = 2
    segment_end_time = 12
    segment_start_sample = int(segment_start_time * sample_rate)
    segment_end_sample = int(segment_end_time * sample_rate)
    segment = denoised_signal[segment_start_sample:segment_end_sample]

    # Extract MFCC features for the segment
    mfcc_features = extract_features(segment, sample_rate)

    # Make prediction
    prediction = predict_rnn(model, label_encoder, mfcc_features)

    print(f"Predicted Bird Species: {prediction}")
