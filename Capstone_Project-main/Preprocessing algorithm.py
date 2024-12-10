import librosa
import librosa.display
import matplotlib.pyplot as plt
import noisereduce as nr
import os
import numpy as np

def reduce_noise(audio_path):
    y, sr = librosa.load(audio_path)
    y_denoised = nr.reduce_noise(y,sr)

    return y_denoised


def extract_features(segment, sample_rate):

    mfcc_features = librosa.feature.mfcc(y=segment, sr=sample_rate, n_mfcc=13)

    return mfcc_features


# Directory containing subdirectories with audio recordings
dataset_directory = "Training Dataset"

# Output directory to save extracted features
output_directory = "Extracted Features"

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Specify the start and end times for the segment (in seconds)
segment_start_time = 2
segment_end_time = 12

# Iterate through each subdirectory in the dataset directory
for root, dirs, files in os.walk(dataset_directory):
    for filename in files:
        # Check if the file is an audio file (you can customize the condition based on your file extensions)
        if filename.lower().endswith(('.ogg')):
            # Construct the full path to the audio file
            audio_path = os.path.join(root, filename)

            # Reduce noise
            denoised_signal = reduce_noise(audio_path)

            # Extract the specified time segment
            sample_rate = librosa.get_samplerate(audio_path)
            segment_start_sample = int(segment_start_time * sample_rate)
            segment_end_sample = int(segment_end_time * sample_rate)
            segment = denoised_signal[segment_start_sample:segment_end_sample]

            # Extract MFCC features for the segment
            mfcc_features = extract_features(segment, sample_rate)

            # Determine the relative path within the 'Training Dataset' directory
            relative_path = os.path.relpath(root, dataset_directory)

            # Construct the subdirectory path within the 'Extracted Features' directory
            output_subdirectory = os.path.join(output_directory, relative_path)

            # Create the subdirectory if it doesn't exist
            os.makedirs(output_subdirectory, exist_ok=True)

            # Save the extracted features
            output_filename = f"{os.path.splitext(filename)[0]}_features.npy"
            output_path = os.path.join(output_subdirectory, output_filename)
            np.save(output_path, mfcc_features)

            print(f"Features extracted and saved for {filename} in {relative_path}.")

print("Feature extraction complete.")