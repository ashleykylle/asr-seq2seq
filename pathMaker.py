import os
import pandas as pd

# Define the root folder where all speaker directories are stored
root_folder = "D:\Schoolshit\Thesis\dataset"

# Load the existing CSV
csv_file = "transcription.csv"
df = pd.read_csv(csv_file)

# Collect all audio file paths
audio_files = []
for speaker in sorted(os.listdir(root_folder)):  # Sort speakers for consistency
    speaker_folder = os.path.join(root_folder, speaker)
    if os.path.isdir(speaker_folder):  # Ensure it's a directory
        for subfolder in sorted(os.listdir(speaker_folder)):  # Iterate over subfolders
            subfolder_path = os.path.join(speaker_folder, subfolder)
            if os.path.isdir(subfolder_path):  # Ensure it's a directory
                for file in sorted(os.listdir(subfolder_path)):  # Get audio files
                    if file.endswith(".wav"):  # Only consider .wav files
                        audio_files.append(os.path.join(subfolder_path, file))

# Ensure we have the same number of audio files as rows in CSV
if len(audio_files) != len(df):
    print("Mismatch! CSV rows:", len(df), "Audio files:", len(audio_files))
else:
    df["audio"] = audio_files  # Assign paths to audio column

    # Save the updated CSV
    df.to_csv("transcription_updated.csv", index=False)
    print("Updated CSV saved as 'transcription_updated.csv'")
