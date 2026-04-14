import os
import librosa
import numpy as np
import pandas as pd

DATASET_PATH = r"C:\Users\Aditya\OneDrive\Desktop\BE Final Year Project\Heart-Murmur-Detection-v2\data\the-circor-digiscope-phonocardiogram-1.0.3"
CSV_PATH = os.path.join(DATASET_PATH, "training_data.csv")
AUDIO_PATH = os.path.join(DATASET_PATH, "training_data")

df = pd.read_csv(CSV_PATH)

print("CSV Columns:", df.columns)

X = []
y = []

for _, row in df.iterrows():

    patient_id = row["Patient ID"]
    locations = row["Recording locations:"]
    murmur = row["Murmur"]

    if pd.isna(locations):
        continue

    locations = locations.split("+")

    for loc in locations:

        file_path = os.path.join(AUDIO_PATH, f"{patient_id}_{loc}.wav")

        if not os.path.exists(file_path):
            continue

        # Load audio
        audio, sr = librosa.load(file_path, sr=22050)

        # Mel Spectrogram
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=64
        )

        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Ensure fixed width
        if mel_db.shape[1] < 128:
            pad = 128 - mel_db.shape[1]
            mel_db = np.pad(mel_db, ((0,0),(0,pad)), mode='constant')

        mel_db = mel_db[:, :128]

        # Normalize spectrogram
        mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())

        X.append(mel_db)

        if murmur == "Present":
            y.append(1)
        else:
            y.append(0)

X = np.array(X)
y = np.array(y)

print("Dataset shape:", X.shape)

np.save("X_spectrogram.npy", X)
np.save("y_labels.npy", y)

print("Spectrogram dataset saved.")