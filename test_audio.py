import os
import librosa

DATASET_PATH = r"C:\Users\Aditya\OneDrive\Desktop\BE Final Year Project\Heart-Murmur-Detection-v2\data\the-circor-digiscope-phonocardiogram-1.0.3"

csv_path = os.path.join(DATASET_PATH, "training_data.csv")
audio_folder = os.path.join(DATASET_PATH, "training_data")

print("Checking dataset path...")
print("CSV exists:", os.path.exists(csv_path))
print("Audio folder exists:", os.path.exists(audio_folder))

# Select only .wav files
wav_files = [f for f in os.listdir(audio_folder) if f.endswith(".wav")]

print("Total wav files found:", len(wav_files))

sample_file = os.path.join(audio_folder, wav_files[0])
print("Loading sample:", sample_file)

try:
    audio, sr = librosa.load(sample_file, sr=22050)
    print("Audio loaded successfully!")
    print("Sample Rate:", sr)
    print("Audio length:", len(audio))
except Exception as e:
    print("Error while loading audio:", e)
