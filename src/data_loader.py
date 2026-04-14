import os
import pandas as pd
import librosa


class DataLoader:

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.csv_path = os.path.join(dataset_path, "training_data.csv")
        self.audio_path = os.path.join(dataset_path, "training_data")

        self.annotations = pd.read_csv(self.csv_path)[["Patient ID", "Murmur"]]

    def load_audio(self, filepath):
        try:
            audio, sr = librosa.load(filepath, sr=22050, mono=True)
            return audio
        except Exception:
            return None

    def load_dataset(self, limit=None):

        rows = []

        for index, row in self.annotations.iterrows():

            patient_id = row["Patient ID"]
            label = 1 if row["Murmur"] == "Present" else 0

            sample = {
                "Patient_ID": patient_id,
                "label": label
            }

            for region in ["AV", "MV", "PV", "TV"]:
                path = os.path.join(self.audio_path, f"{patient_id}_{region}.wav")
                sample[region] = self.load_audio(path)

            rows.append(sample)

            if limit and index >= limit:
                break

        df = pd.DataFrame(rows)
        return df
