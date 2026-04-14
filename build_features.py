import pandas as pd
from src.data_loader import DataLoader
from src.feature_extraction import extract_features

DATASET_PATH = r"C:\Users\Aditya\OneDrive\Desktop\BE Final Year Project\Heart-Murmur-Detection-v2\data\the-circor-digiscope-phonocardiogram-1.0.3"

print("Loading dataset...")
loader = DataLoader(DATASET_PATH)

df = loader.load_dataset()

print("Extracting features for all patients...")

feature_rows = []

for _, row in df.iterrows():

    patient_features = {}

    for region in ["AV", "MV", "PV", "TV"]:

        audio = row[region]

        # Skip if audio is missing
        if audio is None:
            continue

        try:
            features = extract_features(audio)

            for key, value in features.items():
                patient_features[f"{region}_{key}"] = value

        except Exception as e:
            print(f"Error processing {region} for patient {row['Patient_ID']}: {e}")
            continue

    # Skip patient if no features extracted
    if len(patient_features) == 0:
        continue

    patient_features["label"] = row["label"]
    feature_rows.append(patient_features)

feature_df = pd.DataFrame(feature_rows)

print("Feature dataset shape:", feature_df.shape)

# Fill missing values with 0
feature_df = feature_df.fillna(0)

# Save to CSV
feature_df.to_csv("features.csv", index=False)

print("Features saved to features.csv")
