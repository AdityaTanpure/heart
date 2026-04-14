from src.data_loader import DataLoader
from src.feature_extraction import extract_features

DATASET_PATH = r"C:\Users\Aditya\OneDrive\Desktop\BE Final Year Project\Heart-Murmur-Detection-v2\data\the-circor-digiscope-phonocardiogram-1.0.3"

loader = DataLoader(DATASET_PATH)

print("Loading 1 patient...")
df = loader.load_dataset(limit=1)

sample_audio = df.iloc[0]["AV"]

print("Extracting features from AV region...")
features = extract_features(sample_audio)

print("Number of features extracted:", len(features))
print("Features:")
for k, v in features.items():
    print(k, ":", v)
