from src.data_loader import DataLoader

DATASET_PATH = r"C:\Users\Aditya\OneDrive\Desktop\BE Final Year Project\Heart-Murmur-Detection-v2\data\the-circor-digiscope-phonocardiogram-1.0.3"

loader = DataLoader(DATASET_PATH)

print("Loading first 3 patients only...")
df = loader.load_dataset(limit=3)

print(df.head())
