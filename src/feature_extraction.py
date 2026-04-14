import numpy as np
import librosa
from scipy.stats import skew, kurtosis


def extract_features(audio, sr=22050):

    features = {}

    # RMS Energy
    features["rms_mean"] = np.mean(librosa.feature.rms(y=audio))

    # Zero Crossing Rate
    features["zcr_mean"] = np.mean(librosa.feature.zero_crossing_rate(y=audio))

    # Spectral Centroid
    features["centroid_mean"] = np.mean(
        librosa.feature.spectral_centroid(y=audio, sr=sr)
    )

    # Spectral Bandwidth
    features["bandwidth_mean"] = np.mean(
        librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    )

    # Total Energy
    features["energy"] = np.sum(audio ** 2)

    # Skewness & Kurtosis
    features["skew"] = skew(audio)
    features["kurtosis"] = kurtosis(audio)

    # MFCC (13)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

    for i in range(13):
        features[f"mfcc_{i+1}"] = np.mean(mfccs[i])

    return features
