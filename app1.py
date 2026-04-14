from flask import Flask, render_template, request, redirect, url_for, session
import os
import io
import base64
import sqlite3
from datetime import timedelta
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import joblib
import librosa
import librosa.display
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.feature_extraction import extract_features

# ---------------- APP SETUP ---------------- #
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev_key")
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=10)

@app.before_request
def make_session_permanent():
    session.permanent = True

UPLOAD_FOLDER = 'temp_uploads'
ALLOWED_EXTENSIONS = {'wav'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------------- DATABASE ---------------- #
def get_db_connection():
    db_path = os.path.join(os.getcwd(), 'database.db')
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        full_name TEXT NOT NULL,
        role TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL)''')

    conn.execute('''CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        patient_name TEXT NOT NULL,
        patient_age INTEGER NOT NULL,
        patient_gender TEXT NOT NULL,
        model_used TEXT NOT NULL,
        result TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

    conn.commit()
    conn.close()

init_db()

# ---------------- LOAD MODELS ---------------- #
rf_model, cnn_model = None, None

try:
    rf_model = joblib.load("murmur_model.joblib")
    cnn_model = tf.keras.models.load_model("cnn_model/cnn_best_model.keras")
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"⚠️ Model loading error: {e}")

# ---------------- ML FUNCTIONS ---------------- #
def predict_rf(av_path, mv_path, pv_path, tv_path):
    if rf_model is None:
        return "❌ RF model not loaded"

    feature_vector = {}

    for region, path in zip(["AV", "MV", "PV", "TV"], [av_path, mv_path, pv_path, tv_path]):
        if path is None:
            continue
        try:
            audio, sr = librosa.load(path, sr=22050)
            features = extract_features(audio)
            for key, value in features.items():
                feature_vector[f"{region}_{key}"] = value
        except Exception as e:
            return f"Error processing {region}: {str(e)}"

    if len(feature_vector) == 0:
        return "Please upload at least one audio file."

    try:
        X = pd.DataFrame([feature_vector]).fillna(0)
        probs = rf_model.predict_proba(X)[0]
        prediction = np.argmax(probs)
        confidence = probs[prediction] * 100

        if prediction == 1:
            return f"🔴 Murmur Detected (RF)<br>Confidence: {confidence:.2f}%"
        else:
            return f"🟢 No Murmur Detected (RF)<br>Confidence: {confidence:.2f}%"
    except Exception as e:
        return f"Model Prediction Error: {e}"

def predict_cnn(audio_path):
    if cnn_model is None:
        return "❌ CNN model not loaded"

    if audio_path is None:
        return "Please upload an audio file."

    try:
        audio, sr = librosa.load(audio_path, sr=22050)
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64)
        mel_db = librosa.power_to_db(mel)

        if mel_db.shape[1] < 128:
            mel_db = np.pad(mel_db, ((0, 0), (0, 128 - mel_db.shape[1])))

        mel_db = mel_db[:, :128]
        mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())
        X = mel_db.reshape(1, 64, 128, 1)

        probs = cnn_model.predict(X, verbose=0)[0]
        prediction = np.argmax(probs)
        confidence = probs[prediction] * 100

        if prediction == 1:
            return f"🔴 Murmur Detected (CNN)<br>Confidence: {confidence:.2f}%"
        else:
            return f"🟢 No Murmur Detected (CNN)<br>Confidence: {confidence:.2f}%"
    except Exception as e:
        return f"Error: {str(e)}"

def get_spectrogram_base64(audio_path, title):
    if not audio_path or not os.path.exists(audio_path):
        return None

    audio, sr = librosa.load(audio_path, sr=22050)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64)
    mel_db = librosa.power_to_db(mel)

    fig, ax = plt.subplots(figsize=(5, 3))
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    img = librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel", ax=ax)
    ax.set_title(f"Spectrogram: {title}", color='white')

    plt.tight_layout()
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='png', transparent=True)
    img_buf.seek(0)

    img_base64 = base64.b64encode(img_buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64

# ---------------- ROUTES ---------------- #
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("logged_in"):
        return redirect(url_for("home"))

    error = None

    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()

        if user and check_password_hash(user['password'], password):
            session["logged_in"] = True
            session["user"] = username
            session["full_name"] = user["full_name"]
            return redirect(url_for("home"))
        else:
            error = "❌ Invalid username or password."

    return render_template("login.html", error=error)

@app.route("/register", methods=["GET", "POST"])
def register():
    error = None

    if request.method == "POST":
        full_name = request.form.get("full_name")
        role = request.form.get("role")
        email = request.form.get("email")
        username = request.form.get("username")
        password = request.form.get("password")

        conn = get_db_connection()

        if conn.execute('SELECT * FROM users WHERE username = ? OR email = ?', (username, email)).fetchone():
            error = "❌ Username or Email already exists."
        else:
            conn.execute(
                'INSERT INTO users (full_name, role, email, username, password) VALUES (?, ?, ?, ?, ?)',
                (full_name, role, email, username, generate_password_hash(password))
            )
            conn.commit()
            conn.close()
            return redirect(url_for("login"))

        conn.close()

    return render_template("register.html", error=error)

@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    username = session["user"]
    result = None
    spectrograms = {}
    error_msg = None

    if request.method == "POST":
        model_type = request.form.get("model_type")

        file_paths = {"AV": None, "MV": None, "PV": None, "TV": None}

        for region in file_paths:
            file = request.files.get(f"{region}_audio")
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                path = os.path.join(UPLOAD_FOLDER, f"{region}_{filename}")
                file.save(path)
                file_paths[region] = path

        if model_type == "rf":
            result = predict_rf(*file_paths.values())
        elif model_type == "cnn":
            audio = next((v for v in file_paths.values() if v), None)
            result = predict_cnn(audio)

        for region, path in file_paths.items():
            if path:
                spectrograms[region] = get_spectrogram_base64(path, region)

        for path in file_paths.values():
            if path and os.path.exists(path):
                os.remove(path)

    return render_template("dashboard.html", user=username, result=result, spectrograms=spectrograms)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

# ---------------- RUN ---------------- #
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)