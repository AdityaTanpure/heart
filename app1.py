from flask import Flask, render_template, request, redirect, url_for, session, flash
import os
import io
import base64
import sqlite3
from datetime import datetime, timedelta
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

app = Flask(__name__)
app.secret_key = "super_secret_key_change_this"
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=10)

@app.before_request
def make_session_permanent():
    session.permanent = True

UPLOAD_FOLDER = 'temp_uploads'
ALLOWED_EXTENSIONS = {'wav'} # Only allow .wav files
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------------- DATABASE SETUP ---------------- #
def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, full_name TEXT NOT NULL, role TEXT NOT NULL, email TEXT UNIQUE NOT NULL, username TEXT UNIQUE NOT NULL, password TEXT NOT NULL)''')
    conn.execute('''CREATE TABLE IF NOT EXISTS history (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT NOT NULL, patient_name TEXT NOT NULL, patient_age INTEGER NOT NULL, patient_gender TEXT NOT NULL, model_used TEXT NOT NULL, result TEXT NOT NULL, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

init_db()

# ---------------- LOAD REAL MODELS ---------------- #
# Make sure these files are in your main project folder!
try:
    rf_model = joblib.load("murmur_model.joblib")
    cnn_model = tf.keras.models.load_model("cnn_model/cnn_best_model.keras")
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"⚠️ Warning: Could not load models. Error: {e}")

# ---------------- ML FUNCTIONS ---------------- #
def predict_rf(av_path, mv_path, pv_path, tv_path):
    feature_vector = {}
    for region, path in zip(["AV", "MV", "PV", "TV"], [av_path, mv_path, pv_path, tv_path]):
        if path is None: continue
        try:
            audio, sr = librosa.load(path, sr=22050)
            features = extract_features(audio)
            for key, value in features.items():
                feature_vector[f"{region}_{key}"] = value
        except Exception as e:
            return f"Error processing {region}: {str(e)}"

    if len(feature_vector) == 0: return "Please upload at least one audio file."

    try:
        X = pd.DataFrame([feature_vector]).fillna(0)
        probs = rf_model.predict_proba(X)[0]
        prediction = np.argmax(probs)
        confidence = probs[prediction] * 100

        if prediction == 1:
            return f"🔴 Murmur Detected (Random Forest)<br>Confidence: {confidence:.2f}%"
        else:
            return f"🟢 No Murmur Detected (Random Forest)<br>Confidence: {confidence:.2f}%"
    except Exception as e:
        return f"Model Prediction Error: Make sure feature extraction matches training data. Detail: {e}"

def predict_cnn(audio_path):
    if audio_path is None: return "Please upload an audio file."
    try:
        audio, sr = librosa.load(audio_path, sr=22050)
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64)
        mel_db = librosa.power_to_db(mel)

        if mel_db.shape[1] < 128:
            pad_width = 128 - mel_db.shape[1]
            mel_db = np.pad(mel_db, ((0,0),(0,pad_width)))

        mel_db = mel_db[:, :128]
        mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())
        X = mel_db.reshape(1,64,128,1)
        
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
    if not audio_path or not os.path.exists(audio_path): return None
    audio, sr = librosa.load(audio_path, sr=22050)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64)
    mel_db = librosa.power_to_db(mel)

    fig, ax = plt.subplots(figsize=(5, 3))
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    img = librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel", ax=ax)
    ax.set_title(f"Spectrogram: {title}", color='white')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    cbar = fig.colorbar(img, ax=ax)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    plt.tight_layout()
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='png', transparent=True)
    img_buf.seek(0)
    img_base64 = base64.b64encode(img_buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64

# ---------------- ROUTES ---------------- #
@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/analytics")
def analytics():
    if not session.get("logged_in"): return redirect(url_for("login"))
    return render_template("analytics.html", user=session["user"])

@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("logged_in"): return redirect(url_for("home"))
    error = None
    if request.method == "POST":
        username, password = request.form.get("username"), request.form.get("password")
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()
        if user and check_password_hash(user['password'], password):
            session["logged_in"], session["user"], session["full_name"] = True, username, user["full_name"]
            return redirect(url_for("home"))
        else:
            error = "❌ Invalid username or password."
    return render_template("login.html", error=error)

@app.route("/register", methods=["GET", "POST"])
def register():
    if session.get("logged_in"): return redirect(url_for("home"))
    error = None
    if request.method == "POST":
        full_name, role, email, username, password = request.form.get("full_name"), request.form.get("role"), request.form.get("email"), request.form.get("username"), request.form.get("password")
        conn = get_db_connection()
        if conn.execute('SELECT * FROM users WHERE username = ? OR email = ?', (username, email)).fetchone():
            error = "❌ Username or Email already exists."
        else:
            conn.execute('INSERT INTO users (full_name, role, email, username, password) VALUES (?, ?, ?, ?, ?)', (full_name, role, email, username, generate_password_hash(password)))
            conn.commit()
            conn.close()
            return redirect(url_for("login"))
        conn.close()
    return render_template("register.html", error=error)

@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if not session.get("logged_in"): return redirect(url_for("login"))
    username = session["user"]
    result, spectrograms, error_msg = None, {}, None

    if request.method == "POST":
        patient_name, patient_age, patient_gender, model_type = request.form.get("patient_name"), request.form.get("patient_age"), request.form.get("patient_gender"), request.form.get("model_type")
        file_paths = {"AV": None, "MV": None, "PV": None, "TV": None}
        
        # Safe File Upload
        for region in ["AV", "MV", "PV", "TV"]:
            file = request.files.get(f"{region}_audio")
            if file and file.filename != '':
                if allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(UPLOAD_FOLDER, f"{region}_{filename}")
                    file.save(filepath)
                    file_paths[region] = filepath
                else:
                    error_msg = f"❌ Invalid file type for {region}. Please upload .WAV files only."

        if not error_msg:
            if model_type == "rf":
                result = predict_rf(file_paths["AV"], file_paths["MV"], file_paths["PV"], file_paths["TV"])
            elif model_type == "cnn":
                audio_to_analyze = file_paths["AV"] or file_paths["MV"] or file_paths["PV"] or file_paths["TV"]
                result = predict_cnn(audio_to_analyze) if audio_to_analyze else "Please upload at least one audio file."

            if result and "Error" not in result and "Please upload" not in result:
                conn = get_db_connection()
                clean_result = result.replace("<br>", " - ") 
                conn.execute('INSERT INTO history (username, patient_name, patient_age, patient_gender, model_used, result) VALUES (?, ?, ?, ?, ?, ?)', (username, patient_name, patient_age, patient_gender, "Random Forest" if model_type == "rf" else "CNN", clean_result))
                conn.commit()
                conn.close()

            for region, path in file_paths.items():
                if path: spectrograms[region] = get_spectrogram_base64(path, f"{region} Heart Sound")
                
        for path in file_paths.values():
            if path and os.path.exists(path): os.remove(path)

    conn = get_db_connection()
    history = conn.execute('SELECT * FROM history WHERE username = ? ORDER BY timestamp DESC', (username,)).fetchall()
    conn.close()

    return render_template("dashboard.html", user=username, result=result, spectrograms=spectrograms, history=history, error_msg=error_msg)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

if __name__ == "__main__":
    app.run(debug=True)