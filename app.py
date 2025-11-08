# app.py
import os
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from io import BytesIO
from functools import wraps
import logging

from flask import (
    Flask, render_template, request, redirect, url_for, flash,
    send_file, jsonify, session, send_from_directory
)
from flask_wtf import FlaskForm
from wtforms import PasswordField
from wtforms.validators import DataRequired
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image, ImageFilter
import numpy as np
import tensorflow as tf
import pandas as pd
from flask import Flask
import os

import requests
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify, session
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# --- OpenWeather API Key ---
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY")


# -----------------------
# Create Flask App
# -----------------------
app = Flask(__name__)

# -----------------------
# Configuration
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
DB_PATH = os.path.join(BASE_DIR, "app.db")
MODEL_PATH = os.path.join(BASE_DIR, "models", "mold_model_final.keras")
ALLOWED_EXT = {"png", "jpg", "jpeg", "bmp"}
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "ChangeThisAdminPass!")

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()
# Flask config
app.config.update(
    SECRET_KEY=os.environ.get("SECRET_KEY", secrets.token_urlsafe(32)),
    UPLOAD_FOLDER=UPLOAD_FOLDER,
    DB_PATH=DB_PATH,
    MODEL_PATH=MODEL_PATH,
    ALLOWED_EXT=ALLOWED_EXT,
    ADMIN_PASSWORD=ADMIN_PASSWORD,
    MAX_CONTENT_LENGTH=8 * 1024 * 1024,  # 8 MB
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SECURE=True,          # True in production with HTTPS
    SESSION_COOKIE_SAMESITE="Lax",
    PERMANENT_SESSION_LIFETIME=timedelta(hours=2),
    
    # Flask-Mail settings
    MAIL_SERVER=os.environ.get("MAIL_SERVER", "smtp.gmail.com"),
    MAIL_PORT=int(os.environ.get("MAIL_PORT", 465)),
    MAIL_USE_SSL=os.environ.get("MAIL_USE_SSL", "True") == "True",
    MAIL_USERNAME=os.environ.get("MAIL_USERNAME"),
    MAIL_PASSWORD=os.environ.get("MAIL_PASSWORD"),
    MAIL_DEFAULT_SENDER=os.environ.get("MAIL_DEFAULT_SENDER", os.environ.get("MAIL_USERNAME")),
)

# -----------------------
# Database Helper Functions 
# -----------------------

import sqlite3
from datetime import datetime, timedelta, timezone
from werkzeug.security import generate_password_hash, check_password_hash


# -----------------------
# Config
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ALLOWED_EXT = {'png', 'jpg', 'jpeg'}
MODEL_PATH = os.path.join(BASE_DIR, "models", "mold_model_final.keras")
MODEL = None

# -----------------------
# Load model
# -----------------------
MODEL_PATH = os.path.join("models", "mold_model_final.keras")
if os.path.exists(MODEL_PATH):
    try:
        MODEL = load_model(MODEL_PATH, compile=False)
        print(f"[INFO] Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"[ERROR] Could not load model: {e}")
        MODEL = None
else:
    print(f"[WARNING] Model file not found at {MODEL_PATH}")

# -----------------------
# Utilities
# -----------------------
def predict_image_fullpath(path):
    """
    Predict if the image at `path` contains mold or not.
    Returns: (label:str, confidence:float, confidence_display:str)
    """
    try:
        # Load and convert image to RGB (force 3 channels)
        img = Image.open(path).convert("RGB")

        # Resize to exact model input size
        IMG_SIZE = (224, 224)
        img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)

        # Convert to numpy array and normalize
        arr = np.array(img, dtype=np.float32) / 255.0  # shape: (224,224,3)

        # Ensure it has 3 channels
        if arr.ndim == 2:  # fallback for grayscale
            arr = np.stack([arr]*3, axis=-1)
        elif arr.shape[-1] != 3:  # remove alpha or extra channels
            arr = arr[:, :, :3]

        # Add batch dimension
        arr = np.expand_dims(arr, axis=0)  # shape: (1, 224, 224, 3)
        print("Debug: input shape for model:", arr.shape)

        # Predict using loaded model
        if MODEL is not None:
            pred = MODEL.predict(arr, verbose=0).flatten()
            score = float(np.clip(pred[0], 0.0, 1.0))
            label = "mold" if score >= 0.5 else "clean"
            confidence_float = score if label == "mold" else (1 - score)
            confidence_display = f"{confidence_float*100:.2f}%"
            return label, confidence_float, confidence_display

        # Fallback when model not loaded
        avg = float(arr.mean())
        label = "clean" if avg > 0.6 else "mold"
        confidence_float = float(np.clip(abs(0.5 - avg) * 2, 0, 1))
        confidence_display = f"{confidence_float*100:.2f}%"
        return label, confidence_float, confidence_display

    except Exception as e:
        print(f"[ERROR] Prediction failed for {path}: {e}")
        return "unknown", 0.0, "0.00%"


# -----------------------
# Init DB & Model (updated)
# -----------------------
import sqlite3
from datetime import datetime, timezone
import os

DB_PATH = "database.db"

def get_db_conn():
    """Return a SQLite connection with dict-style row access and timeout."""
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize main database tables safely."""
    with get_db_conn() as conn:
        c = conn.cursor()
        # Users table
        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT DEFAULT 'user',
                failed_attempts INTEGER DEFAULT 0,
                locked_until TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                last_login TEXT
            )
        """)
        # Uploads table (with new environmental columns)
        c.execute("""
            CREATE TABLE IF NOT EXISTS uploads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                prediction TEXT,
                confidence REAL,
                shap_path TEXT,
                uploaded_at TEXT NOT NULL,
                user_id INTEGER,
                ground_truth TEXT,
                notes TEXT,
                ventilation TEXT DEFAULT 'moderate',
                leak TEXT DEFAULT 'no',
                health TEXT DEFAULT 'no',
                status TEXT,
                location TEXT,
                weather TEXT
            )
        """)
        # Feedback table
        c.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                rating INTEGER,
                comment TEXT,
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()
def list_records(limit=None):
    """Return all records from uploads table, optionally limited."""
    conn = get_db_conn()
    c = conn.cursor()
    query = "SELECT * FROM uploads ORDER BY uploaded_at DESC"
    if limit:
        c.execute(query + " LIMIT ?", (int(limit),))
    else:
        c.execute(query)
    rows = c.fetchall()
    conn.close()
    return rows

def save_prediction(filename, prediction, confidence, location, weather, status, user_id,
                    ventilation="moderate", leak="no", health="no"):
    """Save a prediction record with environmental info."""
    conn = get_db_conn()
    c = conn.cursor()
    c.execute("""
        INSERT INTO uploads
        (filename, prediction, confidence, location, weather, status, user_id, ventilation, leak, health, uploaded_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        filename, prediction, confidence, location, weather, status, user_id,
        ventilation, leak, health, datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    ))
    conn.commit()
    conn.close()


def ensure_uploads_columns():
    """Ensure all new environmental columns exist (safe migration)."""
    try:
        with get_db_conn() as conn:
            c = conn.cursor()
            c.execute("PRAGMA table_info(uploads)")
            cols = [r[1] for r in c.fetchall()]

            for col, default in [
                ("ventilation", "'moderate'"),
                ("leak", "'no'"),
                ("health", "'no'"),
                ("status", "''"),
                ("location", "''"),
                ("weather", "''")
            ]:
                if col not in cols:
                    c.execute(f"ALTER TABLE uploads ADD COLUMN {col} TEXT DEFAULT {default}")
                    print(f"[INFO] Added '{col}' column to uploads")
            conn.commit()
    except Exception as e:
        print(f"[ERROR] Schema migration failed: {e}")


def init_contacts_table():
    """Ensure contacts table exists."""
    with get_db_conn() as conn:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS contacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL,
                message TEXT NOT NULL,
                submitted_at TEXT NOT NULL
            )
        """)
        conn.commit()


# -----------------------
# Call at app startup
# -----------------------
init_db()
ensure_uploads_columns()
init_contacts_table()



# -----------------------
# Contact Form Helpers
# -----------------------
def save_contact(name, email, message):
    """Store contact form submissions in the database."""
    now = datetime.now(timezone.utc).isoformat()
    with get_db_conn() as conn:
        c = conn.cursor()
        c.execute(
            "INSERT INTO contacts (name, email, message, submitted_at) VALUES (?, ?, ?, ?)",
            (name, email, message, now)
        )
        conn.commit()
    print(f"[INFO] Contact saved: {name} <{email}>")


def send_contact_email(name, email, message):
    """Send contact email to admin via SMTP (safe fallback if not configured)."""
    import smtplib
    from email.message import EmailMessage

    ADMIN_EMAIL = os.environ.get("ADMIN_EMAIL", "admin@moldkit.com")
    SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
    SMTP_PORT = int(os.environ.get("SMTP_PORT", 587))
    SMTP_USER = os.environ.get("SMTP_USER")
    SMTP_PASS = os.environ.get("SMTP_PASS")

    if not (SMTP_USER and SMTP_PASS):
        print("[WARNING] SMTP credentials not configured. Email not sent.")
        return

    try:
        msg = EmailMessage()
        msg["Subject"] = "New Contact Form Submission"
        msg["From"] = SMTP_USER
        msg["To"] = ADMIN_EMAIL
        msg.set_content(f"""
Name: {name}
Email: {email}
Message:
{message}
""")
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
        print("[INFO] Contact email sent successfully.")

    except Exception as e:
        print(f"[ERROR] Failed to send contact email: {e}")


# -----------------------
# Forms
# -----------------------
class AdminLoginForm(FlaskForm):
    """Simple admin login form with password validation."""
    password = PasswordField("Password", validators=[DataRequired()])


# -----------------------
# Decorators
# -----------------------
def login_required(f):
    """Ensure user is logged in before accessing protected routes."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        if session.get("user_id"):
            return f(*args, **kwargs)
        flash("Please log in first.", "warning")
        return redirect(url_for("login"))
    return wrapper


def admin_required(f):
    """Restrict access to admin-only routes."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        if session.get("role") == "admin":
            return f(*args, **kwargs)
        flash("Admin access only.", "warning")
        return redirect(url_for("admin_login"))
    return wrapper


# -----------------------
# Metrics
# -----------------------
def compute_metrics():
    """
    Compute system-wide stats from prediction records.
    Returns a dictionary containing total uploads, counts, and accuracy if verified.
    """
    rows = list_records()
    total = len(rows)
    mold_count = sum(1 for r in rows if r["prediction"] == "mold")
    clean_count = total - mold_count

    verified = sum(
        1 for r in rows if r["ground_truth"] not in (None, "", "NULL")
    )
    correct = sum(
        1 for r in rows
        if r["ground_truth"] == r["prediction"]
        and r["ground_truth"] not in (None, "", "NULL")
    )

    accuracy = round((correct / verified * 100), 2) if verified else None

    return {
        "total": total,
        "mold": mold_count,
        "clean": clean_count,
        "verified": verified,
        "accuracy": accuracy,
    }


# -----------------------
# Routes - User
# -----------------------
@app.route("/", methods=["GET", "POST"])
def index():

    # Admin → Admin Dashboard
    if session.get("user_id") and session.get("role") == "admin":
        metrics = compute_metrics()

        conn = get_db_conn()
        c = conn.cursor()
        c.execute("SELECT * FROM feedback ORDER BY id DESC LIMIT 20")
        feedback_rows = [dict(row) for row in c.fetchall()]
        conn.close()

        return render_template(
            "admin_dashboard.html",
            metrics=metrics,
            feedback=feedback_rows,
            email=session.get("email")
        )

    # Handle Image Upload - Only for logged-in users
    if request.method == "POST":
        if not session.get("user_id"):
            flash("You must be logged in to upload an image.", "warning")
            return redirect(url_for("login"))

        file = request.files.get("image")
        if not file or not allowed_file(file.filename):
            flash("Invalid or missing image file.", "danger")
            return redirect(url_for("index"))

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        filename = secure_filename(f"{timestamp}_{file.filename}")
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(save_path)

        label, confidence = predict_image_fullpath(save_path)
        add_record(filename, label, confidence, user_id=session["user_id"]) # type: ignore

        log_action("prediction", extra=f"{filename}, label={label}, conf={confidence}") # type: ignore

        return redirect(url_for("result_page", filename=filename))

    # GET Requests — Load Index for users & guests
    recent_rows = list_records(limit=8)
    recent_uploads = [
        {
            "id": r["id"],
            "filename": r["filename"],
            "uploaded": r["uploaded_at"],
            "prediction": r["prediction"],
            "confidence": r["confidence"],
        }
        for r in recent_rows
    ]
    metrics = compute_metrics()

    return render_template(
        "index.html",
        recent=recent_uploads,
        metrics=metrics,
        email=session.get("email")
    )

# -----------------------
# Routes - Predict & Result
# -----------------------
@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    if request.method == "GET":
        return render_template("predict.html")

    is_ajax = request.is_json or request.headers.get("X-Requested-With") == "XMLHttpRequest"

    try:
        file = request.files.get("file")
        if not file or not allowed_file(file.filename):
            err = {"error": "No valid image uploaded"}
            return (jsonify(err), 400) if is_ajax else redirect(url_for("predict"))

        # -------------------------------
        # Process uploaded image
        # -------------------------------
        img = Image.open(file).convert("RGB")  # Force 3 channels
        IMG_SIZE = (224, 224)
        img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)

        # Ensure numpy array has shape (224,224,3)
        arr = np.array(img, dtype=np.uint8)
        if arr.ndim == 2:  # grayscale fallback
            arr = np.stack([arr]*3, axis=-1)
        elif arr.shape[-1] != 3:  # enforce 3 channels
            arr = arr[:, :, :3]

        # Convert back to PIL image for saving
        img = Image.fromarray(arr)

        # Save processed image
        filename = secure_filename(file.filename)
        ext = filename.rsplit(".", 1)[-1].lower()
        new_filename = f"{datetime.utcnow().timestamp()}.{ext}"
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], new_filename)
        img.save(save_path)

        # DEBUG: verify shape
        print("Preprocessed image shape:", arr.shape, "mode:", img.mode)

        # -------------------------------
        # Predict
        # -------------------------------
        label, conf_float, conf_display = predict_image_fullpath(save_path)

        # -------------------------------
        # Environmental metadata
        # -------------------------------
        location = request.form.get("location", "").strip()
        ventilation = request.form.get("ventilation", "moderate").lower()
        leak = request.form.get("leak", "no").lower()
        health = request.form.get("health", "no").lower()

        # Fetch weather
        humidity = None
        weather_data = "No location provided"
        if location:
            try:
                res = requests.get(
                    "https://api.openweathermap.org/data/2.5/weather",
                    params={"q": location, "appid": OPENWEATHER_API_KEY, "units": "metric"},
                    timeout=4
                )
                data = res.json()
                if res.ok and "main" in data:
                    temperature = data["main"].get("temp")
                    humidity = data["main"].get("humidity")
                    weather_data = f"{temperature}°C, {humidity}% humidity"
                else:
                    weather_data = "Invalid location"
            except Exception as ex:
                print(f"[WARN] Weather fetch error: {ex}")
                weather_data = "Weather fetch error"

        # -------------------------------
        # Risk score calculation
        # -------------------------------
        ventilation_risk = {"poor": 2, "moderate": 1, "good": 0}
        risk_score = ventilation_risk.get(ventilation, 1)
        if leak == "yes":
            risk_score += 2
        if humidity is not None:
            risk_score += 2 if humidity > 70 else (1 if humidity > 50 else 0)
        if health == "yes":
            risk_score += 1

        final_status = (
            "high" if label == "mold" and risk_score >= 3 else
            "moderate" if label == "mold" else
            "safe" if risk_score <= 2 else "high"
        )

        # Save prediction to database
        save_prediction(
            filename=new_filename,
            prediction=label,
            confidence=conf_float,
            location=location or "Unknown",
            weather=weather_data,
            status=final_status,
            user_id=session.get("user_id"),
            ventilation=ventilation,
            leak=leak,
            health=health
        )

        # Return AJAX response if requested
        if is_ajax:
            return jsonify({
                "success": True,
                "filename": new_filename,
                "prediction": label,
                "confidence": conf_display,
                "status": final_status,
                "weather": weather_data
            })

        return redirect(url_for("result_page"))

    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        if is_ajax:
            return jsonify({"error": "Prediction failed", "details": str(e)}), 500
        flash("Prediction failed.", "danger")
        return redirect(url_for("predict"))



@app.route("/result")
@login_required
def result_page():
    user_id = session.get("user_id")
    filename = request.args.get("filename")  # get filename from query string if provided

    conn = get_db_conn()
    cursor = conn.cursor()

    if filename:
        cursor.execute("""
            SELECT 
                filename,
                prediction,
                confidence,
                location,
                weather,
                status,
                ventilation,
                leak,
                health,
                uploaded_at
            FROM uploads
            WHERE user_id = ? AND filename = ?
            LIMIT 1
        """, (user_id, filename))
    else:
        cursor.execute("""
            SELECT 
                filename,
                prediction,
                confidence,
                location,
                weather,
                status,
                ventilation,
                leak,
                health,
                uploaded_at
            FROM uploads
            WHERE user_id = ?
            ORDER BY uploaded_at DESC
            LIMIT 1
        """, (user_id,))

    latest = cursor.fetchone()
    conn.close()

    if not latest:
        flash("No predictions found yet.", "info")
        return redirect(url_for("predict"))

    latest = dict(latest)

    env_result = {
        "weather": latest.get("weather", "Unknown"),
        "ventilation": (latest.get("ventilation") or "Unknown").title(),
        "leak": "Yes" if latest.get("leak") == "yes" else "No",
        "symptoms": "Yes" if latest.get("health") == "yes" else "No",
        "final_status": latest.get("status", "Unknown")
    }

    return render_template(
        "result.html",
        filename=latest["filename"],
        label=latest["prediction"],
        confidence=latest["confidence"],
        upload_date=latest["uploaded_at"],
        result=env_result,
        user=session.get("email"),
        explain=True,
        download_report=True
    )



@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    upload_folder = app.config.get("UPLOAD_FOLDER")
    if not upload_folder:
        return "Upload folder missing.", 500

    file_path = os.path.join(upload_folder, filename)
    if not os.path.isfile(file_path):
        return "File not found.", 404

    return send_from_directory(upload_folder, filename)

# -----------------------
# Feedback Route
# -----------------------
@app.route("/feedback/<filename>", methods=["POST"])
@login_required
def feedback(filename):
    user_id = session.get("user_id")
    if not user_id:
        flash("You must be logged in to submit feedback", "warning")
        return redirect(url_for("login"))

    fb = request.form.get("feedback")
    comment = request.form.get("comment", "")

    # Save feedback in DB
    conn = get_db_conn()
    c = conn.cursor()
    c.execute("""
        INSERT INTO feedback (user_id, filename, feedback, comment, submitted_at)
        VALUES (?, ?, ?, ?, datetime('now'))
    """, (user_id, filename, fb, comment))
    conn.commit()
    conn.close()

    flash("Thank you for your feedback!", "success")
    # Redirect back to the result page for the same file
    return redirect(url_for("result_page"))


# -----------------------
# User Registration
# -----------------------
import secrets
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash, check_password_hash

# -----------------------
# Helper for sending emails
# -----------------------
import smtplib
from email.mime.text import MIMEText

def send_email(to_email, subject, html_body):
    msg = MIMEText(html_body, "html")
    msg["Subject"] = subject
    msg["From"] = app.config["MAIL_DEFAULT_SENDER"]
    msg["To"] = to_email

    with smtplib.SMTP(app.config["MAIL_SERVER"], app.config["MAIL_PORT"]) as server:
        server.starttls()
        server.login(app.config["MAIL_USERNAME"], app.config["MAIL_PASSWORD"])
        server.send_message(msg)

# -----------------------
# Register
# -----------------------

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password")
        role = request.form.get("role", "user")  # Only admin/user values allowed

        # Validate duplicate email
        if get_user_by_email(email): # type: ignore
            flash("Email already registered.", "danger")
            return redirect(url_for("register"))

        password_hash = generate_password_hash(password)
        create_user(email, password_hash, role=role) # type: ignore

        # Fetch new user as dict
        user = dict(get_user_by_email(email)) # type: ignore

        # Auto-login after registration
        session["user_id"] = user["id"]
        session["email"] = user["email"]
        session["role"] = user["role"]

        flash("Account created successfully.", "success")
        return redirect(url_for("index"))

    return render_template("register.html")


# -----------------------
# LOGIN (Users + Admins)
# -----------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        user_row = get_user_by_email(email) # type: ignore
        if not user_row:
            flash("Email not found.", "danger")
            return redirect(url_for("login"))

        user = dict(user_row)

        # Validate password
        if not check_password_hash(user["password_hash"], password):
            increment_failed_attempts(user["id"]) # type: ignore
            flash("Incorrect password.", "danger")
            return redirect(url_for("login"))

        # Success: set session
        session["user_id"] = user["id"]
        session["email"] = user["email"]
        session["role"] = user["role"]

        update_user_login_success(user["id"]) # type: ignore
        return redirect(url_for("index"))  # Index handles dashboard logic

    return render_template("login.html")


# -----------------------
# OPTIONAL: SEPARATE ADMIN LOGIN PAGE
# -----------------------
@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        user_row = get_user_by_email(email) # type: ignore
        if not user_row:
            flash("Invalid admin credentials.", "danger")
            return redirect(url_for("admin_login"))

        user = dict(user_row)

        if user.get("role") != "admin" or \
           not check_password_hash(user["password_hash"], password):
            flash("Invalid admin credentials.", "danger")
            return redirect(url_for("admin_login"))

        session["user_id"] = user["id"]
        session["email"] = user["email"]
        session["role"] = user["role"]

        update_user_login_success(user["id"]) # type: ignore
        return redirect(url_for("index"))

    return render_template("admin_login.html")


# -----------------------
# VERIFY EMAIL (Optional)
# -----------------------
@app.route("/verify_email/<token>")
def verify_email(token):
    conn = get_db_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT user_id, expires_at FROM email_verifications WHERE token=?", (token,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        flash("Invalid or expired token.", "danger")
        return redirect(url_for("index"))

    expires_at = datetime.strptime(row["expires_at"], "%Y-%m-%d %H:%M:%S")
    if datetime.utcnow() > expires_at:
        flash("Verification link expired.", "danger")
        return redirect(url_for("index"))

    user_id = row["user_id"]

    conn = get_db_conn()
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET email_verified=1 WHERE id=?", (user_id,))
    cursor.execute("DELETE FROM email_verifications WHERE token=?", (token,))
    conn.commit()
    conn.close()

    user = dict(get_user_by_id(user_id)) # type: ignore

    # Log user in after verify
    session["user_id"] = user["id"]
    session["email"] = user["email"]
    session["role"] = user.get("role", "user")

    flash("Email verified. You are now logged in.", "success")
    return redirect(url_for("index"))


# -----------------------
# LOGOUT
# -----------------------
@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.", "info")
    return redirect(url_for("index"))

# -----------------------
# Profile
# -----------------------
@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    user_id = session.get("user_id")
    user = get_user_by_id(user_id) # type: ignore
    if not user:
        flash("User not found", "danger")
        return redirect(url_for("logout"))

    if request.method == "POST":
        new_email = request.form.get("email", "").strip().lower()
        current_password = request.form.get("current_password", "")
        new_password = request.form.get("new_password", "")

        if new_email:
            try:
                set_user_email(user_id, new_email) # type: ignore
                session["email"] = new_email
                flash("Email updated", "success")
            except sqlite3.IntegrityError:
                flash("Email already taken", "danger")

        if current_password and new_password:
            if check_password_hash(user["password_hash"], current_password):
                set_user_password(user_id, new_password) # type: ignore
                flash("Password updated", "success")
            else:
                flash("Current password incorrect", "danger")

        return redirect(url_for("profile"))

    uploads = list_records()
    user_uploads = [r for r in uploads if r.get("user_id") == user_id]

    user_info = {
        "email": user.get("email"),
        "role": user.get("role"),
        "last_login": user.get("last_login")
    }

    return render_template("profile_tabs.html", user=user_info, uploads=user_uploads)


@app.route("/profile/delete_account", methods=["POST"])
@login_required
def delete_account():
    user_id = session.get("user_id")
    conn = get_db_conn()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM users WHERE id=?", (user_id,))
        cursor.execute("UPDATE uploads SET user_id=NULL WHERE user_id=?", (user_id,))
        conn.commit()
    finally:
        conn.close()
    session.clear()
    flash("Account deleted", "info")
    return redirect(url_for("index"))


# -----------------------
# Admin Dashboard & Users
# -----------------------
@app.route("/admin/dashboard")
@admin_required
def admin_dashboard():
    rows = list_records(limit=200)
    metrics = compute_metrics()
    conn = get_db_conn()
    c = conn.cursor()
    c.execute("SELECT * FROM feedback ORDER BY id DESC LIMIT 20")
    feedback_rows = c.fetchall()
    conn.close()
    return render_template("admin_dashboard.html", rows=rows, metrics=metrics, feedback=feedback_rows)


@app.route("/admin/users", methods=["GET", "POST"])
@admin_required
def admin_users():
    conn = get_db_conn()
    c = conn.cursor()
    if request.method == "POST":
        action = request.form.get("action")
        target_id = request.form.get("user_id")
        if action == "make_admin":
            c.execute("UPDATE users SET role='admin' WHERE id=?", (target_id,))
        elif action == "remove_admin":
            c.execute("UPDATE users SET role='user' WHERE id=?", (target_id,))
        elif action == "update_email":
            new_email = request.form.get("email", "").strip().lower()
            try:
                c.execute("UPDATE users SET email=? WHERE id=?", (new_email, target_id))
            except sqlite3.IntegrityError:
                flash("Email already taken", "danger")
        conn.commit()
    c.execute("SELECT id,email,role,last_login FROM users ORDER BY id DESC")
    users = c.fetchall()
    conn.close()
    return render_template("admin_users.html", users=users)

@app.route("/admin/delete/<int:record_id>", methods=["POST"])
@admin_required
def admin_delete(record_id):
    conn = get_db_conn()
    c = conn.cursor()
    c.execute("DELETE FROM uploads WHERE id=?", (record_id,))
    conn.commit()
    conn.close()
    flash("Record deleted", "info")
    log_action("delete_upload", user_id=session.get("user_id"), extra=f"record_id={record_id}") # type: ignore
    return redirect(url_for("admin_dashboard"))

@app.route("/admin/download_csv")
@admin_required
def admin_download_csv():
    data = export_csv_bytes() # pyright: ignore[reportUndefinedVariable]
    return send_file(BytesIO(data), as_attachment=True, download_name="uploads.csv", mimetype="text/csv")

# -----------------------
# Public Static Pages
# -----------------------
@app.route("/about")
def about():
    return render_template("about.html", title="About Us")

@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        message = request.form.get("message")
        # save to database or send email
        flash("Message sent successfully!", "success")
        return redirect(url_for("contact"))

    return render_template("contact.html", title="Contact Us")


# -----------------------
# Password Reset (Stub)
# -----------------------
import smtplib
from email.mime.text import MIMEText

def send_email(to_email, subject, html_body):
    msg = MIMEText(html_body, "html")
    msg["Subject"] = subject
    msg["From"] = app.config["MAIL_DEFAULT_SENDER"]
    msg["To"] = to_email

    with smtplib.SMTP(app.config["MAIL_SERVER"], app.config["MAIL_PORT"]) as server:
        server.starttls()
        server.login(app.config["MAIL_USERNAME"], app.config["MAIL_PASSWORD"])
        server.send_message(msg)

@app.route("/forgot", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        user = get_user_by_email(email) # type: ignore
        if not user:
            flash("If that email exists, a reset link was sent.", "info")
            return redirect(url_for("login"))

        # Generate secure token and expiration
        token = secrets.token_urlsafe(32)
        expires_at = datetime.utcnow() + timedelta(hours=1)

        conn = get_db_conn()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO password_resets (user_id, token, expires_at)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET token=?, expires_at=?
        """, (user["id"], token, expires_at, token, expires_at))
        conn.commit()
        conn.close()

        # Send email
        reset_url = url_for('reset_password', token=token, _external=True)
        html_body = f"""
        <p>You requested a password reset. Click the link below to reset your password:</p>
        <p><a href="{reset_url}">{reset_url}</a></p>
        <p>If you did not request this, please ignore this email.</p>
        """
        send_email(email, "Password Reset Request", html_body)

        flash("Password reset link sent to your email.", "info")
        return redirect(url_for("login"))

    return render_template("forgot.html")

@app.route("/reset/<token>", methods=["GET", "POST"])
def reset_password(token):
    conn = get_db_conn()
    cursor = conn.cursor()

    cursor.execute("SELECT user_id, expires_at FROM password_resets WHERE token=?", (token,))
    row = cursor.fetchone()
    if not row or datetime.utcnow() > datetime.strptime(row["expires_at"], "%Y-%m-%d %H:%M:%S"):
        flash("Invalid or expired token.", "danger")
        conn.close()
        return redirect(url_for("forgot_password"))

    user_id = row["user_id"]

    if request.method == "POST":
        new_pw = request.form.get("new_password", "")
        confirm = request.form.get("confirm", "")
        if not new_pw or new_pw != confirm:
            flash("Passwords missing or do not match.", "warning")
            return redirect(url_for("reset_password", token=token))

        cursor.execute(
            "UPDATE users SET password_hash=? WHERE id=?",
            (generate_password_hash(new_pw), user_id)
        )
        # Delete token after use
        cursor.execute("DELETE FROM password_resets WHERE token=?", (token,))
        conn.commit()
        conn.close()

        flash("Password reset successfully! Please log in.", "success")
        return redirect(url_for("login"))

    conn.close()
    return render_template("reset.html", token=token)


@app.route("/profile_settings", methods=["GET", "POST"])
@login_required
def profile_settings():
    """View and update profile settings with history and notifications."""
    user_id = session.get("user_id")

    if not user_id:
        flash("Please log in to manage your profile.", "warning")
        return redirect(url_for("login"))

    try:
        conn = get_db_conn()
        cursor = conn.cursor()

        # Handle profile update
        if request.method == "POST":
            new_email = request.form.get("email")
            current_password = request.form.get("current_password")
            new_password = request.form.get("new_password")

            cursor.execute("SELECT password_hash FROM users WHERE id=?", (user_id,))
            row = cursor.fetchone()
            if not row:
                flash("User not found.", "danger")
                return redirect(url_for("index"))

            stored_hash = row[0]
            if current_password and not check_password_hash(stored_hash, current_password):
                flash("Current password is incorrect.", "danger")
            else:
                if new_password:
                    cursor.execute(
                        "UPDATE users SET email=?, password_hash=? WHERE id=?",
                        (new_email, generate_password_hash(new_password), user_id)
                    )
                else:
                    cursor.execute(
                        "UPDATE users SET email=? WHERE id=?",
                        (new_email, user_id)
                    )
                conn.commit()
                flash("Profile updated successfully!", "success")
                session["email"] = new_email
                return redirect(url_for("profile_settings"))

        # Fetch profile info
        cursor.execute("SELECT email, role, last_login FROM users WHERE id=?", (user_id,))
        user = cursor.fetchone()  # (email, role, last_login)

        # Fetch notifications
        cursor.execute("SELECT notify_upload, notify_news FROM notifications WHERE user_id=?", (user_id,))
        notif_row = cursor.fetchone()
        notifications = {"upload": bool(notif_row[0]) if notif_row else False,
                         "news": bool(notif_row[1]) if notif_row else False}

        # Fetch upload history with prediction, confidence, Grad-CAM, environment, feedback
        cursor.execute("""
            SELECT u.filename, u.uploaded_at, p.prediction, p.confidence,
                   p.gradcam_file, p.environment_data, f.feedback_json
            FROM uploads u
            LEFT JOIN predictions p ON u.id = p.upload_id
            LEFT JOIN feedback f ON u.id = f.upload_id
            WHERE u.user_id=?
            ORDER BY u.uploaded_at DESC
        """, (user_id,))
        uploads_raw = cursor.fetchall()
        uploads = []
        for row in uploads_raw:
            filename, uploaded_at, pred, conf, gradcam, env_json, fb_json = row
            env = json.loads(env_json) if env_json else {} # type: ignore
            feedback = json.loads(fb_json) if fb_json else [] # type: ignore
            uploads.append((filename, uploaded_at, pred, conf, gradcam or '', env, feedback))

        conn.close()
        return render_template("profile_settings.html",
                               user=user,
                               notifications=notifications,
                               uploads=uploads)

    except Exception as e:
        print(f"Profile settings error: {e}")
        flash("An error occurred while loading profile settings.", "danger")
        return redirect(url_for("index"))

from datetime import datetime

@app.route("/privacy")
def privacy():
    """
    Render the Privacy Policy page with a dynamic last-updated date.
    """
    last_updated = datetime.now().strftime("%B %d, %Y")  
    return render_template("privacy.html", last_updated=last_updated)


# ----------------------
# Printable Report Route
# ----------------------
@app.route("/print_report/<filename>")
def print_report(filename):
    # Fetch the main upload record
    conn = get_db_conn()
    c = conn.cursor()
    c.execute("SELECT label, confidence, final_status, temperature, humidity, ventilation, leaks, symptoms, notes FROM uploads WHERE filename=?",
              (filename,))
    row = c.fetchone()
    
    if not row:
        flash("Record not found", "warning")
        return redirect(url_for("index"))

    label, confidence, final_status, temp, humidity, ventilation, leaks, symptoms, notes = row

    # Fetch user feedback
    c.execute("SELECT feedback, comment, submitted_at FROM feedback WHERE filename=? ORDER BY submitted_at DESC", (filename,))
    feedback_rows = c.fetchall()
    feedbacks = [
        {"feedback": fb[0], "comment": fb[1], "submitted_at": fb[2]} for fb in feedback_rows
    ]

    conn.close()

    # Environmental context dictionary
    env = {
        "temperature": temp,
        "humidity": humidity,
        "ventilation": ventilation,
        "leaks": leaks,
        "symptoms": symptoms
    }

    return render_template(
        "print_report.html",
        filename=filename,
        label=label,
        confidence=confidence,
        final_status=final_status,
        env=env,
        feedbacks=feedbacks,
        notes=notes
    )


# -----------------------
# Run
# -----------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
