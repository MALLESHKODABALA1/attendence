import base64
import csv
import os
import re
import shutil
import smtplib
from datetime import datetime, timedelta
from email.message import EmailMessage
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, flash, jsonify, redirect, render_template, request, url_for
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename


BASE_DIR = Path(__file__).resolve().parent
TRAINING_DIR = BASE_DIR / "training img"
REGISTRATION_CSV = BASE_DIR / "registered_students.csv"
ATTENDANCE_CSV = BASE_DIR / "attendance.csv"
EMAIL_LOG_CSV = BASE_DIR / "attendance_email_log.csv"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
FACE_SIZE = (120, 120)
FACE_MATCH_THRESHOLD = 0.48
FACE_PIXEL_THRESHOLD = 0.85
LBPH_MATCH_THRESHOLD = float(os.environ.get("ATTENDANCE_LBPH_THRESHOLD", "90"))
MIN_STABLE_NAME_FRAMES = 1
MIN_MATCH_FRAMES = 1
MAX_LIVENESS_FRAMES = 10
MIN_REAL_MOTION_FRAMES = 2
MIN_FACE_CENTER_SHIFT = 0.025
MIN_FACE_AREA_SHIFT = 0.08
MIN_FACE_TEXTURE_VARIANCE = 18.0
MAX_FACE_GLARE_RATIO = 0.12
EYE_MOTION_THRESHOLD = 0.05
MOUTH_MOTION_THRESHOLD = 0.05
CONTROL_MOTION_WEIGHT = 0.45
ACTION_CONTROL_RATIO = 1.02
MAX_CONTROL_MOTION_FOR_ACTION = 9.0
ASSET_VERSION = "20260707-idempotent-email-v1"
REGISTRATION_FIELDS = [
    "registered_at",
    "student_name",
    "phone",
    "email",
    "password_hash",
    "training_folder",
    "captured_image",
    "uploaded_images",
]

try:
    from email_config import EMAIL_SETTINGS
except ImportError:
    EMAIL_SETTINGS = {}

SMTP_HOST = os.environ.get("ATTENDANCE_SMTP_HOST", EMAIL_SETTINGS.get("SMTP_HOST", ""))
SMTP_PORT = int(os.environ.get("ATTENDANCE_SMTP_PORT", EMAIL_SETTINGS.get("SMTP_PORT", "587")))
SMTP_USER = os.environ.get("ATTENDANCE_SMTP_USER", EMAIL_SETTINGS.get("SMTP_USER", ""))
SMTP_PASSWORD = os.environ.get("ATTENDANCE_SMTP_PASSWORD", EMAIL_SETTINGS.get("SMTP_PASSWORD", ""))
SMTP_FROM = os.environ.get("ATTENDANCE_SMTP_FROM", EMAIL_SETTINGS.get("SMTP_FROM", SMTP_USER))

app = Flask(__name__)
app.secret_key = "attendance-registration-local-secret"
FACE_CASCADE = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"))
EYE_CASCADE = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_eye.xml"))
SMILE_CASCADE = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_smile.xml"))
ATTENDANCE_SESSIONS = {}
FACE_MODEL_CACHE = {"signature": None, "profiles": []}


def slugify_name(name):
    cleaned = re.sub(r"[^\w\s.-]", "", name, flags=re.UNICODE).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned or "unknown"


def person_dir(name):
    directory = TRAINING_DIR / slugify_name(name)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def enhance_photo(image):
    resized = image
    height, width = resized.shape[:2]
    max_side = 900
    if max(height, width) > max_side:
        scale = max_side / max(height, width)
        resized = cv2.resize(resized, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    improved_lightness = clahe.apply(lightness)
    improved = cv2.merge((improved_lightness, a_channel, b_channel))
    improved = cv2.cvtColor(improved, cv2.COLOR_LAB2BGR)
    blurred = cv2.GaussianBlur(improved, (0, 0), 1.0)
    return cv2.addWeighted(improved, 1.35, blurred, -0.35, 0)


def decode_capture(data_url):
    if not data_url or "," not in data_url:
        return None
    _, encoded = data_url.split(",", 1)
    image_bytes = base64.b64decode(encoded)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(image_array, cv2.IMREAD_COLOR)


def largest_face(gray):
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(70, 70))
    if len(faces) == 0:
        return None
    return max(faces, key=lambda face: face[2] * face[3])


def face_descriptor(image, face):
    x, y, width, height = face
    padding_x = int(width * 0.12)
    padding_y = int(height * 0.15)
    y1 = max(0, y - padding_y)
    y2 = min(image.shape[0], y + height + padding_y)
    x1 = max(0, x - padding_x)
    x2 = min(image.shape[1], x + width + padding_x)
    face_image = image[y1:y2, x1:x2]
    gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    gray_face = cv2.equalizeHist(cv2.resize(gray_face, FACE_SIZE, interpolation=cv2.INTER_AREA))

    center = gray_face[1:-1, 1:-1]
    lbp = np.zeros_like(center, dtype=np.uint8)
    neighbors = [
        gray_face[:-2, :-2],
        gray_face[:-2, 1:-1],
        gray_face[:-2, 2:],
        gray_face[1:-1, 2:],
        gray_face[2:, 2:],
        gray_face[2:, 1:-1],
        gray_face[2:, :-2],
        gray_face[1:-1, :-2],
    ]
    for index, neighbor in enumerate(neighbors):
        lbp |= ((neighbor >= center).astype(np.uint8) << index)

    lbp_hist = cv2.calcHist([lbp], [0], None, [64], [0, 256]).flatten()
    tone_hist = cv2.calcHist([gray_face], [0], None, [32], [0, 256]).flatten()
    descriptor = np.concatenate([lbp_hist, tone_hist]).astype("float32")
    descriptor /= descriptor.sum() + 1e-6
    return descriptor


def face_pixel_vector(image, face):
    x, y, width, height = face
    padding_x = int(width * 0.08)
    padding_y = int(height * 0.1)
    y1 = max(0, y - padding_y)
    y2 = min(image.shape[0], y + height + padding_y)
    x1 = max(0, x - padding_x)
    x2 = min(image.shape[1], x + width + padding_x)
    gray_face = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    gray_face = cv2.equalizeHist(cv2.resize(gray_face, FACE_SIZE, interpolation=cv2.INTER_AREA))
    vector = gray_face.astype("float32").flatten() / 255.0
    return (vector - vector.mean()) / (vector.std() + 1e-6)


def lbph_face_sample(image, face):
    x, y, width, height = face
    padding_x = int(width * 0.08)
    padding_y = int(height * 0.1)
    y1 = max(0, y - padding_y)
    y2 = min(image.shape[0], y + height + padding_y)
    x1 = max(0, x - padding_x)
    x2 = min(image.shape[1], x + width + padding_x)
    gray_face = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    gray_face = cv2.equalizeHist(cv2.resize(gray_face, FACE_SIZE, interpolation=cv2.INTER_AREA))
    return gray_face


def augmented_face_samples(face_sample):
    samples = [face_sample]
    center = (FACE_SIZE[0] / 2, FACE_SIZE[1] / 2)
    for angle in (-7, 7):
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(face_sample, matrix, FACE_SIZE, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        samples.append(rotated)
    samples.append(cv2.GaussianBlur(face_sample, (3, 3), 0))
    return samples


def training_signature():
    signature = []
    if not TRAINING_DIR.exists():
        return tuple(signature)
    for directory in sorted(TRAINING_DIR.iterdir(), key=lambda item: item.name.lower()):
        if not directory.is_dir():
            continue
        for image_path in sorted(directory.iterdir()):
            if image_path.suffix.lower() in ALLOWED_EXTENSIONS:
                stat = image_path.stat()
                signature.append((str(image_path), stat.st_mtime_ns, stat.st_size))
    return tuple(signature)


def trained_profiles():
    signature = training_signature()
    if FACE_MODEL_CACHE["signature"] == signature:
        return FACE_MODEL_CACHE["profiles"]

    samples = []
    labels = []
    label_names = {}
    next_label = 0
    for image_path, _, _ in signature:
        path = Path(image_path)
        image = cv2.imread(str(path))
        if image is None:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face = largest_face(gray)
        if face is None:
            continue
        name = path.parent.name
        label = next((key for key, value in label_names.items() if value == name), None)
        if label is None:
            label = next_label
            label_names[label] = name
            next_label += 1
        for sample in augmented_face_samples(lbph_face_sample(image, face)):
            samples.append(sample)
            labels.append(label)

    profiles = {"recognizer": None, "names": label_names}
    if samples:
        recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
        recognizer.train(samples, np.array(labels, dtype=np.int32))
        profiles["recognizer"] = recognizer

    FACE_MODEL_CACHE["signature"] = signature
    FACE_MODEL_CACHE["profiles"] = profiles
    return profiles


def descriptor_distance(left, right):
    return float(0.5 * np.sum(((left - right) ** 2) / (left + right + 1e-6)))


def pixel_distance(left, right):
    return 1 - float(np.dot(left, right) / ((np.linalg.norm(left) * np.linalg.norm(right)) + 1e-6))


def recognize_face(image, face):
    profiles = trained_profiles()
    recognizer = profiles.get("recognizer")
    if recognizer is None:
        return None, 0, None
    label, distance = recognizer.predict(lbph_face_sample(image, face))
    confidence = max(0, min(100, round((1 - (distance / LBPH_MATCH_THRESHOLD)) * 100)))
    if distance > LBPH_MATCH_THRESHOLD:
        return None, confidence, distance
    return profiles["names"].get(label), confidence, distance


def detect_liveness(gray, face):
    x, y, width, height = face
    face_gray = gray[y : y + height, x : x + width]
    upper = face_gray[0 : int(height * 0.55), :]
    lower = face_gray[int(height * 0.48) : height, :]
    eye_band = face_gray[int(height * 0.18) : int(height * 0.52), int(width * 0.12) : int(width * 0.88)]
    mouth_band = face_gray[int(height * 0.55) : int(height * 0.9), int(width * 0.18) : int(width * 0.82)]
    cheek_left = face_gray[int(height * 0.35) : int(height * 0.65), int(width * 0.05) : int(width * 0.28)]
    cheek_right = face_gray[int(height * 0.35) : int(height * 0.65), int(width * 0.72) : int(width * 0.95)]

    eyes = EYE_CASCADE.detectMultiScale(upper, scaleFactor=1.1, minNeighbors=5, minSize=(18, 18))
    smiles = SMILE_CASCADE.detectMultiScale(lower, scaleFactor=1.7, minNeighbors=18, minSize=(35, 18))

    mouth_open = False
    mouth_area = 0
    if len(smiles):
        mouth = max(smiles, key=lambda item: item[2] * item[3])
        mouth_area = int(mouth[2] * mouth[3])
        mouth_open = True

    frame_area = max(1, gray.shape[0] * gray.shape[1])
    face_area_ratio = (width * height) / frame_area
    center_x = (x + width / 2) / max(1, gray.shape[1])
    center_y = (y + height / 2) / max(1, gray.shape[0])
    texture_variance = float(cv2.Laplacian(face_gray, cv2.CV_64F).var())
    glare_ratio = float(np.mean(face_gray > 245))
    texture_ok = texture_variance >= MIN_FACE_TEXTURE_VARIANCE and glare_ratio <= MAX_FACE_GLARE_RATIO
    eye_motion_roi = cv2.resize(cv2.equalizeHist(eye_band), (80, 32), interpolation=cv2.INTER_AREA) if eye_band.size else None
    mouth_motion_roi = cv2.resize(cv2.equalizeHist(mouth_band), (80, 36), interpolation=cv2.INTER_AREA) if mouth_band.size else None
    control_motion_roi = None
    if cheek_left.size and cheek_right.size:
        left = cv2.resize(cv2.equalizeHist(cheek_left), (40, 32), interpolation=cv2.INTER_AREA)
        right = cv2.resize(cv2.equalizeHist(cheek_right), (40, 32), interpolation=cv2.INTER_AREA)
        control_motion_roi = np.hstack([left, right])

    return {
        "eyes_open": len(eyes) >= 1,
        "mouth_open": mouth_open,
        "mouth_area": mouth_area,
        "eye_motion_roi": eye_motion_roi,
        "mouth_motion_roi": mouth_motion_roi,
        "control_motion_roi": control_motion_roi,
        "center": (center_x, center_y),
        "area_ratio": face_area_ratio,
        "texture_variance": texture_variance,
        "glare_ratio": glare_ratio,
        "texture_ok": texture_ok,
    }


def update_liveness(session_id, name, live_state):
    state = ATTENDANCE_SESSIONS.setdefault(
        session_id,
        {
            "name": None,
            "matches": 0,
            "last_eyes": None,
            "last_mouth": None,
            "blink": False,
            "mouth_move": False,
            "candidate_name": None,
            "candidate_count": 0,
            "locked_name": None,
            "completed": False,
            "liveness_frames": 0,
            "eye_seen_open": False,
            "eye_seen_closed": False,
            "mouth_seen_open": False,
            "mouth_seen_closed": False,
            "mouth_open_frames": 0,
            "last_face_center": None,
            "last_face_area_ratio": None,
            "real_motion_frames": 0,
            "real_motion": False,
            "texture_ok_frames": 0,
            "spoof_warning_frames": 0,
            "last_eye_motion_roi": None,
            "last_mouth_motion_roi": None,
            "last_control_motion_roi": None,
            "eye_motion_peak": 0,
            "mouth_motion_peak": 0,
            "control_motion_peak": 0,
            "eye_specific_peak": 0,
            "mouth_specific_peak": 0,
            "blink_frame": 0,
        },
    )
    if state["name"] != name:
        state.update(
            {
                "name": name,
                "matches": 0,
                "last_eyes": None,
                "last_mouth": None,
                "blink": False,
                "mouth_move": False,
                "liveness_frames": 0,
                "eye_seen_open": False,
                "eye_seen_closed": False,
                "mouth_seen_open": False,
                "mouth_seen_closed": False,
                "mouth_open_frames": 0,
                "last_face_center": None,
                "last_face_area_ratio": None,
                "real_motion_frames": 0,
                "real_motion": False,
                "texture_ok_frames": 0,
                "spoof_warning_frames": 0,
                "last_eye_motion_roi": None,
                "last_mouth_motion_roi": None,
                "last_control_motion_roi": None,
                "eye_motion_peak": 0,
                "mouth_motion_peak": 0,
                "control_motion_peak": 0,
                "eye_specific_peak": 0,
                "mouth_specific_peak": 0,
                "blink_frame": 0,
            }
        )

    state["matches"] += 1
    state["liveness_frames"] = state.get("liveness_frames", 0) + 1
    eyes_open = live_state["eyes_open"]
    mouth_open = live_state["mouth_open"]
    state["eye_seen_open"] = state.get("eye_seen_open", False) or eyes_open
    state["eye_seen_closed"] = state.get("eye_seen_closed", False) or not eyes_open
    state["mouth_seen_open"] = state.get("mouth_seen_open", False) or mouth_open
    state["mouth_seen_closed"] = state.get("mouth_seen_closed", False) or not mouth_open
    state["mouth_open_frames"] = state.get("mouth_open_frames", 0) + (1 if mouth_open else 0)

    center = live_state.get("center")
    area_ratio = live_state.get("area_ratio")
    last_center = state.get("last_face_center")
    last_area_ratio = state.get("last_face_area_ratio")
    if center and last_center and area_ratio and last_area_ratio:
        center_shift = float(np.hypot(center[0] - last_center[0], center[1] - last_center[1]))
        area_shift = abs(area_ratio - last_area_ratio) / max(last_area_ratio, 1e-6)
        if center_shift >= MIN_FACE_CENTER_SHIFT or area_shift >= MIN_FACE_AREA_SHIFT:
            state["real_motion_frames"] = state.get("real_motion_frames", 0) + 1
    if state.get("real_motion_frames", 0) >= MIN_REAL_MOTION_FRAMES:
        state["real_motion"] = True
    if live_state.get("texture_ok"):
        state["texture_ok_frames"] = state.get("texture_ok_frames", 0) + 1
    else:
        state["spoof_warning_frames"] = state.get("spoof_warning_frames", 0) + 1

    eye_motion = 0
    mouth_motion = 0
    control_motion = 0
    eye_motion_roi = live_state.get("eye_motion_roi")
    mouth_motion_roi = live_state.get("mouth_motion_roi")
    control_motion_roi = live_state.get("control_motion_roi")
    if control_motion_roi is not None and state.get("last_control_motion_roi") is not None:
        control_motion = float(cv2.absdiff(control_motion_roi, state["last_control_motion_roi"]).mean())
        state["control_motion_peak"] = max(state.get("control_motion_peak", 0), control_motion)
    if eye_motion_roi is not None and state.get("last_eye_motion_roi") is not None:
        eye_motion = float(cv2.absdiff(eye_motion_roi, state["last_eye_motion_roi"]).mean())
        state["eye_motion_peak"] = max(state.get("eye_motion_peak", 0), eye_motion)
    if mouth_motion_roi is not None and state.get("last_mouth_motion_roi") is not None:
        mouth_motion = float(cv2.absdiff(mouth_motion_roi, state["last_mouth_motion_roi"]).mean())
        state["mouth_motion_peak"] = max(state.get("mouth_motion_peak", 0), mouth_motion)
    eye_specific_motion = max(0, eye_motion - (control_motion * CONTROL_MOTION_WEIGHT))
    mouth_specific_motion = max(0, mouth_motion - (control_motion * CONTROL_MOTION_WEIGHT))
    state["eye_specific_peak"] = max(state.get("eye_specific_peak", 0), eye_specific_motion)
    state["mouth_specific_peak"] = max(state.get("mouth_specific_peak", 0), mouth_specific_motion)
    stable_action_frame = control_motion <= MAX_CONTROL_MOTION_FOR_ACTION
    blink_candidate = (
        eye_motion >= EYE_MOTION_THRESHOLD
        or eye_specific_motion >= EYE_MOTION_THRESHOLD
    )
    mouth_candidate = (
        mouth_motion >= MOUTH_MOTION_THRESHOLD
        or mouth_specific_motion >= MOUTH_MOTION_THRESHOLD
    )
    if blink_candidate:
        state["blink"] = True
        state["blink_frame"] = state.get("liveness_frames", 0)
    if mouth_candidate:
        state["mouth_move"] = True
    if eyes_open and state.get("liveness_frames", 0) >= 2:
        state["blink"] = True
    if mouth_open or mouth_motion >= MOUTH_MOTION_THRESHOLD:
        state["mouth_move"] = True
    if state["last_eyes"] is True and eyes_open is False:
        state["blink"] = True
    if state["last_mouth"] is not None and state["last_mouth"] != mouth_open:
        state["mouth_move"] = True

    state["last_eyes"] = eyes_open
    state["last_mouth"] = mouth_open
    state["last_eye_motion_roi"] = eye_motion_roi
    state["last_mouth_motion_roi"] = mouth_motion_roi
    state["last_control_motion_roi"] = control_motion_roi
    state["eye_motion"] = round(eye_motion, 2)
    state["mouth_motion"] = round(mouth_motion, 2)
    state["control_motion"] = round(control_motion, 2)
    state["eye_specific_motion"] = round(eye_specific_motion, 2)
    state["mouth_specific_motion"] = round(mouth_specific_motion, 2)
    state["stable_action_frame"] = stable_action_frame
    state["last_face_center"] = center
    state["last_face_area_ratio"] = area_ratio
    return state


def update_stable_identity(session_id, name):
    state = ATTENDANCE_SESSIONS.setdefault(
        session_id,
        {
            "name": None,
            "matches": 0,
            "last_eyes": None,
            "last_mouth": None,
            "blink": False,
            "mouth_move": False,
            "candidate_name": None,
            "candidate_count": 0,
            "locked_name": None,
            "completed": False,
            "liveness_frames": 0,
            "eye_seen_open": False,
            "eye_seen_closed": False,
            "mouth_seen_open": False,
            "mouth_seen_closed": False,
            "mouth_open_frames": 0,
            "last_face_center": None,
            "last_face_area_ratio": None,
            "real_motion_frames": 0,
            "real_motion": False,
            "texture_ok_frames": 0,
            "spoof_warning_frames": 0,
            "last_eye_motion_roi": None,
            "last_mouth_motion_roi": None,
            "last_control_motion_roi": None,
            "eye_motion_peak": 0,
            "mouth_motion_peak": 0,
            "control_motion_peak": 0,
            "eye_specific_peak": 0,
            "mouth_specific_peak": 0,
            "blink_frame": 0,
        },
    )
    if state.get("locked_name"):
        return state
    if state["candidate_name"] == name:
        state["candidate_count"] += 1
    else:
        state.update(
            {
                "candidate_name": name,
                "candidate_count": 1,
                "name": None,
                "matches": 0,
                "last_eyes": None,
                "last_mouth": None,
                "blink": False,
                "mouth_move": False,
                "locked_name": None,
                "completed": False,
                "liveness_frames": 0,
                "eye_seen_open": False,
                "eye_seen_closed": False,
                "mouth_seen_open": False,
                "mouth_seen_closed": False,
                "mouth_open_frames": 0,
                "last_face_center": None,
                "last_face_area_ratio": None,
                "real_motion_frames": 0,
                "real_motion": False,
                "texture_ok_frames": 0,
                "spoof_warning_frames": 0,
                "last_eye_motion_roi": None,
                "last_mouth_motion_roi": None,
                "last_control_motion_roi": None,
                "eye_motion_peak": 0,
                "mouth_motion_peak": 0,
                "control_motion_peak": 0,
                "eye_specific_peak": 0,
                "mouth_specific_peak": 0,
                "blink_frame": 0,
            }
        )
    return state


def save_capture(name, data_url):
    image = decode_capture(data_url)
    if image is None:
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = person_dir(name) / f"captured_{timestamp}.jpg"
    cv2.imwrite(str(output_path), enhance_photo(image), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return output_path


def save_uploads(name, files):
    saved = []
    directory = person_dir(name)
    for file in files:
        if not file or not file.filename:
            continue
        suffix = Path(file.filename).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            continue
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"upload_{timestamp}_{secure_filename(file.filename)}"
        output_path = directory / filename
        file.save(output_path)
        saved.append(output_path)
    return saved


def display_path(path):
    try:
        return str(path.relative_to(BASE_DIR))
    except ValueError:
        return str(path)


def ensure_registration_schema():
    if not REGISTRATION_CSV.exists():
        return
    with REGISTRATION_CSV.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames == REGISTRATION_FIELDS:
            return
        rows = list(reader)

    with REGISTRATION_CSV.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=REGISTRATION_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in REGISTRATION_FIELDS})


def append_registration(row):
    ensure_registration_schema()
    csv_exists = REGISTRATION_CSV.exists()
    with REGISTRATION_CSV.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=REGISTRATION_FIELDS)
        if not csv_exists:
            writer.writeheader()
        writer.writerow(row)


def registered_students():
    students = {}
    if REGISTRATION_CSV.exists():
        ensure_registration_schema()
        with REGISTRATION_CSV.open("r", newline="", encoding="utf-8") as csv_file:
            for row in csv.DictReader(csv_file):
                name = row.get("student_name", "").strip()
                if name:
                    students[slugify_name(name).casefold()] = row
    return students


def password_matches_student(student, password):
    saved_hash = student.get("password_hash", "").strip()
    if saved_hash:
        return check_password_hash(saved_hash, password)
    return False


def find_registered_student(name, password):
    key = slugify_name(name).casefold()
    student = registered_students().get(key)
    if not student:
        return None, "Student is not registered yet. Please register first."

    if not password:
        return None, "Password is required for login."

    if not password_matches_student(student, password):
        return None, "Password does not match the registered record."

    return student, ""


def remove_registered_student(name, password):
    key = slugify_name(name).casefold()
    students = registered_students()
    student = students.get(key)
    if not student:
        return False, "Student is not registered."
    if not password:
        return False, "Password is required to remove a student."
    if not password_matches_student(student, password):
        return False, "Password does not match the registered record."

    rows = [row for row in students.values() if slugify_name(row.get("student_name", "")).casefold() != key]
    with REGISTRATION_CSV.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=REGISTRATION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    folder = TRAINING_DIR / slugify_name(student.get("student_name", name))
    if folder.exists() and folder.is_dir():
        shutil.rmtree(folder)
    FACE_MODEL_CACHE["signature"] = None
    FACE_MODEL_CACHE["profiles"] = []
    return True, f"{student.get('student_name', name)} was removed."


def is_registered_name(name):
    return slugify_name(name).casefold() in registered_students()


def registered_email_for(name):
    key = slugify_name(name).casefold()
    student = registered_students().get(key)
    if not student:
        return ""
    return student.get("email", "").strip()


def append_email_log(name, recipient, sent, message):
    csv_exists = EMAIL_LOG_CSV.exists()
    fields = ["logged_at", "student_name", "recipient", "sent", "message"]
    with EMAIL_LOG_CSV.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        if not csv_exists:
            writer.writeheader()
        writer.writerow(
            {
                "logged_at": datetime.now().isoformat(timespec="seconds"),
                "student_name": name,
                "recipient": recipient,
                "sent": "yes" if sent else "no",
                "message": message,
            }
        )


def send_attendance_email(name, confidence, marked_at, status):
    recipient = registered_email_for(name)
    if not recipient:
        message = f"No registered email found for {name}. Register this exact name with an email first."
        append_email_log(name, "", False, message)
        return False, message
    if not SMTP_HOST or not SMTP_USER or not SMTP_PASSWORD or not SMTP_FROM:
        message = "Email not sent. Set ATTENDANCE_SMTP_HOST, ATTENDANCE_SMTP_USER, ATTENDANCE_SMTP_PASSWORD, and ATTENDANCE_SMTP_FROM."
        append_email_log(name, recipient, False, message)
        return False, message

    status_label = status.title()
    message = EmailMessage()
    message["Subject"] = f"Live Attendance {status_label}"
    message["From"] = SMTP_FROM
    message["To"] = recipient
    message.set_content(
        "\n".join(
            [
                f"Hello {name},",
                "",
                f"Your live attendance status is: {status_label}.",
                f"Date: {marked_at.date().isoformat()}",
                f"Time: {marked_at.strftime('%H:%M:%S')}",
                f"Face confidence: {confidence}%",
                "",
                "This is an automated message from the AI Attendance System.",
            ]
        )
    )

    try:
        if SMTP_PORT == 465:
            with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=10) as server:
                server.login(SMTP_USER, SMTP_PASSWORD)
                server.send_message(message)
        else:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
                server.starttls()
                server.login(SMTP_USER, SMTP_PASSWORD)
                server.send_message(message)
    except Exception as error:
        failure = f"Email failed: {error}"
        append_email_log(name, recipient, False, failure)
        return False, failure

    success = f"{status_label} email sent to registered email {recipient}."
    append_email_log(name, recipient, True, success)
    return True, success


def already_marked_today(name):
    ensure_attendance_schema()
    if not ATTENDANCE_CSV.exists():
        return False
    today = datetime.now().date().isoformat()
    target_name = slugify_name(name).casefold()
    with ATTENDANCE_CSV.open("r", newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            row_name = slugify_name(row.get("student_name", "")).casefold()
            if row_name == target_name and row.get("date") == today:
                return True
    return False


def ensure_attendance_schema():
    fields = ["marked_at", "date", "time", "student_name", "status", "confidence", "method"]
    if not ATTENDANCE_CSV.exists():
        return
    with ATTENDANCE_CSV.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames and "status" in reader.fieldnames:
            return
        rows = list(reader)

    with ATTENDANCE_CSV.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "marked_at": row.get("marked_at", ""),
                    "date": row.get("date", ""),
                    "time": row.get("time", ""),
                    "student_name": row.get("student_name", ""),
                    "status": row.get("status", "") or "present",
                    "confidence": row.get("confidence", ""),
                    "method": row.get("method", ""),
                }
            )


def append_attendance_once(name, confidence, status):
    ensure_attendance_schema()
    fields = ["marked_at", "date", "time", "student_name", "status", "confidence", "method"]
    now = datetime.now()
    today = now.date().isoformat()
    target_name = slugify_name(name).casefold()
    rows = []
    updated_existing = False

    if ATTENDANCE_CSV.exists():
        with ATTENDANCE_CSV.open("r", newline="", encoding="utf-8") as csv_file:
            rows = list(csv.DictReader(csv_file))

    for row in rows:
        row_name = slugify_name(row.get("student_name", "")).casefold()
        if row_name == target_name and row.get("date") == today:
            if row.get("status", "").casefold() == "present" or status != "present":
                return None, False
            row.update(
                {
                    "marked_at": now.isoformat(timespec="seconds"),
                    "time": now.strftime("%H:%M:%S"),
                    "status": "present",
                    "confidence": confidence,
                    "method": "live_face_eye_lip_corrected",
                }
            )
            updated_existing = True
            break

    if updated_existing:
        with ATTENDANCE_CSV.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return now, True

    csv_exists = ATTENDANCE_CSV.exists()
    with ATTENDANCE_CSV.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        if not csv_exists:
            writer.writeheader()
        writer.writerow(
            {
                "marked_at": now.isoformat(timespec="seconds"),
                "date": now.date().isoformat(),
                "time": now.strftime("%H:%M:%S"),
                "student_name": name,
                "status": status,
                "confidence": confidence,
                "method": "live_face_eye_lip",
            }
        )
    return now, True


def attendance_rows():
    ensure_attendance_schema()
    if not ATTENDANCE_CSV.exists():
        return []
    with ATTENDANCE_CSV.open("r", newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def registration_rows():
    ensure_registration_schema()
    if not REGISTRATION_CSV.exists():
        return []
    with REGISTRATION_CSV.open("r", newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def pdf_escape(value):
    return str(value).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def parse_attendance_date(row):
    try:
        return datetime.strptime(row.get("date", ""), "%Y-%m-%d").date()
    except ValueError:
        return None


def filter_attendance_rows(rows, period):
    today = datetime.now().date()
    if period == "weekly":
        start = today - timedelta(days=today.weekday())
        end = start + timedelta(days=6)
    elif period == "monthly":
        start = today.replace(day=1)
        next_month = (start.replace(day=28) + timedelta(days=4)).replace(day=1)
        end = next_month - timedelta(days=1)
    else:
        return rows, "All Attendance Records"

    filtered = []
    for row in rows:
        row_date = parse_attendance_date(row)
        if row_date and start <= row_date <= end:
            filtered.append(row)
    label = f"{period.title()} Attendance ({start.isoformat()} to {end.isoformat()})"
    return filtered, label


def truncate_cell(value, limit):
    text = str(value or "")
    return text if len(text) <= limit else text[: max(0, limit - 3)] + "..."


def pdf_text(x, y, text, size=8, font="F1"):
    return f"BT /{font} {size} Tf {x:.2f} {y:.2f} Td ({pdf_escape(text)}) Tj ET"


def pdf_line(x1, y1, x2, y2):
    return f"{x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S"


def build_attendance_detail_pdf(student_rows, attendance_rows, period):
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    filtered_attendance, report_label = filter_attendance_rows(attendance_rows, period)
    students = {slugify_name(row.get("student_name", "")).casefold(): row for row in student_rows}
    report_rows = []
    for row in filtered_attendance:
        student = students.get(slugify_name(row.get("student_name", "")).casefold(), {})
        report_rows.append(
            {
                "date": row.get("date", ""),
                "time": row.get("time", ""),
                "name": row.get("student_name", ""),
                "phone": student.get("phone", ""),
                "email": student.get("email", ""),
                "status": row.get("status", ""),
                "confidence": f"{row.get('confidence', '')}%" if row.get("confidence", "") else "",
                "method": row.get("method", ""),
            }
        )

    if not report_rows and student_rows:
        for student in student_rows:
            report_rows.append(
                {
                    "date": "",
                    "time": "",
                    "name": student.get("student_name", ""),
                    "phone": student.get("phone", ""),
                    "email": student.get("email", ""),
                    "status": "No record",
                    "confidence": "",
                    "method": "",
                }
            )

    present_count = sum(1 for row in filtered_attendance if row.get("status", "").casefold() == "present")
    absent_count = sum(1 for row in filtered_attendance if row.get("status", "").casefold() == "absent")
    columns = [
        ("Date", "date", 72, 10),
        ("Time", "time", 56, 8),
        ("Student Name", "name", 120, 18),
        ("Phone", "phone", 90, 13),
        ("Email", "email", 180, 28),
        ("Status", "status", 70, 10),
        ("Conf.", "confidence", 48, 7),
        ("Method", "method", 150, 22),
    ]
    page_width = 842
    page_height = 612
    margin_x = 28
    title_y = 570
    table_top = 500
    row_height = 22
    rows_per_page = 20
    pages = [report_rows[index : index + rows_per_page] for index in range(0, len(report_rows), rows_per_page)] or [[]]
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
    ]
    page_refs = []

    for page_index, page_rows in enumerate(pages, start=1):
        page_number = len(objects) + 1
        content_number = page_number + 1
        page_refs.append(f"{page_number} 0 R")
        stream_lines = ["0.2 w"]
        stream_lines.append(pdf_text(margin_x, title_y, "AI Attendance Detailed Report", 16, "F2"))
        stream_lines.append(pdf_text(margin_x, title_y - 20, report_label, 10, "F2"))
        stream_lines.append(pdf_text(margin_x, title_y - 36, f"Generated at: {generated_at}", 9))
        stream_lines.append(pdf_text(margin_x, title_y - 52, f"Total records: {len(filtered_attendance)}   Present: {present_count}   Absent: {absent_count}", 9))
        stream_lines.append(pdf_text(page_width - 90, 24, f"Page {page_index}/{len(pages)}", 8))

        table_width = sum(column[2] for column in columns)
        row_count = len(page_rows) + 1
        table_height = row_count * row_height
        table_bottom = table_top - table_height

        stream_lines.append(pdf_line(margin_x, table_top, margin_x + table_width, table_top))
        stream_lines.append(pdf_line(margin_x, table_bottom, margin_x + table_width, table_bottom))
        x = margin_x
        for _, _, width, _ in columns:
            stream_lines.append(pdf_line(x, table_top, x, table_bottom))
            x += width
        stream_lines.append(pdf_line(x, table_top, x, table_bottom))

        y = table_top - row_height
        for _ in range(row_count):
            stream_lines.append(pdf_line(margin_x, y, margin_x + table_width, y))
            y -= row_height

        x = margin_x
        for heading, _, width, _ in columns:
            stream_lines.append(pdf_text(x + 4, table_top - 15, heading, 8, "F2"))
            x += width

        y = table_top - row_height - 15
        if page_rows:
            for row in page_rows:
                x = margin_x
                for _, key, width, char_limit in columns:
                    stream_lines.append(pdf_text(x + 4, y, truncate_cell(row.get(key, ""), char_limit), 8))
                    x += width
                y -= row_height
        else:
            stream_lines.append(pdf_text(margin_x + 4, table_top - row_height - 15, "No attendance records found for this period.", 8))

        stream = "\n".join(stream_lines)
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page_width} {page_height}] /Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> /Contents {content_number} 0 R >>")
        objects.append(f"<< /Length {len(stream.encode('latin-1', errors='replace'))} >>\nstream\n{stream}\nendstream")

    objects[1] = f"<< /Type /Pages /Kids [{' '.join(page_refs)}] /Count {len(page_refs)} >>"

    pdf = "%PDF-1.4\n"
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(pdf.encode("latin-1", errors="replace")))
        pdf += f"{index} 0 obj\n{obj}\nendobj\n"

    xref_offset = len(pdf.encode("latin-1", errors="replace"))
    pdf += f"xref\n0 {len(objects) + 1}\n"
    pdf += "0000000000 65535 f \n"
    for offset in offsets[1:]:
        pdf += f"{offset:010d} 00000 n \n"
    pdf += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n"
    return pdf.encode("latin-1", errors="replace")


def existing_people():
    if not TRAINING_DIR.exists():
        return []
    people = []
    for directory in sorted(TRAINING_DIR.iterdir(), key=lambda item: item.name.lower()):
        if directory.is_dir():
            images = [file for file in directory.iterdir() if file.suffix.lower() in ALLOWED_EXTENSIONS]
            people.append({"name": directory.name, "count": len(images)})
    return people


@app.post("/attendance-frame")
def attendance_frame():
    payload = request.get_json(silent=True) or {}
    session_id = payload.get("session_id", "default")
    session_state = ATTENDANCE_SESSIONS.get(session_id)
    if session_state and session_state.get("completed"):
        locked_name = session_state.get("locked_name") or session_state.get("name") or "this person"
        return jsonify(
            {
                "ok": True,
                "status": "session_completed",
                "message": f"Attendance detection already completed for {locked_name}. Start again for another person.",
                "student_name": locked_name,
            }
        )

    image = decode_capture(payload.get("image", ""))
    if image is None:
        return jsonify({"ok": False, "message": "Camera frame was not readable."}), 400

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face = largest_face(gray)
    if face is None:
        return jsonify({"ok": True, "status": "no_face", "message": "Keep your face inside the camera."})

    recognized_name, confidence, match_distance = recognize_face(image, face)
    live_state = detect_liveness(gray, face)
    if recognized_name is None:
        ATTENDANCE_SESSIONS.pop(session_id, None)
        return jsonify(
            {
                "ok": True,
                "status": "unknown",
                "message": "Face not matched strongly enough. Try better light and look straight at the camera.",
                "confidence": confidence,
                "match_distance": round(match_distance, 2) if match_distance is not None else None,
                "match_threshold": LBPH_MATCH_THRESHOLD,
                "eyes_open": live_state["eyes_open"],
                "mouth_open": live_state["mouth_open"],
            }
        )

    candidate_state = update_stable_identity(session_id, recognized_name)
    locked_name = candidate_state.get("locked_name")
    if not locked_name and candidate_state["candidate_count"] >= MIN_STABLE_NAME_FRAMES:
        candidate_state["locked_name"] = candidate_state["candidate_name"]
        locked_name = candidate_state["locked_name"]

    if not locked_name:
        return jsonify(
            {
                "ok": True,
                "status": "confirming",
                "message": f"Confirming face. Keep looking at the camera.",
                "candidate_name": recognized_name,
                "confidence": confidence,
                "match_distance": round(match_distance, 2) if match_distance is not None else None,
                "match_threshold": LBPH_MATCH_THRESHOLD,
                "stable_count": candidate_state["candidate_count"],
                "stable_required": MIN_STABLE_NAME_FRAMES,
                "eyes_open": live_state["eyes_open"],
                "mouth_open": live_state["mouth_open"],
                "blink_done": False,
                "mouth_done": False,
            }
        )

    name = locked_name
    if not is_registered_name(name):
        candidate_state["completed"] = True
        return jsonify(
            {
                "ok": True,
                "status": "unregistered",
                "message": f"{name} matched a training image, but is not registered. Attendance was not saved.",
                "student_name": name,
                "confidence": confidence,
                "match_distance": round(match_distance, 2) if match_distance is not None else None,
                "match_threshold": LBPH_MATCH_THRESHOLD,
                "email_sent": False,
                "email_message": "No email was sent because this person is not in registered_students.csv.",
            }
        )

    state = update_liveness(session_id, name, live_state)
    texture_confirmed = state.get("texture_ok_frames", 0) >= MIN_REAL_MOTION_FRAMES
    ready = (
        state["matches"] >= MIN_MATCH_FRAMES
        and state["blink"]
        and state["mouth_move"]
    )

    if not ready and state.get("liveness_frames", 0) >= MAX_LIVENESS_FRAMES:
        state["completed"] = True
        marked_at, created = append_attendance_once(name, confidence, "absent")
        email_sent = False
        email_message = "Attendance was already recorded today; absent email was not sent again."
        if created:
            email_sent, email_message = send_attendance_email(name, confidence, marked_at, "absent")
        return jsonify(
            {
                "ok": True,
                "status": "spoof_suspected",
                "message": f"{name} matched, but live eye blink and lip movement were not confirmed. Marked absent.",
                "student_name": name,
                "confidence": confidence,
                "match_distance": round(match_distance, 2) if match_distance is not None else None,
                "match_threshold": LBPH_MATCH_THRESHOLD,
                "eyes_open": live_state["eyes_open"],
                "mouth_open": live_state["mouth_open"],
                "blink_done": state["blink"],
                "mouth_done": state["mouth_move"],
                "live_confirmed": ready,
                "real_motion_done": state.get("real_motion", False),
                "texture_ok": texture_confirmed,
                "texture_variance": round(live_state.get("texture_variance", 0), 2),
                "glare_ratio": round(live_state.get("glare_ratio", 0), 3),
                "eye_motion": state.get("eye_specific_motion", 0),
                "mouth_motion": state.get("mouth_specific_motion", 0),
                "matches": state["matches"],
                "liveness_frames": state.get("liveness_frames", 0),
                "attendance_status": "absent",
                "email_sent": email_sent,
                "email_message": email_message,
            }
        )

    marked_now = False
    email_sent = False
    email_message = ""
    if ready:
        marked_at, created = append_attendance_once(name, confidence, "present")
        if not created:
            state["completed"] = True
            return jsonify(
                {
                    "ok": True,
                    "status": "already_marked",
                    "message": f"Attendance already taken today for {name}.",
                    "student_name": name,
                    "confidence": confidence,
                    "match_distance": round(match_distance, 2) if match_distance is not None else None,
                    "match_threshold": LBPH_MATCH_THRESHOLD,
                    "eyes_open": live_state["eyes_open"],
                    "mouth_open": live_state["mouth_open"],
                    "blink_done": state["blink"],
                    "mouth_done": state["mouth_move"],
                    "live_confirmed": True,
                    "real_motion_done": state.get("real_motion", False),
                    "texture_ok": texture_confirmed,
                    "eye_motion": state.get("eye_specific_motion", 0),
                    "mouth_motion": state.get("mouth_specific_motion", 0),
                    "matches": state["matches"],
                    "email_sent": False,
                    "email_message": "Attendance was not saved again and email was not sent again.",
                }
            )
        marked_now = True
        email_sent, email_message = send_attendance_email(name, confidence, marked_at, "present")
        state["completed"] = True

    missing = []
    if not state["blink"]:
        missing.append("blink once")
    if not state["mouth_move"]:
        missing.append("move your lips")

    if ready:
        message = f"Attendance marked for {name}." if marked_now else f"Attendance ready for {name}."
        status = "marked"
    else:
        message = f"Hi {name}. Real face check required: please {' and '.join(missing)}."
        status = "live_check"

    return jsonify(
        {
            "ok": True,
            "status": status,
            "message": message,
            "student_name": name,
            "confidence": confidence,
            "match_distance": round(match_distance, 2) if match_distance is not None else None,
            "match_threshold": LBPH_MATCH_THRESHOLD,
            "eyes_open": live_state["eyes_open"],
            "mouth_open": live_state["mouth_open"],
            "blink_done": state["blink"],
            "mouth_done": state["mouth_move"],
            "live_confirmed": ready,
            "real_motion_done": state.get("real_motion", False),
            "texture_ok": texture_confirmed,
            "texture_variance": round(live_state.get("texture_variance", 0), 2),
            "glare_ratio": round(live_state.get("glare_ratio", 0), 3),
            "eye_motion": state.get("eye_specific_motion", 0),
            "mouth_motion": state.get("mouth_specific_motion", 0),
            "real_motion_frames": state.get("real_motion_frames", 0),
            "real_motion_required": MIN_REAL_MOTION_FRAMES,
            "matches": state["matches"],
            "liveness_frames": state.get("liveness_frames", 0),
            "liveness_limit": MAX_LIVENESS_FRAMES,
            "attendance_status": "present" if ready else "",
            "email_sent": email_sent,
            "email_message": email_message,
        }
    )


@app.post("/remove-student")
def remove_student():
    name = request.form.get("student_name", "").strip()
    password = request.form.get("password", "")
    if not name:
        flash("Student name is required to remove a student.", "error")
        return redirect(url_for("index"))

    removed, message = remove_registered_student(name, password)
    flash(message, "success" if removed else "error")
    return redirect(url_for("index"))


@app.route("/login", methods=["GET", "POST"])
def login():
    flash("Login section was removed. Use live attendance from the home page.", "success")
    return redirect(url_for("index") + "#attendance")


@app.get("/attendance-report.pdf")
def attendance_report_pdf():
    period = request.args.get("period", "all").casefold()
    if period not in {"all", "weekly", "monthly"}:
        period = "all"
    pdf = build_attendance_detail_pdf(registration_rows(), attendance_rows(), period)
    filename = f"attendance_{period}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    return Response(
        pdf,
        mimetype="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.get("/attendance")
def attendance_page():
    return redirect("/#attendance")


@app.get("/report")
@app.get("/pdf")
@app.get("/download")
def attendance_report_alias():
    return redirect("/attendance-report.pdf")


@app.errorhandler(404)
def not_found(error):
    flash("That page was not found, so I opened the attendance home page.", "error")
    return redirect(url_for("index"))


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        name = request.form.get("student_name", "").strip()
        password = request.form.get("password", "")
        if not name:
            flash("Student name is required.", "error")
            return redirect(url_for("index"))
        if not password:
            flash("Password is required for registration.", "error")
            return redirect(url_for("index"))

        if is_registered_name(name):
            flash(f"{name} is already registered. Start live attendance below.", "error")
            return redirect(url_for("index") + "#attendance")

        captured_path = save_capture(name, request.form.get("captured_image", ""))
        uploaded_paths = save_uploads(name, request.files.getlist("training_images"))

        if captured_path is None and not uploaded_paths:
            flash("Capture a photo or upload at least one training image.", "error")
            return redirect(url_for("index"))

        append_registration(
            {
                "registered_at": datetime.now().isoformat(timespec="seconds"),
                "student_name": name,
                "phone": request.form.get("phone", "").strip(),
                "email": request.form.get("email", "").strip(),
                "password_hash": generate_password_hash(password),
                "training_folder": display_path(person_dir(name)),
                "captured_image": display_path(captured_path) if captured_path else "",
                "uploaded_images": "; ".join(display_path(path) for path in uploaded_paths),
            }
        )

        flash(f"{name} registered and saved to CSV.", "success")
        return redirect(url_for("index"))

    return render_template(
        "index.html",
        people=existing_people(),
        csv_file=REGISTRATION_CSV.name,
        attendance_file=ATTENDANCE_CSV.name,
        email_log_file=EMAIL_LOG_CSV.name,
        asset_version=ASSET_VERSION,
    )


if __name__ == "__main__":
    use_https = os.environ.get("ATTENDANCE_HTTPS", "0").lower() in {"1", "true", "yes", "on"}
    default_port = "5443" if use_https else "5000"
    cert_file = os.environ.get("ATTENDANCE_CERT", "")
    key_file = os.environ.get("ATTENDANCE_KEY", "")
    ssl_context = None
    if use_https:
        ssl_context = (cert_file, key_file) if cert_file and key_file else "adhoc"
    app.run(
        host=os.environ.get("ATTENDANCE_HOST", "0.0.0.0"),
        port=int(os.environ.get("ATTENDANCE_PORT", default_port)),
        debug=True,
        use_reloader=False,
        ssl_context=ssl_context,
    )
