"""Streamlit-compatible LBPH face verification for Secure Heart admins.

Place this file in the project root. Authorized profiles are folders under
admin_verification/admins/<profile_name>/*.jpg|jpeg|png.
"""
from pathlib import Path
import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
ADMIN_DIR = BASE / "admin_verification" / "admins"
CASCADE_PATH = BASE / "admin_verification" / "cascades" / "haarcascade_frontalface_default.xml"
FACE_SIZE = (200, 200)
THRESHOLD = 75  # LBPH distance: lower is a closer match; tune only after testing.
EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _cascade():
    detector = cv2.CascadeClassifier(str(CASCADE_PATH))
    if detector.empty():
        raise RuntimeError(f"Could not load face detector: {CASCADE_PATH}")
    return detector


def _face_from_image(image, detector):
    if image is None:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces):
        x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
        gray = gray[y:y+h, x:x+w]
    elif max(gray.shape[:2]) > 400:
        return None
    return cv2.equalizeHist(cv2.resize(gray, FACE_SIZE, interpolation=cv2.INTER_CUBIC))


def _augment(face):
    variants = [face, cv2.flip(face, 1)]
    h, w = face.shape
    for base in (face, cv2.flip(face, 1)):
        for beta in (-25, 25):
            variants.append(cv2.convertScaleAbs(base, alpha=1.0, beta=beta))
        for angle in (-6, 6):
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            variants.append(cv2.warpAffine(base, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE))
    return variants


def train_recognizer():
    detector = _cascade()
    samples, labels, label_names = [], [], {}
    label = 0
    for folder in sorted(ADMIN_DIR.iterdir()):
        if not folder.is_dir():
            continue
        count = 0
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in EXTS:
                continue
            image = cv2.imread(str(path))
            face = _face_from_image(image, detector)
            if face is None:
                continue
            for variant in _augment(face):
                samples.append(variant)
                labels.append(label)
            count += 1
        if count:
            label_names[label] = folder.name.replace('_', ' ').strip()
            label += 1
    if not samples:
        raise RuntimeError("No usable admin face photos found. Add photos to admin_verification/admins/<name>/.")
    recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
    recognizer.train(samples, np.asarray(labels, dtype=np.int32))
    return recognizer, label_names, detector


def verify_camera_image(image_bytes, recognizer, names, detector):
    """Return (matched_name, distance, preview_bgr); None name means reject."""
    import numpy as np
    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        return None, None, None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        return None, None, image
    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    face = cv2.equalizeHist(cv2.resize(gray[y:y+h, x:x+w], FACE_SIZE, interpolation=cv2.INTER_CUBIC))
    label, distance = recognizer.predict(face)
    matched = label in names and distance <= THRESHOLD
    name = names[label] if matched else None
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 180, 0) if matched else (0, 0, 220), 3)
    return name, float(distance), image
