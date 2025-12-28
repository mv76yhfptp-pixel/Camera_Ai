import cv2
import numpy as np
import os
import pickle
import warnings
# Suppress deprecated-pkg_resources UserWarning coming from dependencies at import time
warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API.*",
    category=UserWarning,
)
import face_recognition
from datetime import datetime
from imutils import rotate_bound
# removed unused: deque, threading, time

# We no longer use MediaPipe, pandas or sklearn for the face-only mode.
pd = None
RandomForestClassifier = None

# ---------------- CONFIG ----------------
DATA_DIR = "face_dataset"
ENCODINGS_FILE = os.path.join(os.path.dirname(__file__), "face_encodings.pkl")
os.makedirs(DATA_DIR, exist_ok=True)

# Load face embeddings
if os.path.exists(ENCODINGS_FILE):
    try:
        with open(ENCODINGS_FILE, "rb") as f:
            known_face_encodings, known_face_names = pickle.load(f)
        print(f"Loaded {len(known_face_names)} known faces.")
    except Exception as e:
        print(f"Failed to load encodings: {e}")
        known_face_encodings = []
        known_face_names = []
else:
    known_face_encodings = []
    known_face_names = []

# Hand gesture dataset removed for face-only mode
df = None

cap = cv2.VideoCapture(0)

# ---------------- UTILITIES ----------------
def augment_face(face_img):
    augmented = [face_img]
    for angle in [-15, -10, -5, 5, 10, 15]:
        augmented.append(rotate_bound(face_img, angle))
    augmented.append(cv2.flip(face_img, 1))
    return augmented

def add_face_data(face_img, name):
    global known_face_encodings, known_face_names
    encodings = face_recognition.face_encodings(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
    if not encodings:
        print("No face detected in the selected region. Try again.")
        return

    augmented_faces = augment_face(face_img)
    count_added = 0
    for img in augmented_faces:
        enc = face_recognition.face_encodings(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if enc:
            known_face_encodings.append(enc[0])
            known_face_names.append(name)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            cv2.imwrite(os.path.join(DATA_DIR, f"{name}_{timestamp}.jpg"), img)
            count_added += 1

    # ---------------- ORIGINAL FACE-ONLY SETUP ----------------
    # Simplified setup for original face-only behavior. All hand/gesture
    # detection and related UI have been removed per user request.

    # Single camera window for face display
    cv2.namedWindow("Camera")
    def mouse_callback(event, x, y, flags, param):
        return
    cv2.setMouseCallback("Camera", mouse_callback)

# ---------------- LIVE LABELING ----------------
# (hand gesture labeling removed)
label_map = {}

# ---------------- MAIN LOOP ----------------
adding_face = False
typed_chars = []
frame_count = 0
recognition_interval = 5
face_locations = []
face_names = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame_count += 1
    if frame_count % recognition_interval == 0:
        face_locations_small = face_recognition.face_locations(rgb_small_frame)
        face_encodings_small = face_recognition.face_encodings(rgb_small_frame, face_locations_small)
        face_names = []
        for encoding in face_encodings_small:
            if len(known_face_encodings) == 0:
                face_names.append("Unknown")
                continue
            matches = face_recognition.compare_faces(known_face_encodings, encoding, tolerance=0.45)
            name = "Unknown"
            distances = face_recognition.face_distance(known_face_encodings, encoding)
            if len(distances) > 0:
                best_idx = np.argmin(distances)
                if matches[best_idx]:
                    name = known_face_names[best_idx]
            face_names.append(name)
        face_locations = [(top*4, right*4, bottom*4, left*4) for (top,right,bottom,left) in face_locations_small]

    # HAND PROCESSING removed — running in face-only mode

    # Prepare single display frame for original face-only UI
    display_frame = frame.copy()

    if adding_face:
        typed_name = "".join(typed_chars)
        cv2.putText(display_frame, "Typing Name:", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),2)
        cv2.putText(display_frame, typed_name + "_", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0),2)

    # ANNOTATE FACES on display frame
    for i, ((top,right,bottom,left), name) in enumerate(zip(face_locations, face_names)):
        display_name = name
        if adding_face and i==0 and name == "Unknown":
            display_name = "".join(typed_chars)
        cv2.rectangle(display_frame, (left,top), (right,bottom), (0,255,0), 2)
        cv2.putText(display_frame, display_name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0),2)

    # Draw right-side options panel
    panel_width = 260
    h, w = display_frame.shape[:2]
    cv2.rectangle(display_frame, (w-panel_width, 0), (w, h), (40, 40, 40), -1)
    px = w - panel_width + 15
    py = 40
    cv2.putText(display_frame, "MENU OPTIONS:", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    py += 40
    cv2.putText(display_frame, "A - Add Face", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
    py += 30
    cv2.putText(display_frame, "Q - Quit", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
    py += 30
    cv2.putText(display_frame, "Enter - Save Name", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)
    py += 25
    cv2.putText(display_frame, "Esc - Cancel", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)

    # KEYS
    key = cv2.waitKey(1) & 0xFF

    if key == ord('a') and not adding_face and len(face_locations) > 0:
        adding_face = True
        typed_chars = []
        print("Type new name on-screen. Enter=Save, Esc=Cancel")
    elif adding_face:
        try:
            if key == 13:  # Enter
                if len(face_locations) > 0:
                    top, right, bottom, left = face_locations[0]
                    top = max(0, top); left = max(0, left)
                    bottom = min(frame.shape[0], bottom); right = min(frame.shape[1], right)
                    face_img = frame[top:bottom, left:right]
                    name_to_save = "".join(typed_chars).strip()
                    if name_to_save != "":
                        add_face_data(face_img, name_to_save)
                    else:
                        print("No name typed. Face not saved.")
                else:
                    print("No face detected. Try again.")
                adding_face = False
                typed_chars = []
            elif key == 27:  # Esc
                adding_face = False
                typed_chars = []
            elif key != 255:
                if key == 8:  # Backspace
                    if typed_chars:
                        typed_chars.pop()
                elif 32 <= key <= 126:
                    typed_chars.append(chr(key))
        except Exception as e:
            print(f"Error adding face: {e}")
            adding_face = False
            typed_chars = []

    if key == ord('q'):
        break

    # SHOW: present single Camera window
    cv2.imshow("Camera", display_frame)

cap.release()
cv2.destroyAllWindows()