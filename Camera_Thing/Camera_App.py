import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import speech_recognition as sr
from collections import deque
import threading
import os
from sklearn.ensemble import RandomForestClassifier
import time

# ---------------- CONFIG ----------------
DATA_FILE = "hand_gestures.csv"  # Always use original CSV
wrist_history = deque(maxlen=15)

# ---------------- HAND SETUP ----------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
cap = cv2.VideoCapture(0)

# ---------------- LOAD ORIGINAL DATA ----------------
if os.path.exists(DATA_FILE):
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded original dataset with {len(df)} samples.")
else:
    df = pd.DataFrame(columns=["landmarks", "label"])
    print("Original dataset not found. Creating new one.")

# ---------------- AI SETUP ----------------
def train_model():
    if df.empty:
        return None
    X = np.array([np.fromstring(s, sep=',') for s in df['landmarks']])
    y = df['label']
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X, y)
    return clf

clf = train_model()

# ---------------- FUNCTIONS ----------------
def extract_landmarks(hand_landmarks):
    return np.array([coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y, lm.z)])

def predict_gesture(hand_landmarks):
    global clf
    if clf is None:
        return "Unknown Gesture"
    landmarks = extract_landmarks(hand_landmarks).reshape(1, -1)
    return clf.predict(landmarks)[0]

def detect_wave():
    if len(wrist_history) < wrist_history.maxlen:
        return False
    x_positions = list(wrist_history)
    if max(x_positions) - min(x_positions) > 0.1:
        wrist_history.clear()
        return True
    return False

# ---------------- VOICE THREAD ----------------
voice_command_active = False
voice_lock = threading.Lock()

def listen_for_commands():
    global voice_command_active
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Voice listener ready. Say 'Start' to start or 'stop' to deactivate...")
    while True:
        try:
            with mic as source:
                audio = recognizer.listen(source, phrase_time_limit=3)
            text = recognizer.recognize_google(audio)
            print("\nHeard:", text.lower())
            with voice_lock:
                if "start" in text.lower():
                    voice_command_active = True
                    print("Gesture detection ACTIVATED!")
                elif "stop" in text.lower():
                    voice_command_active = False
                    print("Gesture detection DEACTIVATED!")
        except sr.UnknownValueError:
            continue
        except sr.RequestError as e:
            print("Speech Recognition error:", e)
            continue

voice_thread = threading.Thread(target=listen_for_commands, daemon=True)
voice_thread.start()

# ---------------- LIVE LABELING ----------------
label_map = {
    ord('1'): "Fist",
    ord('2'): "Open Hand",
    ord('3'): "Thumbs Up",
    ord('4'): "Peace",
    ord('5'): "Middle Finger",
    ord('6'): "Waving"
}

print("Press keys 1-6 to label gestures while hand is visible, 'q' to quit")

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    gesture = "No Hand Detected"

    with voice_lock:
        active = voice_command_active

    if active and result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # AI Gesture Prediction
            gesture = predict_gesture(hand_landmarks)

            # Wrist tracking for waving
            wrist_x = hand_landmarks.landmark[0].x
            wrist_history.append(wrist_x)
            if detect_wave():
                gesture = "Waving 👋"

            # ---------------- LIVE DATA LABELING ----------------
            key = cv2.waitKey(1) & 0xFF
            if key in label_map:
                label = label_map[key]
                landmarks_str = ','.join(map(str, extract_landmarks(hand_landmarks)))
                # Append new sample to ORIGINAL CSV
                df = pd.concat([df, pd.DataFrame({'landmarks':[landmarks_str],'label':[label]})], ignore_index=True)
                df.to_csv(DATA_FILE, index=False)
                print(f"\nSaved sample for gesture: {label}")

                # Retrain AI with updated dataset
                clf = train_model()

    # Display gesture and voice status
    cv2.putText(frame, f"Gesture: {gesture}", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    if active:
        cv2.putText(frame, "Voice Activated ✅", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    else:
        cv2.putText(frame, "Say 'start' to activate, 'stop' to deactivate", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    print("\rGesture:", gesture, end="")

    cv2.imshow("AI Hand Gesture (Voice Controlled, Original Data)", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
