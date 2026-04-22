"""
FINAL DATA COLLECTION SCRIPT
============================
- Collects A-Z + SPACE + DELETE + NOTHING
- Resumes automatically where left off
- Avoids duplicate full re-collection
"""

import cv2
import mediapipe as mp
import os
import csv
import time
import numpy as np

# ─── CONFIG ────────────────────────────────────────────────
LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["SPACE", "DELETE", "NOTHING"]

DATA_DIR = "dataset"
CSV_PATH = os.path.join(DATA_DIR, "landmarks.csv")

SAMPLES = 200
COUNTDOWN_SEC = 3
# ────────────────────────────────────────────────────────────

os.makedirs(DATA_DIR, exist_ok=True)

# ─── MEDIAPIPE ─────────────────────────────────────────────
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6,
    running_mode=VisionRunningMode.VIDEO)

landmarker = HandLandmarker.create_from_options(options)

def draw_landmarks(frame, hand_landmarks):
    h, w, _ = frame.shape
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
    ]
    for lm in hand_landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
    for connection in HAND_CONNECTIONS:
        pt1 = hand_landmarks[connection[0]]
        pt2 = hand_landmarks[connection[1]]
        cx1, cy1 = int(pt1.x * w), int(pt1.y * h)
        cx2, cy2 = int(pt2.x * w), int(pt2.y * h)
        cv2.line(frame, (cx1, cy1), (cx2, cy2), (0, 255, 0), 2)


# ─── LOAD EXISTING DATA (RESUME SYSTEM) ───────────────────
label_count = {label: 0 for label in LABELS}

file_exists = os.path.exists(CSV_PATH)

if file_exists:
    with open(CSV_PATH, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                label = row[-1]
                if label in label_count:
                    label_count[label] += 1

print("\n📊 Current dataset status:")
for k, v in label_count.items():
    print(f"{k}: {v}/{SAMPLES}")

# ─── NORMALIZE LANDMARKS ───────────────────────────────────
def normalize(hand_landmarks):
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
    wrist = pts[0]
    pts = pts - wrist
    scale = np.max(np.abs(pts)) + 1e-6
    pts = pts / scale
    return pts.flatten().tolist()

# ─── UI ────────────────────────────────────────────────────
def draw_ui(frame, label, count, total, state):
    h, w = frame.shape[:2]

    cv2.rectangle(frame, (0, 0), (w, 80), (20, 20, 30), -1)

    cv2.circle(frame, (60, 40), 35, (50, 200, 100), -1)
    cv2.putText(frame, label, (40, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    if state == "waiting":
        msg = "SPACE to start | Q to skip"
    elif state == "countdown":
        msg = "Get ready..."
    else:
        msg = f"Recording {count}/{total}"

    cv2.putText(frame, msg, (120, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    return frame

# ─── CAMERA ────────────────────────────────────────────────
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera not found")
    exit()

# ─── START COLLECTION ──────────────────────────────────────
with open(CSV_PATH, "a", newline="") as f:
    writer = csv.writer(f)

    for label in LABELS:

        # Skip only if already COMPLETE
        if label_count[label] >= SAMPLES:
            print(f"⏭ Skipping {label} (already complete)")
            continue

        print(f"\n🎯 Collecting: {label}")

        # ─── WAIT ───────────────────────────────────────────
        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            frame_timestamp_ms = int(time.time() * 1000)
            result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            if result.hand_landmarks:
                for hl in result.hand_landmarks:
                    draw_landmarks(frame, hl)

            frame = draw_ui(frame, label, label_count[label], SAMPLES, "waiting")

            cv2.imshow("Collector", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):
                break
            if key == ord('q'):
                print("Skipped:", label)
                break
        else:
            continue

        # ─── COUNTDOWN ─────────────────────────────────────
        start = time.time()

        while time.time() - start < COUNTDOWN_SEC:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)

            remaining = COUNTDOWN_SEC - int(time.time() - start)

            cv2.putText(frame, str(remaining), (250, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 4, (100, 220, 255), 5)

            frame = draw_ui(frame, label, label_count[label], SAMPLES, "countdown")

            cv2.imshow("Collector", frame)
            cv2.waitKey(1)

        # ─── RECORDING ─────────────────────────────────────
        while label_count[label] < SAMPLES:

            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            frame_timestamp_ms = int(time.time() * 1000)
            result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            if result.hand_landmarks:
                for hl in result.hand_landmarks:

                    row = normalize(hl)
                    row.append(label)

                    writer.writerow(row)

                    draw_landmarks(frame, hl)

                    label_count[label] += 1

            frame = draw_ui(frame, label, label_count[label], SAMPLES, "recording")

            cv2.imshow("Collector", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print(f"✔ Done {label}: {label_count[label]}/{SAMPLES}")

# ─── CLEANUP ───────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()

print("\n✅ DATA COLLECTION COMPLETE")
print("Saved at:", CSV_PATH)