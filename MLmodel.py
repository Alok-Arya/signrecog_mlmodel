import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque

# ─── LOAD MODEL 
model = joblib.load("sign_model.pkl")

# ─── MEDIAPIPE SETUP 
import time
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


# ─── STATE 
sentence = ""
prediction_history = deque(maxlen=7)

last_label = None
cooldown = 0   # prevents repeated triggers

# ─── NORMALIZATION 
def normalize_landmarks(hand_landmarks):
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
    wrist = pts[0]
    pts = pts - wrist
    scale = np.max(np.abs(pts)) + 1e-6
    pts = pts / scale
    return pts.flatten()

# ─── STABLE PREDICTION 
def stable_prediction(history):
    if not history:
        return None
    return max(set(history), key=history.count)

# ─── CAMERA 
cap = cv2.VideoCapture(0)

print("\n🚀 SIGN LANGUAGE RECOGNITION STARTED")
print("Press 'Q' to quit\n")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        frame_timestamp_ms = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        label = None
        display_label = "..."

        # ─── COOLDOWN UPDATE 
        cooldown = max(0, cooldown - 1)

        # ─── HAND DETECTION 
        if result.hand_landmarks:
            for hl in result.hand_landmarks:

                draw_landmarks(frame, hl)

                data = normalize_landmarks(hl).reshape(1, -1)
                pred = model.predict(data)[0]

                prediction_history.append(pred)
                label = stable_prediction(list(prediction_history))

        else:
            label = None

        # ─── ACTION LOGIC (SIGN → NOTHING → SIGN)
        if label is not None and cooldown == 0:

            if label != last_label:

                if label == "SPACE":
                    sentence += " "
                    display_label = "SPACE"

                elif label == "DELETE":
                    if len(sentence) > 0:
                        sentence = sentence[:-1]
                    display_label = "DELETE"

                elif label == "NOTHING":
                    display_label = "..."

                else:
                    sentence += label
                    display_label = label

                last_label = label
                cooldown = 15   
                
        cv2.rectangle(frame, (0, 0), (1000, 100), (20, 20, 30), -1)

        cv2.putText(frame, f"Prediction: {display_label}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(frame, f"Sentence: {sentence}",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Sign Language AI", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()

    print("\n==============================")
    print("📝 FINAL SENTENCE:")
    print(sentence.strip())
    print("==============================\n")
