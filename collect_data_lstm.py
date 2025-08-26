import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress TF logs (0 = all, 3 = only errors)
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)  # suppress Mediapipe warnings
import cv2
import numpy as np
import os
import mediapipe as mp
import time



# --- CONFIGURATION ---
action_to_collect = "F"   # Change this per run
DATA_PATH = os.path.join('MP_Data')
no_sequences = 30
sequence_length = 30

# --- Setup ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Reduce webcam resolution for speed (optional)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Create directories if not exist
os.makedirs(DATA_PATH, exist_ok=True)
action_path = os.path.join(DATA_PATH, action_to_collect)
os.makedirs(action_path, exist_ok=True)
for sequence in range(no_sequences):
    os.makedirs(os.path.join(action_path, str(sequence)), exist_ok=True)

def extract_keypoints(results):
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        return np.array([[res.x, res.y, res.z] for res in hand.landmark]).flatten()
    else:
        return np.zeros(21*3)

# Use lightweight Hands model
with mp_hands.Hands(
    model_complexity=0,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    for sequence in range(no_sequences):
        print(f"⚡ Starting collection for {action_to_collect}, Video {sequence}")
        time.sleep(1)  # short pause before starting sequence

        for frame_num in range(sequence_length):
            ret, frame = cap.read()
            if not ret:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            # Draw landmarks (optional, comment out if still lagging)
            if results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

            # Minimal text overlay (less laggy than multiple putText)
            cv2.putText(frame, f'{action_to_collect} | Seq {sequence} | Frame {frame_num}',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow('Data Collection', frame)

            # Save keypoints
            keypoints = extract_keypoints(results)
            np.save(os.path.join(action_path, str(sequence), str(frame_num)), keypoints)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()

cap.release()
cv2.destroyAllWindows()
