# app.py

import cv2
import numpy as np
import os
import mediapipe as mp
from collections import deque
from tensorflow.keras.models import load_model
import time

# --- Setup ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

actions = np.array(["A", "B", "C", "D", "E", "F"])
# Ensure you are using the simplified, faster model
model = load_model(os.path.join("Trained_Model", "asl_alphabet_lstm.h5"))

# --- Helper Function ---
def extract_keypoints(results):
    """Extracts hand landmarks and returns a flattened numpy array."""
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        return np.array([[res.x, res.y, res.z] for res in hand.landmark]).flatten()
    return np.zeros(21 * 3)

# --- Real-time Detection Variables ---
sequence = deque(maxlen=30)         # A rolling window of 30 frames for the model input
predictions_history = deque(maxlen=10) # A short history of predictions for smoothing
last_label = ""
last_conf = 0.0
frame_counter = 0                   # Counter to control prediction frequency

# --- Constants for Detection Logic (Tune if needed) ---
CONFIDENCE_THRESHOLD = 0.75         # Minimum confidence to consider a prediction valid.
STABILITY_FRAMES = 5                # How many consecutive identical predictions to be considered "stable".
PREDICTION_INTERVAL = 3             # Run prediction every 3 frames to boost FPS.

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prev_time = time.time()

with mp_hands.Hands(
    model_complexity=0,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1
        # Process frame for detection
        frame = cv2.flip(frame, 1)
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Draw landmarks if a hand is visible
        if results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

        # Always update the sequence with the latest keypoints
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)

        # --- Continuous Prediction Logic ---
        # Only run the expensive prediction logic periodically to save resources and boost FPS.
        if len(sequence) == 30 and frame_counter % PREDICTION_INTERVAL == 0:
            res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
            max_prob = np.max(res)
            current_prediction = actions[np.argmax(res)]

            # Check if the prediction is confident enough
            if max_prob > CONFIDENCE_THRESHOLD:
                predictions_history.append(current_prediction)

                # NEW SMOOTHING LOGIC: Check for stability over the last few frames
                if len(predictions_history) >= STABILITY_FRAMES:
                    # Check if the last STABILITY_FRAMES predictions are all the same
                    if len(set(list(predictions_history)[-STABILITY_FRAMES:])) == 1:
                        # If stable, update the displayed label
                        last_label = current_prediction
                        last_conf = max_prob
            else:
                # If confidence drops, clear the history and the displayed label
                predictions_history.clear()
                last_label = ""

        # Display the prediction on the screen
        # MODIFIED: Removed the "else" block that showed "Starting..." and "Ready..."
        if last_label != "":
            cv2.putText(frame, f"{last_label} ({last_conf*100:.1f}%)",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 0, 0), 2, cv2.LINE_AA)

        # Calculate and display FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)

        cv2.imshow("ASL Detection", frame)

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
