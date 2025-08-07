
import cv2
import numpy as np
import os
import mediapipe as mp

# --- !! CONFIGURATION: CHANGE THIS FOR EACH LETTER !! ---
# ---------------------------------------------------------
# Set the letter you want to collect data for.
# Run this script once for 'A', then change to 'B' and run again, etc.
action_to_collect = "B"
# ---------------------------------------------------------

# --- Setup ---
# Using the lightweight Hands model for better performance
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Path for exported data
DATA_PATH = os.path.join('MP_Data')

# Thirty videos worth of data
no_sequences = 30
# Videos are 30 frames in length
sequence_length = 30

# --- Folder Creation ---
# Create the main data folder if it doesn't exist
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)
# Create the folder for the specific action
action_path = os.path.join(DATA_PATH, action_to_collect)
if not os.path.exists(action_path):
    os.makedirs(action_path)

# Create folders for each sequence
for sequence in range(no_sequences):
    try:
        os.makedirs(os.path.join(action_path, str(sequence)))
    except FileExistsError:
        pass

# --- Data Collection Function ---
def extract_keypoints(results):
    if results.multi_hand_landmarks:
        # We are only using one hand, so we take the first one detected
        hand = results.multi_hand_landmarks[0]
        return np.array([[res.x, res.y, res.z] for res in hand.landmark]).flatten()
    else:
        # If no hand is detected, return an array of zeros
        return np.zeros(21*3)

# --- Main Collection Loop ---
cap = cv2.VideoCapture(0)
# Set up the MediaPipe Hands model for high performance
# max_num_hands=1 tells the model to only look for one hand, which is much faster.
with mp_hands.Hands(
    model_complexity=0,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    # Loop through sequences (videos)
    for sequence in range(no_sequences):
        # Loop through video length (sequence_length)
        for frame_num in range(sequence_length):
            ret, frame = cap.read()
            if not ret:
                break

            # Make detections
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw landmarks if a hand is detected
            if results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

            # Wait logic
            if frame_num == 0:
                cv2.putText(image, 'STARTING COLLECTION', (120,200),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                cv2.putText(image, f'Collecting frames for {action_to_collect} - Video No. {sequence}', (15,12),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', image)
                cv2.waitKey(2000)
            else:
                cv2.putText(image, f'Collecting frames for {action_to_collect} - Video No. {sequence}', (15,12),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', image)

            # Export keypoints
            keypoints = extract_keypoints(results)
            npy_path = os.path.join(DATA_PATH, action_to_collect, str(sequence), str(frame_num))
            np.save(npy_path, keypoints)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()

cap.release()
cv2.destroyAllWindows()