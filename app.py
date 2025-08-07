
import cv2
import numpy as np
import os
import mediapipe as mp
from tensorflow.keras.models import load_model

# --- Setup ---
# Using the lightweight Hands model for better performance
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Load the trained model and define the actions it was trained on
# Make sure this list is IDENTICAL to the one in your training script
actions = np.array(["A", "B", "C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"])
model = load_model(os.path.join('Trained_Model', 'asl_alphabet_lstm.h5'))

# --- Helper Function ---
def extract_keypoints(results):
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        return np.array([[res.x, res.y, res.z] for res in hand.landmark]).flatten()
    else:
        return np.zeros(21*3)

# --- Real-time Detection Logic ---
sequence = []
sentence = []
predictions = []
threshold = 0.8  # Confidence threshold

cap = cv2.VideoCapture(0)
# Set up the MediaPipe Hands model for high performance
with mp_hands.Hands(
    model_complexity=0,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Make detections
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw landmarks
        if results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

        # Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
            predictions.append(np.argmax(res))

            # Visualization logic
            if np.unique(predictions[-10:])[0] == np.argmax(res):
                if res[np.argmax(res)] > threshold:
                    cv2.putText(image, actions[np.argmax(res)], (3, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('OpenCV Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()