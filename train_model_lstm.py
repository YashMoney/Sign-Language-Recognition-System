# train_model_lstm.py
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint

# --- Load Data and Preprocess ---
DATA_PATH = os.path.join("MP_Data")
actions = np.array(list("ABCDEF"))
no_sequences = 30
sequence_length = 30

label_map = {label: num for num, label in enumerate(actions)}

sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy"))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# --- Build SIMPLIFIED LSTM Model (for faster predictions) ---
log_dir = os.path.join("Logs")
os.makedirs("Trained_Model", exist_ok=True)

model = Sequential()
# MODIFIED: Reduced LSTM units to match paper's architecture for better performance[cite: 233].
model.add(LSTM(64, return_sequences=True, activation="relu", input_shape=(30, 63)))
model.add(LSTM(128, return_sequences=True, activation="relu"))
model.add(LSTM(64, return_sequences=False, activation="relu"))

# Dense layers for classification
model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(actions.shape[0], activation="softmax"))

model.compile(
    optimizer="Adam", # Adam optimizer is an effective choice[cite: 172].
    loss="categorical_crossentropy",
    metrics=["categorical_accuracy"],
)

model.summary()

# --- Callbacks ---
tb_callback = TensorBoard(log_dir=log_dir)
early_stop = EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True)
checkpoint = ModelCheckpoint(
    os.path.join("Trained_Model", "asl_alphabet_lstm.h5"),
    monitor="val_loss",
    save_best_only=True,
    verbose=1,
)

# --- Train Model ---
# The paper used 2000 epochs, but EarlyStopping is a modern technique to prevent overfitting[cite: 168].
history = model.fit(
    X_train,
    y_train,
    epochs=2000,
    validation_split=0.1,
    callbacks=[tb_callback, early_stop, checkpoint],
    batch_size=32,
)

print("\n✅ Training complete. Best model saved as asl_alphabet_lstm.h5")

# --- Evaluate on Test Set ---
y_pred = model.predict(X_test)
y_true_classes = np.argmax(y_test, axis=1)
y_pred_classes = np.argmax(y_pred, axis=1)

print("\n🔹 Classification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=actions))