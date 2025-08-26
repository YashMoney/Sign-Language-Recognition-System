# ⚡ Real-Time Sign Language Recognition

<div align="center">

**A Python application for real-time American Sign Language (ASL) alphabet recognition using MediaPipe and LSTM neural networks.**

</div>

## 📖 Overview

This project provides a complete system for recognizing ASL alphabet gestures from a live webcam feed. It leverages high-performance hand tracking to capture the spatial data of hand landmarks and uses a Long Short-Term Memory (LSTM) network to understand the temporal sequence of a gesture.

The primary goal is to create a fast, accurate, and accessible tool that can serve as a foundational component for building more complex sign language translation applications, thereby helping to bridge the communication gap between the deaf and hearing communities. This repository is intended for developers, researchers, and students interested in computer vision, deep learning, and human-computer interaction.

## ✨ Features

* **Real-Time Performance:** Optimized for high FPS, providing smooth and immediate feedback on gesture recognition.
* **High-Fidelity Hand Tracking:** Utilizes Google's MediaPipe to accurately detect 21 3D hand landmarks, forming the basis for gesture analysis.
* **Sequence-Based Recognition:** Employs an LSTM model, which is ideal for interpreting time-series data like the motion involved in forming a sign.
* **Modular & Extensible Workflow:** The project is clearly divided into three distinct scripts for data collection, model training, and live prediction, making it easy to customize and extend with new signs.
* **Data-Driven:** Includes scripts to create your own custom datasets, allowing the model to be trained on diverse hand shapes, sizes, and signing styles.

## 🛠️ Technical Stack

* **Backend / Machine Learning Model:**
    * **Python:** Core programming language.
    * **TensorFlow / Keras:** For building, training, and running the LSTM model.
    * **MediaPipe:** For high-performance, real-time hand landmark detection.
    * **OpenCV:** For handling the webcam feed and rendering video frames.
    * **NumPy:** For efficient numerical and array operations.
    * **Scikit-learn:** For splitting data into training and testing sets.
* **Dataset:**
    * Custom-generated from webcam input, based on ASL alphabet signs.

## 🚀 Setup and Installation

### 1. Prerequisites

* Python 3.8+
* A connected webcam

### 2. Clone the Repository

```bash
git clone [https://github.com/YashMoney/Sign-Language-Recognition-System.git](https://github.com/YashMoney/Sign-Language-Recognition-System.git)
cd Sign-Language-Recognition-System
```

### 3. Create a Virtual Environment (Recommended)

```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies

Install all required packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

*(Note: If `requirements.txt` is not provided, you can install the packages individually)*:

```bash
pip install tensorflow opencv-python mediapipe scikit-learn
```

## Usage Guide

Follow these steps to collect data, train the model, and run the application.

### Step 1: Collect Gesture Data

The `collect_data_lstm.py` script captures the hand landmark data needed for training.

1.  Open `collect_data_lstm.py` in an editor.
2.  Modify the `action_to_collect` variable to the sign you want to record (e.g., `"A"`).
3.  Run the script from your terminal:
    ```bash
    python collect_data_lstm.py
    ```
4.  The script will guide you through recording 30 sequences for that sign. Repeat this for every sign you wish to include in your model. The data will be organized and saved in the `MP_Data/` directory.

### Step 2: Train the LSTM Model

The `train_model_lstm.py` script uses the collected data to train the neural network.

1.  Open `train_model_lstm.py`.
2.  Ensure the `actions` array contains all the signs you collected data for.
3.  Run the training script:
    ```bash
    python train_model_lstm.py
    ```
4.  This process will train the model, evaluate its performance, and save the best-performing version as `asl_alphabet_lstm.h5` in the `Trained_Model/` directory.

### Step 3: Run the Real-Time Application

The `app.py` script launches the live sign language recognizer.

1.  Ensure the trained model `asl_alphabet_lstm.h5` is present in the `Trained_Model/` directory.
2.  Run the application:
    ```bash
    python app.py
    ```
3.  A window will appear with your webcam feed. After a brief "Starting..." phase, it will be "Ready to detect." Form a sign with your hand, and the model will predict and display the corresponding letter.
4.  Press **'q'** to close the application.

## 🧠 Model Architecture

The core of this project is a **Long Short-Term Memory (LSTM)** network, a type of Recurrent Neural Network (RNN) well-suited for sequence data.

The model architecture typically includes:

* **Input Layer:** Expects a sequence of 30 frames, with each frame containing 63 data points (21 landmarks \* 3 coordinates). Shape: `(30, 63)`.
* **LSTM Layers:** Multiple LSTM layers (e.g., with 64 and 128 units) with `relu` activation to process the temporal patterns in the hand movements. `return_sequences=True` is used to pass the output of one LSTM layer to the next.
* **Dense Layers:** Fully connected layers that perform classification based on the features extracted by the LSTM layers.
* **Output Layer:** A `softmax` activation function across a Dense layer with a size equal to the number of signs (classes), which outputs the probability for each sign.

The model is compiled with `categorical_crossentropy` loss and the `Adam` optimizer, which are standard choices for multi-class classification tasks.

## 🔮 Future Enhancements

* **Expand Vocabulary:** Collect data for and train the model on numbers, words, and common phrases, not just individual letters.
* **Text-to-Sign:** Implement a feature where a user can type a word, and an animated model demonstrates how to sign it.
* **UI/UX Improvements:** Build a more user-friendly interface using a framework like Tkinter, PyQt, or Streamlit.
* **Deployment:** Package the application for deployment on web platforms (using Flask/Django) or as a standalone desktop application.
* **Two-Handed Gestures:** Extend MediaPipe's capabilities to track both hands and train the model to recognize two-handed signs.

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements or find any issues, please feel free to:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/YourFeature`).
3.  Commit your changes (`git commit -m 'Add some YourFeature'`).
4.  Push to the branch (`git push origin feature/YourFeature`).
5.  Open a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.

---

<div align="center">


</div>
