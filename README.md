# Real-Time ASL Detection System

A real-time American Sign Language (ASL) detection system that recognizes hand gestures and converts them into text and speech. The system supports letters, numbers, and basic mathematical operations.

## Features

- Real-time hand gesture detection using MediaPipe
- Recognition of:
  - ASL alphabet (A-Z)
  - Numbers (0-9)
  - Mathematical operators (+, -, *, /)
  - Space and Delete commands
- Text-to-Speech conversion
- Real-time calculation of mathematical expressions
- Modern Tkinter-based GUI
- Reference guide for ASL signs
- Save conversation functionality

## 📁 Project Structure

real-time-asl-detection/
├── ASL_Dataset/         # Training data for gestures
├── static/             # Static assets 
│   └── asl_reference.jpg
├── main.py             # Main application with GUI
├── data_collector.py   # Data collection script
├── model_trainer.py    # ML model training script
├── asl_model.pkl       # Trained KNN model
└── requirements.txt    # Project dependencies


## Technical Details

- *Hand Detection*: MediaPipe Hands
- *Classification*: K-Nearest Neighbors (KNN)
- *GUI Framework*: Tkinter
- *Speech Engine*: pyttsx3
- *Computer Vision*: OpenCV-Python
- *Data Processing*: NumPy


## 📖 Usage Guide

1. *Launch Application*: Run main.py
2. *Position Hand*: Place your hand in the camera view
3. *Make Signs*: Form ASL signs within the detection box
4. *Available Actions*:
   - Click "Speak Sentence" to hear the text
   - Use "Clear Text" to reset
   - "Save Conversation" to export text