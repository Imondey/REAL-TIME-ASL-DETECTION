# data_collector.py
import cv2
import mediapipe as mp
import numpy as np
import os
import csv

# --- Configuration ---
DATA_PATH = os.path.join('ASL_Dataset')
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

# Ask user which sign to recollect
print("Available signs:")
print("Letters:", ' '.join(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']))
print("Numbers:", ' '.join(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']))
print("Operations:", 'plus, minus, multiply, divide, space, del')

sign_to_collect = input("Enter the sign you want to collect data for: ").strip()
signs = [sign_to_collect]  # Only collect for one sign
num_samples = 250

# --- MediaPipe Initialization ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# --- Main Data Collection Loop ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Check if sign folder exists
sign_path = os.path.join(DATA_PATH, sign_to_collect)
if os.path.exists(sign_path):
    response = input(f"Folder for '{sign_to_collect}' already exists. Do you want to replace it? (y/n): ")
    if response.lower() == 'y':
        import shutil
        shutil.rmtree(sign_path)
    else:
        print("Data collection cancelled.")
        cap.release()
        cv2.destroyAllWindows()
        exit()

os.makedirs(sign_path, exist_ok=True)

for sign in signs:
    print(f'Collecting data for sign: {sign}')

    sample_count = 0
    while sample_count < num_samples:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        H, W, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            all_landmarks = np.zeros(126) # 63 for right hand, 63 for left

            # Iterate through each detected hand
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness = results.multi_handedness[idx].classification[0].label
                
                # Extract landmarks
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                
                # Place landmarks in the correct slot based on handedness
                if handedness == 'Right':
                    all_landmarks[:63] = landmarks
                elif handedness == 'Left':
                    all_landmarks[63:] = landmarks

                # Add visual feedback to confirm detection
                wrist_coords = tuple(np.multiply(
                    [hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y],
                    [W, H]
                ).astype(int))
                
                cv2.putText(frame, handedness, wrist_coords, 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            # Save landmarks to file
            file_path = os.path.join(sign_path, f'{sample_count}.csv')
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(all_landmarks)
            
            sample_count += 1
            
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        cv2.putText(frame, f'Collecting {sign}: {sample_count}/{num_samples}', 
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Data Collection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

print("Data collection complete!")
cap.release()
cv2.destroyAllWindows()
