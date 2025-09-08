# model_trainer.py
import os
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pickle

# --- Configuration ---
DATA_PATH = 'ASL_Dataset'
MODEL_PATH = 'asl_model.pkl'

# --- Load and Prepare Data ---
X = []  # Features (landmark data)
y = []  # Labels (sign names)

signs = os.listdir(DATA_PATH)

for i, sign in enumerate(signs):
    sign_path = os.path.join(DATA_PATH, sign)
    
    if not os.path.isdir(sign_path):
        continue
    
    for file_name in os.listdir(sign_path):
        file_path = os.path.join(sign_path, file_name)
        
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            landmarks = list(reader)[0]
            X.append([float(val) for val in landmarks])
            y.append(i) # Use the index of the sign folder as the label

print(f"Loaded {len(X)} samples from {len(signs)} signs.")

X = np.array(X)
y = np.array(y)

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# --- Model Training ---
model = KNeighborsClassifier(n_neighbors=5)

print("Training the model...")
model.fit(X_train, y_train)
print("Model training complete.")

# --- Model Evaluation ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# --- Save the Model ---
with open(MODEL_PATH, 'wb') as f:
    pickle.dump({'model': model, 'labels': signs}, f)

print(f"Model saved successfully to {MODEL_PATH}")
