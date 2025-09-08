import tkinter as tk
from tkinter import font, filedialog
import cv2
import mediapipe as mp
import numpy as np
import pickle
import pyttsx3
import time
from PIL import Image, ImageTk
import threading

class ASLApplication:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.configure(bg='#2c3e50')

        # --- Load Model ---
        try:
            with open('asl_model.pkl', 'rb') as f:
                model_data = pickle.load(f)
            self.model = model_data['model']
            self.labels = model_data['labels']
        except FileNotFoundError:
            print("Error: Model file 'asl_model.pkl' not found.")
            self.window.destroy()
            return

        # --- Initialize TTS with better voice settings ---
        self.is_speaking = False
        self.speech_queue = []
        self.speech_thread = None
        
        # --- Initialize MediaPipe ---
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # --- Camera ---
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            self.window.destroy()
            return

        # --- Sentence logic ---
        self.sentence = []
        self.last_prediction_time = time.time()
        self.prediction_delay = 1.5  # seconds

        # Add these variables for stability detection
        self.previous_landmarks = None
        self.stability_threshold = 0.01  # Adjust this value to control sensitivity
        self.stable_frames_required = 10  # Number of stable frames needed
        self.stable_frame_counter = 0

        # --- GUI Setup ---
        self.custom_font = font.Font(family="Helvetica", size=12)
        self.custom_font_bold = font.Font(family="Helvetica", size=14, weight="bold")

        main_frame = tk.Frame(self.window, bg='#2c3e50')
        main_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Add reference image frame on the left
        ref_frame = tk.Frame(main_frame, bg='#34495e', bd=2, relief=tk.SUNKEN)
        ref_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.Y)
        
        # Load and display reference image
        try:
            ref_img = Image.open("static/asl_reference.jpg")
            ref_img = ref_img.resize((200, 400))  # Adjust size as needed
            ref_photo = ImageTk.PhotoImage(ref_img)
            ref_label = tk.Label(ref_frame, image=ref_photo)
            ref_label.image = ref_photo  # Keep a reference
            ref_label.pack(fill=tk.BOTH, expand=True)
        except Exception as e:
            print(f"Error loading reference image: {e}")

        # Video panel
        video_frame = tk.Frame(main_frame, bg='#34495e', bd=2, relief=tk.SUNKEN)
        video_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.video_label = tk.Label(video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # Controls panel
        controls_frame = tk.Frame(main_frame, bg='#2c3e50', width=300)
        controls_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.Y)
        controls_frame.pack_propagate(False)

        self.text_box = tk.Text(
            controls_frame, height=15, width=35,
            font=self.custom_font, bg='#ecf0f1', fg='#2c3e50', wrap=tk.WORD
        )
        self.text_box.pack(pady=(0, 10), fill=tk.BOTH, expand=True)

        btn_speak = tk.Button(
            controls_frame, text="Speak Sentence",
            command=self.speak_sentence, font=self.custom_font_bold,
            bg='#2980b9', fg='white'
        )
        btn_speak.pack(fill=tk.X, pady=5)

        btn_clear = tk.Button(
            controls_frame, text="Clear Text",
            command=self.clear_text, font=self.custom_font_bold,
            bg='#c0392b', fg='white'
        )
        btn_clear.pack(fill=tk.X, pady=5)

        btn_save = tk.Button(
            controls_frame, text="Save Conversation",
            command=self.save_conversation, font=self.custom_font_bold,
            bg='#27ae60', fg='white'
        )
        btn_save.pack(fill=tk.X, pady=5)

        # --- Start update loop ---
        self.update()
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.mainloop()

    def setup_tts_engine(self):
        """Initialize or reinitialize the TTS engine with proper settings"""
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 130)
        self.engine.setProperty('volume', 0.9)
        voices = self.engine.getProperty('voices')
        female_voice = None
        for voice in voices:
            if "female" in voice.name.lower():
                female_voice = voice
                break
        self.engine.setProperty('voice', female_voice.id if female_voice else voices[0].id)
        self.engine.setProperty('pitch', 1.1)

    def update(self):
        success, frame = self.cap.read()
        if not success:
            return
        frame = cv2.flip(frame, 1)
        H, W, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            all_landmarks = np.zeros(126)
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness = results.multi_handedness[idx].classification[0].label
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                if handedness == 'Right':
                    all_landmarks[:63] = landmarks
                elif handedness == 'Left':
                    all_landmarks[63:] = landmarks

                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            # Check for stability
            is_stable = self.check_stability(all_landmarks)
            
            if is_stable:
                self.stable_frame_counter += 1
                if self.stable_frame_counter >= self.stable_frames_required:
                    prediction_index = self.model.predict([all_landmarks])[0]
                    predicted_sign = self.labels[prediction_index]

                    cv2.putText(frame, f"Prediction: {predicted_sign}", (50, 50),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                    current_time = time.time()
                    if current_time - self.last_prediction_time > self.prediction_delay:
                        if predicted_sign == 'space':
                            self.sentence.append(' ')
                        elif predicted_sign == 'del':
                            if self.sentence:
                                self.sentence.pop()
                        # Handle mathematical operators
                        elif predicted_sign in ['plus', 'minus', 'multiply', 'divide']:
                            # Convert operation words to symbols immediately
                            symbol_map = {
                                'plus': '+',
                                'minus': '-',
                                'multiply': '*',
                                'divide': '/'
                            }
                            self.sentence.append(symbol_map[predicted_sign])
                        else:
                            self.sentence.append(predicted_sign)
                        self.last_prediction_time = current_time

                        # Try calculation
                        try:
                            line = "".join(self.sentence)
                            # Look for number-operator-number pattern that isn't part of a completed equation
                            import re
                            # Skip patterns that are part of completed equations (n+n=n)
                            matches = re.finditer(r'(\d+[\s]*[+\-*/][\s]*\d+)(?!\s*=)', line)
                            for match in matches:
                                expr = match.group(1)
                                # Clean up the expression
                                clean_expr = expr.replace(' ', '')
                                
                                # Extract numbers and operator
                                if '+' in clean_expr:
                                    num1, num2 = map(float, clean_expr.split('+'))
                                    result = np.add(num1, num2)
                                    operator_word = "plus"
                                elif '-' in clean_expr:
                                    num1, num2 = map(float, clean_expr.split('-'))
                                    result = np.subtract(num1, num2)
                                    operator_word = "minus"
                                elif '*' in clean_expr:
                                    num1, num2 = map(float, clean_expr.split('*'))
                                    result = np.multiply(num1, num2)
                                    operator_word = "multiplied by"
                                elif '/' in clean_expr:
                                    num1, num2 = map(float, clean_expr.split('/'))
                                    result = np.divide(num1, num2)
                                    operator_word = "divided by"
                                
                                # Format and speak the result, add space after result
                                full_expression = f"{clean_expr} = {result} "  # Added space after result
                                # Replace the matched expression with the full expression
                                line = line.replace(expr, full_expression)
                                
                                # Speak the calculation
                                speech_text = f"{int(num1) if num1.is_integer() else num1} {operator_word} {int(num2) if num2.is_integer() else num2} equals {int(result) if result.is_integer() else result}"
                                
                                def speak_result():
                                    try:
                                        engine = pyttsx3.init()
                                        engine.setProperty('rate', 120)
                                        engine.setProperty('volume', 1.0)
                                        engine.say(speech_text)
                                        engine.runAndWait()
                                    except Exception as e:
                                        print(f"Speech Error: {str(e)}")
                                    finally:
                                        try:
                                            engine.stop()
                                        except:
                                            pass
                                
                                # Speak result in separate thread
                                thread = threading.Thread(target=speak_result, daemon=True)
                                thread.start()
                            
                            # Update the sentence with all calculated results
                            self.sentence = list(line)
                            
                        except Exception as e:
                            print(f"Calculation Error: {str(e)}")
                            pass
            else:
                self.stable_frame_counter = 0
                cv2.putText(frame, "Hand moving...", (50, 50),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            self.previous_landmarks = all_landmarks

        # Update sentence display
        self.text_box.delete("1.0", tk.END)
        self.text_box.insert(tk.END, "".join(self.sentence))

        # Show frame in Tkinter
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = img_tk
        self.video_label.configure(image=img_tk)

        self.window.after(10, self.update)

    def speak_sentence(self):
        """Handles the speak button click event"""
        text = self.text_box.get("1.0", tk.END).strip()
        if text:
            # Create and configure new engine instance for this request
            try:
                engine = pyttsx3.init()
                
                # Configure voice settings
                engine.setProperty('rate', 130)  # Slower speech rate
                engine.setProperty('volume', 0.9)
                
                # Try to set female voice if available
                voices = engine.getProperty('voices')
                for voice in voices:
                    if "female" in voice.name.lower():
                        engine.setProperty('voice', voice.id)
                        break
                
                # Speak in current thread to ensure completion
                engine.say(text)
                engine.runAndWait()
                
            except Exception as e:
                print(f"Speech Error: {str(e)}")
                # Try one more time with basic settings
                try:
                    engine = pyttsx3.init()
                    engine.say(text)
                    engine.runAndWait()
                except:
                    pass
            finally:
                # Always cleanup
                try:
                    engine.stop()
                    del engine
                except:
                    pass

    def clear_text(self):
        self.sentence = []
        self.text_box.delete("1.0", tk.END)

    def save_conversation(self):
        text_to_save = self.text_box.get("1.0", tk.END).strip()
        if text_to_save:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            if file_path:
                with open(file_path, 'w') as f:
                    f.write(text_to_save)

    def on_closing(self):
        self.cap.release()
        self.window.destroy()

    def check_stability(self, current_landmarks):
        if self.previous_landmarks is None:
            return False
            
        # Calculate movement between current and previous landmarks
        movement = np.mean(np.abs(current_landmarks - self.previous_landmarks))
        return movement < self.stability_threshold


if __name__ == "__main__":
    ASLApplication(tk.Tk(), "ASL Detector with Tkinter Frontend")
