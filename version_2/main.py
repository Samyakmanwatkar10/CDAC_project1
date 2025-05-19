import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils

# Load trained model
MODEL_PATH = "D:/samyak/ASL_left_hand/models/asl_model2.keras"
try:
    if not os.path.exists(MODEL_PATH):
        logging.error(f"Model file not found at {MODEL_PATH}. Run train.py first.")
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    logging.info(f"Model loaded from {MODEL_PATH}")
except Exception as e:
    logging.error(f"Error loading model: {str(e)}")
    raise

# Parameters
NUM_FRAMES = 30
LABELS = [chr(i) for i in range(65, 91)]  # A-Z
CONFIDENCE_THRESHOLD = 0.6
SMOOTHING_WINDOW = 5

# Initialize variables
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    logging.error("Could not open webcam")
    raise RuntimeError("Webcam not accessible")
sequence = []
recent_predictions = deque(maxlen=SMOOTHING_WINDOW)
displayed_letter = "None"
last_confidence = 0.0
display_frames = 0

def normalize_landmarks(landmarks):
    """Normalize landmarks relative to wrist."""
    wrist = landmarks[0]
    normalized = [[lm[0] - wrist[0], lm[1] - wrist[1], lm[2] - wrist[2]] for lm in landmarks]
    return normalized

def get_majority_prediction(predictions):
    """Return the most common prediction."""
    if not predictions:
        return None, 0.0
    pred_counts = np.bincount(predictions, minlength=len(LABELS))
    majority_pred = np.argmax(pred_counts)
    confidence = pred_counts[majority_pred] / len(predictions)
    return majority_pred, confidence

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        logging.warning("Failed to capture frame from webcam")
        break
    
    # Process frame with MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    # Draw landmarks and process landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
            landmarks = normalize_landmarks(landmarks)
            sequence.append(landmarks)
            
            # Predict when sequence is full
            if len(sequence) == NUM_FRAMES:
                sequence_array = np.array(sequence).reshape(1, NUM_FRAMES, 63)
                try:
                    prediction = model.predict(sequence_array, verbose=0)
                    predicted_class = np.argmax(prediction, axis=1)[0]
                    confidence = np.max(prediction)
                    
                    recent_predictions.append(predicted_class)
                    majority_pred, majority_conf = get_majority_prediction(recent_predictions)
                    
                    if majority_pred is not None and majority_conf >= CONFIDENCE_THRESHOLD:
                        displayed_letter = LABELS[majority_pred]
                        last_confidence = majority_conf
                        display_frames = 30  # Display for ~1 second
                    else:
                        display_frames = max(0, display_frames - 1)
                    
                    logging.info(f"Prediction: {displayed_letter}, Confidence: {last_confidence:.2f}")
                except Exception as e:
                    logging.error(f"Error during prediction: {str(e)}")
                
                sequence = sequence[5:]  # Slide window
    
    else:
        display_frames = max(0, display_frames - 1)
        logging.info("No hand detected")
    
    # Display prediction
    if display_frames > 0:
        cv2.putText(frame, f"Letter: {displayed_letter} ({last_confidence:.2f})", 
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('ASL Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
logging.info("Real-time inference terminated")


# #new code 
# import cv2
# import numpy as np
# import mediapipe as mp
# from tensorflow.keras.models import load_model
# from collections import deque
# import logging
# import os

# # Setup logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # Initialize MediaPipe Hands
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
# mp_drawing = mp.solutions.drawing_utils

# # Load trained model
# MODEL_PATH = "D:/samyak/asl_recognition/models/asl_model.keras"
# try:
#     if not os.path.exists(MODEL_PATH):
#         logging.error(f"Model file not found at {MODEL_PATH}. Run train.py first.")
#         raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
#     model = load_model(MODEL_PATH)
#     logging.info(f"Model loaded from {MODEL_PATH}")
# except Exception as e:
#     logging.error(f"Error loading model: {str(e)}")
#     raise

# # Parameters
# NUM_FRAMES = 30
# LABELS = [chr(i) for i in range(65, 91)]  # A-Z
# CONFIDENCE_THRESHOLD = 0.5
# SMOOTHING_WINDOW = 10
# DISPLAY_SIZE = (640, 480)  # Size for display window

# # Initialize variables
# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# if not cap.isOpened():
#     logging.error("Could not open webcam")
#     raise RuntimeError("Webcam not accessible")
# sequence = []
# recent_predictions = deque(maxlen=SMOOTHING_WINDOW)
# displayed_letter = "None"
# last_confidence = 0.0
# display_frames = 0

# def normalize_landmarks(landmarks):
#     """Normalize landmarks relative to wrist."""
#     wrist = landmarks[0]
#     normalized = [[lm[0] - wrist[0], lm[1] - wrist[1], lm[2] - wrist[2]] for lm in landmarks]
#     return normalized

# def get_majority_prediction(predictions):
#     """Return the most common prediction."""
#     if not predictions:
#         return None, 0.0
#     pred_counts = np.bincount(predictions, minlength=len(LABELS))
#     majority_pred = np.argmax(pred_counts)
#     confidence = pred_counts[majority_pred] / len(predictions)
#     return majority_pred, confidence

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         logging.warning("Failed to capture frame from webcam")
#         break
    
#     # Create a copy for display
#     display_frame = frame.copy()
#     display_frame = cv2.resize(display_frame, DISPLAY_SIZE)
    
#     # Resize frame for model input to match training preprocessing
#     model_frame = cv2.resize(frame, (50, 50))
#     frame_rgb = cv2.cvtColor(model_frame, cv2.COLOR_BGR2RGB)
#     results = hands.process(frame_rgb)
    
#     # Draw landmarks and process landmarks
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             # Draw landmarks on display frame (scaled to display size)
#             mp_drawing.draw_landmarks(display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#             landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
#             landmarks = normalize_landmarks(landmarks)
#             logging.info(f"Landmarks shape: {np.array(landmarks).shape}")
#             sequence.append(landmarks)
            
#             # Predict when sequence is full
#             if len(sequence) >= NUM_FRAMES:
#                 sequence = sequence[-NUM_FRAMES:]  # Keep latest 30 frames
#                 sequence_array = np.array(sequence).reshape(1, NUM_FRAMES, 63)
#                 try:
#                     prediction = model.predict(sequence_array, verbose=0)
#                     predicted_class = np.argmax(prediction, axis=1)[0]
#                     confidence = np.max(prediction)
                    
#                     recent_predictions.append(predicted_class)
#                     majority_pred, majority_conf = get_majority_prediction(recent_predictions)
                    
#                     if majority_pred is not None and majority_conf >= CONFIDENCE_THRESHOLD:
#                         displayed_letter = LABELS[majority_pred]
#                         last_confidence = majority_conf
#                         display_frames = 30  # Display for ~1 second
#                     else:
#                         display_frames = max(0, display_frames - 1)
                    
#                     logging.info(f"Prediction: {displayed_letter}, Confidence: {last_confidence:.2f}")
#                 except Exception as e:
#                     logging.error(f"Error during prediction: {str(e)}")
    
#     else:
#         display_frames = max(0, display_frames - 1)
#         logging.info("No hand detected")
    
#     # Display prediction on larger frame
#     if display_frames > 0:
#         cv2.putText(display_frame, f"Letter: {displayed_letter} ({last_confidence:.2f})", 
#                     (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
#     cv2.imshow('ASL Recognition', display_frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
# hands.close()
# logging.info("Real-time inference terminated")