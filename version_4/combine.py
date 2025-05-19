import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
import logging
import zipfile
import os
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)

# ASL Alphabet Classes
CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
SEQUENCE_LENGTH = 30
IMG_SIZE = (64, 64, 1)
LEFT_MODEL_PATH = Path("D:/samyak/got_it_left/models/asl_model1.keras")
RIGHT_MODEL_PATH = Path("D:/samyak/as_recog_duplicate/models/asl_model2.keras")
FRAME_WIDTH, FRAME_HEIGHT = 640, 480
DEBUG_MODE = True  # Print model summaries and debug info

def create_dummy_model():
    """Create a dummy model for testing prediction logic."""
    model = Sequential([
        Dense(64, activation='relu', input_shape=(SEQUENCE_LENGTH * 64 * 64,)),
        Dense(len(CLASSES), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    logger.warning("Using dummy model; predictions will be random")
    return model

def validate_keras_file(filepath):
    """Check if the file exists, is readable, and is a valid ZIP archive."""
    filepath = Path(filepath)
    if not filepath.exists():
        logger.error(f"File does not exist: {filepath}")
        return False
    if not filepath.is_file():
        logger.error(f"Path is not a file: {filepath}")
        return False
    try:
        file_size = filepath.stat().st_size / (1024 * 1024)  # Size in MB
        logger.info(f"File size: {file_size:.2f} MB for {filepath}")
        if not os.access(filepath, os.R_OK):
            logger.error(f"No read permission for {filepath}")
            return False
        with open(filepath, 'rb') as f:
            f.seek(0)
        with zipfile.ZipFile(filepath, 'r') as zf:
            if zf.testzip() is not None:
                logger.error(f"Corrupted ZIP archive: {filepath}")
                return False
        logger.info(f"Valid .keras file: {filepath}")
        return True
    except (IOError, zipfile.BadZipFile, OSError) as e:
        logger.error(f"Invalid or corrupted .keras file: {filepath}, error: {e}")
        return False

def list_directory(filepath):
    """Log contents of the directory containing the file."""
    directory = Path(filepath).parent
    try:
        files = [f.name for f in directory.iterdir() if f.is_file()]
        logger.info(f"Files in {directory}: {', '.join(files)}")
    except Exception as e:
        logger.error(f"Failed to list directory {directory}: {e}")

def normalize_landmarks(landmarks):
    wrist = landmarks[0]
    normalized = []
    for lm in landmarks:
        norm_lm = [lm[0] - wrist[0], lm[1] - wrist[1], lm[2] - wrist[2]]
        normalized.extend(norm_lm)
    max_val = max([abs(coord) for coord in normalized] + [1e-5])
    return np.array([coord / max_val for coord in normalized])

def landmarks_to_image(landmarks):
    img = np.zeros((64, 64), dtype=np.float32)
    for i in range(0, len(landmarks), 3):
        x = int((landmarks[i] + 1) * 32)
        y = int((landmarks[i + 1] + 1) * 32)
        if 0 <= x < 64 and 0 <= y < 64:
            img[y, x] = 1.0
    return img.reshape(64, 64, 1)

def preprocess_frame(frame, is_mirrored=False):
    if is_mirrored:
        frame = cv2.flip(frame, 1)
    frame = cv2.convertScaleAbs(frame, alpha=1.5, beta=20)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks and results.multi_handedness:
        hand_landmarks = results.multi_hand_landmarks[0]
        handedness = results.multi_handedness[0].classification[0].label  # 'Left' or 'Right'
        landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
        normalized = normalize_landmarks(landmarks)
        return landmarks_to_image(normalized), handedness
    return None, None

def main():
    # List directory contents
    list_directory(LEFT_MODEL_PATH)
    list_directory(RIGHT_MODEL_PATH)
    
    # Validate and load models
    left_model = None
    right_model = None
    
    # Try loading as .keras
    if validate_keras_file(LEFT_MODEL_PATH):
        try:
            logger.info(f"Loading left-hand model from {LEFT_MODEL_PATH} as .keras")
            left_model = tf.keras.models.load_model(LEFT_MODEL_PATH)
            if DEBUG_MODE:
                logger.info(f"Left model input shape: {left_model.input_shape}")
        except Exception as e:
            logger.error(f"Failed to load left-hand model as .keras: {e}")
    else:
        logger.warning("Left-hand model (.keras) not loaded")
    
    # Try loading as .h5 if .keras fails
    if left_model is None and LEFT_MODEL_PATH.with_suffix('.h5').exists():
        try:
            logger.info(f"Attempting to load left-hand model from {LEFT_MODEL_PATH.with_suffix('.h5')} as .h5")
            left_model = tf.keras.models.load_model(LEFT_MODEL_PATH.with_suffix('.h5'), compile=False)
            if DEBUG_MODE:
                logger.info(f"Left model input shape: {left_model.input_shape}")
        except Exception as e:
            logger.error(f"Failed to load left-hand model as .h5: {e}")
    
    if left_model is None:
        logger.warning("Left-hand model not loaded; left-hand predictions will use dummy model")
        left_model = create_dummy_model()
    
    # Same for right model
    if validate_keras_file(RIGHT_MODEL_PATH):
        try:
            logger.info(f"Loading right-hand model from {RIGHT_MODEL_PATH} as .keras")
            right_model = tf.keras.models.load_model(RIGHT_MODEL_PATH)
            if DEBUG_MODE:
                logger.info(f"Right model input shape: {right_model.input_shape}")
        except Exception as e:
            logger.error(f"Failed to load right-hand model as .keras: {e}")
    
    if right_model is None and RIGHT_MODEL_PATH.with_suffix('.h5').exists():
        try:
            logger.info(f"Attempting to load right-hand model from {RIGHT_MODEL_PATH.with_suffix('.h5')} as .h5")
            right_model = tf.keras.models.load_model(RIGHT_MODEL_PATH.with_suffix('.h5'), compile=False)
            if DEBUG_MODE:
                logger.info(f"Right model input shape: {right_model.input_shape}")
        except Exception as e:
            logger.error(f"Failed to load right-hand model as .h5: {e}")
    
    if right_model is None:
        logger.warning("Right-hand model not loaded; right-hand predictions will use dummy model")
        right_model = create_dummy_model()
    
    # Warn about different directories
    if LEFT_MODEL_PATH.parent != RIGHT_MODEL_PATH.parent:
        logger.warning(f"Models are in different directories: {LEFT_MODEL_PATH.parent} and {RIGHT_MODEL_PATH.parent}")
        logger.warning("Consider consolidating to D:/samyak/got_it_left/models/")
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Failed to open webcam")
        raise RuntimeError("Webcam not accessible")
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    sequence = []
    prediction = "None"
    hand_type = "Unknown"
    prev_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        frame=cv2.flip(frame, 1)
        if not ret:
            logger.warning("Failed to capture frame")
            break
        
        # Process frame
        landmark_image, detected_hand = preprocess_frame(frame, is_mirrored=False)
        
        if landmark_image is not None and detected_hand is not None:
            sequence.append(landmark_image)
            hand_type = detected_hand
            if len(sequence) > SEQUENCE_LENGTH:
                sequence.pop(0)
            
            # Predict when sequence is complete
            if len(sequence) == SEQUENCE_LENGTH:
                sequence_array = np.array(sequence).reshape(1, SEQUENCE_LENGTH, 64, 64, 1)
                try:
                    if hand_type == "Left" and left_model is not None:
                        # Flatten input for dummy model
                        if isinstance(left_model, Sequential):
                            pred = left_model.predict(sequence_array.reshape(1, -1), verbose=0)
                        else:
                            pred = left_model.predict(sequence_array, verbose=0)
                        pred_idx = np.argmax(pred, axis=1)[0]
                        prediction = CLASSES[pred_idx]
                    elif hand_type == "Right" and right_model is not None:
                        if isinstance(right_model, Sequential):
                            pred = right_model.predict(sequence_array.reshape(1, -1), verbose=0)
                        else:
                            pred = right_model.predict(sequence_array, verbose=0)
                        pred_idx = np.argmax(pred, axis=1)[0]
                        prediction = CLASSES[pred_idx]
                    else:
                        prediction = "Model not loaded"
                        logger.warning(f"No model available for {hand_type} hand")
                    logger.info(f"Hand: {hand_type}, Predicted ASL Alphabet: {prediction}")
                except Exception as e:
                    logger.warning(f"Prediction failed: {e}")
                    prediction = "Error"
        else:
            logger.debug("No hand detected in frame")
        
        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        
        # Display on frame
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Hand: {hand_type}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Sign: {prediction}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw landmarks if detected
        if landmark_image is not None:
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        cv2.imshow("ASL Alphabet Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == "__main__":
    main()