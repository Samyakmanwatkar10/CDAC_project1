import os
import cv2
import numpy as np
import mediapipe as mp
import pickle
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import logging
from multiprocessing import Pool

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)

# Dataset paths
TRAIN_PATH = "D:/samyak/asl_recognition(new_proto)/data/asl_alphabet_train"
TEST_PATH = "D:/samyak/asl_recognition(new_proto)/data/asl_alphabet_test"
OUTPUT_DIR = "D:/samyak/as_recog_duplicate/processed_data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "processed_data.pkl")



# Parameters
MAX_IMAGES_PER_CLASS = None
NUM_FRAMES = 30
LABELS = [chr(i) for i in range(65, 91)]  # A-Z
NUM_CLASSES = len(LABELS)
label_to_index = {label: idx for idx, label in enumerate(LABELS)}
USE_TRAIN_SPLIT_FOR_TEST = False
IMAGE_SIZE = (100, 100)

def normalize_landmarks(landmarks):
    """Normalize landmarks relative to wrist."""
    wrist = landmarks[0]
    normalized = [[lm[0] - wrist[0], lm[1] - wrist[1], lm[2] - wrist[2]] for lm in landmarks]
    return normalized

def extract_landmarks(image, img_path, hands_processor):
    """Extract normalized hand landmarks from an image."""
    try:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands_processor.process(image_rgb)
        if results.multi_hand_landmarks:
            landmarks = [[lm.x, lm.y, lm.z] for lm in results.multi_hand_landmarks[0].landmark]
            return normalize_landmarks(landmarks)
        else:
            logging.warning(f"No landmarks detected in {img_path}")
            return None
    except Exception as e:
        logging.error(f"Error processing {img_path}: {str(e)}")
        return None

def process_image(args):
    """Process a single image for a class."""
    class_dir, img_name, max_images = args
    img_path = os.path.join(class_dir, img_name)
    try:
        image = cv2.imread(img_path)
        if image is None:
            logging.warning(f"Failed to load image: {img_path}")
            return None, None
        image = cv2.resize(image, IMAGE_SIZE)
        landmarks = extract_landmarks(image, img_path, hands)
        if landmarks is None:
            return None, None
        sequence = [landmarks] * NUM_FRAMES
        # Reshape to (30, 63): 30 frames, 21 landmarks * 3 coordinates
        sequence_array = np.array(sequence).reshape(NUM_FRAMES, -1)
        return sequence_array, label_to_index[os.path.basename(class_dir)]
    except Exception as e:
        logging.error(f"Error processing {img_path}: {str(e)}")
        return None, None

def process_split(data_dir, split_name, max_images_per_class):
    """Process images for a split (train or test)."""
    X, y = [], []
    if split_name == "test" and not os.path.isdir(data_dir):
        logging.warning(f"Test directory {data_dir} not found. Skipping test set.")
        return np.array([]), np.array([])
    
    for label in tqdm(LABELS, desc=f"Processing {split_name} classes"):
        class_dir = os.path.join(data_dir, label)
        if not os.path.exists(class_dir):
            logging.warning(f"Class directory not found: {class_dir}")
            continue
        img_names = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if max_images_per_class:
            img_names = img_names[:max_images_per_class]
        args = [(class_dir, img_name, max_images_per_class) for img_name in img_names]
        with Pool() as pool:
            results = pool.map(process_image, args)
        for sequence, label_idx in results:
            if sequence is not None and label_idx is not None:
                X.append(sequence)
                y.append(label_idx)
    return np.array(X), np.array(y)

def preprocess_dataset():
    """Preprocess the entire dataset."""
    logging.info("Starting preprocessing...")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logging.info(f"Created directory: {OUTPUT_DIR}")

    # Process training data
    logging.info("Processing training data...")
    X_train_full, y_train_full = process_split(TRAIN_PATH, "train", MAX_IMAGES_PER_CLASS)
    
    if USE_TRAIN_SPLIT_FOR_TEST:
        X_train_temp, X_test, y_train_temp, y_test = train_test_split(
            X_train_full, y_train_full, test_size=0.1, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_temp, y_train_temp, test_size=0.2, random_state=42
        )
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=0.2, random_state=42
        )
        logging.info("Processing test data...")
        X_test, y_test = process_split(TEST_PATH, "test", None)
    
    logging.info(f"Train samples: {len(X_train)}, Validation samples: {len(X_val)}, Test samples: {len(X_test)}")
    if len(X_train) == 0 or len(X_val) == 0:
        logging.error("No training or validation data generated.")
        raise ValueError("Empty dataset")
    if len(X_test) == 0:
        logging.warning("No test data generated. Using train split or check test directory.")
    
    data = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test
    }
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(data, f)
    logging.info(f"Processed data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    try:
        preprocess_dataset()
    except Exception as e:
        logging.error(f"Preprocessing failed: {str(e)}")
    finally:
        hands.close()