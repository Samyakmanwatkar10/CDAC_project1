import os
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)

# Parameters
SEQUENCE_LENGTH = 30
LANDMARKS_PER_HAND = 21 * 3
DATA_DIR = Path("data")
TRAIN_DIR = DATA_DIR / "asl_alphabet_train"
TEST_DIR = DATA_DIR / "asl_alphabet_test"
OUTPUT_DIR = DATA_DIR / "processed"
CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
IMG_SIZE = (64, 64)

def normalize_landmarks(landmarks):
    wrist = landmarks[0]
    normalized = []
    for lm in landmarks:
        norm_lm = [lm[0] - wrist[0], lm[1] - wrist[1], lm[2] - wrist[2]]
        normalized.extend(norm_lm)
    max_val = max([abs(coord) for coord in normalized] + [1e-5])
    return np.array([coord / max_val for coord in normalized])

def landmarks_to_image(landmarks):
    img = np.zeros(IMG_SIZE, dtype=np.float32)
    for i in range(0, len(landmarks), 3):
        x = int((landmarks[i] + 1) * IMG_SIZE[0] / 2)
        y = int((landmarks[i + 1] + 1) * IMG_SIZE[1] / 2)
        if 0 <= x < IMG_SIZE[0] and 0 <= y < IMG_SIZE[1]:
            img[y, x] = 1.0
    return img.reshape(*IMG_SIZE, 1)

def preprocess_image(image, is_mirrored=False, img_path="unknown"):
    if is_mirrored:
        image = cv2.flip(image, 1)
    # Enhance contrast to improve landmark detection
    image = cv2.convertScaleAbs(image, alpha=1.5, beta=20)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    if results.multi_hand_landmarks:
        landmarks = [[lm.x, lm.y, lm.z] for lm in results.multi_hand_landmarks[0].landmark]
        normalized = normalize_landmarks(landmarks)
        return landmarks_to_image(normalized)
    else:
        logger.warning(f"No landmarks detected in {img_path}")
        return None

def create_sequences(data_dir, split, is_test=False):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / split).mkdir(exist_ok=True)
    
    for class_name in CLASSES:
        class_dir = data_dir / class_name
        if not class_dir.exists():
            logger.error(f"Class directory {class_dir} does not exist")
            continue
        
        images = list(class_dir.glob("*.jpg"))
        if len(images) < SEQUENCE_LENGTH:
            logger.warning(f"Class {class_name} has only {len(images)} images, need {SEQUENCE_LENGTH}")
            continue
        
        logger.info(f"Processing {class_name} with {len(images)} images for {split}")
        np.random.shuffle(images)
        
        if not is_test:
            train_size = int(0.8 * len(images))
            image_list = images[:train_size] if split == "train" else images[train_size:]
        else:
            image_list = images
        
        for hand_type in ["right", "left"]:
            sequences = []
            valid_indices = []
            # Pre-check images to avoid wasting time on invalid ones
            for i in range(len(image_list)):
                img_path = image_list[i]
                img = cv2.imread(str(img_path))
                if img is None:
                    logger.error(f"Failed to load image {img_path}")
                    continue
                landmark_img = preprocess_image(img, is_mirrored=(hand_type == "left"), img_path=str(img_path))
                if landmark_img is not None:
                    valid_indices.append(i)
            
            # Generate sequences from valid indices
            for i in range(0, len(valid_indices) - SEQUENCE_LENGTH + 1, 1):
                sequence = []
                valid = True
                for j in range(SEQUENCE_LENGTH):
                    idx = valid_indices[i + j]
                    img_path = image_list[idx]
                    img = cv2.imread(str(img_path))
                    landmark_img = preprocess_image(img, is_mirrored=(hand_type == "left"), img_path=str(img_path))
                    if landmark_img is not None:
                        sequence.append(landmark_img)
                    else:
                        valid = False
                        break
                if valid and len(sequence) == SEQUENCE_LENGTH:
                    sequences.append(sequence)
                else:
                    logger.debug(f"Incomplete sequence for {class_name}_{hand_type} at index {i}")
            
            if sequences:
                np.save(OUTPUT_DIR / split / f"{class_name}_{hand_type}.npy", np.array(sequences))
                logger.info(f"Saved {len(sequences)} {hand_type} sequences for {class_name} in {split}")
            else:
                logger.warning(f"No sequences saved for {class_name}_{hand_type} in {split}")

def main():
    if TRAIN_DIR.exists():
        logger.info("Processing training data")
        create_sequences(TRAIN_DIR, "train", is_test=False)
        create_sequences(TRAIN_DIR, "test", is_test=False)
    
    if TEST_DIR.exists():
        logger.info("Processing test data")
        create_sequences(TEST_DIR, "test", is_test=True)
    
    hands.close()

if __name__ == "__main__":
    main()