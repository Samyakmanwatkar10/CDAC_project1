import os
import cv2
import numpy as np
from utils import extract_hand_landmarks, preprocess_landmarks, ALPHABET
import mediapipe as mp
from sklearn.model_selection import train_test_split
import pickle
from tqdm import tqdm

def process_dataset(data_dir, output_dir='processed_data', test_size=0.2):
    """
    Process the sign language dataset and extract landmarks using MediaPipe
    
    Args:
        data_dir: Directory containing train and test folders
        output_dir: Directory to save processed data
        test_size: Test split ratio for the training data
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize MediaPipe hands model
    hands = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    )
    
    # Process training and testing data
    for dataset_type in ['train', 'test']:
        print(f"Processing {dataset_type} data...")
        dataset_path = os.path.join(data_dir, dataset_type)
        
        if not os.path.exists(dataset_path):
            print(f"Directory {dataset_path} not found. Skipping.")
            continue
            
        X = []  # Features (hand landmarks)
        y = []  # Labels (alphabet letters)
        
        # Loop through each alphabet folder
        for idx, letter in enumerate(ALPHABET):
            letter_dir = os.path.join(dataset_path, letter)
            
            if not os.path.exists(letter_dir):
                print(f"Directory for letter {letter} not found. Skipping.")
                continue
                
            print(f"Processing letter {letter}...")
            
            # Process each image in the letter folder
            image_files = [f for f in os.listdir(letter_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            for img_file in tqdm(image_files):
                img_path = os.path.join(letter_dir, img_file)
                
                # Read and process image
                image = cv2.imread(img_path)
                if image is None:
                    print(f"Could not read image: {img_path}")
                    continue
                
                # Convert to RGB for MediaPipe
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Extract hand landmarks
                landmarks = extract_hand_landmarks(image_rgb, hands)
                
                if landmarks is not None:
                    # Preprocess landmarks (normalize and flatten)
                    processed_landmarks = preprocess_landmarks(landmarks)
                    
                    if processed_landmarks is not None:
                        X.append(processed_landmarks)
                        y.append(idx)  # Use index as label
        
        if not X:
            print(f"No valid data found for {dataset_type}.")
            continue
            
        X = np.array(X)
        y = np.array(y)
        
        # If this is training data, split into train and validation
        if dataset_type == 'train':
            # Split the data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            # Save the data
            np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
            np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
            np.save(os.path.join(output_dir, 'X_val.npy'), X_val)
            np.save(os.path.join(output_dir, 'y_val.npy'), y_val)
            
            print(f"Saved training data: {X_train.shape[0]} samples")
            print(f"Saved validation data: {X_val.shape[0]} samples")
        else:
            # Save test data
            np.save(os.path.join(output_dir, 'X_test.npy'), X)
            np.save(os.path.join(output_dir, 'y_test.npy'), y)
            print(f"Saved test data: {X.shape[0]} samples")
    
    # Save the alphabet mapping
    with open(os.path.join(output_dir, 'alphabet.pkl'), 'wb') as f:
        pickle.dump(ALPHABET, f)
    
    # Close MediaPipe hands model
    hands.close()
    
    print("Dataset processing complete!")

if __name__ == "__main__":
    process_dataset('data')