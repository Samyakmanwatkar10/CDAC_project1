import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
from utils import extract_hand_landmarks, preprocess_landmarks, ALPHABET
import pickle
import os

class SignLanguagePredictor:
    def __init__(self, model_path='sign_language_model.h5', alphabet_path='processed_data/alphabet.pkl'):
        """
        Initialize the sign language predictor
        
        Args:
            model_path: Path to the trained model
            alphabet_path: Path to the alphabet mapping
        """
        # Load the model
        self.model = tf.keras.models.load_model(model_path)
        
        # Load alphabet mapping if exists
        if os.path.exists(alphabet_path):
            with open(alphabet_path, 'rb') as f:
                self.alphabet = pickle.load(f)
        else:
            self.alphabet = ALPHABET
        
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        self.mp_drawing = mp.solutions.drawing_utils
    
    def predict_from_image(self, image):
        """
        Predict the sign language letter from an image
        
        Args:
            image: RGB image
        
        Returns:
            letter: Predicted letter
            confidence: Prediction confidence
            processed_image: Image with hand landmarks drawn
        """
        # Convert image to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
        
        # Extract hand landmarks
        results = self.hands.process(rgb_image)
        processed_image = image.copy()
        
        if not results.multi_hand_landmarks:
            return None, 0.0, processed_image
        
        # Draw landmarks on image
        for hand_landmarks in results.multi_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                processed_image,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS
            )
        
        # Extract and preprocess landmarks
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Convert to numpy array (21 landmarks with x, y, z coordinates)
        landmarks_array = np.zeros((21, 3))
        for i, landmark in enumerate(hand_landmarks.landmark):
            landmarks_array[i] = [landmark.x, landmark.y, landmark.z]
        
        # Preprocess landmarks
        processed_landmarks = preprocess_landmarks(landmarks_array)
        
        if processed_landmarks is None:
            return None, 0.0, processed_image
        
        # Reshape for prediction
        processed_landmarks = np.expand_dims(processed_landmarks, axis=0)
        
        # Make prediction
        predictions = self.model.predict(processed_landmarks)[0]
        
        # Get the predicted class index and confidence
        predicted_idx = np.argmax(predictions)
        confidence = predictions[predicted_idx]
        
        # Convert to letter
        predicted_letter = self.alphabet[predicted_idx]
        
        return predicted_letter, confidence, processed_image
    
    def close(self):
        """Close the MediaPipe hands model"""
        self.hands.close()