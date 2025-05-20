import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import os

# Define the alphabet labels
ALPHABET = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

# Initialize MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def extract_hand_landmarks(image, hands_model):
    """
    Extract hand landmarks from an image using MediaPipe
    
    Args:
        image: RGB image
        hands_model: MediaPipe hands model
    
    Returns:
        landmarks_array: numpy array of landmarks (21x3) or None if no hand detected
    """
    results = hands_model.process(image)
    
    if not results.multi_hand_landmarks:
        return None
    
    # Get the first detected hand
    hand_landmarks = results.multi_hand_landmarks[0]
    
    # Convert to numpy array (21 landmarks with x, y, z coordinates)
    landmarks_array = np.zeros((21, 3))
    for i, landmark in enumerate(hand_landmarks.landmark):
        landmarks_array[i] = [landmark.x, landmark.y, landmark.z]
    
    return landmarks_array

def preprocess_landmarks(landmarks_array):
    """
    Normalize landmarks to make them translation and scale invariant
    
    Args:
        landmarks_array: numpy array of landmarks (21x3)
    
    Returns:
        normalized_landmarks: flattened and normalized landmarks
    """
    if landmarks_array is None:
        return None
    
    # Calculate center of palm (using wrist and middle finger MCP as reference)
    wrist = landmarks_array[0]
    middle_mcp = landmarks_array[9]  # Middle finger MCP
    palm_center = (wrist + middle_mcp) / 2
    
    # Translate to origin
    centered_landmarks = landmarks_array - palm_center
    
    # Scale normalization
    max_distance = np.max(np.linalg.norm(centered_landmarks, axis=1))
    normalized_landmarks = centered_landmarks / max_distance
    
    # Flatten to 1D array (63 values)
    return normalized_landmarks.flatten()

def draw_landmarks_on_image(image, hand_landmarks):
    """
    Draw hand landmarks and connections on the image
    
    Args:
        image: RGB image
        hand_landmarks: MediaPipe hand landmarks
    
    Returns:
        image_with_landmarks: image with landmarks drawn
    """
    if hand_landmarks is None:
        return image
    
    image_with_landmarks = image.copy()
    
    # Convert the image to BGR for drawing
    if len(image_with_landmarks.shape) == 3 and image_with_landmarks.shape[2] == 3:
        # Image is already RGB/BGR
        img_to_draw = image_with_landmarks.copy()
    else:
        # Convert grayscale to BGR
        img_to_draw = cv2.cvtColor(image_with_landmarks, cv2.COLOR_GRAY2BGR)
    
    # Draw landmarks and connections
    mp_drawing.draw_landmarks(
        img_to_draw,
        mp.solutions.hands.HandLandmark(hand_landmarks),
        mp_hands.HAND_CONNECTIONS
    )
    
    return img_to_draw

def display_prediction(frame, prediction, prob):
    """
    Display the predicted letter and confidence on the frame
    
    Args:
        frame: The frame to display on
        prediction: Predicted letter
        prob: Prediction confidence/probability
    
    Returns:
        frame with prediction text
    """
    # Create a rectangle at the top of the frame
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (245, 117, 16), -1)
    
    # Display prediction text
    prediction_text = f'Prediction: {prediction} ({prob:.2f})'
    cv2.putText(frame, prediction_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    return frame

def plot_training_history(history):
    """
    Plot training and validation accuracy/loss
    
    Args:
        history: Keras training history
    """
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()