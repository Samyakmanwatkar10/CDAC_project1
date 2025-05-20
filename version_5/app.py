import cv2
import numpy as np
import time
from predict import SignLanguagePredictor
from utils import display_prediction

def main():
    """Main application for sign language detection"""
    print("Loading Sign Language Detection System...")
    
    # Initialize predictor
    try:
        predictor = SignLanguagePredictor()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure to run the training script first!")
        return
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Set webcam properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Variables for smoothing predictions
    prediction_history = []
    history_size = 5
    
    # Variables for text display
    detected_text = ""
    last_letter = None
    last_letter_time = time.time()
    letter_pause = 1.0  # Seconds to wait before registering the same letter again
    
    print("Press 'q' to quit, 'c' to clear text.")
    print("Ready! Detecting sign language...")
    
    try:
        while cap.isOpened():
            # Read frame from webcam
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Mirror the frame horizontally
            frame = cv2.flip(frame, 1)
            
            # Make prediction
            letter, confidence, processed_frame = predictor.predict_from_image(frame)
            
            # Store prediction in history
            if letter is not None:
                prediction_history.append((letter, confidence))
                if len(prediction_history) > history_size:
                    prediction_history.pop(0)
            
            # Get the most common prediction
            current_prediction = None
            if prediction_history:
                # Sort by confidence and take the highest
                prediction_history.sort(key=lambda x: x[1], reverse=True)
                current_prediction = prediction_history[0][0]
                current_confidence = prediction_history[0][1]
                
                # Only accept predictions with sufficient confidence
                if current_confidence < 0.7:
                    current_prediction = None
            
            # Update detected text if stable prediction
            current_time = time.time()
            if current_prediction and (current_prediction != last_letter or 
                                     current_time - last_letter_time > letter_pause):
                detected_text += current_prediction
                last_letter = current_prediction
                last_letter_time = current_time
            
            # Display prediction on frame
            if current_prediction:
                processed_frame = display_prediction(processed_frame, current_prediction, current_confidence)
            
            # Display detected text
            cv2.rectangle(processed_frame, (0, processed_frame.shape[0]-40), 
                         (processed_frame.shape[1], processed_frame.shape[0]), (16, 117, 245), -1)
            cv2.putText(processed_frame, detected_text[-30:], (10, processed_frame.shape[0]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Display the frame
            cv2.imshow('Sign Language Detection', processed_frame)
            
            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Clear detected text
                detected_text = ""
            
    finally:
        # Release resources
        predictor.close()
        cap.release()
        cv2.destroyAllWindows()
        print("Application closed.")

if __name__ == "__main__":
    main()