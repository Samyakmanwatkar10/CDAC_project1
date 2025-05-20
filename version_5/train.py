import numpy as np
import os
import pickle
from model import create_cnn_lstm_model, get_callbacks
from utils import plot_training_history, ALPHABET
import tensorflow as tf

def train_model(data_dir='processed_data', batch_size=32, epochs=50):
    """
    Train the sign language recognition model
    
    Args:
        data_dir: Directory containing processed data
        batch_size: Batch size for training
        epochs: Number of training epochs
    """
    # Load the data
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
    y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    
    # Check if we have a saved alphabet mapping
    alphabet_path = os.path.join(data_dir, 'alphabet.pkl')
    if os.path.exists(alphabet_path):
        with open(alphabet_path, 'rb') as f:
            alphabet = pickle.load(f)
        print(f"Loaded alphabet mapping: {alphabet}")
    else:
        alphabet = ALPHABET
        print(f"Using default alphabet mapping: {alphabet}")
    
    # Create the model
    model = create_cnn_lstm_model(X_train.shape[1], num_classes=len(alphabet))
    model.summary()
    
    # Get callbacks
    callbacks = get_callbacks()
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Save the final model
    model.save('sign_language_model.h5')
    print("Model saved as 'sign_language_model.h5'")
    
    # Evaluate on validation set
    val_loss, val_acc = model.evaluate(X_val, y_val)
    print(f"Validation accuracy: {val_acc:.4f}")
    
    # Try to evaluate on test set if available
    try:
        X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
        y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
        test_loss, test_acc = model.evaluate(X_test, y_test)
        print(f"Test accuracy: {test_acc:.4f}")
    except:
        print("No test data found.")

if __name__ == "__main__":
    train_model()
