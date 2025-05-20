import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input, Conv1D, MaxPooling1D, Flatten, Reshape, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from utils import ALPHABET
import os

def create_cnn_lstm_model(input_shape, num_classes=len(ALPHABET)):
    """
    Create a CNN+LSTM model for sign language recognition
    
    Args:
        input_shape: Shape of input features (integer)
        num_classes: Number of output classes
    
    Returns:
        model: Compiled Keras model
    """
    # Input shape is the number of features (63 for flattened landmarks)
    
    model = Sequential([
        # Reshape the input to work with Conv1D
        Reshape((21, 3), input_shape=(input_shape,)),
        
        # CNN layers
        Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        
        Conv1D(128, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        
        # LSTM layers
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        
        # Output layers
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def get_callbacks(checkpoint_path='model_checkpoints'):
    """
    Get callbacks for model training
    
    Args:
        checkpoint_path: Path to save model checkpoints
    
    Returns:
        callbacks: List of Keras callbacks
    """
    os.makedirs(checkpoint_path, exist_ok=True)
    
    checkpoint = ModelCheckpoint(
        os.path.join(checkpoint_path, 'model-{epoch:02d}-{val_accuracy:.4f}.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )
    
    return [checkpoint, early_stopping, reduce_lr]