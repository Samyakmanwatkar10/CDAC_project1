import pickle
import numpy as np
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, GaussianNoise, Bidirectional
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load preprocessed data
DATA_PATH = "D:/samyak/ASL_left_hand/processed_data/processed_data.pkl"
try:
    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    logging.info(f"Loaded data: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test samples")
except FileNotFoundError:
    logging.error(f"Preprocessed data not found at {DATA_PATH}. Run preprocess.py first.")
    raise
except Exception as e:
    logging.error(f"Error loading {DATA_PATH}: {str(e)}")
    raise

# Verify data shapes
if len(X_train) == 0 or len(X_val) == 0:
    logging.error("Empty training or validation data. Check preprocess.py output.")
    raise ValueError("Empty dataset")
if len(X_test) == 0:
    logging.warning("Empty test data. Evaluation may be unreliable.")
if X_train.shape[1:] != (30, 63):
    logging.error(f"Invalid X_train shape: {X_train.shape}. Expected (samples, 30, 63).")
    raise ValueError("Shape mismatch in training data")

# Convert labels to one-hot encoding
NUM_CLASSES = 26
LABELS = [chr(i) for i in range(65, 91)]  # A-Z
try:
    y_train_one_hot = to_categorical(y_train, NUM_CLASSES)
    y_val_one_hot = to_categorical(y_val, NUM_CLASSES)
    y_test_one_hot = to_categorical(y_test, NUM_CLASSES) if len(X_test) > 0 else np.array([])
except Exception as e:
    logging.error(f"Error converting labels to one-hot: {str(e)}")
    raise

# Define CNN + LSTM model using functional API
inputs = Input(shape=(30, 63))
x = GaussianNoise(0.1)(inputs)
x = Conv1D(filters=64, kernel_size=3, activation='relu')(x)
x = MaxPooling1D(pool_size=2)(x)
x = Conv1D(filters=128, kernel_size=3, activation='relu')(x)
x = MaxPooling1D(pool_size=2)(x)
x = Conv1D(filters=256, kernel_size=3, activation='relu')(x)
x = MaxPooling1D(pool_size=2)(x)
x = Bidirectional(LSTM(128, return_sequences=False))(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.6)(x)
outputs = Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(inputs=inputs, outputs=outputs)

# Compile model with additional metrics
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy', 'top_k_categorical_accuracy']
)

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# Train model
try:
    history = model.fit(
        X_train, y_train_one_hot,
        validation_data=(X_val, y_val_one_hot),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping]
    )
    logging.info("Training completed")
except Exception as e:
    logging.error(f"Error during training: {str(e)}")
    raise

# Evaluate model on test set
try:
    if len(X_test) > 0:
        # Basic metrics
        test_loss, test_accuracy, test_top3_accuracy = model.evaluate(X_test, y_test_one_hot)
        logging.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test Top-3 Accuracy: {test_top3_accuracy:.4f}")

        # Detailed metrics
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = y_test

        # Precision, Recall, F1-Score
        precision = precision_score(y_test_classes, y_pred_classes, average='macro')
        recall = recall_score(y_test_classes, y_pred_classes, average='macro')
        f1 = f1_score(y_test_classes, y_pred_classes, average='macro')
        logging.info(f"Macro-Averaged Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

        # Classification report
        report = classification_report(y_test_classes, y_pred_classes, target_names=LABELS)
        logging.info("Classification Report:\n" + report)

        # Save classification report
        os.makedirs("D:/samyak/ASL_left_hand/results", exist_ok=True)
        with open("D:/samyak/ASL_left_hand/results/classification_report.txt", "w") as f:
            f.write(report)

        # Confusion matrix
        cm = confusion_matrix(y_test_classes, y_pred_classes)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=LABELS, yticklabels=LABELS)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig("D:/samyak/ASL_left_hand/results/confusion_matrix.png")
        plt.close()

        # Per-class accuracy
        per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
        for i, acc in enumerate(per_class_accuracy):
            logging.info(f"Class {LABELS[i]} Accuracy: {acc:.4f}")
    else:
        logging.warning("No test data available. Skipping evaluation.")
except Exception as e:
    logging.error(f"Error evaluating model: {str(e)}")
    raise

# Save model
MODEL_PATH = "D:/samyak/ASL_left_hand/models/asl_model2.keras"
try:
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    logging.info(f"Model saved to {MODEL_PATH}")
except Exception as e:
    logging.error(f"Error saving model: {str(e)}")
    raise

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("D:/samyak/ASL_left_hand/results/training_history.png")
plt.close()