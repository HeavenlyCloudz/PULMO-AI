# ==================== COMPLETE LUNG SOUND CLASSIFIER ====================
# This block of code was run on Kaggle

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import librosa
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

print("🚀 Starting Lung Sound Classification...")
print("TensorFlow version:", tf.__version__)

# ==================== CONFIGURATION ====================
# Updated path to handle the nested directory structure
BASE_DATASET_PATH = "/kaggle/input/asthma-detection-dataset-version-2/Asthma Detection Dataset Version 2/Asthma Detection Dataset Version 2"
YAMNET_PATH = "/kaggle/input/yamnet/tensorflow2/yamnet/1"
CLASSES_TO_USE = ['asthma', 'copd', 'pneumonia', 'healthy', 'Bronchial']  # All 5 classes

# ==================== 1. LOAD DATASET ====================
print("\n📁 Loading dataset...")
print(f"Looking in: {BASE_DATASET_PATH}")

def load_audio_files(dataset_path, target_classes):
    """Load all audio files from the nested dataset structure"""
    audio_files = []
    labels = []
    
    for class_name in target_classes:
        class_path = os.path.join(dataset_path, class_name)
        print(f"Checking: {class_path}")
        
        if os.path.exists(class_path):
            files_in_class = [f for f in os.listdir(class_path) if f.endswith(('.wav', '.mp3', '.flac', '.m4a'))]
            print(f"  ✅ {class_name}: {len(files_in_class)} files")
            
            for audio_file in files_in_class:
                full_path = os.path.join(class_path, audio_file)
                audio_files.append(full_path)
                labels.append(class_name)
        else:
            print(f"  ❌ Folder not found: {class_path}")
    
    return audio_files, labels

# Load the data
audio_files, labels = load_audio_files(BASE_DATASET_PATH, CLASSES_TO_USE)
print(f"\n✅ Total loaded: {len(audio_files)} audio files")

# Show class distribution
label_counts = pd.Series(labels).value_counts()
print("\n📊 Class Distribution:")
for class_name, count in label_counts.items():
    print(f"  {class_name}: {count} samples")

# ==================== 2. INITIALIZE YAMNET ====================
print("\n🧠 Loading YAMNet model...")
yamnet_model = hub.load(YAMNET_PATH)
print("✅ YAMNet loaded successfully!")

# ==================== 3. PREPROCESSING FUNCTIONS ====================
def load_and_preprocess_audio(audio_path, target_sr=16000):
    """Load and preprocess single audio file"""
    try:
        audio, sr = librosa.load(audio_path, sr=target_sr)
        
        # Pad if too short (YAMNet needs at least 0.96s)
        if len(audio) < target_sr:
            padding = target_sr - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')
        # If too long, take the first 3 seconds (YAMNet processes in 0.96s windows)
        elif len(audio) > 3 * target_sr:
            audio = audio[:3 * target_sr]
        
        return audio.astype(np.float32)
    except Exception as e:
        print(f"❌ Error loading {audio_path}: {e}")
        return None

def extract_yamnet_embeddings(audio):
    """Extract embeddings using YAMNet"""
    # Ensure audio is the right format
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    
    # Get embeddings from YAMNet
    scores, embeddings, spectrogram = yamnet_model(audio)
    
    # Return mean embedding across time
    return np.mean(embeddings.numpy(), axis=0)

# ==================== 4. EXTRACT FEATURES ====================
print("\n🔧 Extracting features from audio files...")

X_features = []
y_processed = []
failed_files = []

for i, (audio_file, label) in tqdm(enumerate(zip(audio_files, labels)), total=len(audio_files)):
    # Load audio
    audio = load_and_preprocess_audio(audio_file)
    
    if audio is not None:
        # Extract embeddings
        embeddings = extract_yamnet_embeddings(audio)
        X_features.append(embeddings)
        y_processed.append(label)
    else:
        failed_files.append(audio_file)

X = np.array(X_features)
y = np.array(y_processed)

print(f"✅ Features extracted: {X.shape}")
print(f"❌ Failed files: {len(failed_files)}")

if len(failed_files) > 0:
    print("First few failed files:")
    for f in failed_files[:3]:
        print(f"  - {f}")

# ==================== 5. PREPARE DATA FOR TRAINING ====================
print("\n📊 Preparing data for training...")

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_processed)

print("Final Class Distribution:")
for i, class_name in enumerate(label_encoder.classes_):
    count = np.sum(y_encoded == i)
    print(f"  {class_name}: {count} samples")

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X, y_encoded, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_encoded
)

print(f"📈 Training set: {X_train.shape[0]} samples")
print(f"📊 Validation set: {X_val.shape[0]} samples")

# ==================== 6. BUILD MODEL ====================
print("\n🏗️ Building model...")

model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_shape=(1024,)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("✅ Model built successfully!")
model.summary()

# ==================== 7. TRAIN MODEL ====================
print("\n🎯 Training model...")

# Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        patience=15,
        restore_best_weights=True,
        monitor='val_accuracy',
        min_delta=0.001
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        patience=8,
        factor=0.5,
        min_lr=1e-7
    )
]

# Train
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=100,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)

# ==================== 8. EVALUATE MODEL ====================
print("\n📈 Evaluating model...")

# Final evaluation
train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)

print(f"✅ Final Training Accuracy: {train_accuracy:.4f}")
print(f"✅ Final Validation Accuracy: {val_accuracy:.4f}")
print(f"📉 Final Training Loss: {train_loss:.4f}")
print(f"📉 Final Validation Loss: {val_loss:.4f}")

# ==================== 9. PLOT RESULTS ====================
print("\n📊 Plotting results...")

plt.figure(figsize=(15, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.title('Model Accuracy', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('Model Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ==================== 10. SAVE EVERYTHING ====================
print("\n💾 Saving model and results...")

# Save model
model.save('/kaggle/working/lung_sound_classifier.keras')

# Save training history
history_df = pd.DataFrame(history.history)
history_df.to_csv('/kaggle/working/training_history.csv', index=False)

# Save label mapping
label_mapping = pd.DataFrame({
    'encoded_value': range(len(label_encoder.classes_)),
    'class_name': label_encoder.classes_
})
label_mapping.to_csv('/kaggle/working/label_mapping.csv', index=False)

# Save failed files list
if failed_files:
    failed_df = pd.DataFrame({'failed_files': failed_files})
    failed_df.to_csv('/kaggle/working/failed_files.csv', index=False)

print("✅ All files saved to /kaggle/working/")

# ==================== 11. PREDICTION DEMO ====================
print("\n🔮 Prediction Demo...")

if len(X_val) > 0:
    # Test on 5 random samples
    indices = np.random.choice(len(X_val), min(5, len(X_val)), replace=False)
    
    print("Sample Predictions:")
    print("-" * 50)
    
    for i, idx in enumerate(indices):
        sample_features = X_val[idx:idx+1]
        true_label = y_val[idx]
        
        prediction = model.predict(sample_features, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = np.max(prediction[0])
        
        true_class_name = label_encoder.inverse_transform([true_label])[0]
        pred_class_name = label_encoder.inverse_transform([predicted_class])[0]
        
        status = "✅ CORRECT" if true_label == predicted_class else "❌ WRONG"
        
        print(f"Sample {i+1}:")
        print(f"  True: {true_class_name}")
        print(f"  Pred: {pred_class_name} ({confidence:.3f})")
        print(f"  {status}")
        print()

# ==================== 12. CONFUSION MATRIX ====================
print("\n📋 Generating confusion matrix...")

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Predict on validation set
y_pred = model.predict(X_val, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)

# Confusion matrix
cm = confusion_matrix(y_val, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Classification report
print("\n📊 Classification Report:")
print(classification_report(y_val, y_pred_classes, target_names=label_encoder.classes_))

print("\n" + "="*60)
print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
print("📁 Check /kaggle/working/ for all saved files")
print("="*60)
