import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
import os

# --- Configuration ---
DATA_DIR = r"e:\Srishti\Student project haritha\Dorsal hand vein ST Thomas\dorsalhandveins-main\Data\han non hand"
MODEL_SAVE_PATH = r"e:\Srishti\Student project haritha\Dorsal hand vein ST Thomas\dorsalhandveins-main\Models\hand_detection_model.h5"
IMG_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 10

def train_hand_detector():
    print(f"Loading data from: {DATA_DIR}")
    
    # Data Augmentation for training
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_generator = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training'
    )

    validation_generator = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation'
    )

    # Base model with VGG16
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    base_model.trainable = False  # Freeze base layers

    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid') # Binary classification: Hand vs Non-Hand
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    print("Starting training...")
    model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator
    )

    # Save the model
    if not os.path.exists(os.path.dirname(MODEL_SAVE_PATH)):
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH))
    
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    print(f"Classes: {train_generator.class_indices}") # Usually {'hand': 0, 'non hand': 1} or vice versa

if __name__ == "__main__":
    train_hand_detector()
