import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import cv2
from tqdm import tqdm

def main():
    base_dir = r"e:\Srishti\Student project haritha\Dorsal hand vein ST Thomas\dorsalhandveins-main"
    # NOW WE POINT TO THE PRE-PROCESSED IMAGES
    data_dir = os.path.join(base_dir, "Processed_Vein_Images")
    
    if not os.path.exists(data_dir) or len(os.listdir(data_dir)) == 0:
        print(f"Error: {data_dir} not found or empty. PLEASE RUN 'preprocess_data.py' FIRST.")
        return

    # Image parameters
    img_size = (224, 224)
    
    # Feature Extractor
    print("Initializing CNN feature extractor...")
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    feature_extractor = Model(inputs=base_model.input, outputs=x)
    feature_extractor.save('cnn_model.h5')

    images = []
    labels = []
    
    user_folders = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])
    print(f"Loading pre-processed images from {len(user_folders)} persons...")
    
    # Augmentation setup for skeletons (Fast)
    datagen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, height_shift_range=0.08, fill_mode='nearest')

    # 1. Feature Extraction Loop
    for i, user_folder in enumerate(tqdm(user_folders, desc="Extracting Features")):
        user_path = os.path.join(data_dir, user_folder)
        image_files = [f for f in os.listdir(user_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_name in image_files:
            img_path = os.path.join(user_path, img_name)
            img_cv = cv2.imread(img_path)
            if img_cv is None: continue
            img_cv = cv2.resize(img_cv, (224, 224))
            
            # Simple shifts for robustness (Fast Augment)
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            
            # Original + 2 tiny shifts
            for dx, dy in [(0,0), (5,5), (-5,-5)]:
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                shifted = cv2.warpAffine(img_rgb, M, (224, 224))
                
                # Preprocess
                proc_img_blur = cv2.GaussianBlur(shifted, (5, 5), 0)
                img_batch = np.expand_dims(proc_img_blur, axis=0).astype(np.float32)
                img_preprocessed = tf.keras.applications.vgg16.preprocess_input(img_batch)
                
                feat = feature_extractor.predict(img_preprocessed, verbose=0).flatten()
                all_features.append(feat)
                all_labels.append(user_folder)

        # Clear memory occasionally
        if (i + 1) % 50 == 0:
            tf.keras.backend.clear_session()
            base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
            x_layer = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
            feature_extractor = Model(inputs=base_model.input, outputs=x_layer)

    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels)
    del all_features
    del all_labels

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)

    print(f"\nTraining SVM on {X.shape[0]} samples...")
    X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.15, stratify=y_encoded, random_state=42)
    
    svm = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True)
    svm.fit(X_train, y_train)
    
    print(f"Validation Accuracy: {accuracy_score(y_val, svm.predict(X_val)) * 100:.2f}%")
    
    print("Finalizing model...")
    svm.fit(X, y_encoded)
    with open('Svm_model.pkl', 'wb') as f:
        pickle.dump(svm, f)
    
    print("SUCCESS: Full training complete!")

if __name__ == "__main__":
    main()
