import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
import cv2
import pickle
from tqdm import tqdm

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "Data", "Processed_Vein_Images")
    model_dir = os.path.join(base_dir, "Models")
    
    if not os.path.exists(data_dir):
        print("Error: Run preprocess_data.py first!")
        return

    print("Initializing AI Feature Extractor...")
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    feature_extractor = Model(inputs=base_model.input, outputs=x)
    feature_extractor.save(os.path.join(model_dir, 'cnn_model.h5'))

    all_features = []
    all_labels = []
    
    user_folders = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])
    
    for user_folder in tqdm(user_folders, desc="Extracting Features"):
        user_path = os.path.join(data_dir, user_folder)
        image_files = [f for f in os.listdir(user_path) if f.lower().endswith('.png')]
        
        for img_name in image_files:
            img_path = os.path.join(user_path, img_name)
            img_cv = cv2.imread(img_path)
            if img_cv is None: continue
            
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            
            # Create 3 versions (Original + 2 shifts) for high accuracy
            for dx, dy in [(0,0), (6,6), (-6,-6)]:
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                shifted = cv2.warpAffine(img_rgb, M, (224, 224))
                
                # Blur to help AI see patterns
                proc = cv2.GaussianBlur(shifted, (5, 5), 0)
                img_batch = np.expand_dims(proc, axis=0).astype(np.float32)
                img_pre = tf.keras.applications.vgg16.preprocess_input(img_batch)
                
                feat = feature_extractor.predict(img_pre, verbose=0).flatten()
                all_features.append(feat)
                all_labels.append(user_folder)

    # Save features to disk so we don't lose work
    with open(os.path.join(model_dir, 'extracted_features.pkl'), 'wb') as f:
        pickle.dump({'X': np.array(all_features), 'y': np.array(all_labels)}, f)
    
    print(f"\nSUCCESS: All features extracted and saved to {os.path.join(model_dir, 'extracted_features.pkl')}")

if __name__ == "__main__":
    main()
