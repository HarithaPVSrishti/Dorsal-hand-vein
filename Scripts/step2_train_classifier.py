import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import os

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, "Models")

    print("Loading extracted features...")
    try:
        with open(os.path.join(model_dir, 'extracted_features.pkl'), 'rb') as f:
            data = pickle.load(f)
    except:
        print(f"Error: Run step1_extract_features.py first! Could not find {os.path.join(model_dir, 'extracted_features.pkl')}")
        return

    X = data['X']
    y = data['y']

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    with open(os.path.join(model_dir, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(le, f)

    print(f"Training 'Brain' (SVM) on {X.shape[0]} samples...")
    X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.15, stratify=y_encoded, random_state=42)
    
    svm = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True)
    svm.fit(X_train, y_train)
    
    acc = accuracy_score(y_val, svm.predict(X_val))
    print(f"Final Accuracy: {acc * 100:.2f}%")
    
    print("Saving final model...")
    svm.fit(X, y_encoded)
    with open(os.path.join(model_dir, 'Svm_model.pkl'), 'wb') as f:
        pickle.dump(svm, f)
    
    print("SUCCESS: System is now fully trained and ready!")

if __name__ == "__main__":
    main()
