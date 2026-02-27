# Dorsal Hand Vein Biometric Authentication System

A high-security biometric identification system that uses the unique vascular patterns on the back of the hand. 

## 🚀 Accuracy: 99.13%
This system utilizes a hybrid AI approach combining **Miura Maximum Curvature** for vein extraction and **VGG16 + SVM** for classification.

## 🛠️ Technology Stack
- **Python 3.12**
- **TensorFlow/Keras** (VGG16 Feature Extraction)
- **Scikit-Learn** (SVM Classification)
- **OpenCV** (Image Processing & Miura Algorithm)
- **Streamlit** (Premium Web Browser UI)

## 📁 System Architecture
1. **Preprocessing**: Miura Method isolates vein "skeletons" in 4 directions.
2. **Feature Extraction**: VGG16 CNN turns skeletons into numeric "fingerprints".
3. **Classification**: SVM identifies the person from a database of 251 subjects.

## 📦 How to Run
1. Install requirements: `pip install tensorflow opencv-python scikit-learn streamlit tqdm pillow scipy`
2. Run the App: `streamlit run app_streamlit.py`

## 📊 Results
- **Identification Speed**: ~800ms
- **Top-1 Accuracy**: 99.13%
- **Identities Supported**: 251 Subjects
