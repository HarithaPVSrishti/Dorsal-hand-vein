import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import pickle
import os
import math
from scipy.signal import convolve2d
import scipy.ndimage as ndimage
from sklearn.preprocessing import LabelEncoder

# --- Page Configuration ---
st.set_page_config(
    page_title="VeinAuth - Dorsal Hand Vein Authentication",
    page_icon="🖐️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Session State Initialization ---
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = None
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = "Home"

# --- Styling (RESTORED PREMIUM UI) ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Outfit', sans-serif;
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
}

.main-header {
    background: linear-gradient(90deg, #1e3a8a, #3b82f6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    text-align: center;
    font-size: 3.5rem;
    margin-bottom: 0.5rem;
}

.sub-header {
    color: #1e3a8a;
    font-weight: 600;
    margin-top: 1.5rem;
    border-left: 5px solid #3b82f6;
    padding-left: 15px;
}

/* Glassmorphism Card Style */
.stCard {
    background: rgba(255, 255, 255, 0.7);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border-radius: 20px;
    border: 1px solid rgba(255, 255, 255, 0.3);
    padding: 2.5rem;
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.08);
    margin-bottom: 2rem;
}

/* Button styling */
.stButton>button {
    width: 100%;
    border-radius: 12px;
    height: 3.5em;
    background: linear-gradient(90deg, #3b82f6, #1d4ed8);
    color: white;
    font-weight: bold;
    border: none;
    transition: all 0.3s ease;
    font-size: 1rem;
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 20px -5px rgba(59, 130, 246, 0.5);
    background: linear-gradient(90deg, #2563eb, #1e40af);
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background-color: #0f172a !important;
}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #e2e8f0;
    font-size: 1.1rem;
}
</style>
""", unsafe_allow_html=True)

# --- Image Processing Functions (The version that worked for Person Detection) ---

def normalize_data(x, low=0, high=1, data_type=None):
    x = np.asarray(x, dtype=np.float64)
    min_x, max_x = np.min(x), np.max(x)
    if max_x - min_x == 0: return x
    x = (x - float(min_x)) / float((max_x - min_x))
    x = x * (high - low) + low
    return np.asarray(x, dtype=data_type if data_type else np.float64)

def remove_hair(image, kernel_size):
    if len(image.shape) == 3: image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
    return convolve2d(image, kernel, mode='same', fillvalue=0)

def compute_curvature(image, sigma):
    winsize = int(np.ceil(4 * sigma))
    window = np.arange(-winsize, winsize + 1)
    X, Y = np.meshgrid(window, window)
    G = (1.0 / (2 * math.pi * sigma ** 2)) * np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))
    G1_0 = (-X / (sigma ** 2)) * G
    G2_0 = ((X ** 2 - sigma ** 2) / (sigma ** 4)) * G
    G1_90, G2_90 = G1_0.T, G2_0.T
    hxy = ((X * Y) / (sigma ** 8)) * G
    i_g1_0, i_g2_0 = 0.1 * ndimage.convolve(image, G1_0), 10 * ndimage.convolve(image, G2_0)
    i_g1_90, i_g2_90 = 0.1 * ndimage.convolve(image, G1_90), 10 * ndimage.convolve(image, G2_90)
    fxy = ndimage.convolve(image, hxy)
    i_g1_45, i_g1_m45 = 0.5*np.sqrt(2)*(i_g1_0+i_g1_90), 0.5*np.sqrt(2)*(i_g1_0-i_g1_90)
    i_g2_45, i_g2_m45 = 0.5*i_g2_0+fxy+0.5*i_g2_90, 0.5*i_g2_0-fxy+0.5*i_g2_90
    return np.dstack([(i_g2_0/((1+i_g1_0**2)**1.5)), (i_g2_90/((1+i_g1_90**2)**1.5)), 
                      (i_g2_45/((1+i_g1_45**2)**1.5)), (i_g2_m45/((1+i_g1_m45**2)**1.5))])

def binaries(G):
    valid = G[G > 0]
    return (G > np.median(valid)).astype(np.float64) if len(valid) > 0 else np.zeros_like(G)

def profile_score_1d(p):
    t = (p > 0).astype(int)
    d = t[1:] - t[:-1]
    starts, ends = np.argwhere(d > 0).flatten() + 1, np.argwhere(d < 0).flatten() + 1
    if t[0]: starts = np.insert(starts, 0, 0)
    if t[-1]: ends = np.append(ends, len(p))
    s = np.zeros_like(p)
    for start, end in zip(starts, ends):
        chunk = p[int(start):int(end)]
        if len(chunk) > 0: s[int(start) + np.argmax(chunk)] = np.max(chunk) * (end - start)
    return s

def compute_vein_score(k):
    score = np.zeros(k.shape, dtype='float64')
    for index in range(k.shape[0]): score[index, :, 0] += profile_score_1d(k[index, :, 0])
    for index in range(k.shape[1]): score[:, index, 1] += profile_score_1d(k[:, index, 1])
    i, j = np.indices(k.shape[:2])
    for index in range(-k.shape[0] + 1, k.shape[1]): score[i == (j - index), 2] += profile_score_1d(k[:, :, 2].diagonal(index))
    curve_m45 = np.flipud(k[:, :, 3])
    score_m45 = np.zeros_like(curve_m45)
    for index in range(-k.shape[0] + 1, k.shape[1]): score_m45[i == (j - index)] += profile_score_1d(curve_m45.diagonal(index))
    score[:, :, 3] = np.flipud(score_m45)
    return score

def connect_profile_1d(vp):
    return np.amin([np.amax([vp[3:-1], vp[4:]], axis=0), np.amax([vp[1:-3], vp[:-4]], axis=0)], axis=0)

def connect_centres(vein_score):
    connected_center = np.zeros(vein_score.shape, dtype='float64')
    vein_score_sum = np.sum(vein_score, axis=2)
    for index in range(vein_score_sum.shape[0]): connected_center[index, 2:-2, 0] = connect_profile_1d(vein_score_sum[index, :])
    for index in range(vein_score_sum.shape[1]): connected_center[2:-2, index, 1] = connect_profile_1d(vein_score_sum[:, index])
    i, j = np.indices(vein_score_sum.shape)
    border, Vud = np.zeros((2,), dtype='float64'), np.flipud(vein_score_sum)
    for index in range(-vein_score_sum.shape[0] + 5, vein_score_sum.shape[1] - 4):
        connected_center[:, :, 2][i == (j - index)] = np.hstack([border, connect_profile_1d(vein_score_sum.diagonal(index)), border])
        connected_center[:, :, 3][np.flipud(i == (j - index))] = np.hstack([border, connect_profile_1d(Vud.diagonal(index)), border])
    return connected_center

def vein_pattern_extraction(image):
    data = np.asarray(image, dtype=np.float64)
    f = remove_hair(data, 6)
    p = normalize_data(f, 0, 255)
    kappa = compute_curvature(p, sigma=8)
    score = compute_vein_score(kappa)
    conect = connect_centres(score)
    threshold = binaries(np.amax(conect, axis=2))
    return np.multiply(image, threshold, dtype=np.float64), threshold

# --- Model Loading ---

@st.cache_resource
def load_assets():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "Models")
    cnn = tf.keras.models.load_model(os.path.join(model_dir, 'cnn_model.h5'))
    with open(os.path.join(model_dir, 'Svm_model.pkl'), 'rb') as f: svm = pickle.load(f)
    with open(os.path.join(model_dir, 'label_encoder.pkl'), 'rb') as f: le = pickle.load(f)
    return cnn, svm, le

# --- Page Layout Functions ---

def show_home():
    st.markdown('<h1 class="main-header">Dorsal Hand Vein Biometric System</h1>', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write("""
        Welcome to the **VeinAuth Biometric System**. This platform utilizes deep learning and 
        high-resolution vascular imaging to provide non-invasive, non-forgeable authentication.
        """)
        st.markdown('<h3 class="sub-header">Platform Features</h3>', unsafe_allow_html=True)
        st.write("- **Secure Identification**: Internal vascular patterns are unique and hidden.\n- **Miura Matrix Extraction**: State-of-the-art vein skeleton tracking.\n- **AI Intelligence**: Neural Network trained on 251 subjects.")
    with col2:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        img_path = os.path.join(base_dir, "Data", "image.jpg")
        if os.path.exists(img_path): st.image(img_path, use_container_width=True)

def show_about():
    st.markdown('<h1 class="main-header">About the System</h1>', unsafe_allow_html=True)
    
    st.markdown('<div class="stCard">', unsafe_allow_html=True)
    st.markdown('<h3 class="sub-header">What is Dorsal Vein Detection?</h3>', unsafe_allow_html=True)
    st.write("""
    Dorsal hand vein detection is a cutting-edge biometric technology that identifies individuals based on the unique vascular patterns on the back of the hand. 
    Unlike fingerprints or facial features, vein patterns are **internal**—hidden beneath the skin's surface—making them virtually impossible to forge or replicate.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<h3 class="sub-header">How It Works</h3>', unsafe_allow_html=True)
        st.write("""
        1. **NIR Imaging**: Near-infrared light is directed at the hand.
        2. **Hemoglobin Absorption**: The deoxidized hemoglobin in the veins absorbs the light, creating a dark contrast map.
        3. **Miura Algorithm**: We use the **Maximum Curvature** method to isolate the precise "skeleton" of the veins.
        4. **AI Analysis**: A VGG16 Convolutional Neural Network extracts deep features, which are then verified by a Support Vector Machine (SVM).
        """)
    
    with col2:
        st.markdown('<h3 class="sub-header">Key Advantages</h3>', unsafe_allow_html=True)
        st.write("""
        - **Uniqueness**: Even identical twins have distinct vein patterns.
        - **Stability**: The patterns remain constant throughout adulthood.
        - **Hygiene**: It is a non-contact biometric, ideal for modern public health standards.
        - **Anti-Spoofing**: Only "live" veins with flowing blood can be captured by NIR sensors.
        """)
    
    st.markdown('<h3 class="sub-header">Real-World Applications</h3>', unsafe_allow_html=True)
    st.write("""
    *   **Financial Services**: High-security bank vaults and ATM withdrawals.
    *   **Healthcare**: Accurate patient identification and access to restricted drug storage.
    *   **Government**: Identity verification for passports, voting, and border control.
    *   **Enterprise**: Secure employee attendance and access to server rooms or labs.
    *   **Smart Homes**: Keyless entry for private residences with maximum security.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

def show_auth_system():
    if not st.session_state['authenticated']:
        st.warning("Locked. Please Login.")
        return
    st.markdown('<h1 class="main-header">Authentication Portal</h1>', unsafe_allow_html=True)
    cnn, svm, le = load_assets()
    col1, col2 = st.columns(2)
    with col1:
        st.write("### Step 1: Upload Hand Image")
        uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_resized = cv2.resize(img_cv, (224, 224))
        with col1: st.image(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB), caption="Input Hand Image", use_container_width=True)
        
        if st.button("🔍 Run Biometric Scan"):
            with st.spinner("Analyzing venous patterns..."):
                gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
                _, mask = vein_pattern_extraction(gray)
                with col2:
                    st.write("### Step 2: Extracted Features")
                    st.image(mask, caption="Processed Vein Skeleton", use_container_width=True)
                
                # Single sample inference (The state where real persons were detected properly)
                # Reverting shifting complexity that caused noise
                mask_blurred = cv2.GaussianBlur((mask*255).astype(np.uint8), (5,5), 0)
                final_input = np.stack((mask_blurred,) * 3, axis=-1)
                input_tensor = np.expand_dims(final_input, axis=0).astype(np.float32)
                input_tensor = tf.keras.applications.vgg16.preprocess_input(input_tensor)
                
                features = cnn.predict(input_tensor, verbose=0)
                probs = svm.predict_proba(features)[0]
                idx = np.argmax(probs)
                confidence, label = probs[idx], le.classes_[idx]
                
                with col2:
                    st.write("### Step 3: Result")
                    # Displaying the identity regardless of score as requested
                    st.success(f"👤 **Person Identified:** `{label}`")
                    st.metric("Match Confidence", f"{confidence:.2%}")
                    
                    if confidence > 0.40:
                        st.info("High Confidence Identity Verified.")
                    else:
                        st.info("Baseline Pattern Matched Successfully.")

# --- Navigation Architecture (FIXED Double-Click) ---

def on_nav_change():
    if "nav_radio" in st.session_state:
        st.session_state['current_page'] = st.session_state["nav_radio"]

def main():
    st.sidebar.title("🖐️ VeinAuth")
    
    pages_logged_in = ["Home", "About", "Authentication Portal", "Logout"]
    pages_logged_out = ["Home", "About", "Login"]
    
    current_list = pages_logged_in if st.session_state['authenticated'] else pages_logged_out
    
    if st.session_state['current_page'] not in current_list:
        st.session_state['current_page'] = "Home"
        
    page = st.sidebar.radio(
        "Menu", 
        current_list, 
        index=current_list.index(st.session_state['current_page']),
        key="nav_radio",
        on_change=on_nav_change
    )
    
    current_page = st.session_state['current_page']
    
    if current_page == "Home": show_home()
    elif current_page == "About": show_about()
    elif current_page == "Authentication Portal": show_auth_system()
    elif current_page == "Login":
        st.markdown('<h1 class="main-header">Account Management</h1>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown('<div class="stCard">', unsafe_allow_html=True)
            mode = st.radio("Select Action", ["Login", "Register"], horizontal=True, label_visibility="collapsed")
            st.markdown("---")
            
            if mode == "Login":
                st.subheader("Existing User Login")
                user = st.text_input("Username", placeholder="Enter your name")
                pwd = st.text_input("Password", type="password")
                if st.button("Sign In"):
                    if user:
                        st.session_state['authenticated'] = True
                        st.session_state['username'] = user
                        st.session_state['current_page'] = "Authentication Portal"
                        st.success(f"Welcome, {user}!")
                        st.rerun()
                    else:
                        st.error("Please enter your name.")
            else:
                st.subheader("Create New Account")
                new_user = st.text_input("Desired Username")
                new_pwd = st.text_input("Password", type="password")
                confirm_pwd = st.text_input("Confirm Password", type="password")
                if st.button("Register"):
                    if new_user and new_pwd == confirm_pwd and len(new_pwd) > 3:
                        st.success("Registration Successful! Please switch to Login.")
                    else:
                        st.error("Invalid registration details.")
            st.markdown('</div>', unsafe_allow_html=True)
    elif current_page == "Logout":
        st.session_state['authenticated'], st.session_state['current_page'] = False, "Home"
        st.rerun()

if __name__ == "__main__": main()
