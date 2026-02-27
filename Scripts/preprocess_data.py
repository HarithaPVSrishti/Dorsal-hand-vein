import os
import cv2
import numpy as np
import math
from scipy.signal import convolve2d
import scipy.ndimage as ndimage
from tqdm import tqdm

# --- Miura Method Implementation ---

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

def connect_profile_1d(vp):
    return np.amin([np.amax([vp[3:-1], vp[4:]], axis=0), np.amax([vp[1:-3], vp[:-4]], axis=0)], axis=0)

def connect_centres(vein_score):
    connected_center = np.zeros(vein_score.shape, dtype='float64')
    vein_score_sum = np.sum(vein_score, axis=2)
    for index in range(vein_score_sum.shape[0]):
        connected_center[index, 2:-2, 0] = connect_profile_1d(vein_score_sum[index, :])
    for index in range(vein_score_sum.shape[1]):
        connected_center[2:-2, index, 1] = connect_profile_1d(vein_score_sum[:, index])
    i, j = np.indices(vein_score_sum.shape)
    border = np.zeros((2,), dtype='float64')
    for index in range(-vein_score_sum.shape[0] + 5, vein_score_sum.shape[1] - 4):
        connected_center[:, :, 2][i == (j - index)] = np.hstack([border, connect_profile_1d(vein_score_sum.diagonal(index)), border])
    Vud = np.flipud(vein_score_sum)
    for index in range(-vein_score_sum.shape[0] + 5, vein_score_sum.shape[1] - 4):
        mask = (i == (j - index))
        connected_center[:, :, 3][np.flipud(mask)] = np.hstack([border, connect_profile_1d(Vud.diagonal(index)), border])
    return connected_center

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
    if k.shape[0] == 0 or k.shape[1] == 0: return score
    for index in range(k.shape[0]): score[index, :, 0] += profile_score_1d(k[index, :, 0])
    for index in range(k.shape[1]): score[:, index, 1] += profile_score_1d(k[:, index, 1])
    i, j = np.indices(k.shape[:2])
    for index in range(-k.shape[0] + 1, k.shape[1]): score[i == (j - index), 2] += profile_score_1d(k[:, :, 2].diagonal(index))
    curve_m45 = np.flipud(k[:, :, 3])
    score_m45 = np.zeros_like(curve_m45)
    for index in range(-k.shape[0] + 1, k.shape[1]): score_m45[i == (j - index)] += profile_score_1d(curve_m45.diagonal(index))
    score[:, :, 3] = np.flipud(score_m45)
    return score

def extract_veins(img_path, output_path):
    if os.path.exists(output_path): return
    img = cv2.imread(img_path)
    if img is None: return
    img = cv2.resize(img, (224, 224))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    data = np.asarray(gray, dtype=np.float64)
    f = remove_hair(data, 6)
    p = normalize_data(f, 0, 255)
    kappa = compute_curvature(p, sigma=8)
    score = compute_vein_score(kappa)
    conect = connect_centres(score)
    threshold = binaries(np.amax(conect, axis=2))
    vein_img = (threshold * 255).astype(np.uint8)
    cv2.imwrite(output_path, vein_img)

# --- Execution ---

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_dir = os.path.join(base_dir, "Data", "Total_Vein_Dataset")
    output_dir = os.path.join(base_dir, "Data", "Processed_Vein_Images")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    user_folders = sorted([f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))])
    
    print("Pre-processing images to extract vein patterns...")
    print("This only needs to be done ONCE. It will save progress automatically.")

    for user_folder in tqdm(user_folders, desc="Processing Persons"):
        input_user_path = os.path.join(input_dir, user_folder)
        output_user_path = os.path.join(output_dir, user_folder)
        
        if not os.path.exists(output_user_path):
            os.makedirs(output_user_path)

        image_files = [f for f in os.listdir(input_user_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for img_name in image_files:
            in_file = os.path.join(input_user_path, img_name)
            out_file = os.path.join(output_user_path, img_name)
            extract_veins(in_file, out_file)

    print(f"\nSUCCESS! All vein skeletons saved in: {output_dir}")
    print("You can now run 'train.py' and it will finish in minutes.")

if __name__ == "__main__":
    main()
