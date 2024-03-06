import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import cv2

# ----------------------------------------------------------------------------------------------------------------------
# IMPORTS
# ----------------------------------------------------------------------------------------------------------------------

MASK_DATA_PATH = "./data/Ground Truth/"
IMAGE_DATA_PATH = "./data/Original/"

import os
from PIL import Image
import numpy as np

def load(folder, mask = None):
    images = []
    for filename in sorted(os.listdir(folder)):
        if filename.endswith(".png"):
            img = Image.open(os.path.join(folder, filename))
            if img is not None:
                if mask:
                    img = img.convert('L')
                    img = np.array(img)
                    img = img == 255
                    images.append(img.astype(np.uint8))
                else: 
                    images.append(np.array(img))
    return images

# ----------------------------------------------------------------------------------------------------------------------
# MASK APPLICATION
# ----------------------------------------------------------------------------------------------------------------------

def square_mask(masks):
    img_number = []
    bounding_boxes = []
    all_contours = []
    for i in range(len(masks)):  
        mask_8bit = (masks[i] * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_8bit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            box = np.zeros(mask_8bit.shape)
            box[y:y+h, x:x+w] = 1  
            bounding_boxes.append(box)
            all_contours.append(contour)
            img_number.append(i)
    return img_number, bounding_boxes, all_contours
    

def apply_masks(images, masks):
    # images = load(IMAGE_DATA_PATH)
    # mask = load(MASK_DATA_PATH, mask=False)
    # masks = load(MASK_DATA_PATH, mask=True)
    images = np.array(images).astype(int)
    masks = masks
    polyps = []
    N = len(images)

    for idx in range(N):
        mask_expanded = np.expand_dims(masks[idx], axis=-1)
        # Repeat the expanded masks across the third dimension to get (288, 384, 3)
        rgb_mask = np.repeat(mask_expanded, 3, axis=2)
        polyps.append(np.multiply(images[idx], rgb_mask))

    fig, axe = plt.subplots(1, 3, figsize=(10, 5))
    nr = np.random.randint(0,N)
    axe[0].imshow(images[15])
    axe[1].imshow(masks[15], cmap=plt.cm.gray)
    axe[2].imshow(polyps[15])
    fig.tight_layout()
    plt.show()
    return polyps

def extract_polyps(ind, polyps, contours):
    output_directory = "./cropped_images"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    N = len(contours)
    for i in range(N):
        x, y, w, h = cv2.boundingRect(contours[i])
        polyp = polyps[ind[i]]
        crop = polyp[y:y+h, x:x+w]
        cropped_image_pil = Image.fromarray(crop)
        filename = f"cropped_polyp_image{ind[i]}_contour_{i}.png"
        path = os.path.join(output_directory, filename)
        cropped_image_pil.save(path)

# ----------------------------------------------------------------------------------------------------------------------
# PCA
# ----------------------------------------------------------------------------------------------------------------------

def flatten_array(array):
    return [image.flatten() for image in array]

def pca(polyps):
    NR_COMPONENTS = 3 # ?MK?: Hur ska det väljas?
    flat_polyps = flatten_array(polyps)

    scaler = StandardScaler()
    flat_polyps = scaler.fit_transform(flat_polyps)

    pca = PCA(n_components=NR_COMPONENTS)
    data = pca.fit_transform(flat_polyps)
    return data

# ----------------------------------------------------------------------------------------------------------------------
# K-Means Clustering
# ----------------------------------------------------------------------------------------------------------------------
def k_m_plot(data):
    k = 4

    kmeans = KMeans(n_clusters=k, random_state=0)
    clusters = kmeans.fit_predict(data)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=clusters, cmap='viridis', marker='o')
    ax.set_title('Cluster Visualization')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    # ax.colorbar(label='Cluster')
    plt.show()



def __main__():
    images = load(IMAGE_DATA_PATH)
    masks = load(MASK_DATA_PATH, mask=True)
    ind, _, contours = square_mask(masks)
    # _ = apply_masks(images, masks)
    extract_polyps(ind, images, contours)


    #k_m_plot(pca(polyps))

__main__()

