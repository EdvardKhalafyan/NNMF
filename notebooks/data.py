import os
import cv2
import itertools
import numpy as np
from sklearn.utils import shuffle
from sklearn.datasets import fetch_olivetti_faces

import config

np.random.seed(config.SEED)


# Load swimmer
def load_swimmer():
    
    left_hand = [
        [[1, 5], [2, 5]],
        [[1, 3], [2, 4]],
        [[3, 3], [3, 4]],
        [[4, 4], [5, 3]]
    ]
    left_leg = [
        [[7, 3], [8, 4]],
        [[9, 3], [9, 4]],
        [[10, 4], [11, 3]],
        [[10, 5], [11, 5]]
    ]
    right_hand = [
        [[2, 7], [1, 7]],
        [[2, 8], [1, 9]],
        [[3, 8], [3, 9]],
        [[4, 8], [5, 9]]
    ]
    right_leg = [
        [[8, 8], [7, 9]],
        [[9, 8], [9, 9]],
        [[10, 8], [11, 9]],
        [[10, 7], [11, 7]]
    ]

    def generate_body():
        body = np.zeros((config.IMAGE_SIZE["swimmer"], config.IMAGE_SIZE["swimmer"]))
        body[3:10, 6] = 1
        body[3, 5] = 1
        body[3, 7] = 1
        body[9, 5] = 1
        body[9, 7] = 1
        return body
    
    combinations = list(itertools.product(list(range(4)), repeat=4))
    
    image_list = list()
    for comb in combinations:
        for part in [left_hand, left_leg, right_hand, right_leg]:
            image = generate_body()
            for x, y in left_hand[comb[0]]:
                image[x, y] = 1
            for x, y in left_leg[comb[1]]:
                image[x, y] = 1
            for x, y in right_hand[comb[2]]:
                image[x, y] = 1
            for x, y in right_leg[comb[3]]:
                image[x, y] = 1
        image_list.append(image.reshape(config.IMAGE_SIZE["swimmer"] ** 2,))
    
    image_list = shuffle(image_list, random_state=config.SEED)
    np_image_array = np.array(image_list)
    np_image_array = (np_image_array / np_image_array.max()).astype(np.float64)
    return np_image_array


# Load olivetti
def load_olivetti():
    data = fetch_olivetti_faces(shuffle=True, random_state=config.SEED).images
    np_image_array = data.reshape(data.shape[0], -1)
    np_image_array = (np_image_array / np_image_array.max()).astype(np.float64)
    return np_image_array


# Load UTK
def load_utk():
    image_name_list = shuffle(os.listdir(config.UTK_DATA_PATH), random_state=config.SEED)
    
    source_images = list()
    resized_images_list = list()
    for image_name in image_name_list:
        image_path = os.path.join(config.UTK_DATA_PATH, image_name)
        
        image_source = cv2.imread(image_path)
        image_gray = cv2.cvtColor(image_source, cv2.COLOR_BGR2GRAY)
        image_gray_resized = cv2.resize(image_gray, (config.IMAGE_SIZE["UTK"], config.IMAGE_SIZE["UTK"]), interpolation=cv2.INTER_AREA)
    
        source_images.append(image_gray)
    
        resized_images_list.append(image_gray_resized)
    
    resized_images = np.array(resized_images_list)
    np_image_array = resized_images.reshape(resized_images.shape[0], -1)
    np_image_array = (np_image_array / np_image_array.max()).astype(np.float64)
    return np_image_array