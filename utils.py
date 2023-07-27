
import cv2
import time
import os
from PIL import Image
    

def save_img(dir: str, original_filepath: str or Image.Image, image: Image.Image):
    if not os.path.exists(dir):
        os.makedirs(dir)

    if isinstance(original_filepath, str):
        filename = original_filepath.split('/')[-1]
    else:
        filename = f'temp_{time.time()}.jpg'
    cv2.imwrite(f'{dir}{filename}', image)

def get_images_inside_dir(directory:str):
    images = []
    # loop through each directory and subdirectory in the directory
    for root, dirs, files in os.walk(directory):
        # loop through each file in the current directory
        for file in files:
            if file.endswith('.jpg') or file.endswith('.jpeg'):
                file_path = os.path.join(root, file)
                images.append(file_path)

    return images