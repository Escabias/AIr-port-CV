import cv2
import time
import utils
import numpy as np

from tensorflow import keras
from PIL import Image
from scipy.spatial import distance



class ModelVGG19:

    def __init__(self):
        self.model = keras.models.load_model('VGG19/resources/masknet.h5')

        self.mask_label = {0: 'mask', 1: 'no_mask'}
        self.dist_label = {0: (0, 255, 0), 1: (255, 0, 0)}

        self.face_model = cv2.CascadeClassifier('VGG19/resources/haarcascades/haarcascade_frontalface_default.xml')

        self.MIN_DISTANCE = 130

    def predict(self, images_filepaths: list or str or Image.Image, outdir: str):

        if isinstance(images_filepaths, str) or isinstance(images_filepaths, Image.Image):
            images_filepaths = [images_filepaths]

        imgs_tagged = []
        imgs = []
        for img in images_filepaths:

            if isinstance(img, str):
                img = cv2.imread(img)
            elif isinstance(img, Image.Image):
                img = np.array(img)

            img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)

            start = time.time()
            faces = self.face_model.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4)  # returns a list of (x,y,w,h) tuples
            end_time = time.time() - start

            label = [0 for i in range(len(faces))]

            if len(faces) >= 2:
                for i in range(len(faces) - 1):
                    for j in range(i + 1, len(faces)):
                        dist = distance.euclidean(faces[i][:2], faces[j][:2])
                        if dist < self.MIN_DISTANCE:
                            label[i] = 1
                            label[j] = 1

            new_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # colored output image

            boxes = []
            for i in range(len(faces)):
                (x, y, w, h) = faces[i]
                crop = new_img[y:y + h, x:x + w]
                crop = cv2.resize(crop, (128, 128))
                crop = np.reshape(crop, [1, 128, 128, 3]) / 255.0
                mask_result = self.model.predict(crop)


                label = self.mask_label[mask_result.argmax()]

                colour = (0, 255, 0)
                if label is self.mask_label[1]:
                    colour = (255, 0, 0)

                cv2.putText(new_img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            colour, 2)
                
                cv2.rectangle(new_img, (x, y), (x + w, y + h), colour, 1)

                boxes.append({'coords': (y, x, y+h, x+w), 'label': label})

            imgs.append(new_img)
            imgs_tagged.append((img, new_img, boxes, len(faces), end_time))

        if outdir is not None:
            for img, path in zip(imgs, images_filepaths):
                utils.save_img(outdir, path, img)
        return imgs_tagged

