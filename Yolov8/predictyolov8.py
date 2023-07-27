from ultralytics import YOLO
import numpy as np
from ultralytics.yolo.utils.plotting import Annotator
from PIL import Image
import utils


def detect_classes_image_yolov8(model: YOLO, image_path: str or Image.Image, classes: list):
    if isinstance(image_path, str):
        img = Image.open(image_path)
    elif isinstance(image_path, Image.Image):
        img = image_path
    
    results = model(image_path)  # predict on an image
    annotator = Annotator(np.ascontiguousarray(img))
    tagged_boxes = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
            c = box.cls # Class
            if model.names[int(c)] in classes:
                new_box = {'coords': b, 'label': model.names[int(c)]}
                tagged_boxes.append(new_box)
                annotator.box_label(b, model.names[int(c)])

    img = annotator.result()
    return img, tagged_boxes


def predict_yolov8(images_filepaths: list or str or Image.Image, model, classes: list, outdir: str=None):
    predictions = []

    if isinstance(images_filepaths, str) or isinstance(images_filepaths, Image.Image):
        images_filepaths = [images_filepaths]

    imgs = []
    for img_path in images_filepaths:
        img, coords = detect_classes_image_yolov8(model=model, image_path=img_path, classes=classes)
        predictions.append((img, coords))
        imgs.append(img)

    if outdir is not None:
        for img, path in zip(imgs, images_filepaths):
            utils.save_img(outdir, path, img)

    return predictions
