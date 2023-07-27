import time
from ultralytics import YOLO
from PIL import Image
from predictyolov8 import predict_yolov8


def predict_masks_yolov8(paths: list or str or Image.Image, outdir: str=None):

    if isinstance(paths, str) or isinstance(paths, Image.Image):
        paths = [paths]

    imgs_tagged = []
    for file_path in paths:

        start = time.time()
        mask_detections = predict_yolov8(file_path, model=YOLO('models/yolov8-mask.pt'), classes=['mask', 'no_mask', 'improper_mask'], outdir=outdir)
        
        end_time = time.time() - start
        n_masks = len(mask_detections[0][1])

        img_tagged, boxes = mask_detections[0]

        imgs_tagged.append((file_path, img_tagged, boxes, n_masks, end_time))
    return imgs_tagged