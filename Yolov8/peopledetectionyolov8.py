import time
from ultralytics import YOLO
from PIL import Image
from predictyolov8 import predict_yolov8

def predict_people_yolov8(paths: list or str or Image.Image, outdir: str=None):

    if isinstance(paths, str) or isinstance(paths, Image.Image):
        paths = [paths]

    imgs_tagged = []
    for file_path in paths:

        start = time.time()
        people_detections = predict_yolov8(file_path, model=YOLO('models/yolov8n.pt'), classes=['person'], outdir=outdir)
        
        end_time = time.time() - start
        n_people = len(people_detections[0][1])

        img_tagged, boxes = people_detections[0]

        imgs_tagged.append((file_path, img_tagged, boxes, n_people, end_time))
    return imgs_tagged
