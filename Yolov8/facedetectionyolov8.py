import time
from ultralytics import YOLO
from PIL import Image
from predictyolov8 import predict_yolov8


def predict_faces_yolov8(paths: list or str or Image.Image, outdir: str=None):

    if isinstance(paths, str) or isinstance(paths, Image.Image):
        paths = [paths]

    imgs_tagged = []
    for file_path in paths:

        start = time.time()
        face_detections = predict_yolov8(file_path, model=YOLO('models/yolov8n-face.pt'), classes=['face'], outdir=outdir)
        
        end_time = time.time() - start
        n_faces = len(face_detections[0][1])

        img_tagged, boxes = face_detections[0]

        imgs_tagged.append((file_path, img_tagged, boxes, n_faces, end_time))
    return imgs_tagged
