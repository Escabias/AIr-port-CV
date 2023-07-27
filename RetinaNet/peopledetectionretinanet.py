import time
from PIL import Image
from predictretinanet import predict_retinanet

def predict_people_retinanet(paths: list or str or Image.Image, outdir: str=None):

    if isinstance(paths, str) or isinstance(paths, Image.Image):
        paths = [paths]

    imgs_tagged = []
    for file_path in paths:

        start = time.time()
        people_detections = predict_retinanet(file_path, model=None, classes=['person'], outdir=outdir)
        end_time = time.time() - start

        n_people = len(people_detections[0][1])

        img_tagged, boxes = people_detections[0]

        imgs_tagged.append((file_path, img_tagged, boxes, n_people, end_time))
    return imgs_tagged
