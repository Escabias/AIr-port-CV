import torch
import utils
import numpy as np
from PIL import Image
from ultralytics.yolo.utils.plotting import Annotator
from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights


def detect_classes_image_retinanet(model, image_path: str or Image.Image, target_classes: list):
    tagged_boxes = []
    if isinstance(image_path, str):
        input = Image.open(image_path)
    elif isinstance(image_path, Image.Image):
        input = image_path

    weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
    if model is None:
        model = retinanet_resnet50_fpn_v2(weights=weights, score_thresh=0.35)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    preprocess = weights.transforms()

    batch = [preprocess(input)]
    prediction = model(batch)[0]

    labels = [weights.meta["categories"][i] for i in prediction["labels"]]
    boxes = prediction["boxes"]

    annotator = Annotator(np.ascontiguousarray(input))
    for label, box in zip(labels, boxes):
        if label in target_classes:
            new_box = {'coords': box, 'label': label}
            tagged_boxes.append(new_box)
            annotator.box_label(box, label)

    img = annotator.result()
    return img, tagged_boxes


def predict_retinanet(images_filepaths: list or str or Image.Image, model, classes: list, outdir: str=None):
    predictions = []

    if isinstance(images_filepaths, str) or isinstance(images_filepaths, Image.Image):
        images_filepaths = [images_filepaths]

    imgs = []
    for img_path in images_filepaths:
        img, coords = detect_classes_image_retinanet(model=model, image_path=img_path, target_classes=classes)
        predictions.append((img, coords))
        imgs.append(img)

    if outdir is not None:
        for img, path in zip(imgs, images_filepaths):
            utils.save_img(outdir, path, img)

    return predictions