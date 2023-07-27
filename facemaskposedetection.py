import numpy as np
from ultralytics.yolo.utils.plotting import Annotator
from PIL import Image

from peopledetectionyolov8 import predict_people_yolov8
from peopledetectionretinanet import predict_people_retinanet

from facedetectionyolov8 import predict_faces_yolov8
from maskdetectionyolov8 import predict_masks_yolov8

from modelvgg19 import ModelVGG19

import utils
import math


def calculate_distance(rect1, rect2):
    center1 = [(rect1['coords'][1] + rect1['coords'][3]) / 2, (rect1['coords'][0] + rect1['coords'][2]) / 2]
    center2 = [(rect2['coords'][1] + rect2['coords'][3]) / 2, (rect2['coords'][0] + rect2['coords'][2]) / 2]
    return math.dist(center1, center2)

def is_rect_in_groups(rect, groups):
    for g in groups:
        for r in g:
            if rect['coords'] is r['coords']:
                return True
    return False

def calculate_center_distance(rect1, rect2):
    x1 = (rect1['coords'][1] + rect1['coords'][3]) / 2
    y1 = (rect1['coords'][0] + rect1['coords'][2]) / 2
    x2 = (rect2['coords'][1] + rect2['coords'][3]) / 2
    y2 = (rect2['coords'][0] + rect2['coords'][2]) / 2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

def calculate_rectangle_area(rect):
    return (rect['coords'][2] - rect['coords'][0]) * (rect['coords'][3] - rect['coords'][1])

def knn_grouping(rectangles1, rectangles2, rectangles3):
    groups = []
    for i in range(len(rectangles1)):
        rect1 = rectangles1[i]
        closest_rect2 = None
        closest_rect2_distance = float('inf')
        for rect2 in rectangles2:
            if is_rect_in_groups(rect2, groups):
                continue
            if rect1['coords'][3] <= rect2['coords'][1]:
                continue
            if rect2['coords'][3] <= rect1['coords'][1]:
                continue
            # Check if rect2 is mostly inside rect1
            if rect2['coords'][1] > rect1['coords'][1] and rect2['coords'][3] < rect1['coords'][3]:
                distance = calculate_center_distance(rect1, rect2) * 0.9  # Reduce distance for rect2
            else:
                distance = calculate_center_distance(rect1, rect2)
            if distance < closest_rect2_distance:
                closest_rect2 = rect2
                closest_rect2_distance = distance
        if closest_rect2 is not None:
            group = [rect1, closest_rect2]
            closest_rect3 = None
            closest_rect3_distance = float('inf')
            for rect3 in rectangles3:
                if is_rect_in_groups(rect3, groups):
                    continue
                if closest_rect2['coords'][3] <= rect3['coords'][1]:
                    continue
                if rect3['coords'][3] <= closest_rect2['coords'][1]:
                    continue
                # Check if rect3 is mostly inside rect1
                if rect3['coords'][1] > rect1['coords'][1] and rect3['coords'][3] < rect1['coords'][3]:
                    distance = calculate_center_distance(closest_rect2, rect3) * 0.9  # Reduce distance for rect3
                else:
                    distance = calculate_center_distance(closest_rect2, rect3)
                if distance < closest_rect3_distance:
                    closest_rect3 = rect3
                    closest_rect3_distance = distance
            if closest_rect3 is not None:
                group.append(closest_rect3)
            groups.append(group)
        else:
            groups.append([rect1])
    return groups


def get_color_proximity_box(b, boxes):
    green = "green", (0, 163, 108)
    red = "red", (0, 0, 255)
    color = green

    b_top, b_left, b_bottom, b_right = b['coords']
    b_width = b_right - b_left
    b_height = b_bottom - b_top
    b_center_x = b_left + (b_width / 2)
    b_center_y = b_top + (b_height / 2)
    b_area = b_width * b_height

    for box in boxes:
        if b['coords'] is not box['coords']:
            box_top, box_left, box_bottom, box_right = box['coords']
            box_width = box_right - box_left
            box_height = box_bottom - box_top
            box_center_x = box_left + (box_width / 2)
            box_center_y = box_top + (box_height / 2)
            box_area = box_width * box_height

            # Calculate the distance between the centers of the two boxes
            distance = math.sqrt((b_center_x - box_center_x) ** 2 + (b_center_y - box_center_y) ** 2)

            # Calculate the angle between the boxes
            angle = math.atan2(abs(b_center_y - box_center_y), abs(b_center_x - box_center_x))
            
            # Calculate the proportional threshold based on area and distance
            threshold = b_width * (1 - (abs(b_area - box_area) / max(b_area, box_area))) + distance * (2 / math.pi) * (1 - (angle / (math.pi / 2)))
            
            if distance < threshold:
                color = red
                break

    return color



def predict_facemask_pose_people(paths: list or str or Image.Image, people_detection_model: str = 'yolo', face_detection_model: str = 'yolo', mask_detection_model: str = 'yolo', outdir_people: str=None, outdir_faces: str=None, outdir_facemasks: str=None, outdir_all: str=None, group_predicts:bool=False):

    if isinstance(paths, str) or isinstance(paths, Image.Image):
        paths = [paths]

    if outdir_people is not None:
        outdir_people = f"{outdir_people}{people_detection_model}/"
    
    if outdir_faces is not None:
        outdir_faces = f"{outdir_faces}{face_detection_model}/"
    
    if outdir_facemasks is not None:
        outdir_facemasks = f"{outdir_facemasks}{mask_detection_model}/"

    people_detections = None
    face_detections = None
    mask_detections = None

    if 'yolo' == people_detection_model:
        people_detections = predict_people_yolov8(paths, outdir=outdir_people)
    elif 'retinanet' == people_detection_model:
        people_detections = predict_people_retinanet(paths, outdir=outdir_people)

    if 'yolo' == face_detection_model:
        face_detections = predict_faces_yolov8(paths, outdir=outdir_faces)
    
    if 'yolo' == mask_detection_model:
        mask_detections = predict_masks_yolov8(paths, outdir=outdir_facemasks)
    elif 'vgg19' == mask_detection_model:
        model = ModelVGG19()
        mask_detections = model.predict(paths, outdir=outdir_facemasks)


    imgs_tagged = []
    if None not in [people_detections, face_detections, mask_detections]:
        for path, prediction_people, prediction_faces, prediction_masks in zip(paths, people_detections, face_detections, mask_detections):

            _, _, boxes_people, n_people, end_time_people = prediction_people
            _, _, boxes_faces, n_faces, end_time_faces = prediction_faces
            _, _, boxes_masks, n_masks, end_time_masks = prediction_masks

            n_masks_good = 0
            n_masks_improper = 0
            n_masks_bad = 0
            n_dist_viol = 0

            if isinstance(path, str):
                img = Image.open(path)
            else:
                img = path

            annotator = Annotator(np.ascontiguousarray(img))
            
            if not group_predicts:

                for b in boxes_people:
                    coords = b['coords']
                    label = b['label']

                    ncol, col = get_color_proximity_box(b, boxes_people)
                    if ncol == "red":
                        n_dist_viol += 1

                    annotator.box_label(coords, f'{label}', color=col)

                for b in boxes_masks:
                    coords = b['coords']
                    label = b['label']
                    if label == 'mask':
                        n_masks_good += 1
                        annotator.box_label(coords, f'{label}', color=(160,252,0))
                    elif label == 'no_mask':
                        n_masks_bad += 1
                        annotator.box_label(coords, f'{label}', color=(252,160,0))
                    else:
                        n_masks_improper += 1
                        annotator.box_label(coords, f'{label}', color=(160,160,0))
            
            if group_predicts:
                groups = knn_grouping(boxes_people, boxes_faces, boxes_masks)
                for group in groups:
                    b = group[0]['coords']
                    if len(group) == 3:
                        label = group[2]['label']
                        if label == 'mask':
                            n_masks_good += 1
                            annotator.box_label(b, f'exposed_{label}', color=(160,252,0))
                        elif label == 'no_mask':
                            n_masks_bad += 1
                            annotator.box_label(b, f'exposed_{label}', color=(252,160,0))
                        else:
                            n_masks_improper += 1
                            annotator.box_label(b, f'exposed_{label}', color=(160,160,0))
                    elif len(group) == 2:
                        annotator.box_label(b, f'exposed', color=(160,160,0))
                    elif len(group) == 1:
                        annotator.box_label(b, 'reversed', color=(0,160,252))
                                    
            img = annotator.result()
            times = (end_time_people, end_time_faces, end_time_masks)
            imgs_tagged.append((path, img, n_people, n_faces, (n_masks_good, n_masks_improper, n_masks_bad), int(n_dist_viol/2), times))
        if outdir_all is not None:
            for path_out, img, _, _, _, _, _ in imgs_tagged:
                utils.save_img(outdir_all, path_out, img)
    return imgs_tagged
