import sys
sys.path.append('Yolov8')
sys.path.append('RetinaNet')
sys.path.append('VGG19')

import cv2
import utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from facemaskposedetection import predict_facemask_pose_people


DATA_DIR = 'data/'
OUTPUT_DIR = f'{DATA_DIR}output/'
INPUT_DIR = f'{DATA_DIR}input/'
OUTPUT_PEOPLE_DIR = f'{OUTPUT_DIR}people/'
OUTPUT_FACES_DIR = f'{OUTPUT_DIR}faces/'
OUTPUT_MASKS_DIR = f'{OUTPUT_DIR}masks/'
OUTPUT_ALL_DIR = f'{OUTPUT_DIR}all/'

PLOTS_DIR = f'{DATA_DIR}statistics/plots/'
OUT_PLOT_PNG = f'{PLOTS_DIR}out_statistics_images_plot_PREDICTION.png'

TRULY_TAGGED_CSV = f'{DATA_DIR}statistics/truly_tagged.csv'
PREDICTED_TAGGED_CSV = f'{DATA_DIR}statistics/predicted_tagged.csv'

PEOPLE_DETECTION_MODEL = 'yolo' # 'yolo', 'retinanet'
FACE_DETECTION_MODEL = 'yolo'
MASK_DETECTION_MODEL = 'yolo' # 'yolo', 'vgg19'


def save_metrics(predictions):
    img_names = []
    n_people_predicted = []
    n_faces_predicted = []
    n_masks_good_predicted = []
    n_masks_improper_predicted = []
    n_masks_bad_predicted = []

    times_people = []
    times_faces = []
    times_masks = []

    if predictions is not None:
        for prediction in predictions:

            path_img, new_img, n_people, n_faces, n_masks, n_dist_viol, times = prediction
            n_masks_good, n_masks_improper, n_masks_bad = n_masks

            img_name = path_img.split('/')[-1].split('.')[0]
            img_names.append(img_name)

            n_people_predicted.append(n_people)
            n_faces_predicted.append(n_faces)
            n_masks_good_predicted.append(n_masks_good)
            n_masks_improper_predicted.append(n_masks_improper)
            n_masks_bad_predicted.append(n_masks_bad)

            time_people, time_face, time_masks = times
            times_people.append(time_people)
            times_faces.append(time_face)
            times_masks.append(time_masks)

    new_cols = {
        'image' : img_names,
        'predicted_n_people': n_people_predicted,
        'predicted_n_faces': n_faces_predicted,
        'predicted_n_masks_good': n_masks_good_predicted,
        'predicted_n_masks_improper': n_masks_improper_predicted,
        'predicted_n_masks_bad': n_masks_bad_predicted,
        'time_people': times_people,
        'time_faces': times_faces,
        'time_masks': times_masks
    }

    df1 = pd.read_csv(TRULY_TAGGED_CSV)
    df2 = pd.DataFrame().from_dict(new_cols)

    df = pd.merge(df1, df2, on='image')
    
    df.to_csv(PREDICTED_TAGGED_CSV, index=False)

def do_batch(path_to_files: str):
    images_to_predict = utils.get_images_inside_dir(path_to_files)
    predictions = predict_facemask_pose_people(images_to_predict, \
                                 people_detection_model=PEOPLE_DETECTION_MODEL, face_detection_model=FACE_DETECTION_MODEL, mask_detection_model=MASK_DETECTION_MODEL, \
                                 outdir_people=OUTPUT_PEOPLE_DIR, outdir_faces=OUTPUT_FACES_DIR, \
                                 outdir_facemasks=OUTPUT_MASKS_DIR, outdir_all=OUTPUT_ALL_DIR)
    save_metrics(predictions)
    build_visual_metrics(PREDICTED_TAGGED_CSV)

def build_visual_metrics(dataset):

    df = pd.read_csv(dataset)
    images = df['image']

    col_names = df.columns[1:6]
    for colname in col_names:
        n_of = df[colname]
        predicted_n_of = df[f'predicted_{colname}']
        
        x = np.arange(len(images))
        width = 0.35
        
        fig, ax = plt.subplots()
        ax.bar(x - width/2, n_of, width, label=colname)
        ax.bar(x + width/2, predicted_n_of, width, label=f'predicted_{colname}')
        
        ax.set_xlabel('Image')
        ax.set_ylabel('Number of')
        ax.set_title(f'Comparison of {colname} and predicted_{colname}')
        ax.set_xticks(x)
        ax.set_xticklabels(images, rotation=45)
        ax.legend()
        
        out_filename = OUT_PLOT_PNG.replace('PREDICTION', colname.replace('n_', ''))
        plt.savefig(out_filename)
        plt.clf()

def do_live():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the numpy array to a PIL.Image object
        img = Image.fromarray(frame)
        new_img = None
        if img is not None:
            predictions = predict_facemask_pose_people(img, \
                                 people_detection_model=PEOPLE_DETECTION_MODEL, face_detection_model=FACE_DETECTION_MODEL, mask_detection_model=MASK_DETECTION_MODEL)
            if predictions is not None:
                path_out, new_img, n_people, n_faces, n_masks, n_dist_viol, times = predictions[0]
                n_masks_good, n_masks_improper, n_masks_bad = n_masks
                
                # Get the dimensions of the image
                height, width, _ = new_img.shape
                # Define the size of the bar
                bar_height = 25
                # Create a white bar with the text
                bar = np.zeros((bar_height, width, 3), np.uint8)
                text = f"People: {n_people}  Faces: {n_faces}  Masks Good: {n_masks_good} Masks Bad: {n_masks_bad} Masks Improper: {n_masks_improper} Dist. Viol.: {n_dist_viol}"
                cv2.putText(bar, text, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                # Concatenate the bar and the image vertically
                new_img = np.concatenate((new_img, bar), axis=0)

        if new_img is not None:
            cv2.imshow('Camera', new_img)

        # Wait for the user to press a key
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    

def do_comparison_time():
    yyy_df = pd.read_csv(f'{DATA_DIR}statistics/predicted_tagged_y_y_y.csv')
    ryv_df = pd.read_csv(f'{DATA_DIR}statistics/predicted_tagged_r_y_v.csv')

    common_columns = [column for column in yyy_df.columns if column.startswith("time_") and column != "time_faces"]

    merged_df = yyy_df.merge(ryv_df[common_columns + ['image']], on='image')

    merged_df.to_csv(f'{DATA_DIR}statistics/merged_data.csv', index=False)

    x = np.arange(len(common_columns))
    width = 0.35

    fig, ax = plt.subplots()

    for i, column in enumerate(common_columns):
        yyy_values = merged_df[column + "_x"]
        ryv_values = merged_df[column + "_y"]
        mean_yyy = np.mean(yyy_values)
        mean_ryv = np.mean(ryv_values)
        label_yyy = 'yolo' if column == 'time_people' else 'yolo' if column == 'time_masks' else ''
        label_ryv = 'retinanet' if column == 'time_people' else 'vgg' if column == 'time_masks' else ''
        rects1 = ax.bar(x[i] - width/2, mean_yyy, width, label=label_yyy)
        rects2 = ax.bar(x[i] + width/2, mean_ryv, width, label=label_ryv)

    ax.set_xticks(x)
    ax.set_xticklabels(common_columns)
    ax.legend()

    ax.set_xlabel('Columns')
    ax.set_ylabel('Mean Time')
    ax.set_title('Mean Time Comparison')

    plt.savefig(f'{DATA_DIR}statistics/comparison.png')
    plt.close()

def do_comparison_tests_time():
    yyy_df = pd.read_csv(f'{DATA_DIR}statistics/predicted_tagged_y_y_y.csv')
    ryv_df = pd.read_csv(f'{DATA_DIR}statistics/predicted_tagged_r_y_v.csv')

    common_columns = [column for column in yyy_df.columns if column.startswith("time_")]

    merged_df = yyy_df.merge(ryv_df[common_columns + ['image']], on='image')

    merged_df.to_csv(f'{DATA_DIR}statistics/merged_data.csv', index=False)

    for column in common_columns:
        fig, ax = plt.subplots()
        yyy_values = merged_df[column + "_x"]
        ryv_values = merged_df[column + "_y"]
        ax.plot(merged_df['image'], yyy_values, label='yolo' if column == 'time_people' else 'yolo' if column == 'time_masks' else '')
        ax.plot(merged_df['image'], ryv_values, label='retinanet' if column == 'time_people' else 'vgg' if column == 'time_masks' else '')
        ax.legend()
        ax.set_xlabel('Image')
        ax.set_ylabel(column)
        ax.set_title(f'{column} Comparison')
        ax.set_xticklabels(merged_df['image'], rotation=45)  # Rotate x-labels by 45 degrees
        plt.savefig(f'{DATA_DIR}statistics/{column}_comparison.png')
        plt.close()

if __name__ == '__main__':
    # do_batch(INPUT_DIR)
    do_live()