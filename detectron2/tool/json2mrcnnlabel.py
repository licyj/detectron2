import argparse
from genericpath import exists, isdir, isfile
import json
import os
import numpy as np

# Fixed dataset labels
label_classes = ['head', 'hand']
categories = [
    {'supercategory': label, 'id': idx, 'name': label}
    for idx, label in enumerate(label_classes, 1)
]


def process_annotation(image_id, anno_id, segmentation, category):
    area = 0
    left_most = int(min(segmentation[::2]))
    top_most = int(min(segmentation[1:][::2]))
    right_most = int(max(segmentation[::2]))
    bot_most = int(max(segmentation[1:][::2]))
    bbox = [left_most, top_most, right_most - left_most,
            bot_most - top_most]   # find min max (x, y)
    if category == 'left_hand' or category == 'right_hand':
        category = 'hand'
    return {
        # segmentation is a list contain multiple regions that express one instance
        'segmentation': [segmentation.tolist()],
        'area': area,
        'iscrowd': 0,
        'image_id': image_id,
        'bbox': bbox,
        # category id start from 1,
        'category_id': label_classes.index(category) + 1,
        'id': anno_id
    }


anno_id = 0


def convert(args, dirname: str, filename: str, image_id: int = 0):
    global anno_id
    annotations = []
    frame_folder = args.frame_folder
    assert filename.endswith('.json')
    with open(f'{args.label_folder}/{dirname}/{filename}', 'r') as fp:
        src_json = json.load(fp)

    # set image annotation
    # image filename without extension
    img_filename = src_json['image_filename']
    assert os.path.exists(
        f'{frame_folder}/{dirname}/{img_filename}.jpg'), f'{frame_folder}/{dirname}/{img_filename}.jpg doesn\'t exists.'
    img = cv2.imread(f'{frame_folder}/{dirname}/{img_filename}.jpg')
    height, width, _ = img.shape
    image = {
        'height': height,
        'width': width,
        'id': image_id,
        'file_name': f'{dirname}/{img_filename}.jpg'
    }

    # set segment annotation
    labels = src_json['labels']
    print(f'parsing {img_filename} with {len(labels)} labels')
    for label in labels:
        label_cls = label['label_class']
        label_type = label['label_type']
        assert label_type == 'polygon', 'currently only support polygon format annotation'
        regions = label['regions']
        print(f'{len(regions)} regions in {label_cls}')
        for region in regions:
            # TODO: segmentation should be [[x_1, y_1, x_2, y_2, ...]] format
            segmentation = np.asarray([
                [pts['x'], pts['y']]
                for pts in region
            ]).flatten()
            annotations.append(process_annotation(
                image_id, anno_id, segmentation, label_cls))
            anno_id += 1
    return image, annotations

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("label_folder", type=str, help="path/to/label/folder")
    parser.add_argument("frame_folder", type=str, help="path/to/frame/folder")    
    parser.add_argument("json_file", type=str, help="path/to/train_json")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    import cv2
    args = get_args()
    image_id = 0
    images = []
    annotations = []
    label_folder = args.label_folder
    for dirname in [name for name in os.listdir(label_folder) if os.path.isdir(f'{label_folder}/{name}') and not name.startswith('.')]:
        for filename in [name for name in os.listdir(f'{label_folder}/{dirname}') if os.path.isfile(f'{label_folder}/{dirname}/{name}') and not name.startswith('.')]:
            image, annotation = convert(args, dirname, filename, image_id=image_id)
            images.append(image)
            annotations.extend(annotation)
            image_id += 1

    with open(args.json_file, 'w') as fp:
        json.dump({
            'images': images,
            'categories': categories,
            'annotations': annotations
        }, fp)
