import os
import cv2
import json
import yaml
from datetime import datetime
import numpy as np
from pathlib import Path
import argparse

jsons_folder = os.path.join('TSL', 'mask', 'json_contour', 'train')
imgs_folder = os.path.join('TSL', 'no_sub', 'train', 'frames')
png_folders = os.path.join('just_hands')

label_order = ['head', 'right_hand', 'left_hand', 'others']
colors = [None, (0, 0, 255), (255, 0, 0)]

vids = ['012', '016', '018', '022', '023', '688', '749']

vid = '000'
filename = '032'

def get_args():
    parser= argparse.ArgumentParser()
    parser.add_argument('frame_root', type=str, help="path/to/root/frame")
    parser.add_argument('mask_root', type=str, help="path/to/root/mask")
    parser.add_argument('label_root', type=str, help="path/to/root/label")
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    for vid in os.listdir(args.label_root):
        img_folder = os.path.join(args.frame_root, vid)
        json_folder = os.path.join(args.label_root, vid)
        png_folder = os.path.join(args.mask_root, vid)
        
        if not os.path.exists(png_folder):
            os.makedirs(png_folder)

        for filename in os.listdir(img_folder):
            filename = Path(filename).stem
            img = cv2.imread(os.path.join(img_folder, filename+'.jpg'))
            json_file = os.path.join(
                json_folder, '{}__labels.json'.format(filename))
            png_file = os.path.join(png_folder, filename+'.png')

            with open(json_file) as f:
                json_data = json.load(f)

            # full_mask = np.zeros(img.shape[:2], dtype=np.uint8)
            full_mask2 = np.zeros(img.shape[:2], dtype=np.uint8)
            for labels in json_data['labels']:
                mask = np.zeros(img.shape[:2], dtype=np.uint8)
                label_class = labels['label_class']
                area_and_pts = []
                for region in labels['regions']:
                    pts = np.array([(min(int(coords['x']), img.shape[1]-1),
                                min(int(coords['y']), img.shape[0]-1)) for coords in region])
                    area_and_pts.append((pts, cv2.contourArea(pts)))
                for pts, area in sorted(area_and_pts, key=lambda x: x[1])[::-1]:
                    if area < 100:
                        continue
                    coords = np.transpose(pts)
                    if (mask[coords[1], coords[0]] != 0).any():
                        cv2.drawContours(mask, [pts], -1, (0, 0, 0), -1)
                    else:
                        cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1)
                full_mask2[np.where(mask)] = label_order.index(label_class) + 1
            cv2.imwrite(png_file, full_mask2)