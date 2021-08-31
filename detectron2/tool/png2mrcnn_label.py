import os
import cv2
import json
from pathlib import Path
import numpy as np
import time
import argparse

'''
	path need to change:
		png_masks_folder
		folder
		json_file_name
'''
def run(args):
	png_masks_folder = args.png_masks_folder
	annotations = []
	images = []
	category_id = 1
	image_id = 1 
	id = 1
	start = 0
	end = 9
	label_classes = ['background', 'head', 'right_hand', 'left_hand']
	categories = [{ 'supercategory': label, 'id': idx, 'name': label } for idx, label in enumerate(label_classes[1:], 1)]
	png_folders = sorted([os.path.join(png_masks_folder, folder) for folder in os.listdir(png_masks_folder)])
	for png_folder in png_folders:
		folder = os.path.join(Path(png_folder).stem)
		print('folder',folder,'\t')
		for p in sorted(os.listdir(png_folder)):
			png_path = os.path.join(png_folder, p)
			file_name = str(os.path.join(folder, Path(p).stem+'.jpg')).replace('\\','/')
			print('\r'+' '*60+'\r'+file_name, end='')
			mask = cv2.imread(png_path, 0)
			height, width = mask.shape[:2]
			images.append({
				'height': height,
				'width': width,
				'id': image_id,
				'file_name': file_name,
				})
			for i in range(1, 4):
				area = 0
				iscrowd = 0
				segmentations = []

				if not (mask == i).any():
					continue

				ys, xs = np.where(mask==i)

				mask_ = mask.copy()
				mask_[mask_ != i] = 0

				contours, hierarchy = cv2.findContours(mask_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
				for cnt in contours:
					segmentation = []
					segmentation.extend([float(pt) for pt in cnt.flatten()])
					area += cv2.contourArea(cnt)
					segmentations.append(segmentation)
				area = float(area)

				bbox = [int(min(xs)), int(min(ys)), int(max(xs)-min(xs)), int(max(ys)-min(ys))]

				# category_id = i - 1
				category_id = i
				annotations.append({
					'segmentation': segmentations,
					'area': area,
					'iscrowd': iscrowd,
					'image_id': image_id,
					'bbox': bbox,
					'category_id': category_id,
					'id': id,
					})

				id += 1

			image_id += 1

	order_list = [i for i in range(start, end+1)]
	with open(args.json_file_name, 'w', encoding='utf-8') as json_file:
		json.dump({
			'images': images,
			'categories': categories,
			'annotations': annotations
		}, json_file)

	print('\n{} done.'.format(args.json_file_name))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("TASK",
                        type=str,
                        help="task name")
    parser.add_argument("png_masks_folder",
                        type=str,
                        help="mask folder")    
    parser.add_argument("json_file_name",
                        type=str,
                        help="model name")
    args = parser.parse_args()
    return args

def main():
	args = get_args()
	run(args)

if __name__ == '__main__':
	main()