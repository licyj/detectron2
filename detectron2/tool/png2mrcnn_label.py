import os
import cv2
import json
from pathlib import Path
import numpy as np
import time

############################### README ##################################
'''
	path need to change:
		png_masks_folder
		folder
		json_file_name
'''
#########################################################################
png_masks_folder = os.path.join('mask_png0507_new', 'train')
png_folders = sorted([os.path.join(png_masks_folder, folder) for folder in os.listdir(png_masks_folder)])
images = []
# label_classes = ['head', 'right_hand', 'left_hand']
label_classes = ['background', 'head', 'right_hand', 'left_hand']
categories = [{ 'supercategory': label, 'id': idx, 'name': label } for idx, label in enumerate(label_classes[1:], 1)]
#[{'supercategory': 'head', 'id': 1, 'name': 'head'}, {'supercategory': 'right_hand', 'id': 2, 'name': 'right_hand'}, {'supercategory': 'left_hand', 'id': 3, 'name': 'left_hand'}]
# categories = [{ 'supercategory': 'hand', 'id': 1, 'name': 'hand' }]
annotations = []
category_id = 1
image_id = 1 
id = 1
start = 0
end = 9
# folders = os.listdir('just_hands') + ['064', '387', '515', '444']
# print('png_folders is: ',[f for f in png_folders[start:end+1] if Path(f).stem not in folders])
# time.sleep(10)
# png_folders = [f for f in png_folders[start:end+1] if Path(f).stem not in folders] + [os.path.join(png_masks_folder, folder) for folder in folders] + \
# 				[os.path.join('TSL', 'mask', 'mask_png0504', 'train', '{}'.format(i)) for i in ['064', '387', '515', '444']]
# print('png_folders is: ',png_folders)
# png_filter = [os.path.join('just_hands', '688', '{:03d}.png'.format(idx)) for idx in range(28, 36)]

for png_folder in png_folders:
	# folder = os.path.join('Datasets/TSL','no_sub', 'train', 'frames', Path(png_folder).stem)
	folder = os.path.join(Path(png_folder).stem)
	print('folder',folder,'\t')
	# print( Path(png_folder).stem)
	for p in sorted(os.listdir(png_folder)):
		png_path = os.path.join(png_folder, p)
		# print('png_path',png_path,'\t')
		# if png_path in png_filter:
		# 	continue
		file_name = str(os.path.join(folder, Path(p).stem+'.jpg')).replace('\\','/')
		# file_name = str(os.path.join(Path(p).stem+'.jpg')).replace('\\','/')
		print('\r'+' '*60+'\r'+file_name, end='')

		# print(file_name)
		mask = cv2.imread(png_path, 0)
		height, width = mask.shape[:2]
		images.append({
			'height': height,
			'width': width,
			'id': image_id,
			'file_name': file_name,
			})

		# area = 0
		# iscrowd = 0
		# segmentations = []

		# if not (mask > 1).any():
		# 	image_id += 1
		# 	continue

		# ys, xs = np.where(mask > 1)

		# mask_ = mask.copy()
		# mask_[mask_ <= 1] = 0

		# contours, hierarchy = cv2.findContours(mask_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		# for cnt in contours:
		# 	segmentation = []
		# 	segmentation.extend([float(pt) for pt in cnt.flatten()])
		# 	area += cv2.contourArea(cnt)
		# 	segmentations.append(segmentation)
			
		# 	area = float(area)

		# 	bbox = [int(min(xs)), int(min(ys)), int(max(xs)-min(xs)), int(max(ys)-min(ys))]

		# 	annotations.append({
		# 		'segmentation': segmentations,
		# 		'area': area,
		# 		'iscrowd': iscrowd,
		# 		'image_id': image_id,
		# 		'bbox': bbox,
		# 		'category_id': category_id,
		# 		'id': id,
		# 		})

		# 	id += 1
		#########################################################################################
		# for i in range(2, 4):
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
				# for x, y in zip(xs, ys):
				# 	if cv2.pointPolygonTest(cnt, (x, y), False) >= 0:
				# 		segmentation.append(float(x))
				# 		segmentation.append(float(y))
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
# json_file_name = '{}-{}_{}.json'.format(start, end, '_'.join(sorted([f for f in folders if int(Path(f).stem) not in order_list])))
json_file_name = 'TSL_segmentation/train/train.json'
with open(json_file_name, 'w', encoding='utf-8') as json_file:
	json.dump({
		'images': images,
		'categories': categories,
		'annotations': annotations
		}, json_file)

print('\n{} done.'.format(json_file_name))