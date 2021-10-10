import os
import pathlib
import cv2
import json
import numpy as np
from pathlib import Path
import argparse
# img_name = 'tt.png'
# img_path = os.path.join('.', img_name)
# img = cv2.imread(img_name)

# def process(img_path, save_folder, mask_folder):
    # file_name = save_folder[save_folder.rindex(os.sep)+1:]
    # file_name = file_name[:file_name.rindex('.')]
    # cv2.imwrite(os.path.join(save_folder, file_name) + '.jpg', img)
    # print(os.path.join(save_folder, file_name) + '.jpg')
def process(data):
    img_path, save_folder, json_mask_folder, yaml_mask_folder, png_mask_folder = data
    file_name = img_path[img_path.rindex(os.sep)+1:]
    file_name = file_name[:file_name.rindex('.')]
    file_name = int(file_name)
    file_name = '{:03d}'.format(file_name)

    yaml_mask_file = os.path.join(yaml_mask_folder, file_name+'.yaml')
    json_mask_file = os.path.join(json_mask_folder, file_name+'__labels.json')
    png_mask_file = os.path.join(png_mask_folder, file_name+'.png')

    file_name += '.jpg'

    img = cv2.imread(img_path)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    skin_lower = np.array([0, 47, 90], dtype='uint8')
    skin_upper = np.array([40, 255, 255], dtype='uint8')
    mask_skin = cv2.inRange(hsv, skin_lower, skin_upper)

    lip_lower = np.array([160, 90, 100], dtype='uint8')
    lip_upper = np.array([180, 255, 255], dtype='uint8')
    mask_lip = cv2.inRange(hsv, lip_lower, lip_upper)

    mask = cv2.bitwise_or(mask_lip, mask_skin)

    indices_one = mask != 0
    indices_zero = mask == 0

    mask[indices_zero] = 0
    mask[indices_one] = 1

    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)
    sizes = stats[1:, -1]
    nb_components -= 1
    min_size = 500
    for i in range(0, nb_components):
        if sizes[i] < min_size:
            mask[output == i + 1] = 0
    # mask = cv2.GaussianBlur(mask, (3, 3), 0)
    mask[mask != 0] = 255

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for i, cnt in enumerate(contours):
        cv2.drawContours(mask,[cnt], 0, (255), -1)

    # write mask file
    labels = []
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mask_pt = np.where(mask)
    # print(mask_pt)
    # unmask = np.zeros(mask.shape, dtype='uint8')
    # unmask[mask_pt] = 255
    # cv2.imshow('mask', unmask)
    # cv2.waitKey(0)
    contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])
    max_idx = contours.index(max(contours, key=cv2.contourArea))
    contours = [contours[max_idx]] + contours[:max_idx] + contours[max_idx+1:]
    if len(contours) > 2:
        c = contours[-1]
        contours.pop()
        contours.insert(2, c)
    mask_pt = list(zip(mask_pt[1], mask_pt[0]))
    label_classes = ['head', 'right_hand', 'left_hand', 'others']
    yaml_data = {}
    png_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for i, cnt in enumerate(contours):
        regions = []
        xs, ys = [], []
        for x, y in mask_pt:
            if cv2.pointPolygonTest(cnt, (x, y), False) >= 0:
                x, y = int(x), int(y)
                # regions.append({'x': int(x), 'y': int(y)})
                xs.append(int(x))
                ys.append(int(y))
        png_mask[ys, xs] = i + 1

        label_class = label_classes[i] if i < len(label_classes) - 1 else label_classes[-1]
        yaml_data[label_class] = { 'x': xs, 'y': ys }

        for i in range(len(cnt)):
            regions.append({'x': int(cnt[i, 0, 0]), 'y': int(cnt[i, 0, 1])})
        labels.append({
            "label_type": "polygon",
            "label_class": label_class,
            "source": "manual",
            "anno_data": {
                "visibility": "full",
                "good_quality": True,
                "material": "human"
            },
            'regions': [regions]
        })
    x = {
        'image_filename': file_name,
        'complete': True,
        'labels': labels
    }
    # print(x)
    cv2.imwrite(png_mask_file, png_mask)

    with open(yaml_mask_file, 'w') as f:
        yaml.dump(yaml_data, f)

    with open(json_mask_file, 'w') as json_file:
        json.dump(x, json_file)

    img = cv2.bitwise_and(img, img, mask=mask)

    cv2.imwrite(os.path.join(save_folder, file_name), img)
    return img


def paste_prev(img, save_folder, mask, prev, i):
    for idx in zip(mask[0], mask[1]):
        img[idx] = prev[idx]
    cv2.imwrite(os.path.join(save_folder, '{:03d}.jpg'.format(i)), img)
    return img

def worker_init():
    # ignore the SIGINI in sub process, just print a log
    def sig_int(signal_num, frame):
        # print('signal: {}'.format(signal_num))
        print('KeyInterrupt')
        sys.exit(0)
    signal(SIGINT, sig_int)


def run(data):
    file_name, path, process_video_dir, process_frame_dir, json_mask_dir, yaml_mask_dir, png_mask_dir = data
    file_path = os.path.join(path, file_name)
    save_folder = os.path.join(process_frame_dir, file_name)
    json_mask_folder = os.path.join(json_mask_dir, file_name)
    yaml_mask_folder = os.path.join(yaml_mask_dir, file_name)
    png_mask_folder = os.path.join(png_mask_dir, file_name)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = 30
    pathlib.Path(save_folder).mkdir(exist_ok=True)
    pathlib.Path(json_mask_folder).mkdir(exist_ok=True)
    pathlib.Path(yaml_mask_folder).mkdir(exist_ok=True)
    pathlib.Path(png_mask_folder).mkdir(exist_ok=True)


    files = sorted([os.path.join(file_path, file) for file in os.listdir(file_path)])
    files_len = len(files)
    save_folders = [save_folder] * files_len
    json_mask_folders = [json_mask_folder] * files_len
    yaml_mask_folders = [yaml_mask_folder] * files_len
    png_mask_folders = [png_mask_folder] * files_len

    thread_data = zip(files, save_folders, json_mask_folders, yaml_mask_folders, png_mask_folders)
    # use multiple threading to process IO problem
    max_workers = 10
    concurrent = futures.ThreadPoolExecutor(max_workers)
    with concurrent as ex:
        for _ in enumerate(ex.map(process, thread_data)):
            pass

    frame0 = cv2.imread(files[0])
    video = cv2.VideoWriter(os.path.join(process_video_dir, file_name) + '.avi', 
        fourcc, fps, frame0.shape[:2][::-1])

    for file in sorted(os.listdir(save_folder)):
        result = cv2.imread(os.path.join(save_folder, file))
        video.write(result)
    video.release()

def get_args():
    parser= argparse.ArgumentParser()
    parser.add_argument('mask_root', type=str, help='path/to/mask_root')
    parser.add_argument('label_root', type=str, help='path/to/label_root')
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    png_mask_folder = args.mask_root
    label_root =args.label_root
    label_classes = ['head', 'right_hand', 'left_hand', 'others']

    for v in os.listdir(png_mask_folder):
        png_mask_dir = os.path.join(png_mask_folder, v)
        json_mask_dir = os.path.join(label_root, v)
        pathlib.Path(json_mask_dir).mkdir(exist_ok=True)

        for path in sorted(os.listdir(png_mask_dir)):
            file_name = Path(path).stem
            png_mask_file = os.path.join(png_mask_dir, file_name+'.png')
            print('\r'+' '*60+'\r'+png_mask_file, end='')
            json_mask_file = os.path.join(json_mask_dir, file_name+'__labels.json')
            png = cv2.imread(png_mask_file, 0)
            labels = []
            for i, label_class in enumerate(label_classes, 1):
                mask = png.copy()
                mask[mask != i] = 0
                if not mask.any():
                    continue
                contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                for i, cnt in enumerate(contours):
                    regions = []
                    for i in range(len(cnt)):
                        regions.append({'x': int(cnt[i, 0, 0]), 'y': int(cnt[i, 0, 1])})
                    labels.append({
                        "label_type": "polygon",
                        "label_class": label_class,
                        "source": "manual",
                        "anno_data": {
                            "visibility": "full",
                            "good_quality": True,
                            "material": "human"
                        },
                        'regions': [regions]
                    })
            x = {
                'image_filename': file_name,
                'complete': True,
                'labels': labels
            }
            with open(json_mask_file, 'w') as json_file:
                json.dump(x, json_file)




if __name__ == '__main__':
    main()