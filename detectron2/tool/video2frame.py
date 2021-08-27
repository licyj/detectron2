import os
from types import FrameType
import cv2
import time,random, shutil,json
########################### README ###########################
'''
    this file is to split video to frames
    and random choice frame move to label folder
'''
##############################################################
TASK = 'all_videos'
src_video_dir = os.path.join('../../../Datasets/','all_videos')
tar_frame_dir = os.path.join('../../../Datasets/','all_videos_frame')
## for move images to label
need_label_dir = 're_selected_0822/'

def spliting():
    if not os.path.exists(tar_frame_dir):
        os.mkdir(tar_frame_dir)
    for video in sorted(os.listdir(src_video_dir)):
        print(video, 'start splitting.')
        frame_cnt = 0
        video_path = os.path.join(src_video_dir, video)
        dst_video_folder = os.path.join(tar_frame_dir, video.split('.')[0])
        if not os.path.exists(dst_video_folder):
            os.mkdir(dst_video_folder)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(video,"fps:",fps)
        success, frame = cap.read()

        while success:
            frame = cv2.resize(frame, (1280, 720))
            cv2.imwrite(os.path.join(dst_video_folder,\
                        '{:04d}.jpg'.format(frame_cnt)), frame)
            frame_cnt += 1
            success, frame = cap.read()

        with open("ASL-video-{}.txt".format(TASK),'a') as f:
            f.write(video+'\t'+'frames:'+str(frame_cnt)+'\t'+'fps:'+str(fps)+'\n')
        cap.release()

def moving_frame2label():
    # for name in os.listdir(need_label_dir):
    #     os.remove(os.path.join(need_label_dir,name))
    if not os.path.exists(need_label_dir):
        os.mkdir(need_label_dir)
    videos = os.listdir(tar_frame_dir)

    img_size_dict = {}
    selected_coount = 1

    idx = 200
    SELECT_NUM = 400
    while idx <= SELECT_NUM:
        video = random.choice(videos)
        frames = os.listdir(os.path.join(tar_frame_dir, video))
        frame_count = len(frames)
        floor = int(frame_count*0.2)
        ceiling = int(frame_count*0.5)
        # print(type(floor),ceiling)
        # print(type(frames))
        selected_frame = random.choice(frames[floor:ceiling])
        selected_frame_path = os.path.join(tar_frame_dir,video,selected_frame)
        # label_frame_path = os.path.join(need_label_dir,\
        #     '{}_{}_{}.jpg'.format(str(selected_coount),video,selected_frame.replace('.jpg','')))
        label_frame_path = os.path.join(need_label_dir,'{}.jpg'.format(str(idx)))
        shutil.copy(selected_frame_path, label_frame_path)
        print(label_frame_path)
        # img = cv2.imread(selected_frame_path)
        # height, width= img.shape[:2]
        # key = 'h_{}w_{}'.format(height, width)
        # if key not in img_size_dict.keys():
        #     img_size_dict[key]=1
        # else:
        #     img_size_dict[key]+=1
        idx += 1
    # with open('imgae_size_distribution.json','w') as j:
    #     json.dump(img_size_dict, j, indent=4)
    # with open("ASL-video-{}.txt".format(TASK),'a') as f:
    #     f.write('max_frame\t'+str(max_frame)+'min_frame\t'+str(min_frame)+'\n')

def main():
    spliting()
    # moving_frame2label()


if __name__ == '__main__':
    main()
