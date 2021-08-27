import os
import json
import shutil

json_folder = 'ASL_label/'
img_folder = 'ASL_img/'
tgt_img_folder = 'imgs/'
tgt_json_folder = 'jsons/'

counter = 0

for video in os.listdir(img_folder):
    for frame in os.listdir(img_folder + video):
        # find json file and rename
        responding_json_path = json_folder + video + '/' + frame.replace('jpg','json')
        frame_img_path = img_folder + video + '/' + frame
        if not os.path.exists(tgt_img_folder):
            os.mkdir(tgt_img_folder)
        if not os.path.exists(tgt_json_folder):
            os.mkdir(tgt_json_folder)
        
        shutil.copyfile(frame_img_path, tgt_img_folder+'/'+str(counter)+'.jpg')
        shutil.copyfile(responding_json_path, tgt_json_folder+'/'+str(counter)+'.json')
        counter+=1