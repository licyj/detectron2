import os
import shutil
spreadthesign_women_root = '/home/Datasets/ASL/train/0901spreadthesign_women/'
long_sleeve_path = os.path.join(spreadthesign_women_root, 'frame_long_sleeve')
short_sleeve_path = os.path.join(spreadthesign_women_root,  'frame_short_sleeve')
long_sleeve_json =  os.path.join(spreadthesign_women_root,  'json_long_sleeve')
short_sleeve_json =  os.path.join(spreadthesign_women_root, 'json_short_sleeve')

if not os.path.exists(long_sleeve_path):
    os.mkdir(long_sleeve_path)
if not os.path.exists(short_sleeve_path):
    os.mkdir(short_sleeve_path)
if not os.path.exists(long_sleeve_json):
    os.mkdir(long_sleeve_json)
if not os.path.exists(short_sleeve_json):
    os.mkdir(short_sleeve_json)


def main():
    wowen_short_sleeve=['00002','00017','00204','00206','00207','00228','00239',\
                        '00266','00295','00296','00338','00342','00468',\
                        '00472','00481','00495','00519','00560']
    
    women_half_sleeve=['00035','00101','00124','00125','00126','00137','00138',\
                        '00139','00141','00146','00159','00169','00173','00195',\
                        '00198','00205','00238','00274','00285','00290','00310',\
                        '00383','00401','00467','00502','00506']
    
    frames_root = os.path.join(spreadthesign_women_root, 'frame')
    jsons_root = os.path.join(spreadthesign_women_root, 'json')
    #frame
    with open(spreadthesign_women_root+'frame_short_classified.txt','w') as f:
        for video in sorted(os.listdir(frames_root)):
            if video in wowen_short_sleeve or video in women_half_sleeve:
                print(video)
                src = os.path.join(frames_root, video)
                tgt = os.path.join(short_sleeve_path, video)
                f.write(src+ '\t' +tgt+ '\n')
                shutil.move(src, tgt)
            else:
                src = os.path.join(frames_root, video)
                tgt = os.path.join(long_sleeve_path, video)
                f.write(src+ '\t' +tgt+ '\n')
                shutil.move(src, tgt) 
    #json
    with open(spreadthesign_women_root+'json_short_classified.txt','w') as file:
       for video in sorted(os.listdir(jsons_root)):
            if video in wowen_short_sleeve or video in women_half_sleeve:
                src = os.path.join(jsons_root, video)
                tgt = os.path.join(short_sleeve_json, video)
                file.write(src+ '\t' +tgt+ '\n')
                shutil.move(src, tgt)
            else:
                src = os.path.join(jsons_root, video)
                tgt = os.path.join(long_sleeve_json, video)
                file.write(src+ '\t' +tgt+ '\n')
                shutil.move(src, tgt)
if __name__ == '__main__':
    main()