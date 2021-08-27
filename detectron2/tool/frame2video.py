import os,cv2,random

frames = './vis_meeting/'
target_video_dir = './vis_meeting'

# for dir in os.listdir(frames):
#     for f in os.listdir(os.path.join(frames,dir)):
#         imgpath = os.path.join(frames,dir,f)
#         img = cv2.imread(imgpath)
#         fps = 30
#         size = (img.shape[1],img.shape[0])
#         fourcc = cv2.VideoWriter_fourcc(*"XVID")
#         video_write = cv2.VideoWriter(frames+dir+'.avi',fourcc,fps,size)
dir = 5
imgpath = frames+str(dir)+'/'+'261.jpg'

img = cv2.imread(imgpath)  # 读取保存的任意一张图片

fps = 5 # 根据第二节介绍的方法获取视频的 fps
size = (img.shape[1],img.shape[0])  #获取视频中图片宽高度信息
fourcc = cv2.VideoWriter_fourcc(*"XVID") # 视频编码格式
videoWrite = cv2.VideoWriter(frames+'new_{}.avi'.format(str(dir)),fourcc,fps,size)# 根据图片的大小，创建写入对象 （文件名，支持的编码器，帧率，视频大小（图片大小））

files = os.listdir(frames+'{}/'.format(str(dir)))
out_num = len(files)
for f in sorted(os.listdir( os.path.join(frames,str(dir)))):
    fileName = frames+str(dir)+"/"+f 
    print(fileName)
    img = cv2.imread(fileName)
    videoWrite.write(img)# 将图片写入所创建的视频对象