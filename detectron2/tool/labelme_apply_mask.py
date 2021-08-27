import os,cv2
import numpy as np

#https://blog.csdn.net/xnholiday/article/details/98886062

'''################################ README ###################################
    此檔案是要把 labelme 視覺化後的原圖和mask
    轉乘 video格式
'''


labelme_dir = '../labelme_json/'
vis_video = './vis_meeting/'
if not os.path.exists(vis_video):
    os.mkdir(vis_video)
for dir in os.listdir(labelme_dir):
    # load img 
    print(dir)
    imgpath = os.path.join(labelme_dir,dir)+'/img.png'
    img = cv2.imread(imgpath)

    # load mask
    print(imgpath)
    mask = cv2.imread(os.path.join(labelme_dir,dir)+'/label.png',cv2.IMREAD_GRAYSCALE)
    # save video
    image=cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=mask)
    cv2.imwrite(vis_video+dir.split('_')[0]+'.jpg',image)
    cv2.imwrite(vis_video+dir.split('_')[0]+'.jpg',image)



#===============================EXAMPLE=====================================
# # load img 
# img = cv2.imread('test/00829/020.jpg')

# # load mask
# mask = cv2.imread('output_masks/00829/020.jpg',cv2.IMREAD_GRAYSCALE)
# mask_array = np.array(mask)

# # save video
# image=cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=mask_array)
# cv2.imwrite('result829.jpg',image)