############################################module for final project###############################################
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def preprocess_image(std,image,channel_avg):
                   
    clear_image = clear_image_make(std,image)
    #clear_image = image
    # if image is so bright, it will be fixed
    if((channel_avg[0][0]>=0.53) & (channel_avg[0][1]>=0.53) & (channel_avg[0][2]>=0.53)): 
        print('Channel is so bright. So it will be fixed')    
        correction_value = (channel_avg[0][1]-0.45)
        new_image = clear_image - correction_value
        return new_image  
    # if image is so dark, it will be fixed
    elif((channel_avg[0][0]<=0.3) & (channel_avg[0][1]<=0.3) & (channel_avg[0][2]<=0.3)):
        print('Channel is so dark. So it will be fixed')
        correction_value = (0.45-channel_avg[0][1])
        new_image = clear_image + correction_value
        return new_image
    else:
        return image

#test module for openCV 
def image_print():
    npImage=np.array(Image.open("../data/samples/0000000000.png").convert("L"))
    print(npImage.shape)
    image_bgr = cv2.imread('../data/samples/0000000000.png', cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    #plt.imshow(image_rgb); plt.show()
    print(image_rgb,image_rgb.shape)

def clear_image_make_2(std,image):
    # This module is for test
    if((std[0]<=0.14) & (std[1]<=0.14) & (std[2]<=0.14)):
        for image_batch in image:
            new_image_batch = np.transpose(image_batch, axes=(1, 2, 0))
            image_yuv = cv2.cvtColor(new_image_batch, cv2.COLOR_BGR2YUV)
            image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])
            image_rgb = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)
            image_batch = np.transpose(image_rgb, axes=(2, 0, 1))
        return image
    else:
        return image

def clear_image_make(std,image):
    # if image is unclear, it will be fixed
    if((std[0]<=0.14) & (std[1]<=0.14) & (std[2]<=0.14)):
        print('There is Fog, So image will be fixed')
        clear_image = ((image - image.min()) / (image.max()-image.min()))
        #checking std
        after_channle_std = np.std(clear_image[0][0])
        print('after_channle_std:{}'.format(after_channle_std)) 
        return clear_image 
    else:
        return image

def get_diff_image():
    npImage_1=np.array(Image.open("../data/samples/0000000000.png").convert("L"))
    npImage_2=np.array(Image.open("../data/samples/0000000000_2.jpg").convert("L"))
    npImage_3 = npImage_1- npImage_2
    print(npImage_3.shape)
    clear_image = ((npImage_2 - npImage_2.min()) / (npImage_2.max()-npImage_2.min()))
    npImage_4 = npImage_1- clear_image
    print(npImage_4)
    Image.fromarray(npImage_3).save('result.png')

#image_print()

#get_diff_image()
#####################################################end of module ###############################################