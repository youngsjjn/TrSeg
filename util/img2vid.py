import cv2
import numpy as np
import os

from os import listdir
from os.path import isfile, join
import glob

# mypath = 'RUGD_all/danet_color'
# onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

img_array = []
# rgb_path = '/home/ispl3/Documents/cityscapes/leftImg8bit/test/all_test/'
# result_path = "/home/ispl3/PycharmProjects/pytorch/PSPNet/exp/cityscapes/gateseg101_aspp8170_tv/result/epoch_220_retrainval/test/ms/color/"
rgb_path = '/home/ispl3/Documents/ECCV_result/rgb/'
result_path = "/home/ispl3/Documents/ECCV_result/gateseg_color_sub/"
onlyfiles = [f for f in listdir(result_path) if isfile(join(result_path, f))]

# out = cv2.VideoWriter('project.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
onlyfiles.sort()
for filename in onlyfiles:
    img1 = cv2.imread(rgb_path + filename)
    img4 = cv2.imread(result_path + filename)
    # vis = (0.7*img1 + 0.3*img4).astype('uint8')
    vis = np.concatenate((img1, img4), axis=1)
    height, width, layers = vis.shape
    size = (width,height)
    img_array.append(vis)

out2 = cv2.VideoWriter('RUGD_video_4fps.avi', cv2.VideoWriter_fourcc(*'DIVX'), 4, size)

for i in range(len(img_array)):
    out2.write(img_array[i])
out2.release()