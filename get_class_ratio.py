import glob
import numpy as np
from PIL import Image

data_path = "/Users/miao/Downloads/eyedata/Edema_trainingset/label_images/*/*.bmp"

class_gray = [0, 128, 191, 255]

all_files = glob.glob(data_path)

print("Image Number: {}".format(len(all_files)))

sum_pixel = 0

sum_per_class = [0] * len(class_gray)

for one_img_path in all_files:
    one_img = np.array(Image.open(one_img_path), np.uint8)
    sum_pixel += one_img.shape[0] * one_img.shape[1]
    for i in range(len(class_gray)):
        sum_per_class[i] += np.sum(one_img == class_gray[i])


for i in range(len(class_gray)):
    print("Class {}: {:.2f}%".format(class_gray[i], float(sum_per_class[i])/sum_pixel*100))

print("Others: {:.2f}%".format(float(sum_pixel-sum(sum_per_class))/sum_pixel*100))