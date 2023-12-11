import tools.image_processing as process
from tools.calculator import calculator
#from tools.test_tool.cnn_model_collect import inference
from tools.cnn_model import inference
from PIL import Image
from matplotlib import pyplot as plt

import numpy as np
import cv2
import os
import datetime

def image_cut(img_dir):

    path = img_dir
    image_cuts = process.get_image_cuts(
        path, dir = img_dir +"/"+ img_dir.split('.')[0]+'_cut', count=0, data_needed=True)
    return image_cuts


img = image_cut('./dataset/300/test_4.jpeg')

formula = ''
print(np.size(img, 0))
for i in range(np.size(img, 0)):

    # image = np.reshape(img[i], (1, SIZE, SIZE, 1))

    # prediction = model.predict(image)
    # index = np.argmax(prediction[0])
    # print("predicted value is " + str(index), prediction[0][index])

    # index = inference(img[i])
    formula += SYMBOL[index]