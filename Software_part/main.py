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




img_dir = './dataset/300/test_4.jpeg'

SIZE = 32

SYMBOL = {0: '0',
          1: '1',
          2: '2',
          3: '3',
          4: '4',
          5: '5',
          6: '6',
          7: '7',
          8: '8',
          9: '9',
          10: '+',
          11: '-',
          12: '*',
          13: '/',
          }


# def image_cut(img_dir):
#     for file in os.listdir(img_dir):
#         if file.endswith('jpeg'):
#             path = os.path.join(img_dir, file)
#             image_cuts = process.get_image_cuts(
#                 path, dir = img_dir +"/"+ file.split('.')[0]+'_cut', count=0, data_needed=True)
#             return image_cuts

def img_resize(img_dir):
    image = Image.open(img_dir)
    new_image = image.resize((400,300))
    new_image.save(img_dir)
    return img_dir

def img_threshold(img_dir):
    image = cv2.imread(img_dir, 0)

    ret, thresh1 = cv2.threshold(image, 90, 255, cv2.THRESH_BINARY)
    # th3 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    plt.plot(thresh1)
    # th3 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    cv2.imwrite(img_dir, thresh1)

    return img_dir

def image_cut(img_dir):

    path = img_dir
    image_cuts = process.get_image_cuts(
        path, dir = img_dir +"/"+ img_dir.split('.')[0]+'_cut', count=0, data_needed=True)
    return image_cuts


from tensorflow.keras.models import load_model
def main(mode=1):


    # Replace the path with the path to your .h5 file
    model_path = 'model_new_3.h5'
    model = load_model(model_path)
    model.summary()

    img_dir_new = img_resize(img_dir)
    img_dir_new1 = img_threshold(img_dir_new)
    img = image_cut(img_dir_new1)

    formula = ''
    print(np.size(img, 0))
    for i in range(np.size(img, 0)):
        cv2.imshow('image',img[i])
        image = np.reshape(img[i], (1, SIZE, SIZE, 1))

        prediction = model.predict(image)
        index = np.argmax(prediction[0])
        # print("predicted value is " + str(index), prediction[0][index])

        # index = inference(img[i])
        formula += SYMBOL[index]
        # print(img[i].size)
    print(formula)

    # result = calculator(formula)
    # print(result)



if __name__ == '__main__':
    main(1)
