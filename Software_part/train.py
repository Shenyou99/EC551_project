from tqdm import tqdm
import numpy as np
import queue
import cv2
import os
from PIL import Image

def getListFiles(path):
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(os.path.join(root,filespath))
    return ret

def get_images_labels():
    operators = ['plus', 'sub', 'mul', 'div']
    images = None
    labels = None
    for i, op in enumerate(operators):
        image_file_list = getListFiles('./cfs/' + op + '/')
        print('正在载入 ' + op + ' 运算符...')
        for filename in tqdm(image_file_list):
            image = cv2.imread(filename, 2)
            if image.shape != (28, 28):
                image = cv2.resize(image, (28, 28))
            image = np.resize(image, (1, 28 * 28))
            image = (255 - image) / 255
            label = np.zeros((1, 10 + len(operators)))
            label[0][10 + i] = 1
            if images is None:
                images = image
                labels = label
            else:
                images = np.r_[images, image]
                labels = np.r_[labels, label]
    return images, labels

def resize_images(images):
    resized_images = []
    for image in images:
        img = Image.fromarray(image)
        img = img.resize((32, 32), Image.ANTIALIAS)
        resized_images.append(np.array(img))
    return np.array(resized_images)


op_images, op_labels = get_images_labels()
op_images = op_images.reshape(-1,28,28)

from keras.datasets import mnist
from matplotlib import pyplot
import numpy as np
(trainX, trainY), (testX, testY) = mnist.load_data()

trainX = trainX.astype("float32")
testX = testX.astype("float32")
trainX = trainX/255.0
testX = testX/255.0

from tensorflow.keras.utils import to_categorical
trainY = to_categorical(trainY, 14)
testY = to_categorical(testY, 14)

dataset   = np.vstack((op_images, trainX, testX))
datalabel = np.vstack((op_labels, trainY, testY))

dataset = resize_images(dataset)

dataset = dataset.reshape(-1,32,32,1)
print('所有训练数据：',dataset.shape)
print('所有标签数据：',datalabel.shape)

from sklearn.model_selection import train_test_split

trainX, testX, trainY, testY = train_test_split(dataset,datalabel,test_size=0.1, random_state=0)
print('训练数据：',trainX.shape, trainY.shape)
print('测试数据：',testX.shape, testY.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten

# model = Sequential()
# model.add(Conv2D(28, (3, 3), activation='relu', input_shape=(28, 28, 1)))
# model.add(Conv2D(28, (3, 3), activation='relu'))
# model.add(Conv2D(28, (3, 3), activation='relu'))
# model.add(MaxPooling2D((2, 2)))
# model.add(Flatten())
# model.add(Dense(256, activation='relu'))
# model.add(Dense(14,  activation='softmax'))

# model = Sequential()
# model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(32, 32, 1), padding='same'))
# model.add(MaxPooling2D((2, 2)))
# model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
# model.add(MaxPooling2D((2, 2)))
# model.add(Flatten())
# model.add(Dense(1024, activation='relu'))
# model.add(Dense(14, activation='softmax'))
#
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.summary()
model = Sequential([
    Conv2D(32, (5, 5), activation='relu', padding='same', input_shape=(32, 32, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (5, 5), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    # Conv2D(32, (5, 5), activation='relu', input_shape=(32, 32, 1)),
    # MaxPooling2D((2, 2)),
    # Conv2D(64, (5, 5), activation='relu'),
    # MaxPooling2D((2, 2)),
    Flatten(),
    Dense(1024, activation='relu'),
    Dense(14, activation='softmax')
])

model.compile(
    optimizer='adam',  # Optimizer
    loss='categorical_crossentropy',  # Loss function to use
    metrics=['accuracy']  # List of metrics to evaluate during training and testing
)
model.fit(trainX, trainY ,batch_size=64 ,epochs = 10 ,verbose=1,validation_data=(testX, testY))
model.save('model_new_3.h5')


for i, layer in enumerate(model.layers):
    if len(layer.get_weights()) > 0:
        np.save(f'weights_layer_{i}.bin', layer.get_weights()[0])
        np.save(f'bias_layer_{i}.bin', layer.get_weights()[1])
# Save weights
# weights = []
# biases = []
# for layer in model.layers:
#     weights.append(layer.get_weights()[0].flatten())  # 提取权重
#     biases.append(layer.get_weights()[1].flatten())   # 提取偏置
#
# # 将权重和偏置保存到二进制文件中
# np.save('weights_3.bin', np.concatenate(weights))
# np.save('bias_3.bin', np.concatenate(biases))


