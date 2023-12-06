from tensorflow.python import pywrap_tensorflow
import tensorflow as tf
import numpy as np
#import quantized_function as qf
import os
import struct

DATAWIDTH = 8


def get_max_np(array):
    return np.max(array)

def get_max_tf(tensor):
    tensor = tf.reshape(tensor, [1, -1])
    return tf.reduce_max(tensor, axis=1)


def get_scale(data, data_width=8):
    data_max = np.max(np.abs(data))  # 计算最大值
    scale = data_max / (2**(data_width - 1) - 1)
    return scale, data_max

def quantized_np(array,scale,data_width=8):
    quantized_array= np.round(array/scale)
    quantized_array = np.maximum(quantized_array, -2**(data_width-1))
    quantized_array = np.minimum(quantized_array, 2**(data_width-1)-1)
    return quantized_array

def quantized_tf(tensor,scale,data_width=8):
    data_quantize = tf.round(tensor / scale)
    data_quantize = tf.maximum(data_quantize, -2**(data_width-1))
    data_quantize = tf.minimum(data_quantize, 2**(data_width-1)-1)
    return data_quantize

# path = '../path/to/save/checkpoint/ckpt-1.data-00000-of-00001'
def quantized_parameter_save(save=True):
    reader = tf.compat.v1.train.NewCheckpointReader(
    '../checkpoint/ckpt-1')  # tf.train.NewCheckpointReader

    WEIGHT_KEY = [
        'conv2d/kernel',
        'conv2d_1/kernel',
        'dense/kernel',
        'dense_1/kernel'
    ]
    max_save=[]
    scale_save=[]
    for key in WEIGHT_KEY:
        value = reader.get_tensor(key)
        print(key, value.shape)
        scale,value_max = get_scale(value,data_width=DATAWIDTH)
        max_save.append(key+" "+str(value_max))
        scale_save.append(key+" "+str(scale))
    print(len(scale_save))
    print(len(max_save))
    print(scale_save)
    
    if save == True:
        file_path='./parameter/quantized/scale.txt'
        fileobject=open(file_path,'w')
        for word in scale_save:
            fileobject.write(word)
            fileobject.write("\n")
        fileobject.close()

        file_path='./parameter/quantized/max.txt'
        fileobject=open(file_path,'w')
        for word in max_save:
            fileobject.write(word)
            fileobject.write("\n")
        fileobject.close()
        print("Save Done")
    else:
        return scale_save,max_save

def quantized_save(max_dict):
    scale_save,max_save = quantized_parameter_save(False)

    LAYER = [
        'input_0',
        'output_1',
        'output_2',
        'output_3',
        'output_4',
        'output_5',
        'output_6',
    ]
    scale_dict={}
    for i in range(7):
        print(len(max_dict[LAYER[i]]))
        max_dict[LAYER[i]] = max(max_dict[LAYER[i]])
        print(max_dict[LAYER[i]])
        scale_dict[LAYER[i]] = get_scale(max_dict[LAYER[i]])
        scale_save.append(LAYER[i]+" "+str(scale_dict[LAYER[i]]))
        max_save.append(LAYER[i]+" "+str(max_dict[LAYER[i]]))
    
    file_path='./parameter/quantized/scale.txt'
    fileobject=open(file_path,'w')
    for word in scale_save:
        fileobject.write(word)
        fileobject.write("\n")
    fileobject.close()

    file_path='./parameter/quantized/max.txt'
    fileobject=open(file_path,'w')
    for word in max_save:
        fileobject.write(word)
        fileobject.write("\n")
    fileobject.close()
    print("Save Done")
    
        


def main():
    quantized_parameter_save(True)

if __name__ == "__main__":
    main()
    