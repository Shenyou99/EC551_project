# from keras.models import load_model
# import numpy as np
#
# # 加载保存的模型
# model = load_model('model_new_2.h5')
#
# # 提取权重和偏置
#
# for i, layer in enumerate(model.layers):
#     if len(layer.get_weights()) > 0:
#         np.save(f'weights_layer_{i}.bin', layer.get_weights()[0])
#         np.save(f'bias_layer_{i}.bin', layer.get_weights()[1])
#
from keras.models import load_model

# Load the saved model


import numpy as np

# Replace 'file_path' with the actual path to your file
file_path = 'weights_layer_5.bin.npy'

# Load the file
data = np.load(file_path)

# Check if the data type is float32
if data.dtype == np.float32:
    number_of_elements = data.size
    print("Number of float32 elements:", number_of_elements)
else:
    print("Data is not in float32 format")

