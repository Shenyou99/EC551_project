# import tensorflow as tf
#
# model_path = 'model_new_2.h5'
# model = tf.keras.models.load_model(model_path)
#
# weights = {layer.name: layer.get_weights() for layer in model.layers if layer.get_weights()}
# # ckpt = tf.train.Checkpoint(**weights)
# # ckpt_manager = tf.train.CheckpointManager(ckpt, 'checkpoint', max_to_keep=3)
# # save_path = ckpt_manager.save()
# # print("Checkpoint saved at:", save_path)
#
# for layer_name, layer_weights in weights.items():
#     if layer_weights:  # Check if layer has weights
#         max_value = max([abs(weight).max() for weight in layer_weights])
#         scale = max_value / (2**7 - 1)  # Assuming 8-bit quantization
#         print(f"Layer: {layer_name}, Max Value: {max_value}, Scale: {scale}")
#
# model.summary()
#
# import numpy as np
#
# def save_as_int8(filename, data, max_value, scale):
#     # 将数据缩放并量化到 int8 范围
#     data_scaled = np.clip(data * scale, -max_value, max_value)
#     data_quantized = np.round(data_scaled).astype(np.int8)
#     # 保存到文件
#     data_quantized.tofile(filename)
#
# # 每个层的最大值和缩放因子
# layer_params = {
#     "conv2d": {"max_value": 0.46227312088012695, "scale": 0.003639945833701787},
#     "conv2d_1": {"max_value": 0.6256335973739624, "scale": 0.004926248798220176},
#     "dense": {"max_value": 0.6881120204925537, "scale": 0.00541820488576814},
#     "dense_1": {"max_value": 1.235285997390747, "scale": 0.009726661396777535}
# }
#
# # 保存每个指定层的权重和偏置
# for layer in model.layers:
#     layer_name = layer.name
#     if layer_name in layer_params:
#         weights = layer.get_weights()
#         if weights:
#             # 保存权重
#             save_as_int8(f'{layer_name}_weights_int8.bin', weights[0], **layer_params[layer_name])
#             # 如果存在偏置，则保存偏置
#             if len(weights) > 1:
#                 save_as_int8(f'{layer_name}_biases_int8.bin', weights[1], **layer_params[layer_name])
#


import numpy as np

# def convert_float32_to_int8(input_file, output_file):
#     # Load the data as float32
#     data_float32 = np.load(input_file)
#
#     # Calculate the maximum absolute value for scale calculation
#     max_abs_value = np.max(np.abs(data_float32))
#
#     # Calculate the scale factor
#     scale = 127 / max_abs_value if max_abs_value != 0 else 0
#
#     # Convert the data to int8 using the scale
#     data_int8 = np.clip(data_float32 * scale, -128, 127).astype(np.int8)
#
#     # Save the int8 data to a new file
#     data_int8.tofile(output_file)
#
#     return max_abs_value, scale
#
# # Replace with the path to your 'weights_layer_0.bin.npy'
# input_file = 'bias_layer_6.bin.npy'
#
# # Specify the desired output file path for the int8 data
# output_file = 'int8/bias_layer_6_int8.bin'
#
# # Convert the data and calculate max value and scale
# max_value, scale = convert_float32_to_int8(input_file, output_file)
#
# print("Max Value:", max_value)
# print("Scale:", scale)
#
# import numpy as np
#
# def count_parameters_in_int8_file(file_path):
#     # Load the data as int8
#     data_int8 = np.fromfile(file_path, dtype=np.int8)
#
#     # Count the number of parameters
#     return data_int8.size
#
# # Replace with the path to your 'weights_layer_5_int8.bin'
# int8_file_path = 'int8/weights_layer_5_int8.bin'
#
# # Count the parameters
# num_parameters = count_parameters_in_int8_file(int8_file_path)
#
# print("Number of parameters in the file:", num_parameters)

def merge_int8_weight_files(file_paths, layer_names, output_file):
    """
    Merge multiple int8 weight files into a single file with each part
    prefixed by its corresponding layer name.

    :param file_paths: List of file paths to the int8 weight files.
    :param layer_names: List of layer names corresponding to each file.
    :param output_file: Path to the output binary file.
    """
    concatenated_data = bytearray()

    for layer_name, file_path in zip(layer_names, file_paths):
        with open(file_path, 'rb') as file:
            data = file.read()
            # Concatenate layer name and data
            concatenated_data.extend(f"{layer_name}\n".encode() + data + b'\n')

    # Write the concatenated data to the output file
    with open(output_file, 'wb') as f:
        f.write(concatenated_data)

# File paths for your int8 weight files
file_paths = [
            'int32/weights_layer_int32_0.bin',
            'int32/weights_layer_int32_2.bin',
            'int32/weights_layer_int32_5.bin',
            'int32/weights_layer_int32_6.bin'
]

# Corresponding layer names
layer_names = ['weight', 'weight_1', 'weight_2', 'weight_3']

# Path for the output merged file
output_file = 'int32/merged_int32_weights.bin'

# Merge the files
merge_int8_weight_files(file_paths, layer_names, output_file)
#
# def merge_int8_bias_files(file_paths, output_file):
#     """
#     Merge multiple int8 bias files into a single file.
#
#     :param file_paths: List of file paths to the int8 bias files.
#     :param output_file: Path to the output binary file.
#     """
#     concatenated_data = bytearray()
#
#     for file_path in file_paths:
#         with open(file_path, 'rb') as file:
#             data = file.read()
#             concatenated_data.extend(data)
#
#     # Write the concatenated data to the output file
#     with open(output_file, 'wb') as f:
#         f.write(concatenated_data)
#
# # File paths for your int8 bias files
# file_paths = [
#                 'int32/bias_layer_int32_0.bin',
#                 'int32/bias_layer_int32_2.bin',
#                 'int32/bias_layer_int32_5.bin',
#                 'int32/bias_layer_int32_6.bin'
# ]
#
# # Path for the output merged file
# output_file = 'int32/merged_int32_bias.bin'
#
# # Merge the files
# merge_int8_bias_files(file_paths, output_file)


# import numpy as np
#
# def convert_float32_to_int32(input_file, output_file, scale_factor=1):
#     """
#     Convert data from float32 to int32 format.
#
#     :param input_file: Path to the input file containing float32 data.
#     :param output_file: Path to the output file for int32 data.
#     :param scale_factor: Scaling factor for conversion. Default is 1.
#     """
#     # Load the float32 data
#     data_float32 = np.fromfile(input_file, dtype=np.float32)
#
#     # Apply the scaling factor and convert to int32
#     data_int32 = np.round(data_float32 * scale_factor).astype(np.int32)
#
#     # Save the int32 data to a new file
#     data_int32.tofile(output_file)
#
# # Example usage
# input_file = 'weights_layer_0.bin'  # Replace with your float32 file path
# output_file = 'int32/weights_layer_int32_0.bin'   # Replace with your desired output path
# scale_factor = 1  # Adjust this as needed
#
# # Perform the conversion
# convert_float32_to_int32(input_file, output_file, scale_factor)

# import numpy as np
#
# def convert_float32_to_int32(input_file, output_file):
#     # Load the data as float32
#     data_float32 = np.load(input_file)
#
#     # Calculate the maximum absolute value
#     max_abs_value = np.max(np.abs(data_float32))
#
#     # Calculate the scale factor
#     scale = 2147483647 / max_abs_value if max_abs_value != 0 else 0
#
#     # Convert the data to int32 using the scale
#     data_int32 = np.round(data_float32 * scale).astype(np.int32)
#
#     # Save the int32 data to a new file
#     data_int32.tofile(output_file)
#
#     return max_abs_value, scale
#
# # File paths
# input_file = 'bias_layer_6.bin.npy'  # Replace with your file path
# output_file = 'int32/bias_layer_int32_6.bin'       # Replace with desired output path
#
# # Convert and calculate max value and scale
# max_value, scale = convert_float32_to_int32(input_file, output_file)
#
# print("Max Value:", max_value)
# print("Scale:", scale)






