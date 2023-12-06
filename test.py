# import numpy as np
#
# def merge_numpy_files(file_paths, layer_names, output_file):
#     """
#     Merge multiple numpy files into a single binary file with each part
#     prefixed by its corresponding layer name.
#
#     :param file_paths: List of file paths to the numpy files.
#     :param layer_names: List of layer names corresponding to each file.
#     :param output_file: Path to the output binary file.
#     """
#     concatenated_data = bytearray()
#
#     for layer_name, file_path in zip(layer_names, file_paths):
#         # Load the numpy array from file
#         data = np.load(file_path)
#         # Convert the numpy array to bytes
#         data_bytes = data.tobytes()
#         # Concatenate the layer name and data
#         concatenated_data.extend(f"{layer_name}\n".encode() + data_bytes + b'\n')
#
#     # Write the concatenated data to the output file
#     with open(output_file, 'wb') as f:
#         f.write(concatenated_data)
#
# # File paths for your numpy weight files
# file_paths = [
#     'bias_layer_0.bin.npy',    # Replace with actual file path
#     'bias_layer_2.bin.npy',    # Replace with actual file path
#     'bias_layer_5.bin.npy',    # Replace with actual file path
#     'bias_layer_6.bin.npy'     # Replace with actual file path
# ]
#
# # Corresponding layer names
# layer_names = ['weight', 'weight_1', 'weight_2', 'weight_3']
#
# # Path for the output merged file
# output_file = 'merged_bias.bin'  # Replace with desired output file path
#
# # Merge the files
# merge_numpy_files(file_paths, layer_names, output_file)

# import numpy as np
#
# def calculate_max_value_and_scale(input_file):
#     # Load data as float32
#     data_float32 = np.fromfile(input_file, dtype=np.float32)
#
#     # Calculate the maximum absolute value
#     max_abs_value = np.max(np.abs(data_float32))
#
#     # Calculate the scale factor
#     scale = 127 / max_abs_value if max_abs_value != 0 else 0
#
#     return max_abs_value, scale
#
# # File path to your 'merged_weights.bin'
# input_file = 'merged_bias.bin'
#
# # Calculate max value and scale
# max_value, scale = calculate_max_value_and_scale(input_file)
#
# print("Max Value:", max_value)
# print("Scale:", scale)
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
# int8_file_path = 'int8/merged_int8_biases.bin'
#
# # Count the parameters
# num_parameters = count_parameters_in_int8_file(int8_file_path)
#
# print("Number of parameters in the file:", num_parameters)

import numpy as np

def convert_npy_to_bin(npy_file_path, bin_file_path):
    # Load the data from the .npy file
    data = np.load(npy_file_path)

    # Save the data to a .bin file
    data.tofile(bin_file_path)

# Specify the path to your .bin.npy file
npy_file_path = 'bias_layer_6.bin.npy'

# Specify the desired path for the .bin file
bin_file_path = 'bias_layer_6.bin'

# Convert the file
convert_npy_to_bin(npy_file_path, bin_file_path)

print(f"File converted and saved to {bin_file_path}")
