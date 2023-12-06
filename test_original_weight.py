import tensorflow as tf
model_path = 'model_new_1.h5'
model = tf.keras.models.load_model(model_path)

# weights = {layer.name: layer.get_weights() for layer in model.layers if layer.get_weights()}
# for layer_name, layer_weights in weights.items():
#     if layer_weights:  # Check if layer has weights
#         max_value = max([abs(weight).max() for weight in layer_weights])
#         scale = max_value / (2**7 - 1)  # Assuming 8-bit quantization
#         print(f"Layer: {layer_name}, Max Value: {max_value}, Scale: {scale}")

model.summary()