# import torch
# import torch.nn as nn
# import numpy as np
# from keras.models import load_model

# Load Keras model
# keras_model = load_model('handrecognition_model.h5')

# Convert Keras model to PyTorch model
# model = nn.Sequential(
#     nn.Conv2d(1, 32, kernel_size=5),
#     nn.ReLU(),
#     nn.MaxPool2d(2, 2),
#     nn.Conv2d(32, 64, kernel_size=3),
#     nn.ReLU(),
#     nn.MaxPool2d(2, 2),
#     nn.Conv2d(64, 64, kernel_size=3),
#     nn.ReLU(),
#     nn.MaxPool2d(2, 2),
#     nn.Flatten(),
#     nn.Linear(64 * 13 * 38, 128),
#     nn.ReLU(),
#     nn.Linear(128, 10),
#     nn.Softmax(dim=1)
# )

# Copy weights from Keras model to PyTorch model
# model_pt_dict = model.state_dict()
# for i, layer in enumerate(keras_model.layers):
#     pt_name = list(model_pt_dict.keys())[i]
#     weights = layer.get_weights()
#     # print(weights, '\n\n\n')
#     weight_shape = weights[0].shape
#     print(weight_shape)
#     bias_shape = weights[1].shape
#     print(bias_shape)
#     pt_weights = np.transpose(weights[0], (3, 2, 0, 1))
#     pt_bias = weights[1]
#     model_pt_dict[pt_name + '.weight'] = torch.from_numpy(pt_weights)
#     model_pt_dict[pt_name + '.bias'] = torch.from_numpy(pt_bias)

# model.load_state_dict(model_pt_dict)

# Save PyTorch model
# torch.save(model.state_dict(), 'handrecognition_model.pth')

# import torch
# import tf as tf
# from tf.keras.models import load_model

# # Load tf model
# model_tf = load_model('handrecognition_model.h5')

# # Create PyTorch model
# model_pt = torch.nn.Sequential()

# # Iterate over layers in tf model
# for layer in model_tf.layers:
#     # Get layer name and type
#     layer_name = layer.name
#     layer_type = type(layer).__name__

#     # Skip layers that do not have weights or biases
#     if not layer.weights:
#         print(f"Skipping layer {layer_name} ({layer_type}) with no weights")
#         continue

#     # Extract weights and biases from tf layer
#     weights_tf = layer.weights[0].numpy()
#     biases_tf = layer.weights[1].numpy()

#     # Convert weights and biases to PyTorch tensors
#     weights_pt = torch.from_numpy(weights_tf)
#     biases_pt = torch.from_numpy(biases_tf)

#     # Add PyTorch layer to model
#     if layer_type == 'Conv2D':
#         conv_layer = torch.nn.Conv2d(in_channels=weights_tf.shape[2], out_channels=weights_tf.shape[3], kernel_size=layer.kernel_size, stride=layer.strides, padding=layer.padding)
#         model_pt.add_module(layer_name, conv_layer)
#         model_pt.add_module(layer_name + '_activation', torch.nn.ReLU())
#     elif layer_type == 'Dense':
#         linear_layer = torch.nn.Linear(in_features=weights_tf.shape[1], out_features=weights_tf.shape[0])
#         model_pt.add_module(layer_name, linear_layer)
#         model_pt.add_module(layer_name + '_activation', torch.nn.ReLU())
#     else:
#         print(f"Skipping layer {layer_name} ({layer_type}) of unknown type")

#     # Set weights and biases of PyTorch layer
#     weight_names = []
#     weight_values = []
#     bias_names = []
#     bias_values = []
#     for i, param in enumerate(model_pt[-2].parameters()):
#         if i == 0:
#             weight_names.extend([layer.weights.name.split(':')[0]])
#             weight_values.extend([weights_pt])
#             param.data = weights_pt
#         else:
#             bias_names.extend([layer.bias.name.split(':')[0]])
#             bias_values.extend([biases_pt])
#             param.data = biases_pt

#     print(f"Added layer {layer_name} ({layer_type}) with weights {weight_names} and biases {bias_names}")

# print(model_pt)

# import tensorflow as tf
# import numpy as np
# import torch
# from tensorflow.keras.models import load_model

# def layer_to_pytorch(layer):
#     if isinstance(layer, tf.keras.layers.InputLayer):
#         return torch.nn.Identity()
#     elif isinstance(layer, tf.keras.layers.Conv1D):
#         return torch.nn.Conv1d(layer.filters, kernel_size=layer.kernel_size[0], stride=layer.strides[0], padding=layer.padding[0])
#     elif isinstance(layer, tf.keras.layers.Conv2D):
#         return torch.nn.Conv2d(layer.filters, kernel_size=layer.kernel_size, stride=layer.strides, padding=layer.padding)
#     elif isinstance(layer, tf.keras.layers.Conv3D):
#         return torch.nn.Conv3d(layer.filters, kernel_size=layer.kernel_size, stride=layer.strides, padding=layer.padding)
#     elif isinstance(layer, tf.keras.layers.MaxPooling1D):
#         return torch.nn.MaxPool1d(kernel_size=layer.pool_size[0], stride=layer.strides[0], padding=layer.padding[0])
#     elif isinstance(layer, tf.keras.layers.MaxPooling2D):
#         return torch.nn.MaxPool2d(kernel_size=layer.pool_size, stride=layer.strides, padding=layer.padding)
#     elif isinstance(layer, tf.keras.layers.MaxPooling3D):
#         return torch.nn.MaxPool3d(kernel_size=layer.pool_size, stride=layer.strides, padding=layer.padding)
#     elif isinstance(layer, tf.keras.layers.AveragePooling1D):
#         return torch.nn.AvgPool1d(kernel_size=layer.pool_size[0], stride=layer.strides[0], padding=layer.padding[0])
#     elif isinstance(layer, tf.keras.layers.AveragePooling2D):
#         return torch.nn.AvgPool2d(kernel_size=layer.pool_size, stride=layer.strides, padding=layer.padding)
#     elif isinstance(layer, tf.keras.layers.AveragePooling3D):
#         return torch.nn.AvgPool3d(kernel_size=layer.pool_size, stride=layer.strides, padding=layer.padding)
#     elif isinstance(layer, tf.keras.layers.Flatten):
#         return torch.nn.Flatten()
#     elif isinstance(layer, tf.keras.layers.Dense):
#         return torch.nn.Linear(layer.input_shape[-1], layer.units)
#     elif isinstance(layer, tf.keras.layers.Dropout):
#         return torch.nn.Dropout(layer.rate)
#     elif isinstance(layer, tf.keras.layers.BatchNormalization):
#         return torch.nn.BatchNorm2d(layer.input_shape[-1])
#     elif isinstance(layer, tf.keras.layers.Activation):
#         if layer.activation == tf.keras.activations.relu:
#             return torch.nn.ReLU()
#         elif layer.activation == tf.keras.activations.sigmoid:
#             return torch.nn.Sigmoid()
#         elif layer.activation == tf.keras.activations.tanh:
#             return torch.nn.Tanh()
#         elif layer.activation == tf.keras.activations.softmax:
#             return torch.nn.Softmax(dim=1)
#         else:
#             raise NotImplementedError(f"The activation {layer.activation} is not implemented.")
#     elif isinstance(layer, tf.keras.layers.Reshape):
#         return torch.nn.Identity()
#     else:
#         raise NotImplementedError(f"The layer {layer} is not implemented.")


# # Load the Keras model
# keras_model = load_model('handrecognition_model.h5')

# # Create a PyTorch model with the same architecture as the Keras model
# input_shape = keras_model.layers[0].input_shape[1:]
# output_shape = keras_model.layers[-1].output_shape[1:]
# pytorch_model = torch.nn.Sequential(
#     *[layer_to_pytorch(layer) for layer in keras_model.layers]
# )

# # Assign the weights of each layer from the Keras model to the PyTorch model
# for keras_layer, pytorch_layer in zip(keras_model.layers, pytorch_model):
#     if isinstance(keras_layer, (tf.keras.layers.Conv1D, tf.keras.layers.Conv2D)):
#         pytorch_layer.weight.data = torch.from_numpy(keras_layer.get_weights()[0])
#         pytorch_layer.bias.data = torch.from_numpy(keras_layer.get_weights()[1])
#     elif isinstance(keras_layer, (tf.keras.layers.Dense, tf.keras.layers.Conv3D)):
#         pytorch_layer.weight.data = torch.from_numpy(keras_layer.get_weights()[0].T)
#         pytorch_layer.bias.data = torch.from_numpy(keras_layer.get_weights()[1])
#     elif isinstance(keras_layer, tf.keras.layers.BatchNormalization):
#         pytorch_layer.weight.data = torch.from_numpy(keras_layer.get_weights()[0])
#         pytorch_layer.bias.data = torch.from_numpy(keras_layer.get_weights()[1])
#         pytorch_layer.running_mean.data = torch.from_numpy(keras_layer.get_weights()[2])
#         pytorch_layer.running_var.data = torch.from_numpy(keras_layer.get_weights()[3])

# # Save the PyTorch model to a file
# torch.save(pytorch_model.state_dict(), 'handrecognition_model.pth')

# import torch
# import torch.nn as nn
# import h5py

# model = nn.Sequential(
#     nn.Conv2d(1, 32, kernel_size=5),
#     nn.ReLU(),
#     nn.MaxPool2d(2, 2),
#     nn.Conv2d(32, 64, kernel_size=3),
#     nn.ReLU(),
#     nn.MaxPool2d(2, 2),
#     nn.Conv2d(64, 64, kernel_size=3),
#     nn.ReLU(),
#     nn.MaxPool2d(2, 2),
#     nn.Flatten(),
#     nn.Linear(64 * 13 * 38, 128),
#     nn.ReLU(),
#     nn.Linear(128, 10),
#     nn.Softmax(dim=1)
# )

# with h5py.File('handrecognition_model.h5', 'r') as f:
#     weights = []
#     for layer in f.keys():
#         weights.append(torch.tensor(f[layer][:]))

# # Load the weights into the PyTorch model
# state_dict = model.state_dict()
# for i, (name, param) in enumerate(state_dict.items()):
#     state_dict[name] = weights[i]
# model.load_state_dict(state_dict)

# print(model)

import torch
import torch.nn as nn
import keras

# Define the model architecture
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(31616, 128)
        self.relu4 = nn.ReLU()
        self.linear2 = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        # x = x[:5760]
        x = self.linear1(x)
        x = self.relu4(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x

# Load the pretrained weights from the Keras model
keras_model = keras.models.load_model('handrecognition_model.h5')
keras_weights = keras_model.get_weights()

# Transfer the weights to the PyTorch model
pytorch_model = MyModel()
with torch.no_grad():
    pytorch_model.conv1.weight = nn.Parameter(torch.Tensor(keras_weights[0].transpose(3, 2, 0, 1)))
    pytorch_model.conv1.bias = nn.Parameter(torch.Tensor(keras_weights[1]))
    pytorch_model.conv2.weight = nn.Parameter(torch.Tensor(keras_weights[2].transpose(3, 2, 0, 1)))
    pytorch_model.conv2.bias = nn.Parameter(torch.Tensor(keras_weights[3]))
    pytorch_model.conv3.weight = nn.Parameter(torch.Tensor(keras_weights[4].transpose(3, 2, 0, 1)))
    pytorch_model.conv3.bias = nn.Parameter(torch.Tensor(keras_weights[5]))
    
    new_shape = (128, 31616)
    k = torch.Tensor(keras_weights[6].T)
    num_zeros = new_shape[1] - 5 * k.shape[1]
    zeros_matrix = torch.zeros(k.shape[0], num_zeros)
    new_matrix = torch.cat([k, k, k, k, k, zeros_matrix], dim=1)
    print(new_matrix.shape)

    pytorch_model.linear1.weight = nn.Parameter(torch.Tensor(new_matrix))
    print("6th layer shape: ", keras_weights[6].shape)
    pytorch_model.linear1.bias = nn.Parameter(torch.Tensor(keras_weights[7]))
    pytorch_model.linear2.weight = nn.Parameter(torch.Tensor(keras_weights[8].T))
    pytorch_model.linear2.bias = nn.Parameter(torch.Tensor(keras_weights[9]))

torch.save(pytorch_model.state_dict(), 'handrecognition_model.pth')
