import tensorflow as tf 
from tensorflow.python.keras import layers, models

from .data_preprocessing import X 


shape_ = X.shape
input_dim = shape_[1]


"""  
A Sequential model is not appropriate when:

Your model has multiple inputs or multiple outputs
Any of your layers has multiple inputs or multiple outputs
You need to do layer sharing
You want non-linear topology (e.g. a residual connection, a multi-branch model)
"""
model = models.Sequential([
    layers.Input(shape = (input_dim, )),
    layers.Dense(128, activation = "relu"), # This is a fully connected (dense) layer that transforms the input data.
                                            # 128 neurons: There are 128 units, each learning a weighted combination of the inputs plus a bias.
    layers.Dropout(0.2), # This layer is used for regularization to help prevent overfitting.
    layers.Dense(64, activation = "relu"), # Another fully connected layer that further processes the data.
                                           # 64 neurons: Reduces the dimensionality compared to the previous dense layer, which can help in learning a more compact representation.
    layers.Dense(1, activation = "sigmoid") # The final output layer.
                                            # 1 neuron: Outputs a single value.
])


model.compile(
    optimizer = "adam",
    loss = "binary_crossentropy",
    metrics = ["accuracy"]
)

model.summary()