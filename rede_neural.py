import numpy as np

class RedeNeural:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = np.random.uniform(-1, 1, (input_size, hidden_size))
        self.weights_hidden_output = np.random.uniform(-1, 1, (hidden_size, output_size))

    def forward(self, x):
        hidden_layer_input = np.dot(x, self.weights_input_hidden)
        hidden_layer_output = self.sigmoid(hidden_layer_input)
        output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_output)
        output = self.sigmoid(output_layer_input)
        return output

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def set_weights(self, weights_input_hidden, weights_hidden_output):
        self.weights_input_hidden = weights_input_hidden
        self.weights_hidden_output = weights_hidden_output