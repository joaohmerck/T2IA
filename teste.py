import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Inicialização aleatória dos pesos para as camadas de entrada->oculta e oculta->saída
        self.weights_input_hidden = np.random.uniform(-1, 1, (input_size, hidden_size))
        self.weights_hidden_output = np.random.uniform(-1, 1, (hidden_size, output_size))

    def sigmoid(self, x):
        # Função de ativação Sigmóide
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        # Propagação direta
        # Camada oculta
        hidden_layer_input = np.dot(x, self.weights_input_hidden)
        hidden_layer_output = self.sigmoid(hidden_layer_input)

        # Camada de saída
        output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_output)
        output = self.sigmoid(output_layer_input)

        return output
