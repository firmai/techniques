"""

 NeuralNetworkClass.py  (author: Anson Wong / git: ankonzoid)

"""
import numpy as np
import random
from math import exp

class NeuralNetwork:
    #
    # Initialize
    #
    def __init__(self, n_input=None, n_output=None, n_hidden_nodes=None):
        self.n_input = n_input  # number of features
        self.n_output = n_output  # number of classes
        self.n_hidden_nodes = n_hidden_nodes  # number of hidden nodes/layers
        self.network = self._build_network()

    #
    # Train network
    #
    def train(self, X_train, y_train, l_rate=0.5, n_epochs=1000):

        for epoch in range(n_epochs):
            for (x, y) in zip(X_train, y_train):
                # Forward-pass training example into network (updates node output)
                self._forward_pass(x)
                # Create target output
                y_target = np.zeros(self.n_output, dtype=np.int)
                y_target[y] = 1
                # Backward-pass error into network (updates node delta)
                self._backward_pass(y_target)
                # Update network weights (using updated node delta and node output)
                self._update_weights(x, l_rate=l_rate)

    #
    # Predict most probable class labels for a data set X
    #
    def predict(self, X):

        y_predict = np.zeros(len(X), dtype=np.int)
        for i, x in enumerate(X):
            output = self._forward_pass(x)  # output class probabilities
            y_predict[i] = np.argmax(output)  # predict highest prob class

        return y_predict

    # ==============================
    #
    # Internal functions
    #
    # ==============================

    #
    # Build neural network via settings weights between nodes
    # Note: we have no bias terms here
    #
    def _build_network(self):

        # Connect input nodes with outputs nodes using weights
        def _build_layer(n_input, n_output):
            layer = list()
            for idx_out in range(n_output):
                weights = list()
                for idx_in in range(n_input):
                    weights.append(random.random())
                layer.append({"weights": weights,
                              "output": None,
                              "delta": None})
            return layer

        # Build weights: input layer -> hidden layer(s)  -> output layer
        n_hidden_layers = len(self.n_hidden_nodes)
        network = list()
        if n_hidden_layers == 0:
            network.append(_build_layer(self.n_input, self.n_output))
        else:
            network.append(_build_layer(self.n_input, self.n_hidden_nodes[0]))
            for i in range(1,n_hidden_layers):
                network.append(_build_layer(self.n_hidden_nodes[i-1],
                                            self.n_hidden_nodes[i]))
            network.append(_build_layer(self.n_hidden_nodes[n_hidden_layers-1],
                                        self.n_output))

        return network

    #
    # Forward-pass input -> output and save to network node values
    # This updates: node['output']
    #
    def _forward_pass(self, x):

        # Weighted sum of inputs with no bias term for our activation
        def activate(weights, inputs):
            activation = 0.0
            for i in range(len(weights)):
                activation += weights[i] * inputs[i]
            return activation

        # Perform forward-pass through network and update node outputs
        input = x
        for layer in self.network:
            output = list()
            for node in layer:
                # Compute activation and apply transfer to it
                activation = activate(node['weights'], input)
                node['output'] = self._transfer(activation)
                output.append(node['output'])
            input = output

        return input

    #
    # Backward-pass error into neural network
    # The loss function is assumed to be L2-error.
    # This updates: node['delta']
    #
    def _backward_pass(self, target):

        # Perform backward-pass through network to update node deltas
        n_layers = len(self.network)
        for i in reversed(range(n_layers)):
            layer = self.network[i]

            # Compute errors either:
            # - explicit target output difference on last layer
            # - weights sum of deltas from frontward layers
            errors = list()
            if i == n_layers - 1:
                # Last layer: errors = target output difference
                for j, node in enumerate(layer):
                    error = target[j] - node['output']
                    errors.append(error)
            else:
                # Previous layers: error = weights sum of frontward node deltas
                for j, node in enumerate(layer):
                    error = 0.0
                    for node in self.network[i + 1]:
                        error += node['weights'][j] * node['delta']
                    errors.append(error)

            # Update delta using our errors
            # The weight update will be:
            # dW = learning_rate * errors * transfer' * input
            #    = learning_rate * delta * input
            for j, node in enumerate(layer):
                node['delta'] = errors[j] * self._transfer_derivative(node['output'])

    #
    # Update network weights with error
    # This updates: node['weights']
    #
    def _update_weights(self, x, l_rate=0.3):

        # Update weights forward layer by layer
        for i_layer, layer in enumerate(self.network):

            # Choose previous layer output to update current layer weights
            if i_layer == 0:
                inputs = x
            else:
                inputs = np.zeros(len(self.network[i_layer - 1]))
                for i_node, node in enumerate(self.network[i_layer - 1]):
                    inputs[i_node] = node['output']

            # Update weights using delta rule for single layer neural network
            # The weight update will be:
            # dW = learning_rate * errors * transfer' * input
            #    = learning_rate * delta * input
            for node in layer:
                for j, input in enumerate(inputs):
                    dW = l_rate * node['delta'] * input
                    node['weights'][j] += dW

    # Transfer function (sigmoid)
    def _transfer(self, x):
        return 1.0/(1.0+exp(-x))

    # Transfer function derivative (sigmoid)
    def _transfer_derivative(self, transfer):
        return transfer*(1.0-transfer)