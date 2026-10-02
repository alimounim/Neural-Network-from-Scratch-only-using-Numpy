class PLayer:
    """
    Base class for all layers in the neural network.

    Every layer (Linear, Sigmoid, ReLU, Tanh, loss functions, Sequential)
    inherits from this class and must implement forward() and backward().
    Because all layers share this interface, a network can call forward
    and backward on any layer without knowing its specific type.
    """

    def __init__(self, name= None):
        self.name = name

    def forward(self, x):
        """
        Compute the output of the layer for a given input.

        Parameters:
            x (np.ndarray): input to the layer, shape (n_samples, n_features)

        Returns:
            np.ndarray: output of the layer
        """

        raise NotImplementedError("Forward method not implemented in base Layer class.")

    def backward(self, grad):
        """
        Compute the backward pass of the layer.

        Parameters:
            grad (np.ndarray): gradient of the loss with respect to this
                               layer's output (from the layer ahead)

        Returns:
            np.ndarray: gradient of the loss with respect to this layer's
                        input (passed to the layer behind)
        """

        raise NotImplementedError("Backward method not implemented in base Layer class.")

    