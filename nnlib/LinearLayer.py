import numpy as np
from .PLayer import PLayer

class LinearLayer(PLayer):
    """
    A linear layer is a fully connected layer; every input feature is connected to every output neuron with its own weight,
    and each output neuron adds a bias. It stores two things: the weights and the bias. The weights are stored in a 2D
    tensor of shape (input_features, output_features), and the bias is stored in a 1D tensor of shape (output_features,).
    The forward pass computes the linear transformation of the input features using the weights and bias. The backward pass
    computes the gradients of the weights and bias with respect to the loss, which are used to update the parameters during
    training.
    """

    def __init__(self, in_features, out_features):
        """
        Initialize the LinearLayer with the given number of input and output features.

        Parameters:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
        """
        super().__init__(name="LinearLayer")
        self.W = np.random.randn( in_features, out_features) * np.sqrt(1.0 / in_features)  # Weight matrix of shape (in_features, out_features)
        self.b = np.zeros(out_features, )  # Bias vector of shape (out_features,)
        self.x = None  # To store the input for backward pass
        self.dW = None  # To store the gradient of weights
        self.db = None  # To store the gradient of bias


    def forward(self, x):
        """
        Perform the forward pass of the linear layer.

        Parameters:
            x (np.ndarray): Input data of shape (n_samples, in_features).

        Returns:
            np.ndarray: Output data of shape (n_samples, out_features).
        """
        self.x = x  # Store input for backward pass
        return np.dot(x, self.W) + self.b  # Linear transformation

    def backward(self, grad):
        """
        Perform the backward pass of the linear layer.

        Parameters:
            grad (np.ndarray): Gradient of the loss with respect to the output of this layer,
                               of shape (n_samples, out_features).

        Returns:
            np.ndarray: Gradient of the loss with respect to the input of this layer,
                        of shape (n_samples, in_features).
        """
        self.dW = np.dot(self.x.T, grad)  # Gradient of weights
        self.db = np.sum(grad, axis=0)  # Gradient of bias
        return np.dot(grad, self.W.T)  # Gradient of input

    def params(self):
        """
        Return the weights and bias of the layer together with their gradients.

        The arrays are returned by reference (not copied), so updating them in place
        (e.g. p -= lr * g) changes the layer's own W and b.

        Returns:
            list[tuple[np.ndarray, np.ndarray]]: [(W, dW), (b, db)], where W has shape
                                                 (in_features, out_features) and b has
                                                 shape (out_features,)
        """
        return [(self.W, self.dW), (self.b, self.db)]
    



