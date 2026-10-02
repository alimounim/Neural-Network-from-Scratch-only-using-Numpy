import numpy as np
from PLayer import PLayer

class Tanh(PLayer):
    """
    Hyperbolic tangent (tanh) activation function layer.
    Applies tanh element-wise to the input: tanh(x) = (e^x - e^-x) / (e^x + e^-x).
    Squashes values into the range (-1, 1) and is centered at 0.
    This layer has no learnable parameters.
    """

    def __init__(self):
        """
        Initialize the Tanh layer.

        Attributes:
            out (np.ndarray): Output of the forward pass. Stored so it can be
                              reused to compute the derivative in backward.
        """
        super().__init__(name="Tanh")
        self.out = None  # Will hold the output of the forward pass


    def forward(self, x):
        """
        Perform the forward pass of the tanh activation function.

        Parameters:
            x (np.ndarray): Input data of any shape.

        Returns:
            np.ndarray: tanh(x) applied element-wise, same shape as input.
        """
        self.out = np.tanh(x)  # Compute tanh and store the output
        return self.out # Return the output of the forward pass


    def backward(self, grad):
        """
        Perform the backward pass of the tanh activation function.

        The derivative of tanh is 1 - tanh(x)^2, which is computed from the
        output stored during the forward pass.

        Parameters:
            grad (np.ndarray): Gradient of the loss with respect to the output of this layer,
                               same shape as output.

        Returns:
            np.ndarray: Gradient of the loss with respect to the input of this layer,
                        same shape as input.
        """
        return grad * (1 - self.out ** 2)  # Compute the gradient using the derivative of tanh

